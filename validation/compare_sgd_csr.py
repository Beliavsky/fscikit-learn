"""CSR SGD updates, warm initialization and mixed dense/sparse streams, sklearn 1.6.1."""
import argparse,atexit,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import numpy as np
import sklearn
from sklearn.linear_model import SGDRegressor,SGDClassifier
from scipy.sparse import csr_matrix
from compare_image_graph import fields
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vector(a):return ' '.join(format(v,'.17g') for v in np.asarray(a).ravel(order='F'))+'\n'
REJECTIONS={
 'unallocated':'CSR arrays are unallocated','empty':'X must be nonempty','zero_width':'X must be nonempty',
 'data_nan':'X contains NaN or infinity','data_inf':'X contains NaN or infinity',
 'pointer':'CSR pointer endpoints are invalid','column':'CSR column index is out of bounds',
 'target_count':'X and y sample counts differ','target_nan':'X contains NaN or infinity',
 'weight_count':'SGD sample weight count mismatch','weight_nan':'nonfinite SGD sample weights','weight_negative':'negative SGD sample weights',
 'reg_query_unfitted':'SGDRegressor is not fitted','cls_query_unfitted':'SGDClassifier is not fitted',
 'reg_width':'SGD feature count mismatch','cls_width':'SGDClassifier feature count mismatch',
 'reg_partial_width':'SGD feature count mismatch','cls_partial_width':'SGDClassifier feature count mismatch',
 'cls_target_count':'X and y sample counts differ','cls_initial_shape':'SGDClassifier coefficient initializer shape mismatch',
 'reg_initial_shape':'SGD coefficient initializer shape mismatch','proba_loss':'SGD loss does not provide probabilities',
 'reg_query_nan':'X contains NaN or infinity','cls_query_nan':'X contains NaN or infinity'}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path,required=True)
 parser.add_argument('--profile',default='checked');parser.add_argument('--report',type=Path,default=ROOT/'validation/results/sgd_csr.json')
 args=parser.parse_args();checks=[];fixtures=0;stages_total=0;nonfinite=0
 warnings.simplefilter('ignore')
 process=subprocess.Popen([str(args.executable.resolve()),'batch'],text=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=1)
 atexit.register(lambda:process.terminate() if process.poll() is None else None)
 def run(payload):
  process.stdin.write(payload);process.stdin.flush();out=[]
  while True:
   line=process.stdout.readline();assert line,process.stderr.read()
   if line.strip()=='END':return fields(''.join(out))
   out.append(line)
 def fixture(seed,kind,loss,penalty,rate,average,shuffle,index,workflow='sparse'):
  nonlocal fixtures,stages_total,nonfinite
  rng=np.random.RandomState(seed+index*13);n=17;p=8
  dense=rng.normal(scale=.15,size=(n,p));dense[rng.uniform(size=dense.shape)<.75]=0;dense[0]=0
  pattern=index%5
  if pattern==4:dense[:]=0
  x=csr_matrix(dense)
  if pattern==1:
   x=csr_matrix((dense.ravel(),np.tile(np.arange(p),n),np.arange(0,n*p+1,p)),shape=(n,p))
  if pattern==2:
   for row in range(n):
    lo,hi=x.indptr[row:row+2];order=rng.permutation(hi-lo)
    x.indices[lo:hi]=x.indices[lo:hi][order];x.data[lo:hi]=x.data[lo:hi][order]
  if pattern==3:
   columns=[];data=[];ptr=[0]
   for row in range(n):
    for at in range(x.indptr[row],x.indptr[row+1]):
     columns.extend([int(x.indices[at])]*2);data.extend([x.data[at]/3,2*x.data[at]/3])
    columns.extend([p-1,p-1]);data.extend([.17,-.17]);ptr.append(len(data))
   x=csr_matrix((np.array(data),np.array(columns),np.array(ptr)),shape=(n,p))
  dense=x.toarray();y=dense@rng.normal(size=p)+.3 if not kind else np.array([3+7*(i%kind) for i in range(n)])
  w=rng.uniform(.1,2,n);w[0]=0;channels=max(1,kind) if kind!=2 else 1
  epsilon=0. if index%7==0 else .23
  intercept=index%2==0
  if workflow=='mixed':intercept=average==0
  options=dict(loss=loss,penalty=None if penalty=='none' else penalty,learning_rate=rate,alpha=.04,eta0=.02,
    power_t=.3,l1_ratio=.37,epsilon=epsilon,random_state=seed,tol=None,n_iter_no_change=2,shuffle=shuffle,
    average=average if average else False,fit_intercept=intercept,max_iter=4)
  source=(SGDClassifier if kind else SGDRegressor)(**options)
  stages=[(0,4,False,index%4),(0,3,True,(index+1)%4),(0,2,False,(index+2)%4)]
  if workflow=='mixed':stages=[(0,2,False,3),(1,1,False,0),(0,3,True,0),(1,1,False,0)]
  if workflow=='public':stages=[(0,2,False,3),(2,3,True,0)]
  payload=vector([n,p,kind,len(stages),seed,x.nnz])+f'{loss} {penalty} {rate}\n'+vector([.04,.02,.3,.37,epsilon,-1])+vector(x.indptr+1)+vector(x.indices+1)+vector(x.data)+vector(y)+vector(w)
  expected=[]
  for stage,(op,epochs,warm,mask) in enumerate(stages):
   if kind==2:
    rank_mode=(index+stage)%5
    if rank_mode==1:mask|=4
    if rank_mode==2:mask|=8
    if rank_mode==3:mask|=16
    if rank_mode==4:mask|=12
   initial=rng.normal(scale=.12,size=(channels,p));bias=rng.normal(scale=.2,size=channels)
   train_format=1 if workflow!='mixed' else stage%2
   query_format=stage%2
   training=x if train_format else dense;query=x if query_format else dense
   payload+=vector([op,epochs,average,int(shuffle),int(intercept),int(warm),mask,train_format,query_format])+vector(initial)+vector(bias)
   source.max_iter=epochs;source.warm_start=warm
   if op==2:source.coef_=initial.copy() if kind else initial[0].copy();source.intercept_=bias.copy()
   if op==1:
    source.partial_fit(training,y,sample_weight=w,**(dict(classes=np.unique(y)) if kind else {}))
   else:
    kwargs={}
    if mask&1:
     kwargs['coef_init']=initial.copy() if kind else initial[0].copy()
     if mask&4:kwargs['coef_init']=initial.ravel().copy()
     if mask&16:kwargs['coef_init']=initial.reshape((2,p//2),order='F').copy()
    if mask&2:kwargs['intercept_init']=float(bias[0]) if mask&8 else bias.copy()
    source.fit(training,y,sample_weight=w,**kwargs)
   expected.extend([np.atleast_2d(source.coef_).copy(),source.intercept_[:,None].copy(),
    source.decision_function(query).reshape(n,-1) if kind else source.predict(query)[:,None],
    source.predict(query)[:,None] if kind else np.empty((0,1)),
    source.predict_proba(query) if kind and loss in ['log_loss','modified_huber'] else np.empty((0,0)),
    source.predict_log_proba(query) if kind and loss in ['log_loss','modified_huber'] else np.empty((0,0)),
    np.array([source.n_iter_,source.t_,source.n_features_in_])[:,None]])
  actual=run(payload);assert len(actual)==len(expected)
  label=f'{workflow}-{pattern}/{seed}/{kind}/{loss}/{penalty}/{rate}/{average}/{shuffle}/{index}'
  names=['coefficients','intercepts','decision_or_prediction','labels','probabilities','log_probabilities','clock']
  for j,(got,want) in enumerate(zip(actual,expected)):
   field=names[j%7];stage=j//7
   np.testing.assert_array_equal(np.isneginf(got),np.isneginf(want),err_msg=f'{label}/{stage}/{field} -inf')
   np.testing.assert_array_equal(np.isnan(got),np.isnan(want),err_msg=f'{label}/{stage}/{field} nan')
   np.testing.assert_allclose(got,want,rtol=3e-9,atol=3e-10,err_msg=f'{label}/{stage}/{field}')
   finite=np.isfinite(want);nonfinite+=int(np.sum(~finite))
   checks.append(dict(fixture=label,stage=stage,field=field,max_abs_error=float(np.max(np.abs(got[finite]-want[finite]),initial=0)),passed=True))
  fixtures+=1;stages_total+=len(stages)
 index=0
 for seed in [0,42,2026]:
  for kind in [0,2,3]:
   losses=['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']
   if kind:losses=['hinge','squared_hinge','perceptron','log_loss','modified_huber',*losses]
   for loss,penalty,rate,average,shuffle in itertools.product(losses,['none','l1','l2','elasticnet'],
       ['constant','invscaling','optimal','adaptive'],[0,1,10],[False,True]):
    fixture(seed,kind,loss,penalty,rate,average,shuffle,index);index+=1
   print(f'seed {seed} kind {kind}: CSR full loss/penalty/schedule grid passed',flush=True)
 # Retain history across partial_fit/warm-fit and replace published arrays.
 for workflow,kind,loss,average,shuffle in itertools.product(['mixed','public'],[0,2,3],
     ['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'],[0,1,10],[False,True]):
  fixture(91,kind,loss,'elasticnet','optimal',average,shuffle,index,workflow);index+=1
 process.stdin.close();assert process.wait(timeout=30)==0,process.stderr.read()
 for name,message in REJECTIONS.items():
  proc=subprocess.run([str(args.executable.resolve()),name],text=True,capture_output=True);output=proc.stdout+proc.stderr
  assert proc.returncode!=0 and message in output and 'SIGFPE' not in output and 'Fortran runtime error' not in output,(name,output)
 source=Path(sklearn.__file__).parent
 report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,fixtures=fixtures,stages=stages_total,
  nonfinite_reference_values=nonfinite,checks=checks,rejections=list(REJECTIONS),executable_sha256=sha(args.executable),
  validator_sha256=sha(__file__),parser_sha256=sha(ROOT/'validation/compare_image_graph.py'),native_source_sha256=sha(ROOT/'src/sklearn_sgd.f90'),
  source_sha256={f:sha(source/f) for f in ['linear_model/_stochastic_gradient.py','linear_model/_sgd_fast.pyx.tp','linear_model/_base.py','utils/_weight_vector.pyx.tp','utils/_seq_dataset.pyx.tp']})
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(f'{len(checks)} groups across {fixtures} fixtures and {stages_total} stages; {len(REJECTIONS)} diagnostics passed',flush=True)
if __name__=='__main__':main()
