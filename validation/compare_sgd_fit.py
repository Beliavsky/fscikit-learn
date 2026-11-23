"""SGD warm/explicit initialization and nine classification losses, sklearn 1.6.1."""
import argparse,atexit,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import numpy as np
import sklearn
from sklearn.linear_model import SGDRegressor,SGDClassifier
from compare_image_graph import fields
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vector(a):return ' '.join(format(v,'.17g') for v in np.asarray(a).ravel(order='F'))+'\n'
REJECTIONS={
 'regressor_coef_shape':'SGD coefficient initializer shape mismatch',
 'regressor_coef_nan':'SGD coefficient initializer must be finite','regressor_coef_inf':'SGD coefficient initializer must be finite',
 'regressor_bias_nan':'SGD intercept initializer must be finite','regressor_bias_inf':'SGD intercept initializer must be finite',
 'regressor_warm_width':'SGD warm_start feature count mismatch',
 'regressor_warm_average':'SGD averaging cannot change during warm_start',
 'classifier_coef_shape':'SGDClassifier coefficient initializer shape mismatch','classifier_bias_shape':'SGDClassifier intercept initializer shape mismatch',
 'classifier_multi_vector':'SGDClassifier coefficient initializer shape mismatch',
 'classifier_multi_scalar':'SGDClassifier intercept initializer shape mismatch',
 'classifier_coef_nan':'SGD coefficient initializer must be finite','classifier_coef_inf':'SGD coefficient initializer must be finite',
 'classifier_bias_nan':'SGD intercept initializer must be finite','classifier_bias_inf':'SGD intercept initializer must be finite',
 'classifier_warm_width':'SGDClassifier coefficient initializer shape mismatch','classifier_warm_classes':'SGDClassifier coefficient initializer shape mismatch',
 'classifier_epsilon_negative':'SGD epsilon must be nonnegative','classifier_epsilon_nan':'nonfinite SGD controls',
 'log_unfitted':'SGDClassifier is not fitted','log_loss_unsupported':'SGD loss does not provide probabilities'}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path,required=True)
 parser.add_argument('--profile',default='checked');parser.add_argument('--report',type=Path,default=ROOT/'validation/results/sgd_fit.json')
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
 def fixture(seed,kind,loss,penalty,rate,average,shuffle,index,workflow='warm'):
  nonlocal fixtures,stages_total,nonfinite
  rng=np.random.RandomState(seed+index*13);n=17;p=4
  x=rng.normal(scale=.15,size=(n,p));y=x@rng.normal(size=p)+.3 if not kind else np.array([3+7*(i%kind) for i in range(n)])
  w=rng.uniform(.1,2,n);w[0]=0;channels=max(1,kind) if kind!=2 else 1
  epsilon=0. if index%7==0 else .23
  intercept=index%2==0
  if workflow=='stream':intercept=average==0
  options=dict(loss=loss,penalty=None if penalty=='none' else penalty,learning_rate=rate,alpha=.04,eta0=.02,
    power_t=.3,l1_ratio=.37,epsilon=epsilon,random_state=seed,tol=None,n_iter_no_change=2,shuffle=shuffle,
    average=average if average else False,fit_intercept=intercept,max_iter=4)
  source=(SGDClassifier if kind else SGDRegressor)(**options)
  stages=[(0,4,False,index%4),(0,3,True,(index+1)%4),(0,2,False,(index+2)%4)]
  if workflow=='stream':stages=[(0,2,False,3),(1,1,False,0),(0,3,True,0),(1,1,False,0)]
  if workflow=='public':stages=[(0,2,False,3),(2,3,True,0)]
  payload=vector([n,p,kind,len(stages),seed])+f'{loss} {penalty} {rate}\n'+vector([.04,.02,.3,.37,epsilon,-1])+vector(x)+vector(y)+vector(w)
  expected=[]
  for stage,(op,epochs,warm,mask) in enumerate(stages):
   if kind==2:
    rank_mode=(index+stage)%5
    if rank_mode==1:mask|=4
    if rank_mode==2:mask|=8
    if rank_mode==3:mask|=16
    if rank_mode==4:mask|=12
   initial=rng.normal(scale=.12,size=(channels,p));bias=rng.normal(scale=.2,size=channels)
   payload+=vector([op,epochs,average,int(shuffle),int(intercept),int(warm),mask])+vector(initial)+vector(bias)
   source.max_iter=epochs;source.warm_start=warm
   if op==2:source.coef_=initial.copy() if kind else initial[0].copy();source.intercept_=bias.copy()
   if op==1:
    source.partial_fit(x,y,sample_weight=w,**(dict(classes=np.unique(y)) if kind else {}))
   else:
    kwargs={}
    if mask&1:
     kwargs['coef_init']=initial.copy() if kind else initial[0].copy()
     if mask&4:kwargs['coef_init']=initial.ravel().copy()
     if mask&16:kwargs['coef_init']=initial.reshape((2,p//2),order='F').copy()
    if mask&2:kwargs['intercept_init']=float(bias[0]) if mask&8 else bias.copy()
    source.fit(x,y,sample_weight=w,**kwargs)
   expected.extend([np.atleast_2d(source.coef_).copy(),source.intercept_[:,None].copy(),
    source.decision_function(x).reshape(n,-1) if kind else source.predict(x)[:,None],
    source.predict(x)[:,None] if kind else np.empty((0,1)),
    source.predict_proba(x) if kind and loss in ['log_loss','modified_huber'] else np.empty((0,0)),
    source.predict_log_proba(x) if kind and loss in ['log_loss','modified_huber'] else np.empty((0,0)),
    np.array([source.n_iter_,source.t_,source.n_features_in_])[:,None]])
  actual=run(payload);assert len(actual)==len(expected)
  label=f'{workflow}/{seed}/{kind}/{loss}/{penalty}/{rate}/{average}/{shuffle}/{index}'
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
   print(f'seed {seed} kind {kind}: warm/initialized full loss grid passed',flush=True)
 # Retain history across partial_fit/warm-fit and replace published arrays.
 for workflow,kind,loss,average,shuffle in itertools.product(['stream','public'],[0,2,3],
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
  source_sha256={f:sha(source/f) for f in ['linear_model/_stochastic_gradient.py','linear_model/_sgd_fast.pyx.tp','utils/_weight_vector.pyx.tp']})
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(f'{len(checks)} groups across {fixtures} fixtures and {stages_total} stages; {len(REJECTIONS)} diagnostics passed',flush=True)
if __name__=='__main__':main()
