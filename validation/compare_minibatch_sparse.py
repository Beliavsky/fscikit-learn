"""Sparse/mixed MiniBatchNMF operation sequences versus installed sklearn."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
import scipy,sklearn
from scipy.sparse import csr_matrix
from sklearn.decomposition import MiniBatchNMF
from compare_nmf_sparse import storage

def base_options(seed):
 return dict(n_components=3,init=None,max_iter=5,batch_size=7,random_state=seed,max_no_improvement=None,fresh_restarts_max_iter=7,transform_max_iter=None,beta_loss=2.,tol=0.,alpha_W=0.,alpha_H='same',l1_ratio=0.,forget_factor=.7,fresh_restarts=False)

def fixtures():
 for seed,n,p in [(0,23,7),(7,31,5),(42,28,11),(2026,19,3)]:
  rng=np.random.RandomState(seed);base=.2+abs(rng.normal(size=(n,p)));query=.2+abs(rng.normal(size=(8,p)))
  for case in range(41):
   x=base.copy();q=query.copy();c=base_options(seed);ops=[]
   def add(op,data,custom=None,style='canonical',dense=False,**changes):
    c.update(changes);a=np.array(data,copy=True)
    if op!=5 and c['beta_loss']>0 and case not in [29]:a[rng.random_sample(a.shape)<.3]=0
    obj=a if dense or op==5 else storage(a,style);ops.append((op,obj,c.copy(),custom))
   if case in [1,2,3,4]:c['init']=['random','nndsvd','nndsvda','nndsvdar'][case-1]
   if case in [5,6,37]:c['init']='custom'
   if case==6:c['n_components']='auto'
   if case==7:c['n_components']=p+3
   if case in [8,9,10,11,12,13,14]:c['beta_loss']=[1.,0.,-.5,.5,1.5,2.5,3.][case-8]
   if case==15:c.update(alpha_W=.03,l1_ratio=1.)
   if case==16:c.update(alpha_W=.02,alpha_H=.05)
   if case==17:c.update(alpha_W=.01,alpha_H=.03,l1_ratio=.6)
   if case==18:c['forget_factor']=0.
   if case==19:c['forget_factor']=1.
   if case==20:c.update(fresh_restarts=True,fresh_restarts_max_iter=5,transform_max_iter=10)
   if case==21:c.update(batch_size=1,max_iter=2)
   if case==22:c['batch_size']=n+10
   if case==23:c.update(tol=1e9,max_iter=1000000000,transform_max_iter=4)
   if case in [24,25]:c.update(max_no_improvement=case-23,max_iter=30,init='random');x[:]=0;q[:]=0
   if case==29:c.update(beta_loss=1.,init='random');x[:]=1e-9;q[:]=2e-9
   if case==30:c.update(beta_loss=1.,init='random');x[rng.random_sample(x.shape)<.4]=0;q[rng.random_sample(q.shape)<.4]=0
   if case==31:c.update(beta_loss=0.,fresh_restarts=True,fresh_restarts_max_iter=3,transform_max_iter=7)
   if case==32:c.update(beta_loss=.5,alpha_W=.01,alpha_H=.02,l1_ratio=.5)
   if case==33:c.update(beta_loss=1.,init='nndsvd')
   if case==34:c.update(beta_loss=3.,init='nndsvdar')
   if case==35:c.update(n_components='auto',max_iter=80,batch_size=1024,tol=1e-4,max_no_improvement=10)
   if case==36:c.update(fresh_restarts=True,forget_factor=0.,alpha_H=.01,beta_loss=1.)
   if case==37:c.update(alpha_W=.01,l1_ratio=.5,forget_factor=1.)
   if case==38:c['init']='random'
   if case==39:x=.2+abs(rng.normal(size=(7,17)));q=.2+abs(rng.normal(size=(8,17)))
   if case==40:c.update(n_components='auto',max_iter=40,max_no_improvement=2,tol=.001);x[:]=0;q[:]=0
   if case in [5,6,37,38]:
    k=4 if case==6 else 3;w0=.1+rng.random_sample((len(x),k));h0=.1+rng.random_sample((k,x.shape[1]))
    if case==37:w0[0]=0;h0[0,0]=0
    add(3,x,custom=(w0,h0))
   elif case==26:
    add(2,x[:5]);add(2,x[5:13],n_components=2,beta_loss=3.,forget_factor=0.,batch_size=3);add(2,x[13:],alpha_W=.03)
   elif case==27:add(1,x);add(2,x[:2],alpha_W=.02,alpha_H=.01,l1_ratio=.3,transform_max_iter=1,random_state=seed+2)
   elif case==28:
    add(2,x[:7]);x=np.c_[x,np.ones(len(x))];q=np.c_[q,np.ones(len(q))];add(3,x,n_components=2,random_state=seed+1)
   else:add(3,x)
   k=x.shape[1] if c['n_components']=='auto' else c['n_components'];k=4 if case==6 else 3 if case==26 else k
   for op,data in [(4,q),(5,rng.normal(size=(5,k))),(2,x[:6]),(4,q)]:add(op,data,dense=case in [0,15,19] and op in [2,4])
   yield dict(group='main',seed=seed,case=case),ops
 # Storage and Source-only acceptance boundaries for streaming and inference.
 for beta in [-1.,0.,.5,1.,1.5,2.,2.5,3.]:
  for style in ['unsorted','duplicates','explicit_zero','empty','singleton']:
   for initial_op in [2,3]:
    rng=np.random.RandomState(21);x=.2+rng.random_sample((11,6));q=.2+rng.random_sample((4,6));x[x<.5]=0;q[q<.5]=0
    c=base_options(21);c.update(n_components=2,init='random',max_iter=3,transform_max_iter=4,beta_loss=beta)
    if style=='singleton':
     raw=csr_matrix(([0.],[0],[0,1,1,1,1]),shape=(4,6));first=csr_matrix(.2+rng.random_sample((11,6)))
    else:raw=storage(q,style);first=storage(x,style)
    ops=[(initial_op,first,c.copy(),None),(4,raw,c.copy(),None),(2,raw,c.copy(),None)]
    yield dict(group='storage',beta=beta,style=style,initial_op=initial_op),ops
 for custom in [False,True]:
  for operation in [2,3]:
   c=base_options(21);c.update(n_components=2,init='custom' if custom else 'random',max_iter=3,beta_loss=2.)
   x=np.ones((7,4));x[0,0]=-.1;h=np.full((2,4),.2);w=np.full((7,2),.3)
   yield dict(group='signed',custom=custom,operation=operation),[(operation,csr_matrix(x),c,(w,h) if custom else None),(4,csr_matrix(np.ones((3,4))),c,None)]

def fl(a):return ' '.join(format(float(v),'.17g') for v in np.asarray(a).ravel())
def serialize(ops):
 lines=[str(len(ops))]
 for op,data,c,custom in ops:
  sparse=isinstance(data,csr_matrix);n,p=data.shape
  lines.append(f'{op} {n} {p} {data.nnz if sparse else 0} {0 if sparse else 1}')
  lines.append(' '.join(map(str,[0 if c['n_components']=='auto' else c['n_components'],c['max_iter'],c['batch_size'],c['random_state'],-1 if c['max_no_improvement'] is None else c['max_no_improvement'],c['fresh_restarts_max_iter'],0 if c['transform_max_iter'] is None else c['transform_max_iter']])))
  lines.append(fl([c['beta_loss'],c['tol'],c['alpha_W'],-1 if c['alpha_H']=='same' else c['alpha_H'],c['l1_ratio'],c['forget_factor']]))
  lines.append(f"{c['init'] or 'auto'} {'T' if c['fresh_restarts'] else 'F'} {0 if custom is None else custom[0].shape[1]}")
  if custom is not None:lines.extend(fl(row) for row in custom[0]);lines.extend(fl(row) for row in custom[1])
  if sparse:lines.extend([' '.join(map(str,data.indptr+1)),' '.join(map(str,data.indices+1)),fl(data.data)])
  else:lines.extend(fl(row) for row in data)
 return '\n'.join(lines)+'\n'

def source(ops):
 model=MiniBatchNMF();expected={};step=0;error=None
 try:
  with warnings.catch_warnings(),np.errstate(all='ignore'):
   warnings.simplefilter('ignore')
   for step,(op,data,c,custom) in enumerate(ops,1):
    model.set_params(**c);data=data.copy();kw={} if custom is None else dict(W=custom[0].copy(),H=custom[1].copy());prefix=f'{step}_';result=None
    if op==1:model.fit(data,**kw)
    elif op==2:model.partial_fit(data,**kw)
    elif op==3:result=model.fit_transform(data,**kw)
    elif op==4:result=model.transform(data)
    else:result=model.inverse_transform(data)
    if result is not None:expected[prefix+'result']=np.asarray(result).ravel(order='F').tolist()
    if isinstance(data,csr_matrix):expected[prefix+'input_preserved']=[1.]
    expected[prefix+'components']=model.components_.ravel(order='F').tolist();expected[prefix+'numerator']=model._components_numerator.ravel(order='F').tolist();expected[prefix+'denominator']=model._components_denominator.ravel(order='F').tolist()
    expected[prefix+'meta']=[model.n_features_in_,model.n_components_,model._batch_size,model._transform_max_iter,model.n_steps_,getattr(model,'n_iter_',0)]
    expected[prefix+'resolved']=[model._beta_loss,model._gamma,model._rho]
    expected[prefix+'convergence']=[np.nan if getattr(model,'_ewa_cost',None) is None else model._ewa_cost,np.nan if getattr(model,'_ewa_cost_min',None) is None else model._ewa_cost_min,getattr(model,'_no_improvement',0)]
    expected[prefix+'error']=[getattr(model,'reconstruction_err_',np.nan)]
 except Exception as exc:error=f'{type(exc).__name__}: {exc}'
 return expected,error,step

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();binary=hashlib.sha256(a.executable.read_bytes()).hexdigest();records=[]
 for index,(config,ops) in enumerate(fixtures()):
  text=serialize(ops);inp=a.report.with_suffix('.in');out=a.report.with_suffix('.out');inp.write_text(text);p=subprocess.run([str(a.executable),str(inp),str(out)],capture_output=True,text=True)
  expected,error,step=source(ops);actual={};differences=[]
  if p.returncode==0 and error is None:
   for line in out.read_text().splitlines():
    key,*values=line.split();actual[key]=np.array(values,dtype=float).tolist()
   if actual.keys()!=expected.keys():differences.append('fields')
   for key in actual.keys()&expected.keys():
    tol=0 if key.endswith(('meta','input_preserved')) else 3e-8
    if np.shape(actual[key])!=np.shape(expected[key]) or not np.allclose(actual[key],expected[key],atol=tol,rtol=tol,equal_nan=True):differences.append(key)
  elif (p.returncode==0)!=(error is None):differences.append('acceptance')
  stages=[int(s.split()[-1]) for s in p.stdout.splitlines() if s.startswith('minibatch_stage ')];native_step=stages[-1] if stages else None
  if error is not None and p.returncode==1 and native_step!=step:differences.append('error_step')
  if p.returncode not in [0,1] or p.returncode==1 and 'fscikit-learn:' not in p.stdout:differences.append('abnormal_native_exit')
  records.append(dict(index=index,config=config,input=text,source=expected,source_error=error,source_step=step,native=actual,native_step=native_step,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,differences=sorted(differences)))
 assert binary==hashlib.sha256(a.executable.read_bytes()).hexdigest()
 path=Path(sklearn.__file__).parent/'decomposition/_nmf.py';report=dict(sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,configurations=len(records),source_sha256={'decomposition/_nmf.py':hashlib.sha256(path.read_bytes()).hexdigest()},native_sha256=binary,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),failures=[r['index'] for r in records if r['differences']],records=records)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('CSR MiniBatchNMF',len(records),'strict failures',report['failures']);raise SystemExit(bool(report['failures']))
if __name__=='__main__':main()
