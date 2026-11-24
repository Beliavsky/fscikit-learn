"""Source-seeded Nystroem: direct full-rank fields and singular SVD invariants."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy.linalg as la
import sklearn
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import pairwise_kernels
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
KERNELS=['rbf','linear','poly','polynomial','cosine','laplacian','sigmoid','chi2','additive_chi2','precomputed','callable','callable']
def callback(a,b):return a@b+.2*a[0]-.1*b[0]+1
def callback_rbf(a,b):return np.exp(-.3*np.sum((a-b)**2))
def config(**kw):return dict(n=9,p=4,m=5,k=3,seed=0,kind=0,gamma=-1.,coef0=1.,degree=3,query_gamma=-1.,mutate=0,scenario='random',mode=0)|kw
def fixtures():
 for seed,kind,(n,p,k),scenario in itertools.product([0,7,42,2026,2147483647],range(12),[(9,4,3),(7,3,10)],['random','zero','constant']):
  if kind==9:
   if (n,p,k)!=(9,4,3):continue
   n=p=5;k=7
  yield config(seed=seed,kind=kind,n=n,p=p,k=k,scenario=scenario)
 for kind,gamma,coef0 in itertools.product(range(9),[0.,.7,3.],[-1.,.5]):
  yield config(kind=kind,gamma=gamma,query_gamma=gamma,coef0=coef0,degree=2)
 for scenario in ['diagonal','rank_diagonal']:
  yield config(kind=9,n=5,p=5,k=7,scenario=scenario)
 for kind in [0,1,4,5,6,7,8]:yield config(kind=kind,query_gamma=.4,mutate=17)
def inputs(c):
 rng=np.random.RandomState(c['seed']%10000+19);x=rng.uniform(.1,3,(c['n'],c['p']));q=rng.uniform(.1,3,(c['m'],c['p']))
 if c['scenario']=='zero':x.fill(0);q.fill(0)
 if c['scenario']=='constant':x.fill(.7);q.fill(1.1)
 if c['kind']==9 and c['scenario']=='random' and c['n']==c['p']:x+=np.eye(c['n'])*3
 if c['scenario'] in ['diagonal','rank_diagonal']:
  x=np.diag([1e10,1e-8,-2.,1. if c['scenario']=='diagonal' else 0.,3.])
 return x,q
def model_for(c):
 kernel=KERNELS[c['kind']] if 0<=c['kind']<len(KERNELS) else 'unknown'
 if c['kind']==10:kernel=callback
 if c['kind']==11:kernel=callback_rbf
 options=dict(kernel=kernel,n_components=c['k'],random_state=c['seed'])
 if c['kind']<9:options.update(gamma=None if c['gamma']==-1 else c['gamma'],coef0=c['coef0'],degree=c['degree'])
 if c['kind']>=9:
  if c['gamma']!=-1:options['gamma']=c['gamma']
  if c['coef0']!=1:options['coef0']=c['coef0']
  if c['degree']!=3:options['degree']=c['degree']
 if c['mode']==7:options['kernel']=lambda a,b:np.nan
 if c['mode']==8:options['kernel']=None
 return Nystroem(**options)
def oracle(c):
 x,q=inputs(c);m=model_for(c);original=x.copy();original_q=q.copy();state=np.random.get_state()
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');z=m.fit_transform(x)
  params=m._get_kernel_params();g=pairwise_kernels(m.components_,metric=m.kernel,filter_params=True,**params)
  singular=la.svdvals(g)
  if c['query_gamma']!=-1 or c['gamma']!=-1:m.gamma=None if c['query_gamma']==-1 else c['query_gamma']
  if c['mutate']:m.n_components=c['mutate']
  query=m.transform(q)
 regular=bool(singular[-1]>max(1e-10,1e-7*singular[0]))
 fields=dict(indices=m.component_indices_+1,components=m.components_,normalization=m.normalization_,norm_gram=m.normalization_.T@m.normalization_,train=z,query=query,train_gram=z@z.T,query_gram=query@query.T,cross_gram=z@query.T,metadata=[m.n_features_in_,len(m.component_indices_),m.n_components],preserved=[int(np.array_equal(x,original)),int(np.array_equal(q,original_q)),int(all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state())))],repeat=[1],svd_status=[0],singular=singular)
 return x,q,g,fields,regular
def payload(c,a):
 x,q,g=a[:3]
 return ' '.join(str(c[k]) for k in ['n','p','m','k','seed','kind','gamma','coef0','degree','query_gamma','mutate','mode'])+'\n'+''.join(' '.join(format(v,'.17g') for v in v.ravel(order='F'))+'\n' for v in [x,q,g])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
 paths=[Path(__file__),args.executable,ROOT/'src/sklearn_kernel_approximation.f90',ROOT/'src/sklearn_nystroem_svd.f90',ROOT/'app/reference_nystroem_driver.f90',Path(sklearn.__file__).parent/'kernel_approximation.py']
 hashes={str(p):sha(p) for p in paths};cases=list(fixtures());assert len({json.dumps(c,sort_keys=True) for c in cases})==len(cases);expected=[oracle(c) for c in cases]
 r=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,expected)),text=True,capture_output=True)
 if r.returncode:raise AssertionError((r.stdout[-800:],r.stderr[-1500:]))
 lines=iter(r.stdout.splitlines());index=-1;checks=[];inventories=[];fields={}
 for line in lines:
  p=line.split()
  if not p:continue
  if p[0]=='BEGIN':index+=1;fields={};assert int(p[1])==index+1
  elif p[0]=='END':
   c=cases[index];x,q,g,want,regular=expected[index];n=len(want['singular']);assert fields.keys()==want.keys()|{'svd_u','svd_v'}
   for name,values in fields.items():
    if name in ['svd_u','svd_v']:continue
    if name in ['normalization','train','query'] and not regular:continue
    target=np.asarray(want[name]).ravel(order='F');assert values.shape==target.shape
    if name in ['indices','components','metadata','preserved','repeat','svd_status']:np.testing.assert_array_equal(values,target,err_msg=str((index,c,name)))
    else:np.testing.assert_allclose(values,target,rtol=3e-8,atol=2e-3 if name=='norm_gram' and not regular else 3e-8,equal_nan=True,err_msg=str((index,c,name)))
    checks.append(dict(fixture=index,field=name,oracle='source' if regular else 'source_svd_invariant',passed=True))
   u=fields['svd_u'].reshape(n,n,order='F');v=fields['svd_v'].reshape(n,n,order='F');s=fields['singular']
   np.testing.assert_allclose(u.T@u,np.eye(n),rtol=1e-11,atol=1e-11);np.testing.assert_allclose(v.T@v,np.eye(n),rtol=1e-11,atol=1e-11)
   np.testing.assert_allclose((u*s)@v.T,g,rtol=2e-10,atol=2e-10)
   checks.extend(dict(fixture=index,field=name,oracle='svd_certificate',passed=True) for name in ['svd_orthogonality','svd_reconstruction'])
   inventories.append(sorted(c['field'] for c in checks if c['fixture']==index))
  else:assert p[0] not in fields;fields[p[0]]=np.fromstring(next(lines),sep=' ');assert len(fields[p[0]])==int(p[1])
 assert index+1==len(cases)
 rejections=[]
 variants=[('unfitted',dict(mode=1)),('zero_components',dict(k=0)),('negative_components',dict(k=-1)),('negative_seed',dict(seed=-1)),('unknown_kernel',dict(kind=12)),('negative_gamma',dict(gamma=-2.)),('nan_gamma',dict(gamma=float('nan'))),('inf_gamma',dict(gamma=float('inf'))),('nan_coef',dict(coef0=float('nan'))),('inf_coef',dict(coef0=float('inf'))),('zero_degree',dict(degree=0)),('negative_degree',dict(degree=-1)),('nan_fit',dict(mode=2)),('inf_fit',dict(mode=3)),('nan_query',dict(mode=4)),('inf_query',dict(mode=5)),('empty_rows',dict(n=0)),('empty_features',dict(p=0)),('empty_query',dict(m=0)),('query_width',dict(mode=6)),('chi2_negative_fit',dict(kind=7)),('chi2_negative_query',dict(kind=8)),('precomputed_shape',dict(kind=9)),('precomputed_count',dict(kind=9,n=4,p=4,k=2)),('precomputed_gamma',dict(kind=9,n=4,p=4,k=4,gamma=.5)),('callback_gamma',dict(kind=10,gamma=.5)),('callback_coef',dict(kind=10,coef0=.5)),('callback_degree',dict(kind=10,degree=2)),('callback_nan',dict(kind=10,mode=7)),('callback_missing',dict(kind=10,mode=8))]
 for name,change in variants:
  c=config(**change);x,q=inputs(c);g=np.zeros((max(0,min(c['n'],c['k'])),)*2)
  if name=='chi2_negative_fit':x[0,0]=-1
  if name=='chi2_negative_query':q[0,0]=-1
  wire=payload(c,(x,q,g))
  try:
   m=model_for(c)
   if c['mode']==2:x[0,0]=np.nan
   if c['mode']==3:x[0,0]=np.inf
   if c['mode']==4:q[0,0]=np.nan
   if c['mode']==5:q[0,0]=np.inf
   if c['mode']!=1:m.fit_transform(x)
   m.transform(q[:,:-1] if c['mode']==6 else q)
  except ValueError as error:source_error=type(error).__name__
  else:raise AssertionError(('source accepts invalid',name))
  r=subprocess.run([str(args.executable)],input='1\n'+wire,text=True,capture_output=True);out=r.stdout+r.stderr
  assert r.returncode and 'fscikit-learn:' in out,(name,out)
  assert not any(t in out for t in ['SIGFPE','Fortran runtime error']),(name,out)
  rejections.append(dict(name=name,passed=True,source_error=source_error))
 assert hashes=={str(p):sha(p) for p in paths}
 d=dict(profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,regular=[a[-1] for a in expected],rejections=rejections,hashes=hashes)
 (args.report or ROOT/'validation/results'/f'nystroem_{args.profile}.json').write_text(json.dumps(d,indent=2)+'\n');print(len(cases),len(checks),sum(d['regular']),len(rejections),'PASS')
if __name__=='__main__':main()
