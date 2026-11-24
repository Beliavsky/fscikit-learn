"""Direct seeded sklearn projection matrices and algebra, without matrix injection."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection,johnson_lindenstrauss_min_dim
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fixtures():
 for kind,seed,(n,p,k),cache in itertools.product([0,1],[0,7,42,2026,2147483647],[(7,1,1),(9,3,2),(6,7,4),(5,5,5),(3,4,7),(2,100,3),(2,101,3),(2,257,2)], [0,1]):
  for density in ([-1.,.001,.01,.2,.5,.99,.999,1.,1e-310] if kind else [-1.]):
   yield dict(kind=kind,seed=seed,n=n,p=p,k=k,cache=cache,density=density,eps=.4,mode=0)
 for kind,seed,cache in itertools.product([0,1],[0,42], [0,1]):
  yield dict(kind=kind,seed=seed,n=2,p=100,k=0,cache=cache,density=-1.,eps=.9,mode=0)
def matrix(c):
 return np.sin(np.arange(c['n']*c['p']).reshape(c['n'],c['p'])+.3)
def payload(c):
 return ' '.join(str(c[k]) for k in ['kind','n','p','k','seed','cache','eps','density','mode'])+'\n'+' '.join(format(v,'.17g') for v in matrix(c).ravel(order='F'))+'\n'
def model_for(c):
 cls=SparseRandomProjection if c['kind'] else GaussianRandomProjection
 options=dict(n_components=c['k'] or 'auto',random_state=c['seed'],eps=c['eps'],compute_inverse_components=bool(c['cache']))
 if c['kind']:options['density']='auto' if c['density']==-1 else c['density']
 return cls(**options)
def oracle(c):
 m=model_for(c);x=matrix(c)
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore');z=m.fit_transform(x);back=m.inverse_transform(z)
 dense=lambda a:a.toarray() if hasattr(a,'toarray') else a
 return dict(components=dense(m.components_),transform=dense(z),inverse=back,pinv=m.inverse_components_ if c['cache'] else [],metadata=[m.n_components_,m.n_features_in_,c['cache']],density=[m.density_] if c['kind'] else [],preserved=[1,1,1],cleared=[1])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
 paths=[Path(__file__),args.executable,*[ROOT/'src'/n for n in ['sklearn_random_projection.f90','sklearn_random_binomial.f90','sklearn_bicluster_datasets.f90','sklearn_legacy_rng.f90']],ROOT/'app/reference_seeded_projection_driver.f90',Path(sklearn.__file__).parent/'random_projection.py',Path(sklearn.__file__).parent/'utils/_random.pyx',ROOT/'validation/results/tools/numpy-2.2.6-distributions.c',ROOT/'validation/results/tools/numpy-2.2.6-legacy-distributions.c']
 hashes={str(p):sha(p) for p in paths};cases=list(fixtures());expected=[oracle(c) for c in cases]
 r=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c) for c in cases),text=True,capture_output=True)
 if r.returncode:raise AssertionError((r.stdout[-500:],r.stderr[-1500:]))
 lines=iter(r.stdout.splitlines());checks=[];inventories=[];index=-1
 for line in lines:
  parts=line.split()
  if not parts:continue
  if parts[0]=='BEGIN':index+=1;fields=set();assert int(parts[1])==index+1
  elif parts[0]=='END':
   assert fields==expected[index].keys();inventories.append(sorted(fields))
  else:
   field=parts[0];assert field not in fields;fields.add(field);size=int(parts[1]);values=np.fromstring(next(lines),sep=' ');assert len(values)==size
   want=np.asarray(expected[index][field]).ravel(order='F');assert values.shape==want.shape
   if field in ['components','density','metadata','preserved','cleared']:np.testing.assert_array_equal(values,want,err_msg=str((index,cases[index],field)))
   else:np.testing.assert_allclose(values,want,rtol=2e-10,atol=2e-11,equal_nan=True,err_msg=str((index,cases[index],field)))
   finite=np.isfinite(want);checks.append(dict(fixture=index,field=field,passed=True,max_abs_error=float(np.max(np.abs(values[finite]-want[finite]),initial=0))))
 assert index+1==len(cases)
 rejections=[]
 invalid=[]
 for kind in [0,1]:
  base=dict(kind=kind,seed=0,n=4,p=3,k=2,cache=0,density=-1.,eps=.4,mode=0)
  changes=[('negative_components',dict(k=-1)),('zero_eps',dict(eps=0.)),('negative_eps',dict(eps=-.1)),('nan_eps',dict(eps=float('nan'))),('inf_eps',dict(eps=float('inf'))),('negative_seed',dict(seed=-1)),('unfitted_transform',dict(mode=1)),('unfitted_inverse',dict(mode=2)),('feature_width',dict(mode=4)),('latent_width',dict(mode=5)),('nan_input',dict(mode=7)),('inf_input',dict(mode=8)),('empty_rows',dict(n=0)),('empty_features',dict(p=0)),('auto_too_wide',dict(k=0)),('auto_single_sample',dict(n=1,k=0))]
  if kind:changes += [(name,dict(density=value)) for name,value in [('zero_density',0.),('negative_density',-2.),('large_density',1.1),('nan_density',float('nan')),('inf_density',float('inf'))]]
  invalid += [(str(kind)+'_'+name,base|change) for name,change in changes]
 for name,eps in [('zero',0.),('negative',-.1),('one',1.),('large',2.),('nan',float('nan')),('inf',float('inf'))]:
  invalid.append(('jl_'+name,dict(kind=0,seed=0,n=4,p=3,k=2,cache=0,density=-1.,eps=eps,mode=3)))
 for name,c in invalid:
  try:
   m=model_for(c);x=matrix(c)
   if c['mode']==1:m.transform(x)
   elif c['mode']==2:m.inverse_transform(x)
   elif c['mode']==3:johnson_lindenstrauss_min_dim(c['n'],eps=c['eps'])
   elif c['mode']==4:m.fit(x).transform(x[:,:-1])
   elif c['mode']==5:m.fit(x).inverse_transform(x)
   else:
    if c['mode']==7:x[0,0]=np.nan
    if c['mode']==8:x[0,0]=np.inf
    m.fit(x)
  except ValueError as error:source_error=type(error).__name__
  else:raise AssertionError(('source accepts invalid',name))
  r=subprocess.run([str(args.executable)],input='1\n'+payload(c),text=True,capture_output=True);out=r.stdout+r.stderr
  assert r.returncode and 'fscikit-learn:' in out,(name,out)
  assert not any(t in out for t in ['SIGFPE','Fortran runtime error']),(name,out)
  rejections.append(dict(name=name,passed=True,source_error=source_error))
 assert hashes=={str(p):sha(p) for p in paths}
 report=dict(profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,rejections=rejections,hashes=hashes)
 path=args.report or ROOT/'validation/results'/f'seeded_projections_{args.profile}.json';path.write_text(json.dumps(report,indent=2)+'\n');print(len(cases),len(checks),len(rejections),'PASS')
if __name__=='__main__':main()
