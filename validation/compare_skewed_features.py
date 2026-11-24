"""Direct scalar-seeded SkewedChi2Sampler reference, including IEEE shifted inputs."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.kernel_approximation import SkewedChi2Sampler
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def config(**kw):return dict(n=7,p=3,m=4,k=5,seed=0,skew=1.,query_skew=1.,mutate=0,mode=0,scenario='random')|kw
def fixtures():
 for seed,(n,p,k),skew in itertools.product([0,7,42,2026,2147483647],[(7,1,1),(9,3,5),(5,7,8),(4,31,17)],[-3.,-.1,0.,.01,1.,10.]):
  yield config(n=n,p=p,k=k,seed=seed,skew=skew,query_skew=skew)
 for seed,scenario,skew in itertools.product([0,7,42,2026,2147483647],['zero','near_boundary','small','large','overflow'],[.1,1.,1e308]):
  yield config(seed=seed,scenario=scenario,skew=skew,query_skew=skew)
 for seed,mutate,query_skew in itertools.product([0,7,42],[1,13],[-1.,.1,10.]):
  yield config(seed=seed,mutate=mutate,query_skew=query_skew)
 for seed in [0,7,42]:yield config(seed=seed,mode=7,scenario='fit_negative')
def inputs(c):
 rng=np.random.RandomState(c['seed']%10000+17)
 x=rng.uniform(.25,4,(c['n'],c['p']))-c['skew'];q=rng.uniform(.25,4,(c['m'],c['p']))-c['query_skew']
 if c['scenario']=='zero':x.fill(0);q.fill(0)
 if c['scenario']=='near_boundary':x.fill(np.nextafter(-c['skew'],np.inf));q.fill(np.nextafter(-c['query_skew'],np.inf))
 if c['scenario']=='small':x.fill(1e-300);q.fill(1e-300)
 if c['scenario']=='large':x.fill(1e154);q.fill(1e154)
 if c['scenario']=='overflow':x.fill(1e308);q.fill(1e308)
 if c['scenario']=='fit_negative':x.fill(-100.)
 # Huge skew has no small representable gap: use large positive values.
 if c['skew']==1e308 and c['scenario']=='random':x.fill(1e308)
 return x,q
def payload(c,arrays):
 return ' '.join(str(c[k]) for k in ['n','p','m','k','seed','skew','query_skew','mutate','mode'])+'\n'+''.join(' '.join(format(v,'.17g') for v in a.ravel(order='F'))+'\n' for a in arrays)
def oracle(c,arrays):
 x,q=arrays;copies=[a.copy() for a in arrays];state=np.random.get_state()
 model=SkewedChi2Sampler(n_components=c['k'],random_state=c['seed'],skewedness=c['skew'])
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore')
  if c['mode']==7:model.fit(x);z=[]
  else:z=model.fit_transform(x)
  if c['mutate']:model.n_components=c['mutate']
  model.skewedness=c['query_skew'];query=model.transform(q)
 result=dict(weights=model.random_weights_,offset=model.random_offset_,train=z,query=query,metadata=[model.n_features_in_,c['k'],model.n_components])
 result['preserved']=[int(np.array_equal(a,b)) for a,b in zip(arrays,copies)]+[int(all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state())))]
 first=model.random_weights_.copy();offset=model.random_offset_.copy();model.n_components=c['k'];model.skewedness=c['skew'];model.fit(x)
 result['repeat']=[int(np.array_equal(first,model.random_weights_,equal_nan=True)),int(np.array_equal(offset,model.random_offset_,equal_nan=True))]
 assert all(result['preserved']) and all(result['repeat'])
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
 paths=[Path(__file__),args.executable,ROOT/'src/sklearn_kernel_approximation.f90',ROOT/'src/sklearn_ieee_math.f90',ROOT/'src/sklearn_legacy_rng.f90',ROOT/'app/reference_skewed_features_driver.f90',Path(sklearn.__file__).parent/'kernel_approximation.py']
 hashes={str(p):sha(p) for p in paths};cases=list(fixtures());arrays=[inputs(c) for c in cases];expected=[oracle(c,a) for c,a in zip(cases,arrays)]
 r=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,arrays)),text=True,capture_output=True)
 if r.returncode:raise AssertionError((r.stdout[-800:],r.stderr[-1500:]))
 lines=iter(r.stdout.splitlines());index=-1;checks=[];inventories=[];nonfinite=0
 for line in lines:
  parts=line.split()
  if not parts:continue
  if parts[0]=='BEGIN':index+=1;fields=set();assert int(parts[1])==index+1
  elif parts[0]=='END':assert fields==expected[index].keys();inventories.append(sorted(fields))
  else:
   field=parts[0];assert field not in fields;fields.add(field);values=np.fromstring(next(lines),sep=' ');assert len(values)==int(parts[1]);want=np.asarray(expected[index][field]).ravel(order='F');assert values.shape==want.shape
   if field in ['weights','offset','metadata','preserved','repeat']:np.testing.assert_array_equal(values,want,err_msg=str((index,cases[index],field)))
   else:np.testing.assert_allclose(values,want,rtol=3e-11,atol=3e-12,equal_nan=True,err_msg=str((index,cases[index],field)))
   finite=np.isfinite(want);nonfinite+=int(np.count_nonzero(~finite));checks.append(dict(fixture=index,field=field,passed=True,max_abs_error=float(np.max(np.abs(values[finite]-want[finite]),initial=0))))
 assert index+1==len(cases)
 rejections=[]
 base=config()
 variants=[('unfitted',dict(mode=1)),('zero_components',dict(k=0)),('negative_components',dict(k=-1)),('negative_seed',dict(seed=-1)),('nan_skew',dict(skew=float('nan'))),('positive_inf_skew',dict(skew=float('inf'))),('negative_inf_skew',dict(skew=-float('inf'))),('nan_fit',dict(mode=2)),('inf_fit',dict(mode=3)),('nan_query',dict(mode=4)),('inf_query',dict(mode=5)),('query_width',dict(mode=6)),('empty_fit_rows',dict(n=0)),('empty_fit_features',dict(p=0)),('empty_query',dict(m=0)),('train_equal',{}),('train_below',{}),('query_equal',{}),('query_below',{})]
 for name,change in variants:
  c=base|change
  # Use finite ordinary input when testing invalid parameters themselves.
  x,q=inputs(base|{k:v for k,v in change.items() if k in ['n','p','m']})
  if name=='train_equal':x.fill(-c['skew'])
  if name=='train_below':x.fill(-c['skew']-1)
  if name=='query_equal':q.fill(-c['query_skew'])
  if name=='query_below':q.fill(-c['query_skew']-1)
  wire=payload(c,(x,q))
  try:
   model=SkewedChi2Sampler(n_components=c['k'],random_state=c['seed'],skewedness=c['skew'])
   if c['mode']==2:x[0,0]=np.nan
   if c['mode']==3:x[0,0]=np.inf
   if c['mode']==4:q[0,0]=np.nan
   if c['mode']==5:q[0,0]=np.inf
   if c['mode']!=1:model.fit_transform(x)
   model.skewedness=c['query_skew'];model.transform(q[:,:-1] if c['mode']==6 else q)
  except ValueError as error:source_error=type(error).__name__
  else:raise AssertionError(('source accepts invalid',name))
  r=subprocess.run([str(args.executable)],input='1\n'+wire,text=True,capture_output=True);out=r.stdout+r.stderr
  assert r.returncode and 'fscikit-learn:' in out,(name,out)
  assert not any(t in out for t in ['SIGFPE','Fortran runtime error']),(name,out)
  rejections.append(dict(name=name,passed=True,source_error=source_error))
 assert hashes=={str(p):sha(p) for p in paths}
 report=dict(profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,rejections=rejections,source_nonfinite_values=nonfinite,hashes=hashes)
 path=args.report or ROOT/'validation/results'/f'skewed_features_{args.profile}.json';path.write_text(json.dumps(report,indent=2)+'\n');print(len(cases),len(checks),len(rejections),nonfinite,'PASS')
if __name__=='__main__':main()
