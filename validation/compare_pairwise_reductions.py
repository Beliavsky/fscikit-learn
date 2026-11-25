"""Streaming nearest-pair reductions against the installed sklearn baseline."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import numpy as np
import sklearn
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances_argmin_min,pairwise_distances_argmin
ROOT=Path(__file__).resolve().parents[1]
METRICS=['euclidean','sqeuclidean','l2','manhattan','cityblock','l1','chebyshev','minkowski','cosine','nan_euclidean','haversine','callback']
def fixture(metric,x,y,axis=1,storage=0,p=-1,squared=-1,tag='regular'):
 return dict(metric=metric,x=np.asarray(x,dtype=float),y=np.asarray(y,dtype=float),axis=axis,storage=storage,p=p,squared=squared,tag=tag)
def fixtures():
 for seed,(n,m,q),scenario,metric,axis,storage in itertools.product([0,7,42],[(1,1,1),(2,3,2),(7,5,7),(17,11,31),(3,4,129)],range(3),METRICS,[0,1],range(4)):
  if metric=='haversine' and q!=2:continue
  if metric=='nan_euclidean' and storage!=0:continue
  if metric=='sqeuclidean' and storage==1:continue
  rng=np.random.RandomState(seed);x=rng.normal(size=(n,q));y=rng.normal(size=(m,q))
  if scenario==1:x[rng.uniform(size=x.shape)<.7]=0;y[rng.uniform(size=y.shape)<.7]=0
  if scenario==2:x=np.round(x);y=np.round(y)
  yield fixture(metric,x,y,axis,storage,tag=f'seed{seed}-scenario{scenario}')
 for metric,axis,storage in itertools.product(METRICS,[0,1],range(4)):
  if metric=='nan_euclidean' and storage!=0:continue
  if metric=='sqeuclidean' and storage==1:continue
  x=np.array([[0.,0.],[1.,0.],[0.,0.] ]);y=np.array([[0.,1.],[0.,0.],[-1.,0.],[1.,0.]])
  yield fixture(metric,x,y,axis,storage,tag='zeros-ties')
 for p,axis,storage in itertools.product([.5,1.,2.,3.,7.,np.inf],[0,1],range(4)):
  rng=np.random.RandomState(2026);yield fixture('minkowski',rng.normal(size=(7,5)),rng.normal(size=(11,5)),axis,storage,p=p,tag='powers')
 for metric,squared,axis,storage in itertools.product(['euclidean','nan_euclidean'],[0,1],[0,1],range(4)):
  if metric=='nan_euclidean' and storage!=0:continue
  rng=np.random.RandomState(19);yield fixture(metric,rng.normal(size=(7,3)),rng.normal(size=(5,3)),axis,storage,squared=squared,tag='squared')
 for seed,axis in itertools.product([0,7,42],[0,1]):
  rng=np.random.RandomState(seed);x=rng.normal(size=(7,5));y=rng.normal(size=(9,5));x[rng.uniform(size=x.shape)<.4]=np.nan;y[rng.uniform(size=y.shape)<.4]=np.nan
  yield fixture('nan_euclidean',x,y,axis,tag='missing')
 for axis in [0,1]:
  yield fixture('nan_euclidean',[[np.nan,1.],[np.nan,np.nan]],[[2.,np.nan],[3.,4.],[np.nan,np.nan]],axis,tag='no-overlap')
 for metric,scale,axis,storage in itertools.product(METRICS,[1e-16,1e-150,1e150,1e200],[0,1],range(4)):
  if metric in ['haversine','callback']:continue
  if metric=='nan_euclidean' and storage!=0:continue
  if metric=='sqeuclidean' and storage==1:continue
  rng=np.random.RandomState(31);yield fixture(metric,scale*rng.normal(size=(2,3)),scale*rng.normal(size=(3,3)),axis,storage,tag=f'scale{scale}')
 for n,m,q in [(257,3,7),(3,513,2)]:
  for metric in METRICS:
   if metric=='haversine' and q!=2:continue
   rng=np.random.RandomState(2026)
   for axis in [0,1]:
    yield fixture(metric,rng.normal(size=(n,q)),rng.normal(size=(m,q)),axis,tag='chunk-boundaries')
 for metric,storage,axis in itertools.product(['euclidean','l2','sqeuclidean','nan_euclidean'],range(4),[0,1]):
  if metric=='nan_euclidean' and storage!=0:continue
  if metric=='sqeuclidean' and storage==1:continue
  yield fixture(metric,[[1e8,1e8],[1e8+5,1e8]],[[1e8+1,1e8],[1e8+2,1e8]],axis,storage,tag='offset-cancellation')
 for axis in [0,1]:
  yield fixture('haversine',[[0.,0.],[np.pi/2,0.]],[[0.,np.pi],[-np.pi/2,0.]],axis,tag='antipodes')
def asymmetric(a,b):return np.sum(np.abs(a-b))+.125*np.sum(a)
def expected(c):
 x,y=c['x'],c['y']
 if c['storage'] in [1,2]:x=csr_matrix(x)
 if c['storage'] in [1,3]:y=csr_matrix(y)
 # sklearn passes sparse rows to callbacks. Normalize those arguments only in
 # the Python adapter to the native contiguous row callback signature.
 def callback(a,b):
  if hasattr(a,'toarray'):a=a.toarray().ravel()
  if hasattr(b,'toarray'):b=b.toarray().ravel()
  return asymmetric(a,b)
 kw={}
 if c['p']!=-1:kw['p']=c['p']
 if c['squared']!=-1:kw['squared']=bool(c['squared'])
 metric=callback if c['metric']=='callback' else c['metric']
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore')
  idx,d=pairwise_distances_argmin_min(x,y,axis=c['axis'],metric=metric,metric_kwargs=kw)
  other=pairwise_distances_argmin(x,y,axis=c['axis'],metric=metric,metric_kwargs=kw)
 assert np.array_equal(idx,other)
 return idx+1,d
def wire(c,id):
 x,y=c['x'],c['y'];return f"{id} {c['metric']} {len(x)} {len(y)} {x.shape[1]} {y.shape[1]} {c['axis']} {c['storage']} {c['p']} {c['squared']}\n"+' '.join(map(repr,x.ravel(order='F').tolist()))+'\n'+' '.join(map(repr,y.ravel(order='F').tolist()))+'\n'
def digest(c):
 return hashlib.sha256(c['x'].tobytes()+c['y'].tobytes()).hexdigest()
def upstream_hashes():
 base=Path(sklearn.__file__).parent
 return {str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [base/'metrics/pairwise.py',base/'metrics/_dist_metrics.pyx.tp',base/'metrics/_pairwise_distances_reduction/_argkmin.pyx.tp',base/'metrics/_pairwise_distances_reduction/_dispatcher.py',base/'utils/_heap.pyx']}
def run(executable,profile,report):
 cases=[];rejections=[]
 for c in fixtures():
  try:idx,d=expected(c)
  except (ValueError,TypeError) as e:rejections.append((c,type(e).__name__));continue
  cases.append((c,idx,d))
 proc=subprocess.run([str(executable)],input=str(len(cases))+'\n'+''.join(wire(c,i) for i,(c,_,_) in enumerate(cases)),text=True,capture_output=True,timeout=240)
 if proc.returncode:raise RuntimeError(proc.stdout[-2000:]+proc.stderr[-4000:])
 lines=proc.stdout.strip().splitlines();assert len(lines)==3*len(cases),(len(lines),len(cases))
 records=[];fail=[];maxerr=0;nonfinite=0;maxscaled=0
 for i,(c,idx,d) in enumerate(cases):
  header=list(map(int,lines[3*i].split()));gotidx=np.fromstring(lines[3*i+1],sep=' ',dtype=int);got=np.fromstring(lines[3*i+2],sep=' ')
  states=header==[i,1,1,1,len(idx)];indices=np.array_equal(idx,gotidx);masks=all(np.array_equal(f(got),f(d)) for f in [np.isnan,np.isposinf,np.isneginf]);finite=np.isfinite(d)
  values=bool(np.allclose(got[finite],d[finite],rtol=3e-12,atol=3e-12));err=float(np.max(np.abs(got[finite]-d[finite]),initial=0));maxerr=max(maxerr,err);maxscaled=max(maxscaled,float(np.max(np.abs(got[finite]-d[finite])/np.maximum(1,np.abs(d[finite])),initial=0)));nonfinite+=int(np.sum(~finite))
  rec={k:v for k,v in c.items() if k not in ['x','y']};rec.update(id=i,data_sha256=digest(c),shape_x=list(c['x'].shape),shape_y=list(c['y'].shape),indices=indices,values=values,masks=masks,state_identity=states,max_error=err,source_value_sha256=hashlib.sha256(d.tobytes()).hexdigest(),source_indices=idx.tolist(),native_indices=gotidx.tolist())
  records.append(rec)
  if not all([states,indices,masks,values]):fail.append(dict(rec,expected=d.tolist(),actual=got.tolist()))
 rejected=[]
 for i,(c,error) in enumerate(rejections):
  p=subprocess.run([str(executable)],input='1\n'+wire(c,i),text=True,capture_output=True,timeout=15)
  rejected.append(dict(id=i,metric=c['metric'],storage=c['storage'],tag=c['tag'],source_error=error,passed=p.returncode!=0,data_sha256=digest(c)))
 result=dict(profile=profile,sklearn=sklearn.__version__,numpy=np.__version__,configurations=len(cases),groups=4*len(cases),source_rejections=len(rejections),nonfinite=nonfinite,max_error=maxerr,max_scaled_error=maxscaled,records=records,rejections=rejected,failures=fail,passed=not fail and all(c['passed'] for c in rejected),sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/sklearn_pairwise_reductions.f90',ROOT/'src/sklearn_metrics.f90',ROOT/'src/sklearn_sparse.f90',ROOT/'src/sklearn_pairwise_measures.f90',ROOT/'src/sklearn_ieee_math.f90',ROOT/'src/sklearn_common.f90',ROOT/'app/reference_pairwise_reductions_driver.f90',Path(__file__)]},executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),upstream_sha256=upstream_hashes())
 report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ['records','rejections','failures','sha256']}));print('failures',len(fail));print(json.dumps(fail[:12],indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--profile',default='checked');p.add_argument('--report',type=Path,required=True);a=p.parse_args();run(a.executable,a.profile,a.report)
