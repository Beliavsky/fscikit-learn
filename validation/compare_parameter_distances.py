"""Parameterized and aliased-input sklearn nearest-pair contracts."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import numpy as np
import scipy,sklearn
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances_argmin_min,pairwise_distances_argmin
from compare_pairwise_reductions import upstream_hashes as basic_upstream_hashes
ROOT=Path(__file__).resolve().parents[1]
def fixture(metric,x,y,axis=1,storage=0,p=-1,squared=-1,self=False,w=None,v=None,vi=None,tag='regular'):
 return dict(metric=metric,x=np.asarray(x,dtype=float),y=np.asarray(y,dtype=float),axis=axis,storage=storage,p=p,squared=squared,self=self,w=None if w is None else np.asarray(w,dtype=float),v=None if v is None else np.asarray(v,dtype=float),vi=None if vi is None else np.asarray(vi,dtype=float),tag=tag)
def fixtures():
 for seed,(n,m,q),axis,storage,scenario in itertools.product([7,42],[(1,1,1),(2,3,2),(7,5,7),(3,4,31)],[0,1],range(4),range(3)):
  rng=np.random.RandomState(seed);x=rng.normal(size=(n,q));y=rng.normal(size=(m,q))
  if scenario==1:x=np.round(x);y=np.round(y)
  if scenario==2:x[rng.uniform(size=x.shape)<.7]=0;y[rng.uniform(size=y.shape)<.7]=0
  for v in [np.ones(q),np.arange(1,q+1),np.linspace(-2,2,q),np.zeros(q)]:yield fixture('seuclidean',x,y,axis,storage,v=v,tag=f'seed{seed}-scenario{scenario}')
  for power in [.5,1.,2.,3.]:
   for w in [np.ones(q),np.zeros(q),np.arange(q,dtype=float)]:yield fixture('minkowski',x,y,axis,storage,p=power,w=w,tag=f'seed{seed}-scenario{scenario}')
  matrix=rng.normal(size=(q,q));spd=matrix@matrix.T+np.eye(q)
  for vi in [np.eye(q),spd,-np.eye(q),matrix]:yield fixture('mahalanobis',x,y,axis,storage,vi=vi,tag=f'seed{seed}-scenario{scenario}')
 for axis,storage in itertools.product([0,1],range(4)):
  for variance in [[np.nan,1.],[np.inf,1.],[-np.inf,1.],[-0.,1.]]:
   yield fixture('seuclidean',[[0.,1.],[1.,0.]],[[1.,0.],[2.,1.]],axis,storage,v=variance,tag='nonfinite-variance')
  for precision in [[[np.nan,0.],[0.,1.]],[[np.inf,0.],[0.,1.]],[[1.,2.],[3.,-1.]]]:
   yield fixture('mahalanobis',[[0.,1.],[1.,0.]],[[1.,0.],[2.,1.]],axis,storage,vi=precision,tag='general-precision')
 for metric in ['euclidean','sqeuclidean','l2','minkowski','manhattan','chebyshev','cosine','nan_euclidean','haversine','callback','correlation','hamming','dice','jaccard','rogerstanimoto','russellrao','sokalmichener','sokalsneath','yule','matching','seuclidean','mahalanobis']:
  for (n,q),scenario,axis,storage in itertools.product([(1,2),(3,2),(7,5)],range(3),[0,1],[0,1]):
   if metric=='haversine'and q!=2:continue
   rng=np.random.RandomState(16);x=rng.normal(size=(n,q))
   if scenario==1:x[:]=0
   if scenario==2:x=np.round(x)
   kw={}
   if metric=='seuclidean':kw['v']=np.arange(1,q+1,dtype=float)
   if metric=='mahalanobis'and scenario==1:kw['vi']=np.eye(q)
   yield fixture(metric,x,x,axis,storage,self=True,tag=f'self-scenario{scenario}',**kw)
 for metric,scale,storage in itertools.product(['minkowski','seuclidean','mahalanobis'],[1e-150,1e150,1e200,1e308],range(4)):
  rng=np.random.RandomState(19);x=scale*rng.uniform(-1,1,(2,3));y=scale*rng.uniform(-1,1,(3,3));kw=dict(w=[0.,1.,2.])if metric=='minkowski'else dict(v=[1.,2.,3.])if metric=='seuclidean'else dict(vi=np.diag([1.,-2.,3.]))
  yield fixture(metric,x,y,storage=storage,tag=f'scale{scale}',**kw)
 for metric in ['infinity','p']:
  for axis,storage in itertools.product([0,1],range(4)):
   yield fixture(metric,[[0.,1.],[2.,3.]],[[1.,0.],[2.,1.]],axis,storage,tag='aliases')
 for metric in ['cosine','correlation','dice','russellrao','sokalsneath','nan_euclidean','mahalanobis']:
  x=np.full((3,2),1e200)
  yield fixture(metric,x,x,self=True,vi=np.eye(2)if metric=='mahalanobis'else None,tag='extreme-self')
 for metric in ['nan_euclidean']:
  for x in [[[np.nan,np.nan]],[[np.nan,1.],[2.,np.nan],[np.nan,np.nan]]]:yield fixture(metric,x,x,self=True,tag='missing-self')
 for n in [1024,1025]:
  x=np.zeros((n,1));y=np.ones((n+1,1));y[0]=0
  yield fixture('seuclidean',x,y,v=[0.],tag='auto-strategy-threshold')

def expected(c):
 x,y=c['x'],c['y']
 if c['storage']in [1,2]:x=csr_matrix(x)
 if c['storage']in [1,3]:y=csr_matrix(y)
 if c['self']:y=x
 kw={}
 for name,key in [('w','w'),('v','V'),('vi','VI')]:
  if c[name]is not None:kw[key]=c[name]
 if c['p']!=-1:kw['p']=c['p']
 if c['squared']!=-1:kw['squared']=bool(c['squared'])
 def callback(a,b):
  if hasattr(a,'toarray'):a=a.toarray().ravel()
  if hasattr(b,'toarray'):b=b.toarray().ravel()
  return np.sum(np.abs(a-b))+.125*np.sum(a)
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore');metric=callback if c['metric']=='callback'else c['metric']
  i,d=pairwise_distances_argmin_min(x,y,axis=c['axis'],metric=metric,metric_kwargs=kw)
  j=pairwise_distances_argmin(x,y,axis=c['axis'],metric=metric,metric_kwargs=kw)
 assert np.array_equal(i,j)
 return i+1,d

def wire(c,id):
 x,y=c['x'],c['y'];w,v,vi=c['w'],c['v'],c['vi'];sizes=[-1 if w is None else len(w),-1 if v is None else len(v),-1 if vi is None else vi.shape[0],-1 if vi is None else vi.shape[1]]
 header=f"{id} {c['metric']} {len(x)} {len(y)} {x.shape[1]} {y.shape[1]} {c['axis']} {c['storage']} {c['p']} {c['squared']} {int(c['self'])} "+' '.join(map(str,sizes))
 arrays=[x,y]+([]if w is None else[w])+([]if v is None else[v])+([]if vi is None else[vi])
 return header+'\n'+'\n'.join(' '.join(map(repr,a.ravel(order='F').tolist()))for a in arrays)+'\n'
def digest(c):return hashlib.sha256(b''.join(c[k].tobytes()for k in ['x','y','w','v','vi']if c[k]is not None)).hexdigest()
def upstream_hashes():
 import scipy.spatial.distance as distance
 import scipy.spatial._distance_wrap as wrap
 result=basic_upstream_hashes();base=Path(sklearn.__file__).parent
 for p in [base/'metrics/_pairwise_distances_reduction/_base.pyx.tp',Path(distance.__file__),Path(wrap.__file__)]:
  result[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return result
def run(executable,profile,report):
 cases=[];rejections=[]
 for c in fixtures():
  try:i,d=expected(c)
  except(ValueError,TypeError,AttributeError,np.linalg.LinAlgError)as e:rejections.append((c,type(e).__name__));continue
  cases.append((c,i,d))
 proc=subprocess.run([str(executable)],input=str(len(cases))+'\n'+''.join(wire(c,i)for i,(c,_,_)in enumerate(cases)),text=True,capture_output=True,timeout=240)
 if proc.returncode:raise RuntimeError(proc.stdout[-1000:]+proc.stderr[-3000:])
 lines=proc.stdout.strip().splitlines();assert len(lines)==3*len(cases)
 records=[];fail=[];maxscaled=0;nonfinite=0
 for i,(c,idx,d)in enumerate(cases):
  head=list(map(int,lines[3*i].split()));gi=np.fromstring(lines[3*i+1],sep=' ',dtype=int);gd=np.fromstring(lines[3*i+2],sep=' ')
  flags=head==[i,1,1,1,len(idx)];indices=np.array_equal(idx,gi);masks=all(np.array_equal(f(gd),f(d))for f in [np.isnan,np.isposinf,np.isneginf]);finite=np.isfinite(d);values=bool(np.allclose(gd[finite],d[finite],rtol=3e-12,atol=3e-12));err=float(np.max(np.abs(gd[finite]-d[finite])/np.maximum(1,np.abs(d[finite])),initial=0));maxscaled=max(maxscaled,err);nonfinite+=int(np.sum(~finite))
  r={k:v for k,v in c.items()if k not in ['x','y','w','v','vi']};r.update(id=i,data_sha256=digest(c),shape_x=list(c['x'].shape),shape_y=list(c['y'].shape),indices=indices,values=values,masks=masks,state_identity=flags,max_scaled_error=err,source_indices=idx.tolist(),source_value_sha256=hashlib.sha256(d.tobytes()).hexdigest())
  records.append(r)
  if not all([flags,indices,masks,values]):fail.append(dict(r,expected=d.tolist(),actual=gd.tolist(),native_indices=gi.tolist()))
 rejected=[];cache={}
 for i,(c,error)in enumerate(rejections):
  key=(c['metric'],c['storage'],c['self'],tuple(c['x'].shape),tuple(c['y'].shape),c['axis'],np.count_nonzero(c['x'])==0,np.count_nonzero(c['y'])==0,tuple(None if c[k]is None else c[k].shape for k in ['w','v','vi']),digest(c)if c['metric']=='mahalanobis'and c['self']and c['vi']is None else None)
  if key not in cache:
   p=subprocess.run([str(executable)],input='1\n'+wire(c,i),text=True,capture_output=True,timeout=15);cache[key]=(len(cache),p.returncode)
  probe,code=cache[key];rejected.append(dict(id=i,metric=c['metric'],storage=c['storage'],self=c['self'],tag=c['tag'],data_sha256=digest(c),source_error=error,native_probe=probe,passed=code!=0))
 result=dict(profile=profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,configurations=len(cases),groups=4*len(cases),source_rejections=len(rejections),unique_rejection_probes=len(cache),nonfinite=nonfinite,max_scaled_error=maxscaled,records=records,rejections=rejected,failures=fail,passed=not fail and all(r['passed']for r in rejected),executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [ROOT/'src/sklearn_pairwise_reductions.f90',ROOT/'app/reference_parameter_distances_driver.f90',Path(__file__)]},upstream_sha256=upstream_hashes(),source_layout='C-contiguous float64',numpy_buffer_size=np.getbufsize(),source_chunk_size=sklearn.get_config().get('pairwise_dist_chunk_size',256),source_strategy=sklearn.get_config().get('pairwise_dist_parallel_strategy','auto'),source_cython_backend=sklearn.get_config().get('enable_cython_pairwise_dist',True))
 report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items()if k not in ['records','rejections','failures','source_sha256','upstream_sha256']});print('Failures',len(fail));print(json.dumps(fail[:12],indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--profile',default='checked');p.add_argument('--report',type=Path,required=True);a=p.parse_args();run(a.executable,a.profile,a.report)
