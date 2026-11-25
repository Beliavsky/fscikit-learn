"""Named/callback/GP/precomputed sklearn pairwise_kernels dispatch contracts."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import numpy as np
import sklearn,scipy
from scipy.sparse import csr_matrix,issparse
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
ROOT=Path(__file__).resolve().parents[1]
NAMES=['linear','rbf','poly','polynomial','sigmoid','cosine','laplacian','additive_chi2','chi2']
def fixture(metric,x,y=None,storage=0,mode=0,gamma=None,degree=None,coef0=None,filter=False,integer=False,tag='regular'):
 x=np.array(x,dtype=float,copy=True,order='C');y=np.array(x if y is None else y,dtype=float,copy=True,order='C')
 return dict(metric=metric,x=x,y=y,storage=storage,mode=mode,gamma=gamma,degree=degree,coef0=coef0,filter=filter,integer=integer,tag=tag)
def fixtures():
 for seed,(n,m,q),scenario,metric,storage in itertools.product([7,42],[(1,1,1),(2,3,2),(7,5,7),(3,4,31),(2,3,129)],range(4),NAMES,range(4)):
  rng=np.random.RandomState(seed);x=rng.normal(size=(n,q));y=rng.normal(size=(m,q))
  if metric in ['chi2','additive_chi2']:x=np.abs(x);y=np.abs(y)
  if scenario==1:x[rng.uniform(size=x.shape)<.7]=0;y[rng.uniform(size=y.shape)<.7]=0
  if scenario==2:x*=1e-16;y*=1e-16
  if scenario==3:x=np.round(x);y=np.round(y)
  yield fixture(metric,x,y,storage=storage,tag=f'seed{seed}-scenario{scenario}')
 for metric,mode,storage in itertools.product(NAMES+['callback'],[1,2],[0,1]):
  for scenario in range(3):
   rng=np.random.RandomState(23);x=rng.normal(size=(7,3))
   if metric in ['chi2','additive_chi2']:x=np.abs(x)
   if scenario==1:x[:]=0
   if scenario==2:x*=1e200
   yield fixture(metric,x,storage=storage,mode=mode,tag=f'self-scenario{scenario}')
 for metric,gamma,degree,coef in itertools.product(['rbf','poly','sigmoid','laplacian','chi2'],[-1.,0.,.2,np.nan,np.inf],[None,.5,0.,-2.],[None,-2.,1.,np.inf]):
  x=np.array([[0.,1.],[2.,0.]]);y=np.array([[1.,0.],[0.,3.]])
  yield fixture(metric,x,y,gamma=gamma,degree=degree,coef0=coef,filter=True,tag='filtered-parameters')
 for degree,integer in itertools.product([-2.,0.,1.,2.,3.,4.5],[False,True]):
  if integer and not float(degree).is_integer():continue
  yield fixture('polynomial',[[1.,-2.],[0.,1.]],[[2.,0.],[-1.,1.]],degree=degree,integer=integer,coef0=-1.,tag='degrees')
 for metric,scale,storage in itertools.product(NAMES,[1e-150,1e150,1e200,1e308],range(4)):
  rng=np.random.RandomState(5);x=scale*rng.uniform(-1,1,(2,3));y=scale*rng.uniform(-1,1,(3,3))
  if metric in ['chi2','additive_chi2']:x=np.abs(x);y=np.abs(y)
  yield fixture(metric,x,y,storage=storage,tag=f'scale{scale}')
 for metric,mode in itertools.product(['gp_rbf','gp_white','gp_sum'],[0,1,2]):
  for n,m,q in [(1,1,1),(3,5,2),(7,4,5)]:
   rng=np.random.RandomState(13);yield fixture(metric,rng.normal(size=(n,q)),rng.normal(size=(m,q)),mode=mode,tag='GP-object')
 for storage,mode in itertools.product(range(4),[0,1,2]):
  if storage>1 and mode!=0:continue
  x=np.array([[1.,-2.,3.],[-1.,0.,2.]])if mode==0 else np.array([[1.,-2.],[-1.,0.]])
  y=np.ones((3,7))if mode==0 else x
  yield fixture('precomputed',x,y,storage=storage,mode=mode,gamma=np.nan,degree=-1.,coef0=np.inf,tag='precomputed')
 for storage in range(4):
  yield fixture('callback',[[0.,1.],[2.,3.]],[[1.,0.],[0.,2.],[3.,1.]],storage=storage,tag='callback-cross')
 for (n,m,q),metric in itertools.product([(1,1,7),(1,3,17),(3,1,31),(2,3,129)],['linear','rbf','poly','sigmoid']):
  rng=np.random.RandomState(41);yield fixture(metric,1e200*rng.normal(size=(n,q)),1e200*rng.normal(size=(m,q)),tag='wide-overflow')
 for degree in [np.nan,np.inf,-np.inf]:
  yield fixture('poly',[[0.,1.],[1.,-1.]],[[0.,2.],[1.,0.]],degree=degree,tag='nonfinite-degree')
 for q,outputs,pattern in itertools.product([3,4,7,8,16,17,31,32,33],[1,3,4,5],range(3)):
  x=np.full((1,q),1e200);y=np.full((outputs,q),1e200)
  if pattern==1:y[:,min(4,q):]*=-1
  if pattern==2:y[:,1::2]*=-1
  yield fixture('linear',x,y,tag='vector-overflow-lanes')

def expected(c):
 x,y=c['x'],c['y']
 if c['storage']in [1,2]:x=csr_matrix(x)
 if c['storage']in [1,3]:y=csr_matrix(y)
 if c['mode']==1:y=None
 if c['mode']==2:y=x
 kw={k:c[k]for k in ['gamma','degree','coef0']if c[k]is not None}
 if c['integer']:kw['degree']=int(c['degree'])
 metric=c['metric']
 if metric.startswith('gp_'):
  metric=RBF(1.2)if metric=='gp_rbf'else WhiteKernel(.3)if metric=='gp_white'else RBF(1.2)+WhiteKernel(.3)
 if c['metric']=='callback':
  def metric(a,b):
   if issparse(a):a=a.toarray().ravel()
   if issparse(b):b=b.toarray().ravel()
   return np.sum(np.abs(a-b))+.125*np.sum(a)
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore');out=pairwise_kernels(x,y,metric=metric,filter_params=c['filter'],**kw)
 sparse=issparse(out)
 return (out.toarray()if sparse else np.asarray(out)),sparse

def wire(c,id):
 x,y=c['x'],c['y'];mask=sum(bit for key,bit in [('gamma',1),('degree',2),('coef0',4)]if c[key]is not None)+(8 if c['integer']else 0)
 g=1. if c['gamma']is None else c['gamma'];d=3. if c['degree']is None else c['degree'];coef=1. if c['coef0']is None else c['coef0']
 return f"{id} {c['metric']} {len(x)} {len(y)} {x.shape[1]} {y.shape[1]} {c['storage']} {c['mode']} {mask} {g} {d} {coef} {int(c['filter'])}\n"+' '.join(map(repr,x.ravel(order='F').tolist()))+'\n'+' '.join(map(repr,y.ravel(order='F').tolist()))+'\n'
def digest(c):return hashlib.sha256(c['x'].tobytes()+c['y'].tobytes()).hexdigest()
def metadata(c):
 r={k:v for k,v in c.items()if k not in ['x','y']}
 for k in ['gamma','degree','coef0']:
  if r[k]is not None and not np.isfinite(r[k]):r[k]=repr(r[k])
 return r

def provenance():
 import sklearn.metrics.pairwise as pairwise
 import sklearn.gaussian_process.kernels as gp
 import scipy.sparse._sparsetools as sparse_tools
 from threadpoolctl import threadpool_info
 files=[Path(pairwise.__file__),Path(gp.__file__),Path(sparse_tools.__file__)]
 pools=threadpool_info()
 files += [Path(p['filepath'])for p in pools if Path(p['filepath']).is_file()]
 return dict(files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},threadpools=pools)
def run(executable,profile,report):
 cases=[];rejections=[]
 for c in fixtures():
  try:d,sparse=expected(c)
  except(ValueError,TypeError)as e:rejections.append((c,type(e).__name__));continue
  cases.append((c,d,sparse))
 p=subprocess.run([str(executable)],input=str(len(cases))+'\n'+''.join(wire(c,i)for i,(c,_,_)in enumerate(cases)),text=True,capture_output=True,timeout=180)
 if p.returncode:raise RuntimeError(p.stdout[-1000:]+p.stderr[-3000:])
 lines=p.stdout.strip().splitlines();assert len(lines)==2*len(cases)
 records=[];fail=[];maxscaled=0;nonfinite=0
 for i,(c,d,sparse)in enumerate(cases):
  head=list(map(int,lines[2*i].split()));got=np.fromstring(lines[2*i+1],sep=' ').reshape(d.shape,order='F')
  state=head==[i,1,1,*d.shape];masks=all(np.array_equal(f(got),f(d))for f in [np.isnan,np.isposinf,np.isneginf]);finite=np.isfinite(d);values=bool(np.allclose(got[finite],d[finite],rtol=3e-12,atol=3e-12));err=float(np.max(np.abs(got[finite]-d[finite])/np.maximum(1,np.abs(d[finite])),initial=0));maxscaled=max(maxscaled,err);nonfinite+=int(np.sum(~finite))
  r=metadata(c);r.update(id=i,data_sha256=digest(c),shape_x=list(c['x'].shape),shape_y=list(c['y'].shape),result_shape=list(d.shape),source_result_sparse=sparse,source_value_sha256=hashlib.sha256(d.tobytes()).hexdigest(),state_identity=state,masks=masks,values=values,max_scaled_error=err);records.append(r)
  if not all([state,masks,values]):fail.append(dict(r,source=d.tolist(),native=got.tolist()))
 rejects=[];cache={}
 for i,(c,error)in enumerate(rejections):
  key=(c['metric'],c['storage'],c['mode'],tuple(c['x'].shape),tuple(c['y'].shape),c['tag'])
  if key not in cache:
   p=subprocess.run([str(executable)],input='1\n'+wire(c,i),text=True,capture_output=True,timeout=15);cache[key]=(len(cache),p.returncode)
  probe,code=cache[key];rejects.append(dict(id=i,metric=c['metric'],storage=c['storage'],mode=c['mode'],tag=c['tag'],source_error=error,native_probe=probe,passed=code!=0,data_sha256=digest(c)))
 result=dict(profile=profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,configurations=len(cases),groups=3*len(cases),source_rejections=len(rejections),unique_rejection_probes=len(cache),nonfinite=nonfinite,max_scaled_error=maxscaled,records=records,rejections=rejects,failures=fail,passed=not fail and all(r['passed']for r in rejects),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [ROOT/'LICENSES/OpenBLAS-small-kernel-BSD',ROOT/'LICENSES/OpenBLAS-LICENSE',ROOT/'src/sklearn_gp_kernels.f90',ROOT/'src/sklearn_sparse.f90',ROOT/'src/sklearn_kernel_dispatch.f90',ROOT/'src/sklearn_metrics.f90',ROOT/'app/reference_kernel_dispatch_driver.f90',Path(__file__)]},executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),upstream=provenance(),numpy_buffer_size=np.getbufsize())
 report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items()if k not in ['records','rejections','failures','source_sha256']});print('Failures',len(fail));print(json.dumps(fail[:10],indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--profile',default='checked');p.add_argument('--report',type=Path,required=True);a=p.parse_args();run(a.executable,a.profile,a.report)
