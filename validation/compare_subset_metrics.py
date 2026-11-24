"""Numeric/mixed-rank accuracy_score and zero_one_loss versus sklearn 1.6.1."""
from pathlib import Path
import argparse,hashlib,io,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import scipy,sklearn
from scipy.sparse import csr_matrix,issparse
from sklearn.metrics import accuracy_score,zero_one_loss
from threadpoolctl import threadpool_limits,threadpool_info
ROOT=Path(__file__).resolve().parents[1]
def form(x,k):
 if k==6:return csr_matrix(x.astype(float))
 dtype=[int,float,bool][k%3];z=x.astype(dtype)
 return z.ravel() if k<3 else z.reshape(len(x),-1) if len(x) else z.reshape(0,x.shape[1])
def fixtures():
 rng=np.random.default_rng(418531)
 for n in [0,1,7,131]:
  y=rng.integers(-2,4,(n,1));p=y.copy();p[::3]=2
  for ky in range(6):
   for kp in range(6):
    for norm in [False,True]:
     for wm in range(3):
      w=None if wm==0 else rng.integers(1,5,n).astype(float) if wm==1 else rng.integers(-3,4,n).astype(float)
      yield dict(family='categorical',ky=ky,kp=kp,normalize=norm,weight=wm),form(y,ky),form(p,kp),w
 for n in [0,7,131]:
  y=rng.integers(0,2,(n,3));p=y.copy();p[::3,1]=1-p[::3,1]
  for ky in range(3,7):
   for kp in range(3,7):
    for norm in [False,True]:
     for wm in range(3):
      w=None if wm==0 else rng.integers(1,5,n).astype(float) if wm==1 else rng.integers(-3,4,n).astype(float)
      yield dict(family='multilabel',ky=ky,kp=kp,normalize=norm,weight=wm),form(y,ky),form(p,kp),w
 for n in list(range(36))+[63,64,65,127,128,129,255,256,257,1023,4095,4096,4097,8191,8192,8193,10000,10001,16383,16384,16385]:
  y=np.zeros(n,int);p=np.zeros(n,int);p[::5]=1
  for wm in range(8):
   if wm==0:w=rng.normal(size=n)
   elif wm==1:w=np.resize([1e16,1.,-1e16,3.],n)
   elif wm==2:w=np.resize([np.finfo(float).max,np.finfo(float).max,-np.finfo(float).max,-np.finfo(float).max],n)
   elif wm==3:w=np.resize([np.nextafter(0.,1.),-0.,0.,np.nextafter(0.,1.)],n)
   elif wm==4:w=np.resize([np.inf,1.,-np.inf,0.],n)
   elif wm==5:w=np.resize([np.nan,1.,0.,2.],n)
   elif wm==6:w=np.zeros(n)
   else:w=np.full(n,-0.)
   for norm in [False,True]:yield dict(family='weights',ky=0,kp=0,normalize=norm,weight=wm),y.copy(),p.copy(),w.copy()
 for n in [4,8,16,17,31,32,33,127,129,8193,10001]:
  y=np.zeros(n,int);p=y.copy();p[::5]=1
  for wm in range(3):
   w=rng.normal(size=n) if wm==0 else np.resize([np.finfo(float).max,np.finfo(float).max,-np.finfo(float).max,-np.finfo(float).max],n) if wm==1 else np.resize([1e16,1.,-1e16,3.],n)
   storage=np.zeros(2*n);storage[::2]=w
   for view in [storage[::2],w[::-1]]:
    for norm in [False,True]:yield dict(family='strided_weights',ky=0,kp=0,normalize=norm,weight=wm),y.copy(),p.copy(),view
 # Raw CSR arithmetic: separate coalescing, explicit zeros and large integer values.
 cases=[]
 for value in [1.,-1.,2.,1e18,np.nextafter(2.**63,0.)]:
  a=csr_matrix((np.array([value,value,0.,value]),[2,2,0,1],[0,3,4]),shape=(2,4))
  cases.extend([(a,a.copy()),(a,csr_matrix((np.array([value*0.5]*6),[2]*4+[1]*2,[0,4,6]),shape=(2,4))), (a,csr_matrix((2,4)))])
 cases += [(csr_matrix((0,2)),csr_matrix((0,2))), (csr_matrix([[0],[1]]),csr_matrix([[0],[1]])), (csr_matrix((2,3)),csr_matrix((2,4)))]
 cases += [(csr_matrix([[2,3],[3,2]],dtype=float),csr_matrix([[2,3],[3,2]],dtype=float)), (csr_matrix([[0,.5],[1,0]]),csr_matrix([[0,0],[1,0]]))]
 for cid,(y,p) in enumerate(cases):
  for norm in [False,True]:
   for weighted in [False,True]:yield dict(family='csr_slots',case=cid,ky=6,kp=6,normalize=norm),y.copy(),p.copy(),np.arange(1,y.shape[0]+1,dtype=float) if weighted else None
 # Dense nonbinary indicators, invalid numeric targets and shape/weight boundaries.
 for y,p in [(np.array([[2,3],[3,2]]),np.array([[2,3],[2,3]])),(np.empty((3,0)),np.empty((3,0))), (np.array([0.,.5]),np.array([0.,1.])),(np.array([np.nan,1.]),np.array([0.,1.])),(np.array([np.inf,1.]),np.array([0.,1.])),(np.array([0.,1.]),np.array([1.])),(np.zeros((3,2)),np.zeros((3,3))), (np.array([[0,1,2]]),np.array([[0,1,2]]))]:
  ky=1 if y.ndim==1 else 4;kp=1 if p.ndim==1 else 4
  for norm in [False,True]:yield dict(family='boundary',ky=ky,kp=kp,normalize=norm),y.copy(),p.copy(),None
 for norm in [False,True]:yield dict(family='bad_weight_length',ky=0,kp=0,normalize=norm),np.array([0,1]),np.array([0,1]),np.ones(3)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--start',type=int,default=0);ap.add_argument('--limit',type=int);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_subset_metrics.f90',ROOT/'app/reference_subset_metrics_driver.f90',ROOT/'LICENSES/OpenBLAS-LICENSE',a.executable,Path(sklearn.__file__).parent/'metrics/_classification.py',Path(sklearn.__file__).parent/'utils/_array_api.py'];hashes={str(p):sha(p) for p in paths};configs=[];checks=[];rejections=[];nonfinite=0
 with threadpool_limits(limits=1,user_api='blas'):
  runtime=threadpool_info()
  for ident,(cfg,y,p,w) in enumerate(fixtures()):
   if ident<a.start:continue
   if a.limit is not None and len(configs)>=a.limit:break
   def digest(x):
    return [hashlib.sha256(v.tobytes()).hexdigest() for v in [x.indptr,x.indices,x.data]] if issparse(x) else hashlib.sha256(x.tobytes()).hexdigest()
   cfg=dict(id=ident,**cfg,shapes=[list(y.shape),list(p.shape)],weight_strides=None if w is None else list(w.strides),data_hashes=[digest(y),digest(p),None if w is None else digest(w)]);configs.append(cfg);error=None
   with warnings.catch_warnings(),np.errstate(all='ignore'):
    warnings.simplefilter('ignore')
    try:expected=dict(accuracy=[accuracy_score(y,p,normalize=cfg['normalize'],sample_weight=w)],loss=[zero_one_loss(y,p,normalize=cfg['normalize'],sample_weight=w)],identity=[1])
    except (ValueError,TypeError,IndexError,ZeroDivisionError) as e:error=e
   mode=0 if w is None else 3 if w.strides[0]<0 else 2 if w.strides[0]>w.itemsize else 1
   py=1 if y.ndim==1 else y.shape[1];pp=1 if p.ndim==1 else p.shape[1];stream=io.StringIO();stream.write(f"{y.shape[0]} {py} {cfg['ky']} {p.shape[0]} {pp} {cfg['kp']} {mode} {0 if w is None else len(w)} {int(cfg['normalize'])}\n")
   for x in [y,p]:
    if issparse(x):
     stream.write(str(x.nnz)+'\n');arrays=[x.indptr+1,x.indices+1,x.data]
    else:arrays=[x.ravel(order='F').astype(int) if x.dtype.kind=='b' else x.ravel(order='F')]
    for v in arrays:stream.write(' '.join(format(value,'.17g') for value in v)+'\n')
   if w is not None:stream.write(' '.join(format(value,'.17g') for value in w)+'\n')
   r=subprocess.run([str(a.executable)],input=stream.getvalue(),capture_output=True,text=True)
   if error is not None:
    assert r.returncode and 'ERROR STOP' in r.stderr and not any(t in r.stderr for t in ['SIGFPE','Segmentation fault','Fortran runtime error']),(cfg,error,r.stdout,r.stderr)
    rejections.append(dict(configuration=ident,source_type=type(error).__name__,message=str(error),passed=True));continue
   assert r.returncode==0,(cfg,r.stdout,r.stderr)
   actual={name:np.array(v,float) for name,*v in [line.split() for line in r.stdout.splitlines()]};assert actual.keys()==expected.keys(),(cfg,actual)
   for name,values in expected.items():
    want=np.array(values,float);got=actual[name];assert got.shape==want.shape
    assert np.array_equal(np.isnan(got),np.isnan(want)) and np.array_equal(np.isposinf(got),np.isposinf(want)) and np.array_equal(np.isneginf(got),np.isneginf(want)),(cfg,name,got,want)
    finite=np.isfinite(want);nonfinite+=int((~finite).sum());np.testing.assert_allclose(got[finite],want[finite],rtol=2e-13,atol=0,err_msg=str((cfg,name)))
    zeros=finite&(want==0);np.testing.assert_array_equal(np.signbit(got[zeros]),np.signbit(want[zeros]),err_msg=str((cfg,name,'zero sign')))
    checks.append(dict(configuration=ident,field=name,passed=True))
   if ident%300==0:print(ident,'configurations checked',flush=True)
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,numpy_buffer_size=np.getbufsize(),blas_runtime=runtime,hashes=hashes,configurations=configs,checks=checks,rejections=rejections,source_nonfinite_values=nonfinite),indent=2)+'\n');print(len(configs),'configurations',len(checks),'groups',len(rejections),'source rejections',nonfinite,'nonfinite PASS')
if __name__=='__main__':main()
