"""Dense/mixed/CSR Hamming loss versus sklearn, including BLAS stride paths."""
from pathlib import Path
import argparse,hashlib,io,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import scipy,sklearn
from scipy.sparse import csr_matrix,issparse
from sklearn.metrics import hamming_loss
from threadpoolctl import threadpool_limits,threadpool_info
from compare_subset_metrics import fixtures as subset_fixtures
ROOT=Path(__file__).resolve().parents[1]

def counted_target(counts,p):
 indices=np.concatenate([np.arange(int(c),dtype=np.int32) for c in counts]) if np.sum(counts) else np.array([],np.int32)
 return csr_matrix((np.ones(len(indices)),indices,np.r_[0,np.cumsum(counts)]),shape=(len(counts),p))

def fixtures():
 for cfg,y,p,w in subset_fixtures():
  if cfg['normalize']:continue
  yield dict(family='retained_'+cfg['family'],ky=cfg['ky'],kp=cfg['kp']),y,p,w
 rng=np.random.default_rng(819513)
 for n in [1,2,3,4,7,8,15,16,17,31,32,33,63,64,65,127,128,129,8191,8192,8193,10000,10001,16384,16385]:
  counts=np.resize([0,1,2,5,8,13,17],n);a=csr_matrix((n,17));b=counted_target(counts,17)
  patterns=[rng.normal(size=n),np.resize([1e16,1.,-1e16,3.],n),np.resize([np.finfo(float).max]*2+[-np.finfo(float).max]*2,n),np.resize([np.inf,1.,-np.inf,0.],n),np.resize([np.nan,1.,0.,2.],n),np.zeros(n),np.resize([np.nextafter(0.,1.),-0.,0.,np.nextafter(0.,1.)],n),np.full(n,-0.)]
  for mode,w in enumerate(patterns):
   storage=np.zeros(2*n);storage[::2]=w
   for stride,view in [(1,w),(2,storage[::2]),(-1,w[::-1])]:
    yield dict(family='multilabel_reductions',ky=6,kp=6,weight=mode,stride=stride),a,b,view
 h=np.finfo(float).max
 for counts in [[2,0,1,0,0],[1,0,2,0,0],[3,1,2,1,0]]:
  a=csr_matrix((5,3));b=counted_target(np.array(counts),3)
  for sign in [1.,-1.]:
   w=sign*np.array([h,0.,-h,0.,1.]);storage=np.zeros(10);storage[::2]=w
   for stride,view in [(1,w),(2,storage[::2]),(-1,w[::-1])]:yield dict(family='fused_stride_overflow',ky=6,kp=6,stride=stride),a,b,view
 for value in [1.,-1.,np.nextafter(2.**63,0.)]:
  a=csr_matrix((np.array([value,value,value,0.]),[2,2,2,0],[0,3,4]),shape=(2,4))
  for p in [a.copy(),csr_matrix((2,4)),csr_matrix((np.array([value,value,0.]),[2,2,1],[0,2,3]),shape=(2,4))]:
   for w in [None,np.array([1.,-1.]),np.array([np.inf,1.])]:yield dict(family='raw_csr',ky=6,kp=6),a,p,w
 for ky in [0,1,2]:
  y=np.array([0,1],dtype=[int,float,bool][ky]);p=csr_matrix([[0,1],[1,0]])
  yield dict(family='mixed_format',ky=ky,kp=6),y,p,None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--start',type=int,default=0);ap.add_argument('--limit',type=int);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 paths=[Path(__file__),ROOT/'validation/compare_subset_metrics.py',ROOT/'src/sklearn_subset_metrics.f90',ROOT/'app/reference_hamming_loss_driver.f90',ROOT/'LICENSES/OpenBLAS-LICENSE',a.executable,Path(sklearn.__file__).parent/'metrics/_classification.py',Path(sklearn.__file__).parent/'utils/sparsefuncs.py']
 hashes={str(p):sha(p) for p in paths};configs=[];checks=[];rejections=[];nonfinite=0
 with threadpool_limits(1,user_api='blas'):
  runtime=threadpool_info()
  for ident,(cfg,y,p,w) in enumerate(fixtures()):
   if ident<a.start:continue
   if a.limit is not None and len(configs)>=a.limit:break
   def digest(x):return [sha_bytes(v.tobytes()) for v in [x.indptr,x.indices,x.data]] if issparse(x) else sha_bytes(x.tobytes())
   cfg=dict(id=ident,**cfg,shapes=[list(y.shape),list(p.shape)],weight_strides=None if w is None else list(w.strides),data_hashes=[digest(y),digest(p),None if w is None else digest(w)]);configs.append(cfg);error=None
   with warnings.catch_warnings(),np.errstate(all='ignore'):
    warnings.simplefilter('ignore')
    try:expected=hamming_loss(y,p,sample_weight=w)
    except (ValueError,TypeError,IndexError,ZeroDivisionError) as e:error=e
   mode=0 if w is None else 3 if w.strides[0]<0 else 2 if w.strides[0]>w.itemsize else 1
   stream=io.StringIO();stream.write(f"{len(y) if not issparse(y) else y.shape[0]} {1 if y.ndim==1 else y.shape[1]} {cfg['ky']} {p.shape[0]} {1 if p.ndim==1 else p.shape[1]} {cfg['kp']} {mode} {0 if w is None else len(w)}\n")
   for x in [y,p]:
    if issparse(x):
     stream.write(str(x.nnz)+'\n');arrays=[x.indptr+1,x.indices+1,x.data]
    else:arrays=[x.ravel(order='F').astype(int) if x.dtype.kind=='b' else x.ravel(order='F')]
    for v in arrays:stream.write(' '.join(format(value,'.17g') for value in v)+'\n')
   if w is not None:stream.write(' '.join(format(value,'.17g') for value in w)+'\n')
   r=subprocess.run([str(a.executable)],input=stream.getvalue(),text=True,capture_output=True)
   if error is not None:
    assert r.returncode and 'ERROR STOP' in r.stderr and not any(t in r.stderr for t in ['SIGFPE','Segmentation fault','Fortran runtime error']),(cfg,error,r.stdout,r.stderr)
    rejections.append(dict(configuration=ident,source_type=type(error).__name__,message=str(error),passed=True));continue
   assert r.returncode==0,(cfg,r.stdout,r.stderr)
   actual={name:float(value) for name,value in [line.split() for line in r.stdout.splitlines()]};assert actual.keys()=={'loss','identity','ieee_state'},(cfg,actual)
   for name,want in zip(['loss','identity','ieee_state'],[float(expected),1.,1.]):
    got=actual[name]
    assert np.isnan(got)==np.isnan(want) and np.isposinf(got)==np.isposinf(want) and np.isneginf(got)==np.isneginf(want),(cfg,name,got,want)
    if np.isfinite(want):
     np.testing.assert_allclose(got,want,rtol=3e-13,atol=0,err_msg=str((cfg,name)))
     if want==0:assert np.signbit(got)==np.signbit(want),(cfg,name,'zero sign',got,want)
    else:nonfinite+=1
    checks.append(dict(configuration=ident,field=name,passed=True,actual=float(got).hex(),expected=want.hex()))
   if ident%300==0:print(ident,'configurations checked',flush=True)
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,numpy_buffer_size=np.getbufsize(),blas_runtime=runtime,hashes=hashes,configurations=configs,checks=checks,rejections=rejections,source_nonfinite_values=nonfinite),indent=2)+'\n')
 print(len(configs),'configurations',len(checks),'groups',len(rejections),'source rejections',nonfinite,'nonfinite PASS')
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':main()
