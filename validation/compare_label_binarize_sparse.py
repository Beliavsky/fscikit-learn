"""CSR target label_binarize: source integer results and native real64 CSR storage."""
from pathlib import Path
import argparse,hashlib,io,json,subprocess,warnings
import numpy as np
import scipy,sklearn
from scipy.sparse import csr_matrix
from sklearn.preprocessing import label_binarize
ROOT=Path(__file__).resolve().parents[1]

def fixtures():
 rng=np.random.default_rng(48333)
 matrices=[]
 for n,p in [(1,2),(7,3),(13,5)]:
  for values in [[0,1],[0,-1],[2],[2,3],[.5],[1e18],[-2.**63],[0,2.**63],[np.nan],[np.inf]]:
   ptr=[0];idx=[];data=[]
   for i in range(n):
    for j in rng.permutation(p):
     if rng.random()<.5:
      for _ in range(1+int(rng.random()<.4)):
       idx.append(j);data.append(rng.choice(values))
    ptr.append(len(idx))
   matrices.append(csr_matrix((np.array(data,float),idx,ptr),shape=(n,p)))
  matrices.append(csr_matrix((n,p),dtype=float))
 # Later-row duplicates do not change the source's one-column binary shortcut.
 for ptr,data in [([0,1,2],[1,2]),([0,2,3],[1,1,2]),([0,1,3],[1,1,1]),([0,0,0],[]),([0,1,2],[.5,.5])]:
  matrices.append(csr_matrix((np.array(data,float),np.zeros(len(data),int),ptr),shape=(2,1)))
 matrices += [csr_matrix((0,3)),csr_matrix((3,0))]
 # Duplicate sums crossing int64 conversion range and positive-zero switching.
 matrices.append(csr_matrix((np.full(12,1e18),np.zeros(12,int),[0,12]),shape=(1,2)))
 matrices.append(csr_matrix((np.array([1.,1.,0.,1.]),[2,2,0,1],[0,3,4]),shape=(2,3)))
 for value in [np.nextafter(2.**63,0.),np.nextafter(-2.**63,-np.inf)]:
  matrices.append(csr_matrix((np.array([value,value]),[0,2],[0,1,2]),shape=(2,3)))
 matrices.append(csr_matrix((np.array([-0.,0.]),[2,0],[0,1,2]),shape=(2,3)))
 for mid,y in enumerate(matrices):
  p=y.shape[1]
  for classes in [np.arange(p),np.arange(p)[::-1],np.ones(p,int),np.array([8]),np.arange(p+1)]:
   for negative,positive in [(0,1),(0,3),(-2,0),(-3,5)]:
    for sparse in [False,True]:yield mid,y.copy(),classes.copy(),negative,positive,sparse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--start',type=int,default=0);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 paths=[Path(__file__),ROOT/'src/sklearn_label_binarize.f90',ROOT/'src/sklearn_numpy_ordering.f90',ROOT/'app/reference_label_binarize_sparse_driver.f90',a.executable,Path(sklearn.__file__).parent/'preprocessing/_label.py',Path(sklearn.__file__).parent/'utils/multiclass.py']
 hashes={str(p):sha(p) for p in paths};configs=[];checks=[];rejections=[]
 for ident,(mid,y,classes,negative,positive,sparse) in enumerate(fixtures()):
  if ident<a.start:continue
  cfg=dict(id=ident,matrix=mid,shape=list(y.shape),classes=classes.tolist(),negative=negative,positive=positive,sparse=sparse,hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('ptr',y.indptr),('idx',y.indices),('data',y.data)]});configs.append(cfg)
  original=[v.copy() for v in [y.indptr,y.indices,y.data]];error=None
  with warnings.catch_warnings(),np.errstate(all='ignore'):
   warnings.simplefilter('ignore')
   try:z=label_binarize(y,classes=classes,neg_label=negative,pos_label=positive,sparse_output=sparse)
   except (ValueError,IndexError,TypeError) as e:error=e
  for old,new in zip(original,[y.indptr,y.indices,y.data]):np.testing.assert_array_equal(old,new)
  stream=io.StringIO();stream.write(f'{y.shape[0]} {y.shape[1]} {len(classes)} {y.nnz} {negative} {positive} {int(sparse)}\n')
  for values in [classes,y.indptr+1,y.indices+1,y.data]:stream.write(' '.join(format(v,'.17g') for v in values)+'\n')
  r=subprocess.run([str(a.executable)],input=stream.getvalue(),capture_output=True,text=True)
  if error is not None:
   assert r.returncode and 'ERROR STOP' in r.stderr and not any(t in r.stderr for t in ['SIGFPE','Segmentation fault','Fortran runtime error']),(cfg,error,r.stdout,r.stderr)
   rejections.append(dict(configuration=ident,source_type=type(error).__name__,message=str(error),passed=True));continue
  assert r.returncode==0,(cfg,r.stdout,r.stderr)
  expected=dict(shape=z.shape,input_ptr=y.indptr+1,input_idx=y.indices+1,input_data=y.data)
  if sparse:
   # The package's CSR data are real64: compare raw source integer slots exactly,
   # and compare its dense view with the explicitly adapted real64 storage.
   expected.update(ptr=z.indptr+1,idx=z.indices+1,data=z.data.astype(float),dense=z.astype(float).toarray().ravel(order='F'))
  else:expected['dense']=z.ravel(order='F')
  actual={name:values for name,*values in [line.split() for line in r.stdout.splitlines()]};assert actual.keys()==expected.keys()
  for name,want in expected.items():
   want=np.asarray(want)
   got=np.array([int(v) for v in actual[name]],dtype=np.int64) if name=='dense' and not sparse else np.array(actual[name],float)
   np.testing.assert_array_equal(got,want,err_msg=str((cfg,name)))
   if name in ['data','dense','input_data'] and got.dtype.kind=='f':np.testing.assert_array_equal(np.signbit(got),np.signbit(want),err_msg=str((cfg,name,'sign bits')))
   oracle='native_real64_storage' if sparse and name=='dense' else 'caller_identity' if name.startswith('input_') else 'sklearn'
   checks.append(dict(configuration=ident,field=name,oracle=oracle,passed=True))
  if ident%200==0:print(ident,'configurations checked',flush=True)
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,hashes=hashes,configurations=configs,checks=checks,rejections=rejections),indent=2)+'\n')
 print(len(configs),'configurations',len(checks),'groups',len(rejections),'source rejections PASS')
if __name__=='__main__':main()
