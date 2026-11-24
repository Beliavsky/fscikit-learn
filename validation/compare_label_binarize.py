"""Exact integer label_binarize dense and raw CSR contracts, sklearn 1.6.1."""
from pathlib import Path
import argparse,hashlib,io,json,subprocess,warnings
import numpy as np
import sklearn
from sklearn.preprocessing import label_binarize
ROOT=Path(__file__).resolve().parents[1]

def fixtures():
 rng=np.random.default_rng(9231)
 classes=[[],[1],[0,1],[1,0],[0,1,2,3],[3,1,0,2],[0,0,1],[1,0,1],[1,1],[-2147483648,2147483647],[2147483647,-2147483648]]
 targets=[[1],[0,1],[0,1,2],[99,99],[-2,0,1,99],[],rng.integers(-2,5,31).tolist(),[-2147483648,0,2147483647]]
 for y in targets:
  for c in classes:
   for negative,positive in [(0,1),(-3,5),(-2,0),(2,3),(0,0),(-2147483648,2147483647),(-2147483648,0)]:
    for sparse in [False,True]:yield np.array(y,int),np.array(c,int),negative,positive,sparse
 for n,p in [(1,1),(7,1),(1,2),(7,3),(31,5),(0,3),(3,0)]:
  for values in [[0,1],[2,3],[-1,0],[-2,3],[1,2,3]]:
   y=rng.choice(values,(n,p))
   for c in [np.arange(p),np.arange(p)[::-1],rng.permutation(p),np.ones(p,int),np.arange(p+1)]:
    for negative,positive in [(0,1),(-3,5),(-2,0),(2,3)]:
     for sparse in [False,True]:yield y.copy(),c.copy(),negative,positive,sparse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--start',type=int,default=0);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 paths=[Path(__file__),ROOT/'src/sklearn_label_binarize.f90',ROOT/'src/sklearn_numpy_ordering.f90',ROOT/'app/reference_label_binarize_driver.f90',a.executable,Path(sklearn.__file__).parent/'preprocessing/_label.py']
 hashes={str(p):sha(p) for p in paths};configs=[];checks=[];rejections=[]
 for ident,(y,classes,negative,positive,sparse) in enumerate(fixtures()):
  if ident<a.start:continue
  cfg=dict(id=ident,shape=list(y.shape),classes=classes.tolist(),negative=negative,positive=positive,sparse=sparse,target_sha256=hashlib.sha256(y.tobytes()).hexdigest());configs.append(cfg)
  source_error=None
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   try:z=label_binarize(y,classes=classes,neg_label=negative,pos_label=positive,sparse_output=sparse)
   except (ValueError,IndexError,TypeError) as e:source_error=e
  n=len(y);p=1 if y.ndim==1 else y.shape[1]
  stream=io.StringIO();stream.write(f'{n} {p} {len(classes)} {negative} {positive} {int(sparse)} {y.ndim}\n');stream.write(' '.join(map(str,classes))+'\n');np.savetxt(stream,y.reshape(n,p),fmt='%d')
  r=subprocess.run([str(a.executable)],input=stream.getvalue(),capture_output=True,text=True)
  if source_error is not None:
   assert r.returncode and 'ERROR STOP' in r.stderr and not any(t in r.stderr for t in ['SIGFPE','Segmentation fault','Fortran runtime error']),(cfg,source_error,r.stdout,r.stderr)
   rejections.append(dict(configuration=ident,source_type=type(source_error).__name__,message=str(source_error),passed=True));continue
  assert r.returncode==0,(cfg,r.stdout,r.stderr)
  expected=dict(shape=z.shape,dense=(z.toarray() if sparse else z).ravel(order='F'))
  if sparse:expected.update(ptr=z.indptr+1,idx=z.indices+1,data=z.data)
  actual={name:np.array(values,float) for name,*values in [line.split() for line in r.stdout.splitlines()]}
  assert actual.keys()==expected.keys(),(cfg,actual.keys(),expected.keys())
  for name,values in expected.items():
   np.testing.assert_array_equal(actual[name],values,err_msg=str((cfg,name)));checks.append(dict(configuration=ident,field=name,passed=True))
  if ident%300==0:print(ident,'configurations checked',flush=True)
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,sklearn=sklearn.__version__,numpy=np.__version__,hashes=hashes,configurations=configs,checks=checks,rejections=rejections),indent=2)+'\n')
 print(len(configs),'configurations',len(checks),'groups',len(rejections),'controlled source rejections PASS')
if __name__=='__main__':main()
