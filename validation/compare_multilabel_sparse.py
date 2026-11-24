"""MultiLabelBinarizer raw CSR fit/transform ordering and sparse inverse tuples."""
from pathlib import Path
import argparse,hashlib,io,json,subprocess,warnings
import numpy as np
import sklearn,scipy
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
ROOT=Path(__file__).resolve().parents[1]
def fixtures():
 rng=np.random.default_rng(314518)
 pool=np.array([-2147483648,2147483647]+list(range(-50,151,7)))
 for n in [0,1,7,31]:
  train=[rng.choice(pool,rng.integers(0,13)).tolist() for _ in range(n)]
  query=[rng.choice(pool,rng.integers(0,13)).tolist() for _ in range(5)]
  classes=np.array([-2147483648]+list(range(-64,65))+[2147483647])
  for cid,ctor in enumerate([None,classes,classes[::-1],np.array([],int),np.array([1,1])]):
   for method in [0,1]:
    for inverse in range(7):
     for refit in [0,1]:
      yield dict(n=n,constructor=cid,method=method,inverse=inverse,refit=refit),train,query,ctor
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_label_binarization.f90',ROOT/'src/sklearn_integer_sets.f90',ROOT/'src/sklearn_numpy_ordering.f90',ROOT/'app/reference_multilabel_sparse_driver.f90',ROOT/'LICENSES/Python-3.13.3-LICENSE',a.executable,Path(sklearn.__file__).parent/'preprocessing/_label.py'];hashes={str(p):sha(p) for p in paths}
 configs=[];checks=[];rejections=[]
 for ident,(cfg,train,query,ctor) in enumerate(fixtures()):
  cfg=dict(id=ident,**cfg,data_sha256=hashlib.sha256(json.dumps([train,query,None if ctor is None else ctor.tolist()]).encode()).hexdigest());configs.append(cfg);expected={};error=None;external=csr_matrix((0,0))
  def mapped(prefix,z):
   expected[prefix+'_shape']=z.shape;expected[prefix+'_ptr']=z.indptr+1;expected[prefix+'_idx']=z.indices+1;expected[prefix+'_data']=z.data;expected[prefix+'_dense']=z.toarray().ravel(order='F')
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   try:
    model=MultiLabelBinarizer(classes=ctor,sparse_output=True)
    if cfg['method']:z=model.fit_transform(train)
    else:model.fit(train);z=model.transform(train)
    expected['classes']=model.classes_;mapped('train',z);expected['legacy_train']=model.transform(train).toarray().ravel(order='F')
    q=model.transform(query);mapped('query',q)
    if cfg['inverse']:
     if cfg['inverse']==1:inverse=z
     elif cfg['inverse']==2:inverse=q
     else:
      width=len(model.classes_)+int(cfg['inverse']==5)
      idx=[];data=[];ptr=[0]
      for i in range(3):
       for j in ([width-1,0,width-1] if width else []):idx.append(j);data.append(float(i%2))
       ptr.append(len(idx))
      if data and cfg['inverse']==4:data[0]=2.
      if data and cfg['inverse']==6:data[0]=np.nan if cfg['n']%2 else np.inf
      external=csr_matrix((np.array(data,float),idx,ptr),shape=(3,width));inverse=external
     decoded=model.inverse_transform(inverse);expected['inverse_size']=[len(row) for row in decoded];expected['inverse_flat']=[v for row in decoded for v in row]
    if cfg['refit']:model.fit(query);expected['refit_classes']=model.classes_;mapped('refit',model.transform(train))
   except (ValueError,TypeError,IndexError) as e:error=e
  classes=[] if ctor is None else ctor;stream=io.StringIO();stream.write(f"{len(train)} {len(query)} {len(classes)} {int(ctor is not None)} {cfg['method']} {cfg['inverse']} {cfg['refit']}\n");stream.write(' '.join(map(str,classes))+'\n')
  for rows in [train,query]:
   for row in rows:stream.write(str(len(row))+'\n'+' '.join(map(str,row))+'\n')
  if cfg['inverse']>=3:
   stream.write(f'{external.shape[0]} {external.shape[1]} {external.nnz}\n')
   for v in [external.indptr+1,external.indices+1,external.data]:stream.write(' '.join(format(x,'.17g') for x in v)+'\n')
  r=subprocess.run([str(a.executable)],input=stream.getvalue(),capture_output=True,text=True)
  if error is not None:
   assert r.returncode and 'ERROR STOP' in r.stderr and not any(t in r.stderr for t in ['SIGFPE','Segmentation fault','Fortran runtime error']),(cfg,error,r.stdout,r.stderr)
   rejections.append(dict(configuration=ident,source_type=type(error).__name__,message=str(error),passed=True));continue
  assert r.returncode==0,(cfg,r.stdout,r.stderr)
  actual={name:np.array(v,float) for name,*v in [line.split() for line in r.stdout.splitlines()]};assert actual.keys()==expected.keys(),(cfg,actual.keys(),expected.keys())
  for name,v in expected.items():np.testing.assert_array_equal(actual[name],v,err_msg=str((cfg,name)));checks.append(dict(configuration=ident,field=name,passed=True))
  if ident%100==0:print(ident,'configurations checked',flush=True)
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,hashes=hashes,configurations=configs,checks=checks,rejections=rejections),indent=2)+'\n');print(len(configs),'configurations',len(checks),'groups',len(rejections),'source rejections PASS')
if __name__=='__main__':main()
