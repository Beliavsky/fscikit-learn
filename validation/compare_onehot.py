"""Numeric OneHotEncoder: grouping/drop/unknown rules, CSR slots and typed inverse objects."""
from pathlib import Path
import argparse,hashlib,io,json,subprocess,warnings
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
ROOT=Path(__file__).resolve().parents[1]
def fixtures():
 rng=np.random.default_rng(8331)
 for n,p in [(7,1),(31,2),(131,3)]:
  x=rng.integers(0,7,size=(n,p)).astype(float);x[::11,0]=np.nan;q=x[:5].copy();q[0,0]=99
  for grouping in range(5):
   for drop in ['none','first','if_binary']:
    for unknown in ['error','ignore','infrequent_if_exist','warn']:
     for sparse in [False,True]:
      c=dict(n=n,p=p,m=5,count=0 if grouping!=1 else 2,fraction=.2 if grouping==2 else 0.,maximum=[0,0,0,3,1][grouping],drop=drop,unknown=unknown,live=unknown,sparse=sparse,named=p==2,callback=False,refit=False,inverse=2)
      yield c,x.copy(),q.copy(),None,None
 # Explicit categories, missing values, unseen fit categories and explicit dropping.
 x=np.array([[0,1],[1,2],[2,3],[np.nan,1],[2,2],[1,3],[2,1]],float);q=np.array([[0,1],[np.nan,3],[7,5]],float)
 for cats in [[np.array([0,1,2,np.nan]),np.array([1,2,3,4.])],[np.array([0,2,np.nan]),np.array([1,3.])],[np.array([0,1,2,np.nan,np.inf]),np.array([-np.inf,1,2,3.])]]:
  for freq in [0,2,4]:
   for drop_values in [None,np.array([0,1.]),np.array([np.nan,3.])]:
    for unknown in ['error','ignore','infrequent_if_exist','warn']:
     for sparse in [False,True]:
      c=dict(n=7,p=2,m=3,count=freq,fraction=0.,maximum=0,drop='none',unknown=unknown,live=unknown,sparse=sparse,named=True,callback=False,refit=False,inverse=2)
      yield c,x.copy(),q.copy(),[v.copy() for v in cats],None if drop_values is None else drop_values.copy()
 # Inverse blocks: soft/signed/cancelling/zero rows, including feature-wise empty drops.
 for case in range(4):
  x=np.array([[0,0],[1,0],[2,1],[2,2],[2,2],[2,2]],float)
  if case==1:x[:,0]=1
  if case==2:x[:]=1
  if case==3:x[0,0]=np.nan
  q=x[:3].copy()
  for grouping in [0,2]:
   for drop in ['none','first','if_binary']:
    for unknown in ['error','ignore','infrequent_if_exist','warn']:
     for inverse in [1,3,4,5]:
      for sparse in [False,True]:
       c=dict(n=6,p=2,m=3,count=grouping,fraction=0.,maximum=0,drop=drop,unknown=unknown,live=unknown,sparse=sparse,named=False,callback=case==3,refit=False,inverse=inverse,inverse_case=case)
       yield c,x.copy(),q.copy(),None,None
 # Live handling and unnamed refits; grouped NaN categories are normal categories.
 x=np.array([[0],[0],[0],[1],[2],[np.nan]],float);q=np.array([[99],[np.nan],[2]],float)
 for unknown in ['error','ignore','infrequent_if_exist','warn']:
  for live in ['error','ignore','infrequent_if_exist','warn']:
   for refit in [False,True]:
    for sparse in [False,True]:
     c=dict(n=6,p=1,m=3,count=2,fraction=0.,maximum=0,drop='none',unknown=unknown,live=live,sparse=sparse,named=True,callback=True,refit=refit,inverse=2)
     yield c,x.copy(),q.copy(),None,None
 # Shortest-roundtrip numeric name formatting and special category labels.
 for values in [[0.,-0.,.1,.10000000000000002,1e-5,1e-4,1e15,1e16,np.nan],[-1e200,1e200,1e-200,np.finfo(float).max,np.nextafter(0.,1.)]]:
  x=np.array(values).reshape(-1,1)
  for sparse in [False,True]:
   c=dict(n=len(x),p=1,m=len(x),count=0,fraction=0.,maximum=0,drop='none',unknown='error',live='error',sparse=sparse,named=False,callback=False,refit=False,inverse=1)
   yield c,x.copy(),x.copy(),None,None

 # Source's undropped if_binary feature can broadcast all categories to zero-sum rows.
 x=np.array([[0,0],[1,1],[2,2],[2,2]],float)
 for sparse in [False,True]:
  for unknown in ['error','infrequent_if_exist','warn']:
   c=dict(n=4,p=2,m=4,count=0,fraction=0.,maximum=0,drop='if_binary',unknown=unknown,live=unknown,sparse=sparse,named=False,callback=False,refit=False,inverse=6)
   yield c,x.copy(),x.copy(),None,None
 # Floating category names exercise shortest round-trip formatting beyond hand-picked labels.
 for case in range(3):
  x=rng.normal(size=(41,2))*[1.,1e20] if case==0 else rng.normal(size=(41,2))*10.**rng.integers(-200,200,size=(41,2))
  for sparse in [False,True]:
   c=dict(n=41,p=2,m=7,count=0,fraction=0.,maximum=0,drop='none',unknown='error',live='error',sparse=sparse,named=False,callback=False,refit=False,inverse=1)
   yield c,x.copy(),x[:7].copy(),None,None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--start',type=int,default=0);ap.add_argument('--limit',type=int);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_one_hot_encoding.f90',ROOT/'src/sklearn_numpy_ordering.f90',ROOT/'app/reference_onehot_driver.f90',a.executable,Path(sklearn.__file__).parent/'preprocessing/_encoders.py',Path(sklearn.__file__).parent/'utils/_encode.py'];hashes={str(p):sha(p) for p in paths}
 configs=[];checks=[];rejections=[];nonfinite=0
 for ident,(c,x,q,cats,drops) in enumerate(fixtures()):
  if ident<a.start:continue
  if a.limit is not None and len(configs)>=a.limit:break
  c=dict(id=ident,**c);c['data_hashes']={name:hashlib.sha256(v.tobytes()).hexdigest() for name,v in [('x',x),('q',q)]+([] if drops is None else [('drop',drops)])+([] if cats is None else [(f'cat{j}',v) for j,v in enumerate(cats)])};configs.append(c)
  expected={};inverse=None;source_error=None;feature_names=[f'feature_{j}' for j in range(x.shape[1])]
  def callback(feature,category):return feature+('__rare' if isinstance(category,str) and category=='infrequent_sklearn' else '__number')
  def names(prefix,values):
   values=list(values);expected[prefix+'_length']=list(map(len,values));expected[prefix+'_codes']=[ord(ch) for v in values for ch in v]
  def state(prefix):
   expected[prefix+'_meta']=[model.n_features_in_,sum(model._n_features_outs)]
   expected[prefix+'_flags']=[model.drop_idx_ is not None,hasattr(model,'infrequent_categories_'),hasattr(model,'feature_names_in_')]
   if model.drop_idx_ is not None:expected[prefix+'_drop']=[0 if v is None else int(v)+1 for v in model.drop_idx_]
   for j,values in enumerate(model.categories_,1):
    expected[prefix+'_cat'+str(j)]=values
    if hasattr(model,'infrequent_categories_'):expected[prefix+'_rare'+str(j)]=[] if model.infrequent_categories_[j-1] is None else model.infrequent_categories_[j-1]
   names(prefix+'_names',model.get_feature_names_out())
  def mapped(prefix,values):
   z=model.transform(values)
   if c['sparse']:
    expected[prefix+'_ptr']=z.indptr+1;expected[prefix+'_idx']=z.indices+1;expected[prefix+'_data']=z.data
   expected[prefix+'_shape']=z.shape;expected[prefix+'_dense']=z.toarray() if c['sparse'] else z
   return z.toarray() if c['sparse'] else z
  with warnings.catch_warnings(),np.errstate(all='ignore'):
   warnings.simplefilter('ignore')
   try:
    model=OneHotEncoder(categories='auto' if cats is None else cats,drop=drops if drops is not None else None if c['drop']=='none' else c['drop'],handle_unknown=c['unknown'],min_frequency=c['count'] or c['fraction'] or None,max_categories=c['maximum'] or None,sparse_output=c['sparse'],feature_name_combiner=callback if c['callback'] else 'concat')
    train=pd.DataFrame(x,columns=feature_names) if c['named'] else x.copy();query=pd.DataFrame(q,columns=feature_names) if c['named'] and not c['refit'] else q.copy()
    model.fit(train);state('fit');train_z=mapped('train',train)
    if c['refit']:model.fit(x+.5);state('refit')
    model.handle_unknown=c['live'];query_z=mapped('query',query)
    if c['inverse']:
     width=sum(model._n_features_outs)
     if c['inverse']==1:inverse=train_z.copy()
     elif c['inverse']==2:inverse=query_z.copy()
     elif c['inverse']==3:inverse=np.tile(np.arange(1,width+1,dtype=float),(3,1))*.17
     elif c['inverse']==4:inverse=np.zeros((2,width))
     elif c['inverse']==6:inverse=np.zeros((3,width))
     else:
      inverse=np.ones((2,width));offset=0
      for k in model._n_features_outs:
       if k>0:inverse[:,offset]=-float(k-1)
       offset+=k
     if c['sparse']:
      nr,nc=inverse.shape;ptr=np.arange(0,2*nr*nc+1,2*nc) if nc else np.zeros(nr+1,int);idx=np.tile(np.repeat(np.arange(nc-1,-1,-1),2),nr)
      data=np.array([v*f for row in inverse for v in row[::-1] for f in [.25,.75]])
      probe=csr_matrix((data,idx,ptr),shape=inverse.shape)
     else:probe=inverse.copy()
     decoded=model.inverse_transform(probe);none=np.equal(decoded,None);rare=np.array([[isinstance(v,str) and v=='infrequent_sklearn' for v in row] for row in decoded])
     values=np.full(decoded.shape,np.nan)
     for i in range(len(decoded)):
      for j in range(decoded.shape[1]):
       if not none[i,j] and not rare[i,j]:values[i,j]=float(decoded[i,j])
     expected['inverse_shape']=decoded.shape;expected['inverse_values']=values;expected['inverse_none']=none;expected['inverse_infrequent']=rare
   except (ValueError,IndexError,TypeError,AttributeError) as e:source_error=e
  ir,ic=(0,0) if inverse is None else inverse.shape
  stream=io.StringIO();stream.write(f"{len(x)} {x.shape[1]} {len(q)} {c['count']} {c['fraction']} {c['maximum']} {int(c['sparse'])} {int(c['named'])} {int(cats is not None)} {int(drops is not None)} {int(inverse is not None)} {ir} {ic} {int(c['callback'])} {int(c['refit'])} {c['unknown']} {c['drop']} {c['live']}\n")
  for v in [x,q]:np.savetxt(stream,v,fmt='%.17g')
  if cats is not None:
   for values in cats:stream.write(str(len(values))+'\n');np.savetxt(stream,np.asarray(values).reshape(1,-1),fmt='%.17g')
  if drops is not None:np.savetxt(stream,drops.reshape(1,-1),fmt='%.17g')
  if inverse is not None:np.savetxt(stream,inverse,fmt='%.17g')
  proc=subprocess.run([str(a.executable)],input=stream.getvalue(),text=True,capture_output=True)
  if source_error is not None:
   assert proc.returncode!=0,(c,'accepted source rejection',type(source_error).__name__,str(source_error),proc.stdout)
   assert 'ERROR STOP' in proc.stderr and not any(t in proc.stderr for t in ['SIGFPE','Segmentation fault','Fortran runtime error']),(c,proc.stdout,proc.stderr)
   rejections.append(dict(configuration=ident,source_type=type(source_error).__name__,source_message=str(source_error),passed=True));continue
  assert proc.returncode==0,(c,proc.stdout,proc.stderr)
  actual={name:np.array(v,float) for name,*v in (line.split() for line in proc.stdout.splitlines())};assert actual.keys()==expected.keys(),(c,actual.keys(),expected.keys())
  for name,v in expected.items():
   want=np.asarray(v).ravel(order='F');got=actual[name];assert got.shape==want.shape,(c,name,got.shape,want.shape)
   assert np.array_equal(np.isnan(got),np.isnan(want)) and np.array_equal(np.isposinf(got),np.isposinf(want)) and np.array_equal(np.isneginf(got),np.isneginf(want)),(c,name,got,want)
   finite=np.isfinite(want);nonfinite+=int((~finite).sum());np.testing.assert_allclose(got[finite],want[finite],rtol=0,atol=0,err_msg=str((c,name)))
   checks.append(dict(configuration=ident,field=name,passed=True))
  if len(configs)%100==0:print(len(configs),'cases checked',flush=True)
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,hashes=hashes,configurations=configs,checks=checks,rejections=rejections,source_nonfinite_values=nonfinite),indent=2,allow_nan=False)+'\n');print(len(configs),'configurations',len(checks),'groups',len(rejections),'rejections',nonfinite,'nonfinite PASS')
if __name__=='__main__':main()
