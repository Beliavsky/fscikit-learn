"""KBinsDiscretizer sampling, one-hot CSR, names, live encoder and inverse contracts."""
from pathlib import Path
import argparse,hashlib,io,json,subprocess,warnings
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy.sparse import csr_matrix,issparse
from sklearn.preprocessing import KBinsDiscretizer
ROOT=Path(__file__).resolve().parents[1]
def fixtures():
 rng=np.random.default_rng(5201)
 for n,p in [(13,1),(31,2),(137,3)]:
  x=rng.normal(size=(n,p));q=rng.normal(size=(7,p))*2
  for strategy in ['uniform','quantile','kmeans']:
   for encode in ['ordinal','onehot-dense','onehot']:
    for sub in [0,7,n,n+1]:
     for external in [False,True]:
      c=dict(n=n,p=p,m=7,strategy=strategy,encode=encode,live=encode,refit=0,refit_encode=encode,subsample=sub,seed=42,external=external,sparse=encode=='onehot',named=p==2,weighted=False)
      yield c,x.copy(),q.copy(),None,None
 # Weighted quantiles and kmeans, including sklearn's post-sampling weight shape.
 for n,p in [(17,1),(33,2)]:
  x=rng.normal(size=(n,p));q=x[:5].copy()
  for strategy in ['quantile','kmeans']:
   for sub in [0,9]:
    for weights in ['positive','zeros','zero','negative','sample_length']:
     w=rng.uniform(.1,2,n);w[::4]=0
     if weights=='zero':w[:]=0
     if weights=='negative':w[::3]*=-1
     if weights=='sample_length':w=w[:9]
     for encode in ['ordinal','onehot','onehot-dense']:
      c=dict(n=n,p=p,m=5,strategy=strategy,encode=encode,live=encode,refit=0,refit_encode=encode,subsample=sub,seed=7,external=True,sparse=encode=='onehot',named=False,weighted=True,weights=weights)
      yield c,x.copy(),q.copy(),w.copy(),None
 # Constant/repeated/very narrow columns and all encoding changes.
 for case in range(4):
  x=np.arange(15,dtype=float).reshape(5,3)
  if case==0:x[:,1]=2
  if case==1:x=np.floor(x/6)
  if case==2:x*=1e-12
  if case==3:x[:]=3
  q=x.copy()
  for strategy in ['uniform','quantile','kmeans']:
   for encode in ['ordinal','onehot-dense','onehot']:
    for live in ['ordinal','onehot-dense','onehot']:
     c=dict(n=5,p=3,m=5,strategy=strategy,encode=encode,live=live,refit=0,refit_encode=encode,subsample=0,seed=0,external=False,sparse=encode=='onehot',named=True,weighted=False,case=case)
     yield c,x.copy(),q.copy(),None,None
 # Cached OneHotEncoder survives ordinal refits and keeps its old categories/names.
 x=rng.normal(size=(23,2));q=rng.normal(size=(6,2))*2
 for encode in ['ordinal','onehot-dense','onehot']:
  for refit_encode in ['ordinal','onehot-dense','onehot']:
   for refit in [2,7]:
    for live in ['ordinal','onehot-dense','onehot']:
     c=dict(n=23,p=2,m=6,strategy='uniform',encode=encode,live=live,refit=refit,refit_encode=refit_encode,subsample=0,seed=1,external=True,sparse=encode=='onehot',named=True,weighted=False)
     yield c,x.copy(),q.copy(),None,None
 # Soft, signed, fractional and Python-negative inverse indices.
 x=np.array([0.,1,2,3,4,5]).reshape(-1,1);q=x[:3].copy()
 for encode in ['ordinal','onehot-dense','onehot']:
  for case in range(6):
   if encode=='ordinal':a=np.array([[[-.7],[1.9],[-1.2]],[[0],[1],[2]],[[-3.9],[-2.1],[-1]],[[3]],[[1e100]],[[-4]]][case],dtype=float)
   else:a=np.array([[[.1,.7,.2],[2,2,1],[-2,-1,-3]],[[1,0,0],[0,1,0],[0,0,1]],[[0,0,0]],[[1,-1,0]],[[.2,.4,.8]],[[np.nan,0,1]]][case],dtype=float)
   c=dict(n=6,p=1,m=3,strategy='uniform',encode=encode,live=encode,refit=0,refit_encode=encode,subsample=0,seed=0,external=False,sparse=encode=='onehot',named=False,weighted=False,inverse_case=case,scalar_bins=3)
   yield c,x.copy(),q.copy(),None,a
 # Full uint32 scalar seeds, source default threshold and large efficient sorting.
 for n,sub in [(401,19),(200003,200000)]:
  x=rng.normal(size=(n,1));q=np.array([[-10.],[0.],[10.]])
  for seed in [0,1,4294967295]:
   c=dict(n=n,p=1,m=3,strategy='quantile',encode='ordinal',live='ordinal',refit=0,refit_encode='ordinal',subsample=sub,seed=seed,external=True,sparse=False,named=False,weighted=False)
   yield c,x.copy(),q.copy(),None,None
def signed_ties():
 rng=np.random.default_rng(931)
 for case in range(25):
  x=rng.integers(0,8,size=(33,1)).astype(float);w=rng.integers(-3,4,33).astype(float);q=np.arange(-1,9,dtype=float).reshape(-1,1)
  for encode in ['ordinal','onehot-dense','onehot']:
   c=dict(n=33,p=1,m=10,strategy='quantile',encode=encode,live=encode,refit=0,refit_encode=encode,subsample=0,seed=0,external=False,sparse=encode=='onehot',named=False,weighted=True,case=case,signed_ties=True)
   yield c,x.copy(),q.copy(),w.copy(),None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--start',type=int,default=0);ap.add_argument('--limit',type=int);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_discretization.f90',ROOT/'src/sklearn_numpy_ordering.f90',ROOT/'app/reference_kbins_driver.f90',a.executable,Path(sklearn.__file__).parent/'preprocessing/_discretization.py',Path(sklearn.__file__).parent/'preprocessing/_encoders.py',Path(sklearn.__file__).parent/'utils/_indexing.py',Path(sklearn.__file__).parent/'utils/stats.py'];hashes={str(p):sha(p) for p in paths}
 checks=[];rejections=[];configs=[];nonfinite=0
 from itertools import chain
 for index,(c,x,q,w,inv) in enumerate(chain(fixtures(),signed_ties())):
  if index<a.start:continue
  if a.limit is not None and len(configs)>=a.limit:break
  c=dict(id=index,**c);c['data_sha256']={name:hashlib.sha256(v.tobytes()).hexdigest() for name,v in [('x',x),('q',q)]+([] if w is None else [('w',w)])+([] if inv is None else [('inverse',inv)])};configs.append(c)
  counts=np.full(x.shape[1],c.get('scalar_bins',4))+np.arange(x.shape[1])%2
  ir,ic=(0,0) if inv is None else inv.shape
  s=io.StringIO();s.write(f"{len(x)} {x.shape[1]} {len(q)} {c['subsample']} {int(w is not None)} {0 if w is None else len(w)} {int(c['sparse'])} {int(c['named'])} {int(c['external'])} {c['seed']} {c['refit']} {ir} {ic} {c['strategy']} {c['encode']} {c['live']} {c['refit_encode']}\n");np.savetxt(s,counts.reshape(1,-1),fmt='%d')
  for v in [x,q]+([] if w is None else [w.reshape(1,-1)])+([] if inv is None else [inv]):np.savetxt(s,v,fmt='%.17g')
  proc=subprocess.run([str(a.executable)],input=s.getvalue(),text=True,capture_output=True)
  expected={};state=np.random.RandomState(c['seed']);features=[f'feature_{j}' for j in range(x.shape[1])];xfit=pd.DataFrame(x,columns=features) if c['named'] else x.copy();qfit=pd.DataFrame(q,columns=features) if c['named'] else q.copy()
  def emit_state(prefix):
   expected[prefix+'_counts']=model.n_bins_;expected[prefix+'_meta']=[model.n_features_in_,len(model.n_bins_) if model.encode=='ordinal' else sum(model.n_bins_)]
   for j,edges in enumerate(model.bin_edges_,1):expected[prefix+'_edges'+str(j)]=edges
   names=model.get_feature_names_out().tolist();expected[prefix+'_names_length']=list(map(len,names));expected[prefix+'_names_codes']=[ord(ch) for name in names for ch in name]
  def mapped(prefix,values,inverse=False):
   z=model.transform(values);sparse=c['sparse'] and model.encode!='ordinal'
   if sparse:
    z=csr_matrix(z);expected[prefix+'_ptr']=z.indptr+1;expected[prefix+'_idx']=z.indices+1;expected[prefix+'_data']=z.data
   expected[prefix+'_shape']=z.shape;expected[prefix+'_dense']=z.toarray() if issparse(z) else z
   if inverse:expected[prefix+'_inverse']=model.inverse_transform(z)
  with warnings.catch_warnings(),np.errstate(all='ignore'):
   warnings.simplefilter('ignore')
   try:
    model=KBinsDiscretizer(n_bins=counts,strategy=c['strategy'],encode=c['encode'],subsample=c['subsample'] or None,random_state=state if c['external'] else c['seed'])
    model.fit(xfit,sample_weight=None if w is None else w.copy());emit_state('fit');mapped('train',xfit,True)
    if c['refit']:
     model.n_bins=c['refit'];model.encode=c['refit_encode'];model.subsample=None;model.fit(xfit+.2);emit_state('refit')
    model.encode=c['live'];mapped('query',qfit)
    if inv is not None:
     if c['sparse']:
      nr,nc=inv.shape;ptr=np.arange(0,2*nr*nc+1,2*nc);idx=np.tile(np.repeat(np.arange(nc-1,-1,-1),2),nr)
      data=np.array([v*f for row in inv for v in row[::-1] for f in [.25,.75]])
      probe=csr_matrix((data,idx,ptr),shape=inv.shape)
     else:probe=inv.copy()
     expected['probe_inverse']=model.inverse_transform(probe)
    if c['external']:expected['rng']=state.randint(0,2**32,size=8,dtype=np.uint32)
   except (ValueError,IndexError,AttributeError) as e:
    assert proc.returncode!=0,(c,'accepted source rejection',type(e).__name__,str(e),proc.stdout)
    assert 'ERROR STOP' in proc.stderr and not any(t in proc.stderr for t in ['SIGFPE','Segmentation fault','Fortran runtime error']),(c,proc.stdout,proc.stderr)
    rejections.append(dict(configuration=index,source_type=type(e).__name__,source_message=str(e),passed=True));continue
  assert proc.returncode==0,(c,proc.stdout,proc.stderr)
  actual={name:np.array(v,float) for name,*v in (line.split() for line in proc.stdout.splitlines())};assert actual.keys()==expected.keys(),(c,actual.keys(),expected.keys())
  for name,v in expected.items():
   want=np.asarray(v).ravel(order='F');got=actual[name];assert got.shape==want.shape,(c,name,got.shape,want.shape)
   assert np.array_equal(np.isnan(got),np.isnan(want)) and np.array_equal(np.isposinf(got),np.isposinf(want)) and np.array_equal(np.isneginf(got),np.isneginf(want)),(c,name,got,want)
   finite=np.isfinite(want);nonfinite+=int((~finite).sum());exact=name=='rng' or name.endswith(('_counts','_meta','_shape','_ptr','_idx','_length','_codes')) or name.endswith(('_dense','_data'))
   np.testing.assert_allclose(got[finite],want[finite],rtol=0 if exact else 2e-10,atol=0 if exact else 2e-10,err_msg=str((c,name)))
   checks.append(dict(configuration=index,field=name,max_abs_error=float(np.max(np.abs(got[finite]-want[finite]),initial=0)),passed=True))
  if len(configs)%100==0:print(len(configs),'configurations checked',flush=True)
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,hashes=hashes,configurations=configs,checks=checks,rejections=rejections,source_nonfinite_values=nonfinite),indent=2,allow_nan=False)+'\n');print(len(configs),'configurations',len(checks),'groups',len(rejections),'rejections',nonfinite,'nonfinite PASS')
if __name__=='__main__':main()
