"""Source SplineTransformer weighted knots, degree zero, CSR slots and live controls."""
from pathlib import Path
import argparse,hashlib,io,json,subprocess,warnings
from contextlib import nullcontext
from unittest.mock import patch
import numpy as np
import pandas as pd
import scipy,sklearn
from sklearn.preprocessing import SplineTransformer
ROOT=Path(__file__).resolve().parents[1]
def fixtures():
 rng=np.random.default_rng(7231)
 for p in [1,2,3]:
  x=rng.normal(size=(31,p));q=rng.normal(size=(7,p))*3
  for degree in [0,1,2,3,5]:
   for mode in ['uniform','quantile','explicit']:
    for extra in ['constant','linear','continue','periodic','error']:
     for sparse in [False,True]:
      bias=bool((degree+p+int(sparse))%2);weighted=mode!='explicit';w=rng.uniform(.1,2,31);w[:3]=0
      c=dict(p=p,n=31,m=7,nk=6,degree=degree,mode=mode,extra=extra,sparse=sparse,bias=bias,weighted=weighted,named=p==2,live_degree=degree,live_bias=bias,live_extra=extra)
      if extra=='error':query=x[:7].copy()
      else:query=q.copy()
      yield c,x,query,w
 # Repeated generated knots, degenerate ranges, zero and negative weights.
 for case in range(7):
  x=np.array([0,0,0,1,2,3,4.]).reshape(-1,1);w=np.ones(7)
  if case==1:x[:]=1
  if case==2:w[:]=0
  if case==3:w[:]=[-1,2,1,1,1,1,1]
  if case==4:w[:]=[-2,-1,-3,-1,-1,-2,-1]
  if case==5:w[:]=[0,0,0,0,0,1,0]
  if case==6:x=np.array([3,1,2,0,0,1,2.]).reshape(-1,1);w[:]=[1,-1,2,1,-2,3,1]
  q=np.array([-1,0,.5,1,2,4,5.]).reshape(-1,1)
  for degree in [0,1,3]:
   for mode in ['uniform','quantile']:
    for extra in ['constant','linear','continue','periodic']:
     for sparse in [False,True]:
      c=dict(p=1,n=7,m=7,nk=4,degree=degree,mode=mode,extra=extra,sparse=sparse,bias=True,weighted=case!=0,named=False,live_degree=degree,live_bias=True,live_extra=extra,case=case)
      yield c,x.copy(),q.copy(),w.copy()
 # Live controls are queried after training with separate fitted options.
 x=rng.normal(size=(17,2));q=rng.normal(size=(5,2))*4
 for degree in [0,1,3]:
  for extra in ['constant','linear','continue','periodic']:
   for live in ['constant','linear','continue','periodic','error']:
    for sparse in [False,True]:
     c=dict(p=2,n=17,m=5,nk=5,degree=degree,mode='uniform',extra=extra,sparse=sparse,bias=True,weighted=False,named=True,live_degree=degree,live_bias=True,live_extra=live)
     yield c,x.copy(),q.copy(),np.ones(17)
 for sparse in [False,True]:
  for bias,lb in [(True,False),(False,True)]:
   for degree,ld in [(3,1),(1,3),(3,3)]:
    c=dict(p=2,n=17,m=5,nk=5,degree=degree,mode='uniform',extra='constant',sparse=sparse,bias=bias,weighted=False,named=False,live_degree=ld,live_bias=lb,live_extra='linear')
    yield c,x.copy(),q.copy(),np.ones(17)
def negative_tie_fixtures():
 rng=np.random.default_rng(931)
 for case in range(25):
  x=rng.integers(0,8,size=(33,1)).astype(float);w=rng.integers(-3,4,33).astype(float)
  q=np.arange(-1,9,dtype=float).reshape(-1,1)
  for degree in [0,1,3]:
   for sparse in [False,True]:
    c=dict(p=1,n=33,m=10,nk=5,degree=degree,mode='quantile',extra='continue',sparse=sparse,bias=True,weighted=True,named=False,live_degree=degree,live_bias=True,live_extra='continue',case=case,endpoint_diagnostic=True)
    yield c,x.copy(),q.copy(),w.copy()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--stable-ties',action='store_true');ap.add_argument('--limit',type=int);ap.add_argument('--start',type=int,default=0);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_spline.f90',ROOT/'app/reference_spline_extended_driver.f90',a.executable,Path(sklearn.__file__).parent/'preprocessing/_polynomial.py',Path(sklearn.__file__).parent/'utils/stats.py',Path(scipy.__file__).parent/'interpolate/_bsplines.py'];hashes={str(p):sha(p) for p in paths}
 checks=[];rejects=[];configs=[];nonfinite=0;endpoints=[]
 from itertools import chain
 def stable_fixtures():
  for c,x,q,w in negative_tie_fixtures():
   c=dict(c);c.pop('endpoint_diagnostic');yield c,x,q,w
 argsort=np.argsort
 def stable_argsort(x,*args,**kwargs):
  kwargs['kind']='stable';return argsort(x,*args,**kwargs)
 selected=stable_fixtures() if a.stable_ties else chain(fixtures(),negative_tie_fixtures())
 for fixture,(c,x,q,w) in enumerate(selected):
  if fixture<a.start:continue
  if a.limit is not None and len(configs)>=a.limit:break
  c=dict(id=fixture,**c);diagnostic=c.get('endpoint_diagnostic',False);c['x_sha256']=hashlib.sha256(x.tobytes()).hexdigest();c['q_sha256']=hashlib.sha256(q.tobytes()).hexdigest();c['w_sha256']=hashlib.sha256(w.tobytes()).hexdigest();configs.append(c)
  explicit=c['mode']=='explicit';knots=x.min(axis=0)+(x.max(axis=0)-x.min(axis=0))*np.linspace(0,1,c['nk'])[:,None]**1.4
  stream=io.StringIO();stream.write(f"{len(x)} {x.shape[1]} {len(q)} {c['nk']} {c['degree']} {int(c['bias'])} {int(c['weighted'])} {int(explicit)} {int(c['sparse'])} {int(c['named'])} {'uniform' if explicit else c['mode']} {c['extra']} {c['live_degree']} {int(c['live_bias'])} {c['live_extra']}\n")
  for array in [x,q,w.reshape(1,-1),knots]:np.savetxt(stream,array,fmt='%.17g')
  proc=subprocess.run([str(a.executable)],input=stream.getvalue(),text=True,capture_output=True)
  expected={}
  def mapped(name,z):
   expected[name+'_shape']=np.array(z.shape)
   if c['sparse']:
    expected[name+'_ptr']=z.indptr+1;expected[name+'_idx']=z.indices+1;expected[name+'_data']=z.data;z=z.toarray()
   expected[name+'_dense']=z
  with warnings.catch_warnings(),np.errstate(all='ignore'):
   warnings.simplefilter('ignore')
   try:
    model=SplineTransformer(n_knots=c['nk'],degree=c['degree'],knots=knots if explicit else c['mode'],extrapolation=c['extra'],include_bias=c['bias'],sparse_output=c['sparse'])
    if c['named']:xfit=pd.DataFrame(x,columns=[f'feature_{i}' for i in range(x.shape[1])])
    else:xfit=x.copy()
    with patch.object(np,'argsort',side_effect=stable_argsort) if a.stable_ties else nullcontext():
     model.fit(xfit,sample_weight=w.copy() if c['weighted'] else None)
    extended=np.column_stack([b.t for b in model.bsplines_]);expected['base']=extended[c['degree']:c['degree']+c['nk']];expected['knots']=extended;expected['meta']=np.array([model.n_features_in_,model.n_features_out_])
    mapped('train',model.transform(xfit))
    model.degree=c['live_degree'];model.include_bias=c['live_bias'];model.extrapolation=c['live_extra']
    mapped('query',model.transform(pd.DataFrame(q,columns=xfit.columns) if c['named'] else q.copy()))
    names=model.get_feature_names_out().tolist();expected['names_length']=np.array(list(map(len,names)));expected['names_codes']=np.array([ord(ch) for name in names for ch in name])
   except (ValueError,IndexError,TypeError) as e:
    if diagnostic:
     if proc.returncode!=0:assert 'ERROR STOP' in proc.stderr and not any(t in proc.stderr for t in ['SIGFPE','Segmentation fault','Fortran runtime error'])
     endpoints.append(dict(configuration=c['id'],field='acceptance',passed=proc.returncode!=0,source_type=type(e).__name__,source_message=str(e),native_accepted=proc.returncode==0));continue
    assert proc.returncode!=0,(c,'accepted source rejection',type(e).__name__,str(e),proc.stdout)
    assert 'ERROR STOP' in proc.stderr and not any(s in proc.stderr for s in ['SIGFPE','Segmentation fault','Fortran runtime error']),(c,proc.stdout,proc.stderr)
    rejects.append(dict(configuration=c['id'],source_type=type(e).__name__,source_message=str(e),passed=True));continue
  if diagnostic and proc.returncode!=0:
   assert 'ERROR STOP' in proc.stderr and not any(t in proc.stderr for t in ['SIGFPE','Segmentation fault','Fortran runtime error'])
   endpoints.append(dict(configuration=c['id'],field='acceptance',passed=False,source_type='accepted',native_accepted=False,native_message=proc.stdout.strip()));continue
  assert proc.returncode==0,(c,proc.stdout,proc.stderr)
  actual={k:np.array(v,float) for k,*v in (line.split() for line in proc.stdout.splitlines())};assert actual.keys()==expected.keys(),(c,actual.keys(),expected.keys())
  for k,want in expected.items():
   want=np.asarray(want).ravel(order='F');got=actual[k]
   if diagnostic and got.shape!=want.shape:
    endpoints.append(dict(configuration=c['id'],field=k,passed=False,source_shape=list(want.shape),native_shape=list(got.shape)));continue
   assert got.shape==want.shape,(c,k,got.shape,want.shape)
   if diagnostic:
    masks=np.isnan(got)==np.isnan(want);masks&=np.isposinf(got)==np.isposinf(want);masks&=np.isneginf(got)==np.isneginf(want)
    finite=np.isfinite(got)&np.isfinite(want);err=float(np.max(np.abs(got[finite]-want[finite]),initial=0))
    endpoints.append(dict(configuration=c['id'],field=k,passed=bool(masks.all() and np.allclose(got[finite],want[finite],rtol=3e-8,atol=3e-8)),nonfinite_mismatch=int((~masks).sum()),max_abs_error=err));continue
   assert np.array_equal(np.isnan(got),np.isnan(want)) and np.array_equal(np.isposinf(got),np.isposinf(want)) and np.array_equal(np.isneginf(got),np.isneginf(want)),(c,k,got,want)
   finite=np.isfinite(want);nonfinite+=int((~finite).sum());exact=k.endswith(('_ptr','_idx','_shape','_length','_codes')) or k=='meta'
   np.testing.assert_allclose(got[finite],want[finite],rtol=0 if exact else 3e-8,atol=0 if exact else 3e-8,err_msg=str((c,k)))
   checks.append(dict(configuration=c['id'],field=k,max_abs_error=float(np.max(np.abs(got[finite]-want[finite]),initial=0)),passed=True))
  if len(configs)%100==0:print(len(configs),'fixtures checked',flush=True)
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,oracle_policy='sklearn with NumPy argsort kind=stable during fit' if a.stable_ties else 'unmodified installed sklearn',sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,hashes=hashes,configurations=configs,checks=checks,rejections=rejects,endpoint_diagnostics=endpoints,source_nonfinite_values=nonfinite),indent=2,allow_nan=False)+'\n')
 print('Endpoint diagnostics:',len(endpoints),'differences:',sum(not d['passed'] for d in endpoints))
 print(len(configs),'configurations',len(checks),'groups',len(rejects),'rejections',nonfinite,'nonfinite PASS')
if __name__=='__main__':main()
