"""Source RobustScaler dense/CSR quantiles, optional statistics and robust_scale."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.preprocessing import RobustScaler,robust_scale
from compare_csr_projections import sparse,csr_payload
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def config(**kw):return dict(n=9,p=7,m=5,seed=0,fit_kind=0,query_kind=0,center=1,scaling=1,unit=0,axis=0,mode=0,action=0,qcenter=1,qscaling=1,low=25.,high=75.,style=0,scenario='random',qwidth=0)|kw
def fixtures():
 for seed,fit,query,center,scaling,unit,axis,(low,high),style in itertools.product([0,42,2026],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[(25.,75.),(0.,100.),(50.,50.),(10.,90.)],[0,1]):
  if fit and center:continue
  yield config(seed=seed,fit_kind=fit,query_kind=query,center=center,qcenter=center,scaling=scaling,qscaling=scaling,unit=unit,axis=axis,low=low,high=high,style=style)
 for fit,query,center,scaling,axis,scenario in itertools.product([0,1],[0,1],[0,1],[0,1],[0,1],['zero','missing','allnan','tiny','huge','huge_span']):
  if fit and center:continue
  yield config(fit_kind=fit,query_kind=query,center=center,qcenter=center,scaling=scaling,qscaling=scaling,axis=axis,scenario=scenario,style=1)
 for fit,query,unit,axis in itertools.product([0,1],[0,1],[0,1],[0,1]):
  yield config(fit_kind=fit,query_kind=query,center=0,qcenter=0,unit=unit,axis=axis,style=2)
 for query,axis in itertools.product([0,1],[0,1]):
  yield config(fit_kind=0,query_kind=query,axis=axis,qcenter=0,qscaling=0)
  if query==1:yield config(fit_kind=1,query_kind=1,center=0,qcenter=1,axis=axis)
 for low,high in [(1e-300,99.),(5e-322,99.),(0.,0.),(100.,100.),(.00001,99.99999)]:
  yield config(low=low,high=high,unit=1)
 for n in [599,600,601]:
  yield config(n=n,p=1,scenario='huge',style=1)
 for query,center,scaling in itertools.product([0,1],[0,1],[0,1]):
  # With disabled operations inverse accepts arbitrary widths; otherwise only
  # a scalar learned statistic can broadcast on dense inverse input.
  if query and scaling:continue
  yield config(query_kind=query,center=center,qcenter=center,scaling=scaling,qscaling=scaling,p=1,qwidth=5,action=1)
def matrix(n,p,style,seed):
 if style<2:return sparse(n,p,style,seed)
 # Raw duplicates fit within source column/row storage limits on both axes.
 data=[];idx=[];ptr=[0]
 for i in range(n):
  if i%3==0 and p:
   j=i%p;data.extend([.25*(i+1),.75*(i+1)]);idx.extend([j,j])
  ptr.append(len(data))
 return sp.csr_matrix((data,idx,ptr),shape=(n,p))
def inputs(c):
 arrays=[matrix(n,p,c['style'],c['seed']+offset) for n,p,offset in [(c['n'],c['p'],0),(c['m'],c['qwidth'] or c['p'],2)]]
 for a in arrays:
  if c['scenario'] in ['zero','allnan','tiny','huge']:a.data.fill(dict(zero=0.,allnan=np.nan,tiny=1e-310,huge=1e308)[c['scenario']])
  if c['scenario']=='huge_span':a.data[::2]=-1e308;a.data[1::2]=1e308
  if c['scenario']=='missing':a.data[::3]=np.nan
 return arrays
def put(f,prefix,a):
 sparse=sp.issparse(a);f.update({prefix+'_shape':a.shape,prefix+'_ptr':a.indptr+1 if sparse else [],prefix+'_idx':a.indices+1 if sparse else [],prefix+'_data':a.data if sparse else a})
def oracle(c,arrays):
 x,q=[a.copy() if kind else a.toarray() for a,kind in zip(arrays,[c['fit_kind'],c['query_kind']])];original=[a.copy() for a in [x,q]];state=np.random.get_state()
 m=RobustScaler(with_centering=bool(c['center']),with_scaling=bool(c['scaling']),quantile_range=(c['low'],c['high']),unit_variance=bool(c['unit']))
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore');m.fit(x);f=dict(center=[] if m.center_ is None else m.center_.copy(),scale=[] if m.scale_ is None else m.scale_.copy(),metadata=[m.n_features_in_,int(m.center_ is not None),int(m.scale_ is not None)])
  put(f,'train',m.transform(x));m.with_centering=bool(c['qcenter']);m.with_scaling=bool(c['qscaling'])
  if c['action']!=1:put(f,'query',m.transform(q))
  put(f,'inverse',m.inverse_transform(q));put(f,'free',robust_scale(x,with_centering=bool(c['center']),with_scaling=bool(c['scaling']),quantile_range=(c['low'],c['high']),unit_variance=bool(c['unit']),axis=c['axis']))
  f['vector']=robust_scale(x.ravel(order='F'),with_centering=bool(c['center']),with_scaling=bool(c['scaling']),quantile_range=(c['low'],c['high']),unit_variance=bool(c['unit']),axis=c['axis']) if not sp.issparse(x) else []
  same=lambda a,b:all(np.array_equal(getattr(a,k),getattr(b,k),equal_nan=True) for k in ['indptr','indices','data']) if sp.issparse(a) else np.array_equal(a,b,equal_nan=True)
  assert all(same(a,b) for a,b in zip([x,q],original)) and all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state()))
  f['preserved']=[1]*3;m.with_centering=bool(c['center']);m.with_scaling=bool(c['scaling']);m.fit(arrays[1].toarray())
  f.update(refit_center=[] if m.center_ is None else m.center_,refit_scale=[] if m.scale_ is None else m.scale_,refit=[m.n_features_in_,int(m.center_ is not None),int(m.scale_ is not None)])
 return f
def payload(c,arrays):return ' '.join(str(c[k]) for k in ['fit_kind','query_kind','center','scaling','unit','axis','mode','action','qcenter','qscaling','low','high'])+'\n'+''.join(csr_payload(a) for a in arrays)
def reject_cases(executable):
 result=[]
 variants=[('unfitted',dict(mode=1)),('inf_fit',dict(mode=2)),('inf_query',dict(mode=3)),('empty_fit',dict(n=0)),('empty_features',dict(p=0)),('empty_query',dict(m=0)),('query_width',dict(qwidth=5)),('inverse_width',dict(qwidth=1,action=1)),('low_negative',dict(low=-1.)),('high_over',dict(high=101.)),('reverse',dict(low=80.,high=20.)),('low_nan',dict(low=float('nan'))),('high_inf',dict(high=float('inf'))),('bad_axis',dict(axis=2)),('missing_center',dict(center=0,qcenter=1)),('missing_scale',dict(scaling=0,qscaling=1))]
 for fit,query in itertools.product([0,1],[0,1]):
  for name,change in variants:
   if name=='missing_center' and query:continue
   c=config(fit_kind=fit,query_kind=query,center=0 if fit else 1,qcenter=0 if fit else 1,style=1)|change
   # Ensure inverse mismatch is active on either representation.
   arrays=inputs(c);wire=payload(c,arrays);x,q=arrays
   if c['mode']==2:x.data[0]=np.inf
   if c['mode']==3:q.data[0]=np.inf
   x=x if fit else x.toarray();q=q if query else q.toarray();m=RobustScaler(with_centering=bool(c['center']),with_scaling=bool(c['scaling']),quantile_range=(c['low'],c['high']))
   error=None
   try:
    with warnings.catch_warnings(),np.errstate(all='ignore'):
     warnings.simplefilter('ignore')
     if c['mode']==1:m.transform(q)
     m.fit(x);m.with_centering=bool(c['qcenter']);m.with_scaling=bool(c['qscaling'])
     if c['action']!=1:m.transform(q)
     m.inverse_transform(q);robust_scale(x,with_centering=bool(c['center']),with_scaling=bool(c['scaling']),quantile_range=(c['low'],c['high']),axis=c['axis'])
   except (ValueError,TypeError,AssertionError) as e:error=type(e).__name__
   assert error is not None,('source accepts invalid',name,c)
   reject(executable,wire,name);result.append(dict(name=f'{fit}_{query}_{name}',passed=True,oracle='sklearn_estimator',source_error=error))
 for name,c in [('sparse_center',config(fit_kind=1,style=1)),('column_raw_count',config(fit_kind=1,center=0,qcenter=0,n=1,p=1,style=2)),('row_raw_count',config(fit_kind=1,center=0,qcenter=0,p=1,style=2,axis=1))]:
  arrays=inputs(c)
  try:
   with warnings.catch_warnings(),np.errstate(all='ignore'):
    warnings.simplefilter('ignore');robust_scale(arrays[0].copy(),with_centering=bool(c['center']),axis=c['axis'])
  except ValueError:pass
  else:raise AssertionError(('source accepts invalid',name))
  reject(executable,payload(c,arrays),name);result.append(dict(name=name,passed=True,oracle='sklearn_estimator',source_error='ValueError'))
 for mode,name in [(4,'bad_index'),(5,'bad_start'),(6,'bad_pointer')]:
  c=config(fit_kind=1,center=0,qcenter=0,style=1,mode=mode);arrays=inputs(c);wire=payload(c,arrays);x=arrays[0]
  if mode==4:x.indices[0]=-1
  if mode==5:x.indptr[0]=-1
  if mode==6:x.indptr[1]=len(x.data)+1
  try:x.check_format(full_check=True)
  except ValueError:pass
  else:raise AssertionError(name)
  reject(executable,wire,name);result.append(dict(name=name,passed=True,oracle='scipy_csr_check_format'))
 c=config(fit_kind=1,center=0,qcenter=0);x=sp.csr_matrix(([1.],[2147483646],[0,1]),shape=(1,2147483647));q=sp.csr_matrix((1,1))
 reject(executable,payload(c,[x,q]),'feature_capacity','overflow');result.append(dict(name='feature_capacity',passed=True,oracle='native_integer_capacity'))
 return result
def reject(executable,wire,name,message='fscikit-learn:'):
 r=subprocess.run([str(executable)],input='1\n'+wire,text=True,capture_output=True);out=r.stdout+r.stderr
 assert r.returncode and 'fscikit-learn:' in out and message in out and not any(v in out for v in ['SIGFPE','Fortran runtime error']),(name,out)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
 paths=[Path(__file__),args.executable,ROOT/'validation/compare_csr_projections.py',ROOT/'src/sklearn_robust_scaling.f90',ROOT/'src/sklearn_preprocessing.f90',ROOT/'src/sklearn_frozen.f90',ROOT/'app/reference_robust_driver.f90',Path(sklearn.__file__).parent/'preprocessing/_data.py'];hashes={str(p):sha(p) for p in paths}
 cases=list(fixtures());assert len({json.dumps(c,sort_keys=True) for c in cases})==len(cases);arrays=[inputs(c) for c in cases];expected=[oracle(c,a) for c,a in zip(cases,arrays)]
 r=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,arrays)),capture_output=True,text=True);assert r.returncode==0,(r.stdout[-800:],r.stderr[-1500:])
 lines=iter(r.stdout.splitlines());index=-1;checks=[];inventories=[];nonfinite=0
 for line in lines:
  p=line.split()
  if not p:continue
  if p[0]=='BEGIN':index+=1;fields=set();assert int(p[1])==index+1
  elif p[0]=='END':assert fields==expected[index].keys();inventories.append(sorted(fields))
  else:
   field=p[0];assert field not in fields;fields.add(field);values=np.fromstring(next(lines),sep=' ');want=np.asarray(expected[index][field]).ravel(order='F');assert len(values)==int(p[1]) and values.shape==want.shape,(index,field,values.shape,want.shape)
   if field.endswith(('_shape','_ptr','_idx')) or field in ['metadata','preserved','refit']:np.testing.assert_array_equal(values,want,err_msg=str((index,cases[index],field)))
   else:np.testing.assert_allclose(values,want,rtol=3e-8,atol=3e-8,equal_nan=True,err_msg=str((index,cases[index],field)))
   finite=np.isfinite(want);nonfinite+=int(np.count_nonzero(~finite));checks.append(dict(fixture=index,field=field,passed=True,oracle='native_input_rng_identity' if field=='preserved' else 'sklearn',max_abs_error=float(np.max(np.abs(values[finite]-want[finite]),initial=0))))
 assert index+1==len(cases) and hashes=={str(p):sha(p) for p in paths}
 rejections=reject_cases(args.executable)
 d=dict(rejections=rejections,profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,source_nonfinite_values=nonfinite,hashes=hashes)
 (args.report or ROOT/'validation/results'/f'robust_{args.profile}.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(len(cases),len(checks),len(rejections),nonfinite,'PASS')
if __name__=='__main__':main()
