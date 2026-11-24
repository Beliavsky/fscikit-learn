"""MaxAbsScaler dense/CSR streaming, NaNs, inverse broadcasting and maxabs_scale."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.preprocessing import MaxAbsScaler,maxabs_scale
from compare_csr_projections import sparse,csr_payload
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def config(**kw):return dict(n=7,p=3,m=4,nb=3,seed=0,fit_kind=1,batch_kind=1,query_kind=1,stream=0,axis=0,mode=0,action=0,style=0,scenario='random',qwidth=0,bwidth=0)|kw
def fixtures():
 for seed,(n,p),fit,batch,query,style,stream,axis in itertools.product([0,7,42,2026,2147483647],[(7,1),(9,3),(5,7)],[0,1],[0,1],[0,1],range(4),[0,1],[0,1]):
  yield config(seed=seed,n=n,p=p,fit_kind=fit,batch_kind=batch,query_kind=query,style=style,stream=stream,axis=axis)
 for fit,batch,query,style,stream,axis,scenario in itertools.product([0,1],[0,1],[0,1],range(3),[0,1],[0,1],['zero','tiny','threshold','huge','allnan','missing','threshold_at','threshold_above']):
  yield config(fit_kind=fit,batch_kind=batch,query_kind=query,style=style,stream=stream,axis=axis,scenario=scenario)
 for fit,batch,query,axis,style,action in itertools.product([0,1],[0,1],[0,1],[0,1],range(4),[1,2]):
  yield config(fit_kind=fit,batch_kind=batch,query_kind=query,axis=axis,style=style,action=action,scenario='zero' if action==1 else 'random')
 for fit,batch,axis,stream in itertools.product([0,1],[0,1],[0,1],[0,1]):
  yield config(fit_kind=fit,batch_kind=batch,query_kind=0,axis=axis,stream=stream,action=3,p=1,qwidth=5)
 for fit,batch,query,axis,style,scenario in itertools.product([0,1],[0,1],[0,1],[0,1],range(3),['batchnan','recover_nan']):
  yield config(fit_kind=fit,batch_kind=batch,query_kind=query,axis=axis,style=style,scenario=scenario,stream=1)
 for query,stream in itertools.product([0,1],[0,1]):yield config(query_kind=query,stream=stream,scenario='overflow',style=2)
def inputs(c):
 arrays=[sparse(n,p,c['style'],c['seed']+offset) for n,p,offset in [(c['n'],c['p'],0),(c['nb'],c['bwidth'] or c['p'],1),(c['m'],c['qwidth'] or c['p'],2)]]
 for a in arrays:
  if c['scenario'] in ['zero','tiny','threshold','huge','allnan','threshold_at','threshold_above']:
   v=dict(zero=0.,tiny=1e-310,threshold=2e-15,huge=1e308,allnan=np.nan,threshold_at=10*np.finfo(float).eps,threshold_above=np.nextafter(10*np.finfo(float).eps,np.inf))[c['scenario']]
   a.data.fill(v/(3 if c['style']==2 else 1))
  if c['scenario']=='missing':
   for i in range(a.shape[0]):
    begin,end=a.indptr[i:i+2];mask=((a.indices[begin:end]+i)%3)==0;a.data[begin:end][mask]=np.nan
 if c['scenario']=='batchnan':arrays[1].data.fill(np.nan)
 if c['scenario']=='recover_nan':arrays[0].data.fill(np.nan)
 if c['scenario']=='overflow':arrays[0].data.fill(1e308);arrays[1].data.fill(1e308)
 return arrays
def same(a,b):
 return all(np.array_equal(getattr(a,k),getattr(b,k),equal_nan=True) for k in ['indptr','indices','data']) if sp.issparse(a) else np.array_equal(a,b,equal_nan=True)
def emit(fields,prefix,a):
 issparse=sp.issparse(a);fields.update({prefix+'_shape':a.shape,prefix+'_ptr':a.indptr+1 if issparse else [],prefix+'_idx':a.indices[:a.nnz]+1 if issparse else [],prefix+'_data':a.data[:a.nnz] if issparse else a})
def oracle(c,arrays):
 x,b,q=[a.copy() if kind else a.toarray() for a,kind in zip(arrays,[c['fit_kind'],c['batch_kind'],c['query_kind']])];original=[a.copy() for a in [x,b,q]];state=np.random.get_state();model=MaxAbsScaler();fields={}
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore');model.fit(x)
  if c['stream']:model.partial_fit(b)
  if c['action']==1:model.scale_.fill(0)
  if c['action']==2:model.scale_=-model.scale_
  fields.update(maximum=model.max_abs_.copy(),scale=model.scale_.copy(),metadata=[model.n_features_in_,model.n_samples_seen_])
  if c['action']==3:back=model.inverse_transform(q)
  else:
   mapped=model.transform(q);emit(fields,'mapped',mapped);back=model.inverse_transform(mapped)
  emit(fields,'inverse',back)
  free_x=x.copy();free_original=free_x.copy();free=maxabs_scale(free_x,axis=c['axis']);emit(fields,'free',free)
  fields['vector']=maxabs_scale(x.ravel(order='F'),axis=c['axis']) if not sp.issparse(x) else []
  mutation=[not same(a,b) for a,b in zip([x,b,q],original)]+[not same(free_x,free_original)]
  assert all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state()))
  fields['preserved']=[1]*4
  model.fit(arrays[2].copy());fields['refit']=[model.n_features_in_,model.n_samples_seen_];fields['refit_maximum']=model.max_abs_
 return fields,mutation
def payload(c,arrays):return ' '.join(str(c[k]) for k in ['fit_kind','batch_kind','query_kind','stream','axis','mode','action'])+'\n'+''.join(csr_payload(a) for a in arrays)
def rejection_cases(executable):
 result=[]
 for fit,batch,query in itertools.product([0,1],[0,1],[0,1]):
  variants=[('unfitted',dict(mode=1)),('inf_fit',dict(mode=2)),('inf_query',dict(mode=3)),('inf_batch',dict(mode=4,stream=1)),('empty_fit',dict(n=0)),('empty_features',dict(p=0)),('empty_query',dict(m=0)),('empty_batch',dict(nb=0,stream=1)),('batch_width',dict(bwidth=5,stream=1)),('query_width',dict(qwidth=5)),('bad_axis',dict(axis=2)),('inverse_width',dict(action=3,qwidth=5))]
  for name,change in variants:
   c=config(style=1,fit_kind=fit,batch_kind=batch,query_kind=query)|change;arrays=inputs(c);wire=payload(c,arrays);x,b,q=arrays;error=None
   if c['mode']==2:x.data[0]=np.inf
   if c['mode']==3:q.data[0]=np.inf
   if c['mode']==4:b.data[0]=np.inf
   x,b,q=[a if kind else a.toarray() for a,kind in zip(arrays,[fit,batch,query])];model=MaxAbsScaler()
   try:
    with warnings.catch_warnings(),np.errstate(all='ignore'):
     warnings.simplefilter('ignore')
     if c['mode']==1:model.transform(q)
     model.fit(x)
     if c['stream']:model.partial_fit(b)
     if c['action']==3:model.inverse_transform(q)
     else:model.transform(q)
     maxabs_scale(x,axis=c['axis'])
   except (ValueError,AssertionError) as err:error=type(err).__name__
   assert error is not None,('source accepts invalid',name,c)
   reject(executable,wire,name);result.append(dict(name=f'{fit}_{batch}_{query}_{name}',passed=True,oracle='sklearn_estimator',source_error=error))
 for mode,name in [(5,'bad_index'),(6,'bad_start'),(7,'bad_pointer')]:
  c=config(mode=mode,style=1);arrays=inputs(c);wire=payload(c,arrays);x=arrays[0]
  if mode==5:x.indices[0]=-1
  if mode==6:x.indptr[0]=-1
  if mode==7:x.indptr[1]=len(x.data)+1
  try:x.check_format(full_check=True)
  except ValueError:pass
  else:raise AssertionError(('scipy accepts malformed CSR',name))
  reject(executable,wire,name);result.append(dict(name=name,passed=True,oracle='scipy_csr_check_format'))
 c=config(mode=8,stream=1);reject(executable,payload(c,inputs(c)),'sample_count','overflow')
 result.append(dict(name='sample_count',passed=True,oracle='native_int64_capacity'))
 c=config(style=2,scenario='overflow',axis=1);arrays=inputs(c)
 try:
  with np.errstate(all='ignore'):maxabs_scale(arrays[0].copy(),axis=1)
 except ValueError:pass
 else:raise AssertionError('source accepts overflowing row canonicalization')
 reject(executable,payload(c,arrays),'row_overflow');result.append(dict(name='row_overflow',passed=True,oracle='sklearn_estimator',source_error='ValueError'))
 return result
def reject(executable,wire,name,message='fscikit-learn:'):
 r=subprocess.run([str(executable)],input='1\n'+wire,text=True,capture_output=True);out=r.stdout+r.stderr
 assert r.returncode and 'fscikit-learn:' in out and message in out and not any(v in out for v in ['SIGFPE','Fortran runtime error']),(name,out)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
 paths=[Path(__file__),args.executable,ROOT/'validation/compare_csr_projections.py',ROOT/'src/sklearn_maxabs_scaling.f90',ROOT/'src/sklearn_preprocessing.f90',ROOT/'src/sklearn_stateless_preprocessing.f90',ROOT/'src/sklearn_sparse.f90',ROOT/'app/reference_maxabs_driver.f90',Path(sklearn.__file__).parent/'preprocessing/_data.py'];hashes={str(p):sha(p) for p in paths}
 cases=list(fixtures());assert len({json.dumps(c,sort_keys=True) for c in cases})==len(cases);arrays=[inputs(c) for c in cases];expected=[oracle(c,a) for c,a in zip(cases,arrays)]
 r=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,arrays)),capture_output=True,text=True)
 assert r.returncode==0,(r.stdout[-800:],r.stderr[-1400:])
 lines=iter(r.stdout.splitlines());index=-1;checks=[];inventories=[];nonfinite=0
 for line in lines:
  p=line.split()
  if not p:continue
  if p[0]=='BEGIN':index+=1;fields=set();assert int(p[1])==index+1
  elif p[0]=='END':assert fields==expected[index][0].keys();inventories.append(sorted(fields))
  else:
   field=p[0];assert field not in fields;fields.add(field);values=np.fromstring(next(lines),sep=' ');want=np.asarray(expected[index][0][field]).ravel(order='F');assert len(values)==int(p[1]) and values.shape==want.shape,(index,field,values.shape,want.shape)
   if field.endswith('_data') or field in ['maximum','scale','vector','refit_maximum']:np.testing.assert_allclose(values,want,rtol=3e-8,atol=3e-8,equal_nan=True,err_msg=str((index,cases[index],field)))
   else:np.testing.assert_array_equal(values,want,err_msg=str((index,cases[index],field)))
   finite=np.isfinite(want);nonfinite+=int(np.count_nonzero(~finite));checks.append(dict(fixture=index,field=field,passed=True,oracle='native_input_rng_identity' if field=='preserved' else 'sklearn',max_abs_error=float(np.max(np.abs(values[finite]-want[finite]),initial=0))))
 assert index+1==len(cases) and hashes=={str(p):sha(p) for p in paths}
 rejections=rejection_cases(args.executable)
 d=dict(rejections=rejections,profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,source_nonfinite_values=nonfinite,source_input_mutation=[a[1] for a in expected],hashes=hashes)
 (args.report or ROOT/'validation/results'/f'maxabs_{args.profile}.json').write_text(json.dumps(d,indent=2)+'\n');print(len(cases),len(checks),len(rejections),nonfinite,'PASS')
if __name__=='__main__':main()
