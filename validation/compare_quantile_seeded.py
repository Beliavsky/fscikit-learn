"""QuantileTransformer source-seeded dense/CSR fits and free quantile maps."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.preprocessing import QuantileTransformer,quantile_transform
from compare_robust import matrix,put
from compare_csr_projections import csr_payload
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def config(**kw):return dict(seed=0,fit=0,query=0,nq=5,subsample=0,distribution=0,ignore=0,axis=0,qdist=0,qignore=0,mode=0,n=31,p=7,m=9,style=1,scenario='random',qwidth=0)|kw

def fixtures():
 for seed,fit,query,ignore,dist,axis,(nq,sub) in itertools.product([0,42,2026],[0,1],[0,1],[0,1],[0,1],[0,1],[(1,0),(1,7),(1,13),(5,0),(5,7),(5,13),(17,0)]):
  yield config(seed=seed,fit=fit,query=query,ignore=ignore,qignore=ignore,distribution=dist,qdist=dist,axis=axis,nq=nq,subsample=sub)
 for scenario,fit,query,ignore,dist,axis,style,nq in itertools.product(['zero','missing','allnan','one_nan_column','tiny','huge','span'],[0,1],[0,1],[0,1],[0,1],[0,1],[0,2],[1,7]):
  yield config(seed=7,fit=fit,query=query,ignore=ignore,qignore=ignore,distribution=dist,qdist=dist,axis=axis,style=style,nq=nq,scenario=scenario)
 for fit,query,dist,axis,style,nq,sub in itertools.product([0,1],[0,1],[0,1],[0,1],[0,2],[1,2],[2,5]):
  yield config(seed=13,n=9,p=17,fit=fit,query=query,ignore=1,qignore=1,distribution=dist,qdist=dist,axis=axis,style=style,nq=nq,subsample=sub,scenario='negative')
 for fit,query,dist,qdist,ignore in itertools.product([0,1],[0,1],[0,1],[0,1],[0,1]):
  yield config(seed=23,fit=fit,query=query,distribution=dist,qdist=qdist,ignore=ignore,qignore=1,mode=1)
 # An interior subnormal interval can overflow numpy.interp's slope.
 for fit,dist in itertools.product([0,1],[0,1]):yield config(n=5,p=1,fit=fit,distribution=dist,qdist=dist,ignore=1,qignore=1,nq=5,scenario='separated')

def inputs(c):
 x=matrix(c['n'],c['p'],c['style'],c['seed']);q=matrix(c['m'],c['qwidth'] or c['p'],c['style'],c['seed']+1)
 if not c['ignore'] or c['scenario']!='negative':x.data=np.abs(x.data);q.data=np.abs(q.data)
 for a in [x,q]:
  if c['scenario']=='zero':a.data[:]=0
  if c['scenario']=='missing':a.data[::3]=np.nan
  if c['scenario']=='tiny':a.data*=1e-310
  if c['scenario']=='huge':a.data[:]=1e308
  if c['scenario']=='span':a.data[::2]=-1e308;a.data[1::2]=1e308
  if c['scenario']=='allnan' and a is x:
   a.data[:]=np.nan
  if c['scenario']=='one_nan_column' and a is x:
   d=a.toarray();d[:,0]=np.nan
   if a is x:x=sp.csr_matrix(d)
   else:q=sp.csr_matrix(d)
 if c['scenario']=='separated':
  x=sp.csr_matrix(np.array([-1.,0.,1e-310,2e-310,1.])[:,None]);q=sp.csr_matrix(np.array([.5e-310,1.5e-310,-.5,0.,1.,-1.,2.,np.nan,1e-311])[:,None])
 p=matrix(7,c['qwidth'] or c['p'],c['style'],c['seed']+2);p.data=np.linspace(-8,8,len(p.data))
 return [x,q,p]

def put_source(f,name,z,original):
 if sp.issparse(z):
  if z.format=='csc':
   tags=sp.csr_matrix((np.arange(len(original.data))+1,original.indices.copy(),original.indptr.copy()),shape=original.shape).tocsc()
   np.testing.assert_array_equal(tags.indptr,z.indptr);np.testing.assert_array_equal(tags.indices,z.indices)
   out=original.copy();out.data=np.empty_like(original.data);out.data[tags.data.astype(int)-1]=z.data
  else:
   np.testing.assert_array_equal(z.indptr,original.indptr);np.testing.assert_array_equal(z.indices,original.indices);out=z
  put(f,name,out)
 else:put(f,name,z)

def oracle(c,arrays):
 x,q,p=[a.copy() if kind else a.toarray() for a,kind in zip(arrays,[c['fit'],c['query'],c['query']])];original=[a.copy() for a in [x,q,p]];state=np.random.get_state();f={}
 options=dict(n_quantiles=c['nq'],subsample=None if c['subsample']==0 else c['subsample'],random_state=c['seed'],output_distribution='normal' if c['distribution'] else 'uniform',ignore_implicit_zeros=bool(c['ignore']))
 if c['mode']==5:options['output_distribution']='invalid'
 if c['mode']==6:quantile_transform(x,output_distribution='uniform         invalid')
 model=QuantileTransformer(**options)
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore');put_source(f,'train',model.fit_transform(x),arrays[0])
  f.update(quantiles=model.quantiles_.copy(),references=model.references_.copy(),metadata=[model.n_quantiles_,model.n_features_in_])
  model.output_distribution='normal' if c['qdist'] else 'uniform';model.ignore_implicit_zeros=bool(c['qignore'])
  if c['mode']==1:model.n_quantiles=0;model.subsample=-1;model.random_state=-1
  put_source(f,'query',model.transform(q),arrays[1]);put_source(f,'inverse',model.inverse_transform(p),arrays[2])
  put_source(f,'free',quantile_transform(x,axis=c['axis'],**options),arrays[0])
  same=lambda a,b:all(np.array_equal(getattr(a,k),getattr(b,k),equal_nan=True) for k in ['data','indices','indptr']) if sp.issparse(a) else np.array_equal(a,b,equal_nan=True)
  assert all(same(a,b) for a,b in zip([x,q,p],original)) and all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state()));f['preserved']=[1]*4
  model.set_params(**options);model.fit(arrays[1].toarray().T)
  f.update(refit_quantiles=model.quantiles_,refit_references=model.references_,refit_metadata=[model.n_quantiles_,model.n_features_in_])
 return f

def payload(c,arrays):return ' '.join(str(c[k]) for k in ['fit','query','nq','subsample','seed','distribution','ignore','axis','qdist','qignore','mode'])+'\n'+''.join(csr_payload(a) for a in arrays)

def reject(executable,wire,name):
 r=subprocess.run([str(executable)],input='1\n'+wire,text=True,capture_output=True);out=r.stdout+r.stderr
 assert r.returncode and 'fscikit-learn:' in out and not any(s in out for s in ['SIGFPE','Fortran runtime error']),(name,out)

def rejections(executable):
 results=[]
 for name,changes in [('count',dict(nq=0)),('subsample',dict(subsample=-1)),('subsample_count',dict(subsample=2)),('seed',dict(seed=-1)),('axis',dict(axis=2)),('unfitted',dict(mode=2)),('empty',dict(n=0)),('empty_features',dict(p=0)),('width',dict(qwidth=3)),('infinite_fit',{}),('infinite_query',{}),('infinite_inverse',{}),('sparse_negative_fit',dict(fit=1)),('sparse_negative_query',dict(query=1)),('distribution',dict(mode=5)),('long_distribution',dict(mode=6)),('raw_column_overflow',dict(fit=1,n=1,p=1,nq=1,style=2)),('raw_row_overflow',dict(fit=1,n=3,p=1,nq=1,style=2,axis=1)),('population_overdraw',dict(fit=1,n=1,p=1,nq=1,subsample=2,ignore=1,qignore=1))]:
  c=config(**changes);a=inputs(c)
  if name=='population_overdraw':a[0]=sp.csr_matrix(([1.,2.,3.,4.],[0,0,0,0],[0,4]),shape=(1,1))
  if name.startswith('infinite'):a[dict(infinite_fit=0,infinite_query=1,infinite_inverse=2)[name]].data[0]=np.inf
  if name=='sparse_negative_fit':a[0].data[0]=-1
  if name=='sparse_negative_query':a[1].data[0]=-1
  if name=='unfitted':
   try:QuantileTransformer().transform(a[1].toarray())
   except Exception:pass
   else:raise AssertionError(name)
  else:
   try:oracle(c,a)
   except (ValueError,TypeError):pass
   else:raise AssertionError(('source accepts',name))
  reject(executable,payload(c,a),name);results.append(dict(name=name,passed=True,oracle='sklearn'))
 for mode,name in [(3,'bad_index'),(4,'bad_start')]:
  c=config(fit=1,mode=mode);a=inputs(c);wire=payload(c,a)
  if mode==3:a[0].indices[0]=-1
  else:a[0].indptr[0]=-1
  try:a[0].check_format(full_check=True)
  except ValueError:pass
  else:raise AssertionError(name)
  reject(executable,wire,name);results.append(dict(name=name,passed=True,oracle='scipy_csr_check_format'))
 return results

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
 source=Path(sklearn.__file__).parent;paths=[Path(__file__),args.executable,ROOT/'src/sklearn_quantile_transform.f90',ROOT/'src/sklearn_preprocessing.f90',ROOT/'app/reference_quantile_seeded_driver.f90',ROOT/'validation/compare_robust.py',ROOT/'validation/compare_csr_projections.py',source/'preprocessing/_data.py',source/'utils/_indexing.py',Path(np.__file__).parent/'lib/_function_base_impl.py'];hashes={str(p):sha(p) for p in paths}
 all_cases=list(fixtures());assert len({json.dumps(c,sort_keys=True) for c in all_cases})==len(all_cases);cases=[];arrays=[];expected=[];source_rejected=[]
 for c in all_cases:
  with np.errstate(all='ignore'):a=inputs(c)
  try:f=oracle(c,a)
  except ValueError as e:
   reject(args.executable,payload(c,a),'source_boundary');source_rejected.append(dict(configuration=c,source_error=str(e),passed=True,oracle='sklearn'))
  else:cases.append(c);arrays.append(a);expected.append(f)
 r=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,arrays)),text=True,capture_output=True);assert r.returncode==0,(r.stdout[-600:],r.stderr[-1800:])
 lines=iter(r.stdout.splitlines());index=-1;checks=[];inventories=[];nonfinite=0
 for line in lines:
  p=line.split()
  if not p:continue
  if p[0]=='BEGIN':index+=1;fields=set();assert int(p[1])==index+1
  elif p[0]=='END':assert fields==expected[index].keys();inventories.append(sorted(fields))
  else:
   field=p[0];assert field not in fields;fields.add(field);got=np.fromstring(next(lines),sep=' ');want=np.asarray(expected[index][field]).ravel(order='F');assert len(got)==int(p[1]) and got.shape==want.shape,(index,field)
   exact=field in ['metadata','refit_metadata','preserved','references','refit_references'] or field.endswith(('_shape','_ptr','_idx'))
   atol=0.
   if exact:np.testing.assert_array_equal(got,want,err_msg=str((index,cases[index],field)))
   else:
    finite_want=want[np.isfinite(want)];atol=min(3e-8,max(64*np.finfo(float).smallest_subnormal,3e-8*float(np.max(np.abs(finite_want),initial=0))))
    np.testing.assert_allclose(got,want,rtol=3e-8,atol=atol,equal_nan=True,err_msg=str((index,cases[index],field)))
   finite=np.isfinite(want);nonfinite+=int(np.count_nonzero(~finite));checks.append(dict(fixture=index,field=field,passed=True,atol=atol,oracle='native_input_rng_identity' if field=='preserved' else 'sklearn',max_abs_error=float(np.max(np.abs(got[finite]-want[finite]),initial=0))))
 assert index+1==len(cases) and hashes=={str(p):sha(p) for p in paths}
 d=dict(profile=args.profile,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,total_configurations=len(all_cases),source_rejected_configurations=source_rejected,rejections=rejections(args.executable)+source_rejected,source_nonfinite_values=nonfinite,hashes=hashes,sklearn=sklearn.__version__)
 (args.report or ROOT/'validation/results'/f'quantile_seeded_{args.profile}.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(len(cases),len(checks),len(d['rejections']),nonfinite,'PASS')
if __name__=='__main__':main()
