"""SimpleImputer numeric dense/CSR strategies, indicators, markers and inverse."""
from pathlib import Path
from collections import defaultdict,deque
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.impute import SimpleImputer
from compare_robust import matrix,put
from compare_csr_projections import csr_payload
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
STRATEGIES=['mean','median','most_frequent','constant','callable']

def config(**kw):
 c=dict(seed=0,fit=0,query=0,strategy=0,keep=0,add=0,named=0,marker=0,fill=0,n=17,p=5,m=7,style=1,scenario='random',mode=0,qwidth=0)
 c.update(kw)
 for key in ['keep','add','strategy','marker']:c.setdefault('q'+key,c[key])
 return c

def fixtures():
 for seed,fit,query,strategy,keep,add,named,marker in itertools.product([0,42,2026],[0,1],[0,1],range(5),[0,1],[0,1],[0,1],range(3)):
  if fit and named:continue
  yield config(seed=seed,fit=fit,query=query,strategy=strategy,keep=keep,add=add,named=named,marker=marker)
 for scenario,fit,query,strategy,keep,add,style in itertools.product(['empty_column','allmissing','huge','span','tiny','ties'],[0,1],[0,1],range(5),[0,1],[0,1],[0,2]):
  yield config(seed=7,scenario=scenario,fit=fit,query=query,strategy=strategy,keep=keep,add=add,style=style)
 for fill,fit,query,keep,add in itertools.product([1,2,3],[0,1],[0,1],[0,1],[0,1]):
  yield config(seed=13,strategy=3,fill=fill,fit=fit,query=query,keep=keep,add=add,scenario='empty_column')
 for fit,query,strategy,keep,add in itertools.product([0,1],[0,1],range(5),[0,1],[0,1]):
  yield config(seed=23,fit=fit,query=query,strategy=strategy,keep=keep,add=add,qkeep=1-keep,qadd=0,qstrategy=3 if strategy==0 else 0,scenario='empty_column')
 for n in [1,2,599,600,601]:
  for strategy in [0,1,2]:yield config(n=n,p=1,strategy=strategy,scenario='huge')

def scalar(code,fill=False):
 return ([0.,-7.,np.nan,np.inf] if fill else [np.nan,-999.,0.,np.inf,-np.inf])[code]

def inputs(c):
 marker=scalar(c['marker']);x=matrix(c['n'],c['p'],c['style'],c['seed']);q=matrix(c['m'],c['qwidth'] or c['p'],c['style'],c['seed']+1)
 for a in [x,q]:
  if c['scenario']=='ties':a.data=np.round(a.data)
  if c['scenario']=='huge':a.data[:]=1e308
  if c['scenario']=='span':a.data[::2]=-1e308;a.data[1::2]=1e308
  if c['scenario']=='tiny':a.data*=1e-310
  a.data[::6]=marker
 if c['scenario'] in ['empty_column','allmissing']:
  d=x.toarray();d[:,0]=marker
  if c['scenario']=='allmissing':d[:]=marker
  x=sp.csr_matrix(d)
 # Dense probe is only used for explicit inverse diagnostics.
 probe=sp.csr_matrix(np.ones((2,2)))
 return [x,q,probe]

def native_layout(z,original,model):
 # Take actual source CSC results, then arrange its values in the specified
 # native CSR order: original retained slots, followed by raw indicator slots.
 coo=z.tocoo();values=defaultdict(deque)
 for r,c,v in zip(coo.row,coo.col,coo.data):values[int(r),int(c)].append(v)
 keep=np.ones(original.shape[1],dtype=bool) if model.strategy=='constant' or model.keep_empty_features else ~np.isnan(model.statistics_)
 columns=np.cumsum(keep)-1;n=int(keep.sum());ind={} if not model.add_indicator else {int(j):n+i for i,j in enumerate(model.indicator_.features_)}
 ptr=[0];idx=[];data=[]
 for r in range(original.shape[0]):
  start,end=original.indptr[r:r+2]
  for k in range(start,end):
   c=int(original.indices[k])
   if keep[c]:idx.append(int(columns[c]));data.append(values[r,int(columns[c])].popleft())
  for k in range(start,end):
   c=int(original.indices[k]);v=original.data[k]
   missing=np.isnan(v) if np.isnan(model.missing_values) else v==model.missing_values
   if c in ind and missing:idx.append(ind[c]);data.append(values[r,ind[c]].popleft())
  ptr.append(len(data))
 assert all(not v for v in values.values()),'source sparse slots were lost'
 return sp.csr_matrix((data,idx,ptr),shape=z.shape)

def put_source(f,name,z,original,model):put(f,name,native_layout(z,original,model) if sp.issparse(z) else z)

def names_output(f,prefix,names):
 f[prefix+'_lengths']=[len(v) for v in names];f[prefix+'_codes']=[ord(c) for v in names for c in v]

def oracle(c,arrays):
 x,q,probe=arrays;x=x.copy() if c['fit'] else x.toarray();q=q.copy() if c['query'] else q.toarray();original=[a.copy() for a in [x,q]];state=np.random.get_state();calls=[0]
 def statistic(v):calls[0]+=1;return float(len(v)+2)
 def strategy(code):return statistic if code==4 else STRATEGIES[code] if 0<=code<4 else 'invalid'
 options=dict(strategy=strategy(c['strategy']),keep_empty_features=bool(c['keep']),add_indicator=bool(c['add']),missing_values=scalar(c['marker']),fill_value=scalar(c['fill'],True))
 if c['mode']==1:options['strategy']='invalid'
 if c['mode']==7:options['strategy']=None
 model=SimpleImputer(**options);f={};names=np.array([f'f{i}' for i in range(arrays[0].shape[1])],dtype=object)
 if c['named']:
  import pandas as pd
  assert not c['fit'];x=pd.DataFrame(x,columns=names)
  if not c['query']:q=pd.DataFrame(q,columns=names)
 def metadata(prefix):
  indicator=[] if model.indicator_ is None else model.indicator_.features_+1
  f[prefix+'_indicator']=indicator;f[prefix+'_metadata']=[model.n_features_in_,int(model.indicator_ is not None),len(indicator),len(getattr(model,'feature_names_in_',[])),calls[0]]
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore')
  if c['mode']==2:model.transform(q)
  put_source(f,'train',model.fit_transform(x),arrays[0],model)
  if c['mode']==6:model.inverse_transform(probe.toarray())
  f['statistics']=model.statistics_.copy();metadata('fit')
  names_output(f,'names_default',model.get_feature_names_out());names_output(f,'names_explicit',model.get_feature_names_out(names))
  model.keep_empty_features=bool(c['qkeep']);model.add_indicator=bool(c['qadd']);model.strategy=strategy(c['qstrategy']);model.missing_values=scalar(c['qmarker'])
  if c['mode']==8:q=q.rename(columns={names[0]:'other'})
  z=model.transform(q);put_source(f,'query',z,arrays[1],model)
  if model.add_indicator:
   v=probe.toarray() if c['mode']==5 else z.toarray() if sp.issparse(z) else z
   inverse=model.inverse_transform(v);f['inverse_shape']=inverse.shape;f['inverse_data']=inverse
  else:f['inverse_shape']=[0,0];f['inverse_data']=[]
  same=lambda a,b:all(np.array_equal(getattr(a,k),getattr(b,k),equal_nan=True) for k in ['data','indices','indptr']) if sp.issparse(a) else np.array_equal(np.asarray(a),b,equal_nan=True)
  assert all(same(a,b) for a,b in zip([x,q],original)) and all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state()));f['preserved']=[1]*4
  model.set_params(**options);calls[0]=0;model.fit(arrays[1].toarray().T);f['refit_statistics']=model.statistics_;metadata('refit');names_output(f,'refit_names',model.get_feature_names_out())
 return f

def payload(c,arrays):
 return ' '.join(str(c[k]) for k in ['fit','query','strategy','keep','add','named','qkeep','qadd','qstrategy','mode'])+' '+str(scalar(c['marker']))+' '+str(scalar(c['fill'],True))+' '+str(scalar(c['qmarker']))+'\n'+''.join(csr_payload(a) for a in arrays)

def reject(executable,wire,name):
 r=subprocess.run([str(executable)],input='1\n'+wire,text=True,capture_output=True);out=r.stdout+r.stderr
 assert r.returncode and 'fscikit-learn:' in out and not any(v in out for v in ['SIGFPE','Fortran runtime error']),(name,out)

def rejections(executable):
 result=[]
 for name,changes in [('strategy',dict(mode=1)),('unfitted',dict(mode=2)),('empty',dict(n=0)),('empty_features',dict(p=0)),('width',dict(qwidth=3)),('dense_nan_marker',dict(marker=1)),('sparse_zero',dict(fit=1,marker=2)),('infinite_fit',{}),('infinite_query',{}),('late_indicator',dict(add=0,qadd=1)),('inverse_without_indicator',dict(mode=6)),('inverse_width',dict(add=1,mode=5,scenario='empty_column')),('callback_none',dict(strategy=4,mode=7)),('name_mismatch',dict(named=1,mode=8))]:
  c=config(**changes);a=inputs(c)
  if name=='inverse_width':a[2]=sp.csr_matrix((2,0))
  if name=='dense_nan_marker':a[0].data[0]=np.nan
  if name=='infinite_fit':a[0].data[0]=np.inf
  if name=='infinite_query':a[1].data[0]=np.inf
  try:oracle(c,a)
  except (ValueError,TypeError,AttributeError,IndexError):pass
  else:raise AssertionError(('source accepts',name))
  reject(executable,payload(c,a),name);result.append(dict(name=name,passed=True,oracle='sklearn'))
 for mode,name in [(3,'bad_index'),(4,'bad_start')]:
  c=config(fit=1,mode=mode);a=inputs(c);wire=payload(c,a)
  if mode==3:a[0].indices[0]=-1
  else:a[0].indptr[0]=-1
  try:a[0].check_format(full_check=True)
  except ValueError:pass
  else:raise AssertionError(name)
  reject(executable,wire,name);result.append(dict(name=name,passed=True,oracle='scipy_csr_check_format'))
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
 source=Path(sklearn.__file__).parent;paths=[Path(__file__),args.executable,ROOT/'src/sklearn_simple_impute.f90',ROOT/'app/reference_simple_imputer_driver.f90',ROOT/'validation/compare_robust.py',ROOT/'validation/compare_csr_projections.py',source/'impute/_base.py',source/'utils/sparsefuncs.py',Path(np.__file__).parent/'ma/core.py',Path(np.__file__).parent/'ma/extras.py'];hashes={str(p):sha(p) for p in paths}
 all_cases=list(fixtures());assert len({json.dumps(c,sort_keys=True) for c in all_cases})==len(all_cases);cases=[];arrays=[];expected=[];source_rejected=[]
 for c in all_cases:
  with np.errstate(all='ignore'):a=inputs(c)
  try:f=oracle(c,a)
  except (ValueError,TypeError,IndexError,AttributeError) as e:
   reject(args.executable,payload(c,a),'source_boundary');source_rejected.append(dict(configuration=c,source_error=type(e).__name__+': '+str(e),passed=True,oracle='sklearn'))
  else:cases.append(c);arrays.append(a);expected.append(f)
 r=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,arrays)),text=True,capture_output=True);assert r.returncode==0,(r.stdout[-600:],r.stderr[-1700:])
 lines=iter(r.stdout.splitlines());index=-1;checks=[];inventories=[];nonfinite=0
 for line in lines:
  p=line.split()
  if not p:continue
  if p[0]=='BEGIN':index+=1;fields=set();assert int(p[1])==index+1
  elif p[0]=='END':assert fields==expected[index].keys();inventories.append(sorted(fields))
  else:
   field=p[0];assert field not in fields;fields.add(field);got=np.fromstring(next(lines),sep=' ');want=np.asarray(expected[index][field]).ravel(order='F');assert len(got)==int(p[1]) and got.shape==want.shape,(index,field)
   exact=field=='preserved' or field.endswith(('_shape','_ptr','_idx','_metadata','_indicator','_codes','_lengths'))
   atol=0.
   if exact:np.testing.assert_array_equal(got,want,err_msg=str((index,cases[index],field)))
   else:
    finite_want=want[np.isfinite(want)];atol=min(3e-8,max(64*np.finfo(float).smallest_subnormal,3e-8*float(np.max(np.abs(finite_want),initial=0))))
    np.testing.assert_allclose(got,want,rtol=3e-8,atol=atol,equal_nan=True,err_msg=str((index,cases[index],field)))
   finite=np.isfinite(want);nonfinite+=int(np.count_nonzero(~finite));checks.append(dict(fixture=index,field=field,passed=True,atol=atol,oracle='native_input_rng_identity' if field=='preserved' else 'sklearn',max_abs_error=float(np.max(np.abs(got[finite]-want[finite]),initial=0))))
 assert index+1==len(cases) and hashes=={str(p):sha(p) for p in paths}
 d=dict(profile=args.profile,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,total_configurations=len(all_cases),source_rejected_configurations=source_rejected,rejections=rejections(args.executable)+source_rejected,source_nonfinite_values=nonfinite,hashes=hashes,sklearn=sklearn.__version__)
 (args.report or ROOT/'validation/results'/f'simple_imputer_{args.profile}.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(len(cases),len(checks),len(d['rejections']),nonfinite,'PASS')
if __name__=='__main__':main()
