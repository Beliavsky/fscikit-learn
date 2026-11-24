"""Source free preprocessing.scale: dense/vector/CSR and precision corrections."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.preprocessing import scale
from compare_robust import matrix,put
from compare_csr_projections import csr_payload
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def config(**kw):return dict(seed=0,kind=0,axis=0,center=1,std=1,style=1,scenario='random',n=17,p=3,mode=0)|kw

def fixtures():
 for seed,kind,axis,center,std,style in itertools.product([0,42,2026],[0,1],[0,1],[0,1],[0,1],[0,1,2]):
  if kind and (axis or center):continue
  yield config(seed=seed,kind=kind,axis=axis,center=center,std=std,style=style)
 for scenario,kind,axis,center,std,style in itertools.product(['zero','constant','missing','allnan','one_nan_column','tiny','variance_tiny','underflow','huge','span','offset','small_offset'],[0,1],[0,1],[0,1],[0,1],[1,2]):
  if kind and (axis or center):continue
  yield config(seed=7,kind=kind,axis=axis,center=center,std=std,style=style,scenario=scenario)
 for n,p,scenario,axis in itertools.product([1,7,8,127,128,129,257],[1,4],['random','offset','missing'],[0,1]):
  yield config(seed=13,n=n,p=p,scenario=scenario,axis=axis)
 for values in ['threshold_below','threshold_equal','threshold_above']:
  for kind in [0,1]:yield config(scenario=values,kind=kind,center=0,n=3,p=1)

def inputs(c):
 a=matrix(c['n'],c['p'],c['style'],c['seed']);scenario=c['scenario']
 if scenario=='zero':a.data[:]=0
 if scenario=='constant':a=sp.csr_matrix(np.full(a.shape,3.5))
 if scenario=='missing':a.data[::3]=np.nan
 if scenario=='allnan':a=sp.csr_matrix(np.full(a.shape,np.nan))
 if scenario=='one_nan_column':
  d=a.toarray();d[:,0]=np.nan;a=sp.csr_matrix(d)
 if scenario=='tiny':a.data*=1e-17
 if scenario=='variance_tiny':a.data*=1e-8
 if scenario=='underflow':a.data*=1e-310
 if scenario=='huge':a.data[:]=1e308
 if scenario=='span':a.data[::2]=-1e308;a.data[1::2]=1e308
 if scenario in ['offset','small_offset']:
  d=np.arange(c['n']*c['p'],dtype=float).reshape(a.shape)
  d=(1e16+d*2) if scenario=='offset' else (1e-8+d*1e-19);a=sp.csr_matrix(d)
 if scenario.startswith('threshold_'):
  # Sparse thresholds variance; dense matrix thresholds its standard deviation.
  t=np.sqrt(15*np.finfo(float).eps) if c['kind'] else 10*np.finfo(float).eps*np.sqrt(1.5)
  if scenario=='threshold_below':t=np.nextafter(t,0.)
  if scenario=='threshold_above':t=np.nextafter(t,np.inf)
  a=sp.csr_matrix(np.array([-t,0.,t])[:,None])
 return a

def source_call(x,c):
 with warnings.catch_warnings(record=True) as caught,np.errstate(all='ignore'):
  warnings.simplefilter('always');z=scale(x,axis=c['axis'],with_mean=bool(c['center']),with_std=bool(c['std']))
 flags=[int(any(phrase in str(w.message) for w in caught)) for phrase in ['when centering the data','when scaling the data']]
 return z,flags

def oracle(c,a):
 x=a.copy() if c['kind'] else a.toarray();original=x.copy();state=np.random.get_state();z,flags=source_call(x,c);f={}
 if c['kind']:
  # Source returns CSC. Its stable CSR->CSC permutation maps values back to the
  # original CSR slots without canonicalizing duplicates or explicit zeros.
  tags=sp.csr_matrix((np.arange(len(a.data))+1,a.indices.copy(),a.indptr.copy()),shape=a.shape).tocsc()
  np.testing.assert_array_equal(tags.indptr,z.indptr);np.testing.assert_array_equal(tags.indices,z.indices)
  values=np.empty_like(a.data);values[tags.data.astype(int)-1]=z.data
  out=a.copy();out.data=values;put(f,'out',out)
 else:put(f,'out',z)
 f['corrections']=flags
 if not c['kind'] and c['axis']==0:f['vector'],f['vector_corrections']=source_call(x.ravel(order='F'),c)
 else:f['vector']=[];f['vector_corrections']=[]
 if c['kind']:
  assert all(np.array_equal(getattr(x,k),getattr(original,k),equal_nan=True) for k in ['data','indices','indptr'])
 else:np.testing.assert_array_equal(x,original)
 assert all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state()));f['preserved']=[1,1]
 return f

def payload(c,a):return ' '.join(str(c[k]) for k in ['kind','axis','center','std','mode'])+'\n'+csr_payload(a)

def rejections(executable):
 results=[]
 cases=[('axis_negative',config(axis=-1)),('axis_two',config(axis=2)),('vector_axis',config(axis=1,mode=1)),('empty_rows',config(n=0)),('empty_columns',config(p=0)),('empty_vector',config(n=0,mode=1)),('empty_sparse_rows',config(n=0,kind=1,center=0)),('empty_sparse_columns',config(p=0,kind=1,center=0)),('infinite_vector',config(mode=1)),('infinite_dense',config()),('infinite_sparse',config(kind=1,center=0)),('sparse_center',config(kind=1)),('sparse_row',config(kind=1,axis=1,center=0)),('sparse_row_noop',config(kind=1,axis=1,center=0,std=0))]
 for name,c in cases:
  a=inputs(c)
  if name.startswith('infinite'):a.data[0]=np.inf
  x=a.copy() if c['kind'] else a.toarray()
  if c['mode']==1:x=x.ravel(order='F')
  try:source_call(x,c)
  except (ValueError,TypeError,np.exceptions.AxisError):pass
  else:raise AssertionError(('source accepts',name))
  reject(executable,payload(c,a),name);results.append(dict(name=name,passed=True,oracle='sklearn'))
 for mode,name in [(2,'bad_index'),(3,'bad_start')]:
  c=config(kind=1,center=0,mode=mode);a=inputs(c);wire=payload(c,a)
  if mode==2:a.indices[0]=-1
  else:a.indptr[0]=-1
  try:a.check_format(full_check=True)
  except ValueError:pass
  else:raise AssertionError(name)
  reject(executable,wire,name);results.append(dict(name=name,passed=True,oracle='scipy_csr_check_format'))
 return results

def reject(executable,wire,name):
 r=subprocess.run([str(executable)],input='1\n'+wire,text=True,capture_output=True);out=r.stdout+r.stderr
 assert r.returncode and 'fscikit-learn:' in out and not any(v in out for v in ['SIGFPE','Fortran runtime error']),(name,out)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
 source=Path(sklearn.__file__).parent;paths=[Path(__file__),args.executable,ROOT/'src/sklearn_standard_scaling.f90',ROOT/'src/sklearn_preprocessing.f90',ROOT/'app/reference_scale_driver.f90',ROOT/'validation/compare_robust.py',ROOT/'validation/compare_csr_projections.py',source/'preprocessing/_data.py',source/'utils/sparsefuncs_fast.pyx',Path(np.__file__).parent/'lib/_nanfunctions_impl.py',Path(np.__file__).parent/'_core/_methods.py'];hashes={str(p):sha(p) for p in paths}
 all_cases=list(fixtures());assert len({json.dumps(c,sort_keys=True) for c in all_cases})==len(all_cases)
 cases=[];arrays=[];expected=[];source_rejected=[]
 for c in all_cases:
  with np.errstate(all='ignore'):a=inputs(c)
  try:f=oracle(c,a)
  except ValueError as e:
   # Densifying a duplicate-heavy huge CSR fixture can overflow before scale.
   # Retain this source rejection as a separately required native diagnostic.
   reject(args.executable,payload(c,a),'generated_infinite_input')
   source_rejected.append(dict(configuration=c,source_error=str(e),passed=True,oracle='sklearn'))
  else:cases.append(c);arrays.append(a);expected.append(f)
 r=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,arrays)),text=True,capture_output=True);assert r.returncode==0,(r.stdout[-600:],r.stderr[-1500:])
 lines=iter(r.stdout.splitlines());index=-1;checks=[];inventories=[];nonfinite=0
 for line in lines:
  p=line.split()
  if not p:continue
  if p[0]=='BEGIN':index+=1;fields=set();assert int(p[1])==index+1
  elif p[0]=='END':assert fields==expected[index].keys();inventories.append(sorted(fields))
  else:
   field=p[0];assert field not in fields;fields.add(field);got=np.fromstring(next(lines),sep=' ');want=np.asarray(expected[index][field]).ravel(order='F');assert len(got)==int(p[1]) and got.shape==want.shape,(index,field)
   exact=field in ['corrections','vector_corrections','preserved'] or field.endswith(('_shape','_ptr','_idx'))
   if exact:np.testing.assert_array_equal(got,want,err_msg=str((index,cases[index],field)))
   else:np.testing.assert_allclose(got,want,rtol=3e-8,atol=3e-8,equal_nan=True,err_msg=str((index,cases[index],field)))
   finite=np.isfinite(want);nonfinite+=int(np.count_nonzero(~finite));checks.append(dict(fixture=index,field=field,passed=True,oracle='native_input_rng_identity' if field=='preserved' else 'sklearn',max_abs_error=float(np.max(np.abs(got[finite]-want[finite]),initial=0))))
 assert index+1==len(cases) and hashes=={str(p):sha(p) for p in paths}
 d=dict(profile=args.profile,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,source_rejected_configurations=source_rejected,total_configurations=len(all_cases),rejections=rejections(args.executable)+source_rejected,source_nonfinite_values=nonfinite,hashes=hashes,sklearn=sklearn.__version__)
 (args.report or ROOT/'validation/results'/f'scale_{args.profile}.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(len(cases),len(checks),len(d['rejections']),nonfinite,'PASS')
if __name__=='__main__':main()
