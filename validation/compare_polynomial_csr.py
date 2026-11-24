"""Source-seeded dense/CSR TensorSketch, storage order and live controls."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy
import sklearn
from sklearn.kernel_approximation import PolynomialCountSketch
from compare_csr_projections import sparse,csr_payload
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def config(**kw):
 return dict(n=7,p=3,m=4,seed=0,fit_kind=1,query_kind=1,degree=2,components=17,gamma=1.,coef0=0.,qdegree=2,qcomponents=17,qgamma=1.,qcoef0=0.,mode=0,style=0,scenario='random',qwidth=0)|kw
def fixtures():
 parameters=[(1,1,1.,0.),(2,3,.4,.3),(3,8,1.,0.),(4,17,2.,.1),(2,31,0.,1.),(7,67,.1,.1),(2,128,1.,0.),(3,257,.3,.2),(2,1009,1.,0.)]
 for seed,(n,p),style,fit,query,(degree,k,g,c) in itertools.product([0,7,42,2026,2147483647],[(7,3),(5,5)],range(4),[0,1],[0,1],parameters):
  yield config(seed=seed,n=n,p=p,style=style,fit_kind=fit,query_kind=query,degree=degree,components=k,gamma=g,coef0=c,qdegree=degree,qcomponents=k,qgamma=g,qcoef0=c)
 for style,fit,query in itertools.product(range(4),[0,1],[0,1]):
  yield config(style=style,fit_kind=fit,query_kind=query,degree=4,qdegree=1,qcomponents=31,qgamma=.7)
  yield config(style=style,fit_kind=fit,query_kind=query,coef0=.2,qcoef0=.8,qgamma=.3)
  yield config(style=style,fit_kind=fit,query_kind=query,coef0=-.2,qcoef0=-.7)
 for style,fit,query,k,degree,scenario in itertools.product(range(3),[0,1],[0,1],[1,2,3,4,5,7,8,11,13,16,17,19,23,29,31],[1,2],['zero','tiny','huge','hugequery']):
  yield config(style=style,fit_kind=fit,query_kind=query,components=k,qcomponents=k,degree=degree,qdegree=degree,scenario=scenario)
 for fit,query,k in itertools.product([0,1],[0,1],[1,2,3,8]):
  yield config(fit_kind=fit,query_kind=query,components=k,qcomponents=k,gamma=1e308,qgamma=1e308,scenario='huge',style=1)
def inputs(c):
 x=sparse(c['n'],c['p'],c['style'],c['seed']);q=sparse(c['m'],c['qwidth'] or c['p'],c['style'],c['seed']+2)
 if c['scenario'] in ['zero','tiny','huge']:
  value=dict(zero=0.,tiny=1e-310,huge=1e308)[c['scenario']]/(3 if c['style']==2 else 1);x.data.fill(value);q.data.fill(value)
 if c['scenario']=='hugequery':q.data.fill(1e308/(3 if c['style']==2 else 1))
 return x,q
def same(a,b):
 return all(np.array_equal(getattr(a,k),getattr(b,k)) for k in ['indptr','indices','data']) if hasattr(a,'indptr') else np.array_equal(a,b)
def model_for(c):return PolynomialCountSketch(degree=c['degree'],n_components=c['components'],random_state=c['seed'],gamma=c['gamma'],coef0=c['coef0'])
def oracle(c,arrays):
 x,q=arrays;a=x.copy() if c['fit_kind'] else x.toarray();b=q.copy() if c['query_kind'] else q.toarray();original_a=a.copy();original_b=b.copy();state=np.random.get_state();model=model_for(c)
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore');z=model.fit_transform(a)
  model.set_params(degree=c['qdegree'],n_components=c['qcomponents'],gamma=c['qgamma'],coef0=c['qcoef0']);query=model.transform(b)
  opposite=x.toarray() if c['fit_kind'] else x.copy();again=model_for(c).fit(opposite)
 assert same(a,original_a) and same(b,original_b) and all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state()))
 return dict(hash_shape=model.indexHash_.shape,hash=model.indexHash_,bits=model.bitHash_,train=z,query=query,metadata=[model.n_features_in_,model._n_features_out,b.shape[0],query.shape[1]],preserved=[1,1,1],environment=[1]*6,repeat=[int(np.array_equal(model.indexHash_,again.indexHash_)),int(np.array_equal(model.bitHash_,again.bitHash_))])
def payload(c,arrays):
 return ' '.join(str(c[k]) for k in ['fit_kind','query_kind','degree','components','seed','gamma','coef0','qdegree','qcomponents','qgamma','qcoef0','mode'])+'\n'+''.join(csr_payload(a) for a in arrays)
def reject_cases(executable):
 result=[]
 variants=[('components_zero',dict(components=0)),('components_negative',dict(components=-1)),('degree_zero',dict(degree=0)),('degree_negative',dict(degree=-1)),('negative_seed',dict(seed=-1)),('negative_gamma',dict(gamma=-1.)),('nan_gamma',dict(gamma=float('nan'))),('inf_gamma',dict(gamma=float('inf'))),('nan_coef',dict(coef0=float('nan'))),('inf_coef',dict(coef0=float('inf'))),('unfitted',dict(mode=1)),('nan_fit',dict(mode=2)),('inf_fit',dict(mode=3)),('nan_query',dict(mode=4)),('inf_query',dict(mode=5)),('empty_fit',dict(n=0)),('empty_features',dict(p=0)),('empty_query',dict(m=0)),('query_width',dict(qwidth=5)),('higher_query_degree',dict(qdegree=3)),('query_bucket_range',dict(qcomponents=1)),('add_constant',dict(qcoef0=1.)),('remove_constant',dict(coef0=1.,qcoef0=0.))]
 for fit,query in itertools.product([0,1],[0,1]):
  for name,change in variants:
   c=config(style=1,fit_kind=fit,query_kind=query)|change;arrays=inputs(c);wire=payload(c,arrays);x,q=arrays;source_error=None
   if c['mode']==2:x.data[0]=np.nan
   if c['mode']==3:x.data[0]=np.inf
   if c['mode']==4:q.data[0]=np.nan
   if c['mode']==5:q.data[0]=np.inf
   a=x if fit else x.toarray();b=q if query else q.toarray();model=model_for(c)
   try:
    with warnings.catch_warnings(),np.errstate(all='ignore'):
     warnings.simplefilter('ignore')
     if c['mode']==1:model.transform(b)
     model.fit(a);model.set_params(degree=c['qdegree'],n_components=c['qcomponents'],gamma=c['qgamma'],coef0=c['qcoef0']);model.transform(b)
   except (ValueError,IndexError) as error:source_error=type(error).__name__
   assert source_error is not None,('source accepts invalid',name,c)
   reject(executable,wire,name)
   result.append(dict(name=f'{fit}_{query}_{name}',passed=True,oracle='sklearn_estimator',source_error=source_error))
 for mode,name in [(6,'bad_index'),(7,'bad_start'),(8,'bad_pointer')]:
  c=config(style=1,mode=mode);arrays=inputs(c);wire=payload(c,arrays);x=arrays[0]
  if mode==6:x.indices[0]=-1
  if mode==7:x.indptr[0]=-1
  if mode==8:x.indptr[1]=len(x.data)+1
  try:x.check_format(full_check=True)
  except ValueError:pass
  else:raise AssertionError(('scipy accepts malformed input',name))
  reject(executable,wire,name);result.append(dict(name=name,passed=True,oracle='scipy_csr_check_format',source_error='ValueError'))
 for name,c in [('output',config(n=1,p=1,m=3,components=1,qcomponents=2147483647)),('workspace',config(n=1,p=1,m=1,degree=3,qdegree=3,components=1,qcomponents=1073741824)),('hash',config(n=1,p=2,m=1,degree=2147483647))]:
  reject(executable,payload(c,inputs(c)),name,'overflow')
  result.append(dict(name='native_'+name,passed=True,oracle='native_integer_capacity'))
 return result
def reject(executable,wire,name,message='fscikit-learn:'):
 r=subprocess.run([str(executable)],input='1\n'+wire,capture_output=True,text=True);out=r.stdout+r.stderr
 assert r.returncode and 'fscikit-learn:' in out and message in out and not any(v in out for v in ['SIGFPE','Fortran runtime error']),(name,out)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
 paths=[Path(__file__),args.executable,ROOT/'validation/compare_csr_projections.py',ROOT/'src/sklearn_polynomial_count_sketch.f90',ROOT/'src/sklearn_count_sketch_fft.f90',ROOT/'src/sklearn_stateless_preprocessing.f90',ROOT/'src/sklearn_sparse.f90',ROOT/'src/sklearn_ieee_math.f90',ROOT/'app/reference_polynomial_csr_driver.f90',Path(sklearn.__file__).parent/'kernel_approximation.py'];hashes={str(p):sha(p) for p in paths}
 cases=list(fixtures());assert len({json.dumps(c,sort_keys=True) for c in cases})==len(cases);arrays=[inputs(c) for c in cases];expected=[oracle(c,a) for c,a in zip(cases,arrays)]
 r=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,arrays)),capture_output=True,text=True)
 assert r.returncode==0,(r.stdout[-800:],r.stderr[-1400:])
 lines=iter(r.stdout.splitlines());index=-1;checks=[];inventories=[];nonfinite=0
 for line in lines:
  p=line.split()
  if not p:continue
  if p[0]=='BEGIN':index+=1;fields=set();assert int(p[1])==index+1
  elif p[0]=='END':assert fields==expected[index].keys();inventories.append(sorted(fields))
  else:
   field=p[0];assert field not in fields;fields.add(field);values=np.fromstring(next(lines),sep=' ');want=np.asarray(expected[index][field]).ravel(order='F');assert len(values)==int(p[1]) and values.shape==want.shape
   if field in ['train','query']:
    np.testing.assert_allclose(values,want,rtol=3e-8,atol=3e-8,equal_nan=True,err_msg=str((index,cases[index],field)))
   else:np.testing.assert_array_equal(values,want,err_msg=str((index,cases[index],field)))
   finite=np.isfinite(want);nonfinite+=int(np.count_nonzero(~finite));checks.append(dict(fixture=index,field=field,passed=True,oracle='native_ieee_environment' if field=='environment' else 'sklearn',max_abs_error=float(np.max(np.abs(values[finite]-want[finite]),initial=0))))
 assert index+1==len(cases) and hashes=={str(p):sha(p) for p in paths}
 rejections=reject_cases(args.executable)
 d=dict(rejections=rejections,profile=args.profile,sklearn=sklearn.__version__,scipy=scipy.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,source_nonfinite_values=nonfinite,hashes=hashes)
 (args.report or ROOT/'validation/results'/f'polynomial_csr_{args.profile}.json').write_text(json.dumps(d,indent=2)+'\n');print(len(cases),len(checks),len(rejections),nonfinite,'PASS')
if __name__=='__main__':main()
