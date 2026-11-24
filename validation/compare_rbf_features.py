"""Seeded RBFSampler dense/CSR generation, scaled variance, IEEE and mutation contracts."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.kernel_approximation import RBFSampler
from compare_csr_projections import sparse,csr_payload
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def config(**kw):
 return dict(seed=0,n=7,p=3,k=5,fit_kind=0,query_kind=0,scale=0,gamma=.7,style=0,scenario='random',mutate=0,mode=0)|kw
def fixtures():
 for seed,(n,p,k),fit_kind,query_kind,style,(scale,gamma) in itertools.product([0,7,42,2026,2147483647],[(7,1,1),(9,3,5),(5,7,8),(4,31,17)],[0,1],[0,1],range(3),[(0,0.),(0,.7),(0,3.),(1,1.)]):
  yield config(seed=seed,n=n,p=p,k=k,fit_kind=fit_kind,query_kind=query_kind,style=style,scale=scale,gamma=gamma)
 for seed,fit_kind,query_kind,scale in itertools.product([0,7,42,2026,2147483647],[0,1],[0,1],[0,1]):
  yield config(seed=seed,fit_kind=fit_kind,query_kind=query_kind,scale=scale,style=3)
 for fit_kind,query_kind,scenario,scale in itertools.product([0,1],[0,1],['zero','constant','near_constant','small','large','huge'],[0,1]):
  yield config(fit_kind=fit_kind,query_kind=query_kind,scenario=scenario,scale=scale,style=1)
 for fit_kind,query_kind,scale in itertools.product([0,1],[0,1],[0,1]):
  yield config(fit_kind=fit_kind,query_kind=query_kind,scale=scale,mutate=13)
  yield config(fit_kind=fit_kind,query_kind=query_kind,scale=scale,gamma=1e308)
def inputs(c):
 x=sparse(c['n'],c['p'],c['style'],c['seed']);q=sparse(4,c['p'],c['style'],c['seed']+3)
 if c['scenario']!='random':
  a=np.full((c['n'],c['p']),dict(zero=0.,constant=3.,near_constant=1.+1e-8,small=1e-300,large=1e154,huge=1e308)[c['scenario']]);
  if c['scenario']=='near_constant':a[-1,-1]+=1e-15
  x=sp.csr_matrix(a)
 return x,q
def payload(c,arrays):
 return ' '.join(str(c[k]) for k in ['k','seed','fit_kind','query_kind','scale','gamma','mode','mutate'])+'\n'+''.join(csr_payload(a) for a in arrays)
def oracle(c,arrays):
 x,q=arrays;x=x if c['fit_kind'] else x.toarray();q=q if c['query_kind'] else q.toarray()
 copies=[a.copy() for a in [x,q]];state=np.random.get_state()
 m=RBFSampler(n_components=c['k'],random_state=c['seed'],gamma='scale' if c['scale'] else c['gamma'])
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore');train=m.fit_transform(x)
  if c['mutate']:m.n_components=c['mutate']
  query=m.transform(q)
 same=lambda a,b:all(np.array_equal(getattr(a,key),getattr(b,key)) for key in ['indptr','indices','data']) if sp.issparse(a) else np.array_equal(a,b)
 preserved=[int(same(a,b)) for a,b in zip([x,q],copies)];preserved.append(int(all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state()))))
 result=dict(weights=m.random_weights_,offset=m.random_offset_,gamma=[m._gamma],train=train,query=query,metadata=[m.n_features_in_,c['k'],m.n_components],preserved=preserved)
 first=m.random_weights_.copy();offset=m.random_offset_.copy();m.n_components=c['k']
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore');m.fit(x)
 result['repeat']=[int(np.array_equal(first,m.random_weights_,equal_nan=True)),int(np.array_equal(offset,m.random_offset_,equal_nan=True))]
 assert all(preserved) and all(result['repeat'])
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
 paths=[Path(__file__),ROOT/'validation/compare_csr_projections.py',args.executable,ROOT/'src/sklearn_kernel_approximation.f90',ROOT/'src/sklearn_sparse.f90',ROOT/'src/sklearn_legacy_rng.f90',ROOT/'src/sklearn_ieee_math.f90',ROOT/'app/reference_rbf_features_driver.f90',Path(sklearn.__file__).parent/'kernel_approximation.py']
 hashes={str(p):sha(p) for p in paths};cases=list(fixtures());arrays=[inputs(c) for c in cases];expected=[oracle(c,a) for c,a in zip(cases,arrays)]
 r=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,arrays)),text=True,capture_output=True)
 if r.returncode:
  begins=[line for line in r.stdout.splitlines() if line.strip().startswith('BEGIN')];index=int(begins[-1].split()[1]) if begins else 0
  raise AssertionError((index,cases[min(index,len(cases)-1)],r.stdout[-700:],r.stderr[-1400:]))
 lines=iter(r.stdout.splitlines());checks=[];inventories=[];index=-1;nonfinite=0
 for line in lines:
  parts=line.split()
  if not parts:continue
  if parts[0]=='BEGIN':index+=1;fields=set();assert int(parts[1])==index+1
  elif parts[0]=='END':assert fields==expected[index].keys();inventories.append(sorted(fields))
  else:
   field=parts[0];assert field not in fields;fields.add(field);values=np.fromstring(next(lines),sep=' ');assert len(values)==int(parts[1])
   want=np.asarray(expected[index][field]).ravel(order='F');assert values.shape==want.shape
   if field in ['weights','offset','gamma','metadata','preserved','repeat']:np.testing.assert_array_equal(values,want,err_msg=str((index,cases[index],field)))
   else:np.testing.assert_allclose(values,want,rtol=3e-11,atol=3e-12,equal_nan=True,err_msg=str((index,cases[index],field)))
   finite=np.isfinite(want);nonfinite+=int(np.count_nonzero(~finite));checks.append(dict(fixture=index,field=field,passed=True,max_abs_error=float(np.max(np.abs(values[finite]-want[finite]),initial=0))))
 assert index+1==len(cases)
 rejections=[]
 for kind in [0,1]:
  base=config(fit_kind=kind,query_kind=kind,style=1)
  variants=[('unfitted',dict(mode=1)),('zero_components',dict(k=0)),('negative_components',dict(k=-1)),('negative_seed',dict(seed=-1)),('negative_gamma',dict(gamma=-1.)),('nan_gamma',dict(gamma=float('nan'))),('inf_gamma',dict(gamma=float('inf'))),('nan_fit',dict(mode=3)),('inf_fit',dict(mode=4)),('nan_query',dict(mode=5)),('inf_query',dict(mode=6)),('empty_rows',dict(n=0)),('empty_features',dict(p=0)),('query_width',{}),('empty_query',{})]
  if kind:variants += [('bad_index',dict(mode=7)),('bad_start',dict(mode=8)),('bad_pointer',dict(mode=9))]
  for name,change in variants:
   c=base|change;a=list(inputs(c))
   if name=='query_width':a[1]=sp.csr_matrix(np.ones((3,2)))
   if name=='empty_query':a[1]=sp.csr_matrix((0,c['p']))
   wire=payload(c,a);x,q=a
   try:
    if c['mode'] in [7,8,9]:
     if c['mode']==7:x.indices[0]=-1
     if c['mode']==8:x.indptr[0]=-1
     if c['mode']==9:x.indptr[1]=len(x.data)+1
     x.check_format(full_check=True)
    else:
     if c['mode']==3:x.data[0]=np.nan
     if c['mode']==4:x.data[0]=np.inf
     if c['mode']==5:q.data[0]=np.nan
     if c['mode']==6:q.data[0]=np.inf
     x=x if kind else x.toarray();q=q if kind else q.toarray()
     model=RBFSampler(n_components=c['k'],gamma=c['gamma'],random_state=c['seed'])
     if c['mode']!=1:model.fit(x)
     model.transform(q)
   except ValueError as error:source_error=type(error).__name__
   else:raise AssertionError(('source accepts invalid',kind,name))
   r=subprocess.run([str(args.executable)],input='1\n'+wire,text=True,capture_output=True);out=r.stdout+r.stderr
   assert r.returncode and 'fscikit-learn:' in out,(kind,name,out)
   assert not any(t in out for t in ['SIGFPE','Fortran runtime error']),(kind,name,out)
   rejections.append(dict(name=str(kind)+'_'+name,passed=True,source_error=source_error,oracle='scipy_csr_check_format' if c['mode'] in [7,8,9] else 'sklearn_estimator'))
 assert hashes=={str(p):sha(p) for p in paths}
 report=dict(profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,rejections=rejections,source_nonfinite_values=nonfinite,hashes=hashes)
 path=args.report or ROOT/'validation/results'/f'rbf_features_{args.profile}.json';path.write_text(json.dumps(report,indent=2)+'\n');print(len(cases),len(checks),len(rejections),nonfinite,'PASS')
if __name__=='__main__':main()
