"""Direct sklearn CSR random projections, including noncanonical sparse arrays."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection,johnson_lindenstrauss_min_dim
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fixtures():
 for kind,seed,(n,p,k),cache,style in itertools.product([0,1],[0,7,42,2026,2147483647],[(5,1,1),(7,4,2),(6,5,5),(3,4,7),(2,100,0)],[0,1],range(4)):
  for density in ([-1.,.1,1.] if kind else [-1.]):
   yield dict(kind=kind,seed=seed,n=n,p=p,k=k,cache=cache,style=style,density=density,eps=.9,mode=0)
def sparse(n,p,style,seed):
 rng=np.random.RandomState(seed%10000+17);dense=rng.normal(size=(n,p));dense[rng.uniform(size=(n,p))<.6]=0;dense[::4]=0
 if style==0:return sp.csr_matrix(dense)
 data=[];indices=[];indptr=[0]
 for i,row in enumerate(dense):
  for j in reversed(range(p)):
   value=row[j]
   if style==1:data.append(value);indices.append(j)
   elif style==2:
    data.extend([.25*value,.75*value,0.]);indices.extend([j,j,j])
   else:
    data.extend([1e8,value,-1e8]);indices.extend([j,j,j])
  indptr.append(len(data))
 return sp.csr_matrix((data,indices,indptr),shape=(n,p))
def model_for(c):
 options=dict(n_components=c['k'] or 'auto',random_state=c['seed'],eps=c['eps'],compute_inverse_components=bool(c['cache']))
 if c['kind']:options['density']='auto' if c['density']==-1 else c['density']
 return (SparseRandomProjection if c['kind'] else GaussianRandomProjection)(**options)
def inputs(c):
 k=c['k'] or int(johnson_lindenstrauss_min_dim(c['n'],eps=c['eps']))
 return sparse(c['n'],c['p'],c['style'],c['seed']),sparse(4,c['p'],c['style'],c['seed']+1),sparse(3,k,c['style'],c['seed']+2)
def csr_payload(a):
 return f'{a.shape[0]} {a.shape[1]} {len(a.data)}\n'+' '.join(map(str,a.indptr+1))+'\n'+' '.join(map(str,a.indices+1))+'\n'+' '.join(format(v,'.17g') for v in a.data)+'\n'
def payload(c,arrays):
 return ' '.join(str(c[k]) for k in ['kind','k','seed','cache','eps','density','mode'])+'\n'+''.join(csr_payload(a) for a in arrays)
def oracle(c,arrays,sparse_components=False):
 m=model_for(c);x,q,v=arrays
 snapshots=[(a.indptr.copy(),a.indices.copy(),a.data.copy().view(np.uint64)) for a in arrays]
 rng_before=np.random.get_state()
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore');z=m.fit_transform(x);query=m.transform(q);back=m.inverse_transform(v)
 dense=lambda a:a.toarray() if sp.issparse(a) else a
 preserved=[int(np.array_equal(a.indptr,b[0]) and np.array_equal(a.indices,b[1]) and np.array_equal(a.data.view(np.uint64),b[2])) for a,b in zip(arrays,snapshots)]
 rng_after=np.random.get_state();preserved.append(int(all(np.array_equal(a,b) for a,b in zip(rng_before,rng_after))))
 assert all(preserved)
 result=dict(components=dense(m.components_),train_transform=dense(z),query_transform=dense(query),inverse=back,pinv=m.inverse_components_ if c['cache'] else [],metadata=[m.n_components_,m.n_features_in_,c['cache']],density=[m.density_] if c['kind'] else [],preserved=preserved,refit=[1,1])
 if sparse_components and c['kind']:
  a=sp.csr_matrix(m.components_)
  result.update(components_indptr=a.indptr+1,components_indices=a.indices+1,components_data=a.data,component_storage=[0,1])
 for prefix,a in [('csr_',query),('csr_train_',z)]:
  if c['kind']:
   a=sp.csr_matrix(a,copy=True);a.sum_duplicates();a.eliminate_zeros();a.sort_indices()
   result.update({prefix+'shape':a.shape,prefix+'indptr':a.indptr+1,prefix+'indices':a.indices+1,prefix+'data':a.data})
  else:result.update({prefix+key:[] for key in ['shape','indptr','indices','data']})
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);ap.add_argument('--sparse-components',action='store_true');args=ap.parse_args()
 command=[str(args.executable)]+(['csr-components'] if args.sparse_components else [])
 paths=[Path(__file__),args.executable,ROOT/'src/sklearn_random_projection.f90',ROOT/'src/sklearn_sparse.f90',ROOT/'src/sklearn_legacy_rng.f90',ROOT/'src/sklearn_random_binomial.f90',ROOT/'app/reference_csr_projection_driver.f90',Path(sklearn.__file__).parent/'random_projection.py']
 hashes={str(p):sha(p) for p in paths};cases=list(fixtures());arrays=[inputs(c) for c in cases];expected=[oracle(c,a,args.sparse_components) for c,a in zip(cases,arrays)]
 r=subprocess.run(command,input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,arrays)),text=True,capture_output=True)
 if r.returncode:raise AssertionError((r.stdout[-1000:],r.stderr[-1700:]))
 lines=iter(r.stdout.splitlines());checks=[];inventories=[];index=-1
 for line in lines:
  parts=line.split()
  if not parts:continue
  if parts[0]=='BEGIN':index+=1;fields=set();assert int(parts[1])==index+1
  elif parts[0]=='END':assert fields==expected[index].keys();inventories.append(sorted(fields))
  else:
   field=parts[0];assert field not in fields;fields.add(field);size=int(parts[1]);values=np.fromstring(next(lines),sep=' ');assert len(values)==size
   want=np.asarray(expected[index][field]).ravel(order='F');assert values.shape==want.shape
   if field!='inverse' and field!='pinv':np.testing.assert_array_equal(values,want,err_msg=str((index,cases[index],field)))
   else:np.testing.assert_allclose(values,want,rtol=2e-9,atol=2e-6 if cases[index]['style']==3 else 2e-11,equal_nan=True,err_msg=str((index,cases[index],field)))
   finite=np.isfinite(want);checks.append(dict(fixture=index,field=field,oracle='native_storage_contract' if field=='component_storage' else ('source_repeat_and_native_cache_contract' if field=='refit' else 'source'),passed=True,max_abs_error=float(np.max(np.abs(values[finite]-want[finite]),initial=0))))
 assert index+1==len(cases)
 rejections=[]
 for kind in [0,1]:
  base=dict(kind=kind,seed=0,n=4,p=3,k=2,cache=0,style=1,density=-1.,eps=.9,mode=0)
  variants=[('unfitted_transform',dict(mode=1)),('unfitted_inverse',dict(mode=2)),('nan_fit',dict(mode=3)),('inf_fit',dict(mode=4)),('bad_index',dict(mode=5)),('bad_start',dict(mode=6)),('bad_pointer',dict(mode=7)),('nan_query',dict(mode=8)),('inf_inverse',dict(mode=9)),('query_width',{}),('latent_width',{}),('empty_fit_rows',dict(n=0)),('empty_fit_columns',dict(p=0)),('empty_query',{}),('empty_inverse',{}),('negative_components',dict(k=-1)),('zero_eps',dict(eps=0.)),('nan_eps',dict(eps=float('nan'))),('inf_eps',dict(eps=float('inf'))),('negative_seed',dict(seed=-1)),('auto_too_wide',dict(k=0))]
  if kind:variants += [('zero_density',dict(density=0.)),('nan_density',dict(density=float('nan')))]
  for name,change in variants:
   c=base|change
   # Invalid parameters still use ordinary latent widths in the wire payload.
   arrays=inputs(c|dict(k=2)) if c['k']<=0 else inputs(c)
   x,q,v=arrays
   if name=='query_width':q=sparse(4,2,1,1)
   if name=='latent_width':v=sparse(3,3,1,1)
   if name=='empty_query':q=sp.csr_matrix((0,3))
   if name=='empty_inverse':v=sp.csr_matrix((0,2))
   arrays=(x,q,v);native_payload=payload(c,arrays)
   try:
    m=model_for(c)
    if c['mode'] in [5,6,7]:
     if c['mode']==5:x.indices[0]=-1
     if c['mode']==6:x.indptr[0]=-1
     if c['mode']==7:x.indptr[1]=len(x.data)+1
     x.check_format(full_check=True)
    elif c['mode']==1:m.transform(q)
    elif c['mode']==2:m.inverse_transform(v)
    else:
     if c['mode']==3:x.data[0]=np.nan
     if c['mode']==4:x.data[0]=np.inf
     if c['mode']==8:q.data[0]=np.nan
     if c['mode']==9:v.data[0]=np.inf
     m.fit(x);m.transform(q);m.inverse_transform(v)
   except ValueError as error:source_error=type(error).__name__
   else:raise AssertionError(('source accepts invalid',kind,name))
   r=subprocess.run(command,input='1\n'+native_payload,text=True,capture_output=True);out=r.stdout+r.stderr
   assert r.returncode and 'fscikit-learn:' in out,(kind,name,out)
   assert not any(t in out for t in ['SIGFPE','Fortran runtime error']),(kind,name,out)
   rejections.append(dict(name=str(kind)+'_'+name,passed=True,source_error=source_error,oracle='scipy_csr_check_format' if c['mode'] in [5,6,7] else 'sklearn_estimator'))
 assert hashes=={str(p):sha(p) for p in paths}
 report=dict(profile=args.profile,component_storage='csr' if args.sparse_components else 'dense',sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,rejections=rejections,hashes=hashes)
 path=args.report or ROOT/'validation/results'/f'csr_projections_{args.profile}.json';path.write_text(json.dumps(report,indent=2)+'\n');print(len(cases),len(checks),len(rejections),'PASS')
if __name__=='__main__':main()
