"""Source Nystroem sparse landmarks, mixed kernels and explicit singular certificates."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('LOKY_MAX_CPU_COUNT','1')
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import sklearn
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import pairwise_kernels
from compare_csr_projections import sparse,csr_payload
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
KERNELS=['rbf','linear','poly','polynomial','cosine','laplacian','sigmoid','callable','callable','chi2','additive_chi2','precomputed']
def row(a):return a.toarray().ravel() if sp.issparse(a) else np.asarray(a).ravel()
def callback(a,b):a,b=row(a),row(b);return a@b+.2*a[0]-.1*b[0]+1
def callback_rbf(a,b):a,b=row(a),row(b);return np.exp(-.3*np.sum((a-b)**2))
def config(**kw):return dict(n=9,p=4,m=5,k=3,seed=0,kind=0,fit_kind=1,query_kind=1,gamma=-1.,coef0=1.,degree=3,query_gamma=-1.,change=0,mode=0,style=0,scenario='random',qwidth=0)|kw
def fixtures():
 for seed,kind,(n,p,k),fit_kind,query_kind,style in itertools.product([0,7,42,2026,2147483647],range(9),[(9,4,3),(7,3,10)],[0,1],[0,1],range(3)):
  if not (fit_kind or query_kind):continue
  yield config(seed=seed,kind=kind,n=n,p=p,k=k,fit_kind=fit_kind,query_kind=query_kind,style=style)
 for kind,fit_kind,query_kind,scenario in itertools.product(range(9),[0,1],[0,1],['zero','tiny']):
  if fit_kind or query_kind:yield config(kind=kind,fit_kind=fit_kind,query_kind=query_kind,scenario=scenario,style=1)
 for fit_kind,query_kind in [(1,0),(1,1),(0,1)]:
  yield config(kind=0,fit_kind=fit_kind,query_kind=query_kind,style=2,change=1,query_gamma=.4)
  yield config(kind=2,fit_kind=fit_kind,query_kind=query_kind,style=2,coef0=-1.,degree=2,gamma=.7,query_gamma=.3)
def inputs(c):
 x=sparse(c['n'],c['p'],c['style'],c['seed']);q=sparse(c['m'],c['qwidth'] or c['p'],c['style'],c['seed']+2)
 if c['scenario']=='zero':x.data.fill(0);q.data.fill(0)
 if c['scenario']=='tiny':x.data.fill(1e-20);q.data.fill(2e-20)
 return x,q
def model_for(c):
 kind=KERNELS[c['kind']] if 0<=c['kind']<len(KERNELS) else 'unknown'
 if c['kind']==7:kind=callback
 if c['kind']==8:kind=callback_rbf
 options=dict(kernel=kind,n_components=c['k'],random_state=c['seed'])
 if c['kind']not in [7,8,11]:options.update(gamma=None if c['gamma']==-1 else c['gamma'],coef0=c['coef0'],degree=c['degree'])
 return Nystroem(**options)
def same(a,b):return all(np.array_equal(getattr(a,k),getattr(b,k)) for k in ['indptr','indices','data']) if sp.issparse(a) else np.array_equal(a,b)
def oracle(c):
 rawx,rawq=inputs(c);x=rawx.copy() if c['fit_kind'] else rawx.toarray();q=rawq.copy() if c['query_kind'] else rawq.toarray();original_x=x.copy();original_q=q.copy();state=np.random.get_state();m=model_for(c)
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore');z=m.fit_transform(x)
  g=pairwise_kernels(m.components_,metric=m.kernel,filter_params=True,**m._get_kernel_params());singular=la.svdvals(g)
  if c['change']:m.kernel='laplacian'
  m.gamma=None if c['query_gamma']==-1 else c['query_gamma'];query=m.transform(q)
 component=m.components_;issparse=sp.issparse(component)
 fields=dict(indices=m.component_indices_+1,component_shape=component.shape,component_ptr=component.indptr+1 if issparse else [],component_idx=component.indices[:component.nnz]+1 if issparse else [],component_data=component.data[:component.nnz] if issparse else component,normalization=m.normalization_,norm_gram=m.normalization_.T@m.normalization_,train=z,query=query,train_gram=z@z.T,query_gram=query@query.T,cross_gram=z@query.T,metadata=[m.n_features_in_,len(m.component_indices_),int(issparse),int(not issparse)],preserved=[1,1,1],repeat=[1],svd_status=[0],singular=singular)
 mutation=[not same(x,original_x),not same(q,original_q)];assert all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state()))
 regular=bool(singular[-1]>max(1e-10,1e-7*singular[0]));return rawx,rawq,g,fields,regular,mutation
def payload(c,a):
 x,q,g=a[:3]
 return ' '.join(str(c[k]) for k in ['fit_kind','query_kind','kind','k','seed','gamma','coef0','degree','query_gamma','change','mode'])+'\n'+csr_payload(x)+csr_payload(q)+' '.join(format(v,'.17g') for v in g.ravel(order='F'))+'\n'
def reject_cases(executable):
 result=[]
 variants=[('zero_components',dict(k=0)),('negative_components',dict(k=-1)),('negative_seed',dict(seed=-1)),('unknown_kernel',dict(kind=12)),('negative_gamma',dict(gamma=-2.)),('nan_gamma',dict(gamma=float('nan'))),('inf_gamma',dict(gamma=float('inf'))),('negative_degree',dict(kind=2,degree=-1)),('nan_coef',dict(kind=2,coef0=float('nan'))),('unfitted',dict(mode=1)),('nan_fit',dict(mode=2)),('inf_fit',dict(mode=3)),('nan_query',dict(mode=4)),('inf_query',dict(mode=5)),('bad_index',dict(mode=6)),('bad_start',dict(mode=7)),('bad_pointer',dict(mode=8)),('query_width',dict(qwidth=5)),('empty_fit',dict(n=0)),('empty_features',dict(p=0)),('empty_query',dict(m=0))]
 variants += [(f'chi_{kind}_{fit}_{query}',dict(kind=kind,fit_kind=fit,query_kind=query,scenario='zero')) for kind in [9,10] for fit,query in [(1,0),(1,1),(0,1)]]
 variants += [('precomputed_fit',dict(kind=11,n=4,p=4,k=4)),('precomputed_query_object',dict(kind=11,n=4,p=4,k=4,fit_kind=0))]
 for name,change in variants:
  c=config(style=1)|change;x,q=inputs(c);g=np.zeros((min(c['n'],max(0,c['k'])),)*2);wire=payload(c,(x,q,g));source_error=None;oracle_name='sklearn_estimator'
  try:
   if c['mode'] in [6,7,8]:
    oracle_name='scipy_csr_check_format'
    if c['mode']==6:x.indices[0]=-1
    if c['mode']==7:x.indptr[0]=-1
    if c['mode']==8:x.indptr[1]=len(x.data)+1
    x.check_format(full_check=True)
   else:
    if c['mode']==2:x.data[0]=np.nan
    if c['mode']==3:x.data[0]=np.inf
    if c['mode']==4:q.data[0]=np.nan
    if c['mode']==5:q.data[0]=np.inf
    a=x if c['fit_kind'] else x.toarray();b=q if c['query_kind'] else q.toarray();model=model_for(c)
    with warnings.catch_warnings(),np.errstate(all='ignore'):
     warnings.simplefilter('ignore')
     if c['mode']==1:model.transform(b)
     model.fit(a);model.gamma=None if c['query_gamma']==-1 else c['query_gamma'];out=model.transform(b)
     if name=='precomputed_query_object':
      assert out.dtype==object;oracle_name='native_numeric_output_guard';source_error='object_array'
  except (ValueError,TypeError) as error:source_error=type(error).__name__
  assert source_error is not None,('source accepts invalid',name)
  r=subprocess.run([str(executable)],input='1\n'+wire,text=True,capture_output=True);out=r.stdout+r.stderr
  assert r.returncode and 'fscikit-learn:' in out and not any(t in out for t in ['SIGFPE','Fortran runtime error']),(name,out)
  result.append(dict(name=name,passed=True,oracle=oracle_name,source_error=source_error))
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
 paths=[Path(__file__),ROOT/'validation/compare_csr_projections.py',args.executable,ROOT/'src/sklearn_kernel_approximation.f90',ROOT/'src/sklearn_sparse.f90',ROOT/'src/sklearn_stateless_preprocessing.f90',ROOT/'app/reference_nystroem_csr_driver.f90',Path(sklearn.__file__).parent/'kernel_approximation.py'];hashes={str(p):sha(p) for p in paths}
 cases=list(fixtures());assert len({json.dumps(c,sort_keys=True) for c in cases})==len(cases);expected=[oracle(c) for c in cases]
 r=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,expected)),text=True,capture_output=True)
 if r.returncode:raise AssertionError((r.stdout[-800:],r.stderr[-1500:]))
 lines=iter(r.stdout.splitlines());index=-1;checks=[];inventories=[];fields={}
 for line in lines:
  p=line.split()
  if not p:continue
  if p[0]=='BEGIN':index+=1;fields={};assert int(p[1])==index+1
  elif p[0]=='END':
   c=cases[index];x,q,g,want,regular,mutation=expected[index];n=len(want['singular']);assert fields.keys()==want.keys()|{'svd_u','svd_v'}
   for name,values in fields.items():
    if name in ['svd_u','svd_v']:continue
    if name in ['normalization','train','query'] and not regular:continue
    target=np.asarray(want[name]).ravel(order='F');assert values.shape==target.shape,(index,c,name,values.shape,target.shape)
    if name in ['indices','component_shape','component_ptr','component_idx','component_data','metadata','preserved','repeat','svd_status']:np.testing.assert_array_equal(values,target,err_msg=str((index,c,name)))
    else:
     # A matrix norm bound handles cancellation in entries of the 1e12 null-space projector.
     atol=3e-8 if regular or name!='norm_gram' else 512*n*np.finfo(float).eps*max(1.,np.linalg.norm(want[name],ord=2))
     np.testing.assert_allclose(values,target,rtol=3e-8,atol=atol,equal_nan=True,err_msg=str((index,c,name)))
    checks.append(dict(fixture=index,field=name,oracle='native_input_identity' if name=='preserved' else 'native_refit_identity' if name=='repeat' else 'source' if regular else 'source_svd_invariant',passed=True))
   u=fields['svd_u'].reshape(n,n,order='F');v=fields['svd_v'].reshape(n,n,order='F');s=fields['singular']
   np.testing.assert_allclose(u.T@u,np.eye(n),rtol=1e-11,atol=1e-11);np.testing.assert_allclose(v.T@v,np.eye(n),rtol=1e-11,atol=1e-11);np.testing.assert_allclose((u*s)@v.T,g,rtol=2e-10,atol=2e-10)
   checks.extend(dict(fixture=index,field=name,oracle='svd_certificate',passed=True) for name in ['svd_orthogonality','svd_reconstruction'])
   inventories.append(sorted(v['field'] for v in checks if v['fixture']==index))
  else:assert p[0] not in fields;fields[p[0]]=np.fromstring(next(lines),sep=' ');assert len(fields[p[0]])==int(p[1])
 assert index+1==len(cases);assert hashes=={str(p):sha(p) for p in paths}
 rejections=reject_cases(args.executable)
 d=dict(rejections=rejections,profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,regular=[a[-2] for a in expected],source_input_mutation=[a[-1] for a in expected],hashes=hashes)
 (args.report or ROOT/'validation/results'/f'nystroem_csr_{args.profile}.json').write_text(json.dumps(d,indent=2)+'\n');print(len(cases),len(checks),sum(d['regular']),len(rejections),'PASS')
if __name__=='__main__':main()
