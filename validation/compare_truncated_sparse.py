"""Dense and operator-based CSR randomized TruncatedSVD against sklearn."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.utils._param_validation import InvalidParameterError
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block

REJECTIONS={'algorithm':'unsupported SVD algorithm','power':'n_iter must be nonnegative',
 'oversampling':'n_oversamples must be positive','seed':'random_state must be nonnegative',
 'normalizer':'unsupported SVD power_iteration_normalizer','rank':'invalid SVD n_components',
 'rank_large':'invalid SVD n_components','capacity':'sampling dimension exceeds integer capacity',
 'single_feature':'randomized SVD requires at least two features','dense_nan':'X contains NaN or infinity',
 'csr_full':'CSR fitting requires randomized algorithm','csr_nan':'CSR data must be finite',
 'csr_empty':'CSR shape must be nonempty','csr_pointer':'CSR pointer endpoints are invalid',
 'csr_column':'CSR column index is out of bounds','csr_unfitted':'truncated_svd is not fitted',
 'csr_features':'CSR feature count mismatch','csr_query_nan':'CSR data must be finite',
 'inverse_features':'X and y sample counts differ'}
NORMS=['auto','none','LU','QR']


def csr_payload(matrix):
 return str(matrix.nnz)+'\n'+' '.join(map(str,matrix.indptr+1))+'\n'+' '.join(map(str,matrix.indices+1))+'\n'+ ' '.join(f'{v:.17g}' for v in matrix.data)+'\n'


def raw_duplicates(matrix):
 # Split stored entries, add explicit zeros and reverse every row. The matrix
 # is unchanged, while canonicality/unique-stored-cell assumptions are exercised.
 pointer=[0];indices=[];values=[]
 for row in range(matrix.shape[0]):
  entries=[]
  for e in range(matrix.indptr[row],matrix.indptr[row+1]):entries +=[(matrix.indices[e],matrix.data[e]/2)]*2
  entries.append((row%matrix.shape[1],0.));entries.reverse()
  indices +=[int(i) for i,v in entries];values +=[v for i,v in entries];pointer.append(len(indices))
 return sp.csr_matrix((values,indices,pointer),shape=matrix.shape)


def exact_reference(x,k):
 u,s,vt=svd(x,full_matrices=False);vt=vt[:k];s=s[:k]
 for row in vt:
  if row[np.argmax(abs(row))]<0:row*=-1
 model=TruncatedSVD(n_components=k);model.components_=vt;model.n_features_in_=x.shape[1];model.singular_values_=s
 z=x@vt.T;model.explained_variance_=z.var(axis=0);total=x.var(axis=0).sum()
 with np.errstate(invalid='ignore'):model.explained_variance_ratio_=model.explained_variance_/total
 return model,z


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
 parser.add_argument('--executable',type=Path);args=parser.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_truncated_sparse_driver')
 options=[(0,2,'auto'),(1,5,'none'),(2,5,'QR'),(5,10,'auto'),(5,4,'LU'),(3,3,'QR')]
 settings=[]
 for n,p in [(30,12),(12,30),(3,9),(1,5)]:
  for seed in [0,42,2026]:
   for power,extra,norm in options:
    for k in [1,min(3,n,p),p]:
     for storage in ['dense','csr','duplicates']:settings.append((n,p,seed,power,extra,norm,k,storage,'randomized','random'))
   for k in [1,min(n,p)]:settings.append((n,p,seed,5,10,'auto',k,'dense','full','random'))
 for storage in ['dense','csr','duplicates']:
  for kind in ['zero','constant']:
   settings.append((20,8,42,5,10,'auto',3,storage,'randomized',kind))
 settings.append((20,1,42,5,10,'auto',1,'dense','full','random'))
 settings.append((5000,100000,42,5,10,'auto',5,'csr','randomized','large'))
 checks=[];fixtures=0;basis_certificates=[];undefined_ratios=[];large=[];duplicates=[];warnings_seen={};errors={};source_parameter_bugs=[]
 for n,p,seed,power,extra,norm,k,storage,algorithm,kind in settings:
  rng=np.random.RandomState(seed)
  if kind=='large':
   rows=np.repeat(np.arange(n),2);cols=rng.randint(p,size=len(rows));data=rng.normal(size=len(rows))
   encoded=sp.csr_matrix((data,(rows,cols)),shape=(n,p));encoded.sum_duplicates();query=encoded[:6].copy()
   source_x=encoded.copy();source_query=query.copy()
  else:
   values=rng.normal(size=(n,p));values[rng.uniform(size=(n,p))<.7]=0
   qvalues=rng.normal(size=(6,p));qvalues[rng.uniform(size=(6,p))<.7]=0
   if kind=='zero':values[:]=0
   if kind=='constant':values[:]=2.5
   if storage=='dense':encoded=values;query=qvalues;source_x=values;source_query=qvalues
   else:
    encoded=sp.csr_matrix(values);query=sp.csr_matrix(qvalues)
    if storage=='duplicates':encoded=raw_duplicates(encoded);query=raw_duplicates(query)
    source_x=encoded.copy();source_query=query.copy();source_x.sum_duplicates();source_query.sum_duplicates()
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always')
   if algorithm=='full':model,source_fit=exact_reference(source_x,k)
   else:
    model=TruncatedSVD(n_components=k,algorithm='randomized',n_iter=power,n_oversamples=extra,
                       power_iteration_normalizer=norm,random_state=seed)
    try:source_fit=model.fit_transform(source_x)
    except InvalidParameterError as exc:
     assert norm=='QR' and "'OR'" in str(exc) and 'power_iteration_normalizer' in str(exc),str(exc)
     source_parameter_bugs.append(dict(n=n,p=p,seed=seed,power=power,extra=extra,k=k,storage=storage,
          requested='QR',source_constraint_typo='OR',error_type=type(exc).__name__,recorded=True))
     with sklearn.config_context(skip_parameter_validation=True):source_fit=model.fit_transform(source_x)
  for w in caught:
   key=w.category.__name__+': '+str(w.message);warnings_seen[key]=warnings_seen.get(key,0)+1
  header=[int(storage!='dense'),n,p,6,k,int(algorithm=='randomized'),power,extra,NORMS.index(norm),seed]
  payload=' '.join(map(str,header))+'\n'
  if storage=='dense':payload+=''.join(' '.join(f'{v:.17g}' for v in a.ravel(order='F'))+'\n' for a in [encoded,query])
  else:payload+=csr_payload(encoded)+csr_payload(query)
  run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
  assert run.returncode==0,(n,p,seed,power,extra,norm,k,storage,algorithm,kind,run.stdout[-2000:],run.stderr)
  lines=iter(run.stdout.splitlines());names=['components','singular','variance','ratio','metadata','fit_transform','train_transform','transform','inverse','unchanged']
  actual={name:block(lines) for name in names};assert next(lines,None) is None
  context=dict(n=n,p=p,seed=seed,power=power,extra=extra,normalizer=norm,k=k,storage=storage,algorithm=algorithm,kind=kind)
  def passed(name):checks.append(dict(**context,name=name,passed=True))
  def close(name,want,tol=3e-6):
   got=actual[name];want=np.asarray(want);assert got.shape==want.shape,(context,name,got.shape,want.shape)
   np.testing.assert_allclose(got,want,atol=tol,rtol=tol,err_msg=str((context,name)))
   errors[name]=max(errors.get(name,0.),float(np.max(abs(got-want),initial=0.)));passed(name)
  close('singular',model.singular_values_);close('variance',model.explained_variance_)
  ratio=model.explained_variance_ratio_.copy()
  if (~np.isfinite(ratio)).any():
   assert n==1 or kind in ['zero','constant'];assert (~np.isfinite(ratio)).all()
   dense_values=source_x.toarray() if sp.issparse(source_x) else source_x
   assert dense_values.var(axis=0).sum()==0
   source_kinds=['nan' if np.isnan(v) else 'positive_infinity' for v in ratio]
   undefined_ratios.append(dict(**context,source_undefined=True,native_zero_convention=True,
        source_total_variance=0.,source_ratio_kinds=source_kinds,passed=True));ratio[:]=0
  close('ratio',ratio);close('metadata',[len(model.components_),p],0);close('unchanged',[1.],0)
  comp=actual['components'];count=len(comp);np.testing.assert_allclose(comp@comp.T,np.eye(count),atol=1e-9,rtol=1e-9);passed('right_orthogonality')
  null=model.singular_values_<=model.singular_values_.max(initial=0.)*1e-10
  operator=TruncatedSVD(n_components=k);operator.n_features_in_=p
  if null.any():
   assert kind!='large'
   full=source_x.toarray() if sp.issparse(source_x) else source_x
   u,s,vt=svd(full,full_matrices=False);positive=s>s.max(initial=0.)*1e-10
   projector=np.eye(p)-vt[positive].T@vt[positive]
   residual=float(np.max(abs(comp[null]-comp[null]@projector),initial=0.))
   assert residual<=3e-6,(context,'null-space membership',residual)
   basis_certificates.append(dict(**context,residual=residual,bound=3e-6,orthonormal=True,passed=True))
   if (~null).any():
    inner=comp[~null]@model.components_[~null].T
    rows,perm=linear_sum_assignment(2-2*abs(inner));sign=np.where(inner[rows,perm]<0,-1.,1.)
    np.testing.assert_allclose(comp[~null],model.components_[~null][perm]*sign[:,None],rtol=3e-6,atol=3e-6)
   operator.components_=comp.copy();passed('components')
  else:
   inner=comp@model.components_.T;rows,perm=linear_sum_assignment(2-2*abs(inner));sign=np.where(inner[rows,perm]<0,-1.,1.)
   operator.components_=model.components_[perm]*sign[:,None];close('components',operator.components_)
  close('fit_transform',operator.transform(source_x));close('train_transform',operator.transform(source_x))
  close('transform',operator.transform(source_query));close('inverse',operator.inverse_transform(actual['transform']))
  np.testing.assert_array_equal(actual['fit_transform'],actual['train_transform']);passed('fit_uses_actual_projection')
  if storage=='duplicates':duplicates.append(dict(**context,nnz=encoded.nnz,canonical_nnz=source_x.nnz,input_preserved=True,passed=True))
  if kind=='large':large.append(dict(**context,nnz=encoded.nnz,dense_input_entries=n*p,dense_input_bytes=8*n*p,passed=True))
  fixtures+=1
  if fixtures%36==0:print(f'{fixtures} truncated sparse fixtures passed',flush=True)
 results=ROOT/'validation/results';results.mkdir(exist_ok=True);source=Path(sklearn.__file__).parent;digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 report=dict(profile=args.profile,fixtures=fixtures,checks=checks,basis_certificates=basis_certificates,
  undefined_ratio_certificates=undefined_ratios,duplicate_cases=duplicates,large_sparse_cases=large,
  source_warnings=warnings_seen,source_parameter_bugs=source_parameter_bugs,max_absolute_errors=errors,native_sha256=digest(exe),validator_sha256=digest(Path(__file__)),
  sklearn=sklearn.__version__,numpy=np.__version__,source_sha256={p:digest(source/p) for p in ['decomposition/_truncated_svd.py','utils/extmath.py','utils/sparsefuncs.py']})
 (results/f'truncated_sparse_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejected=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True)
  assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and message in run.stdout,(case,run.stdout,run.stderr)
  rejected.append(dict(name=case,expected=message,passed=True))
 (results/f'truncated_sparse_invalid_{args.profile}.json').write_text(json.dumps(dict(profile=args.profile,checks=rejected,
  native_sha256=report['native_sha256'],validator_sha256=report['validator_sha256']),indent=2,allow_nan=False)+'\n')
 print(f'{len(checks)} truncated sparse groups / {fixtures} fixtures, {len(duplicates)} duplicate cases, {len(rejected)} rejections passed')


if __name__=='__main__':main()
