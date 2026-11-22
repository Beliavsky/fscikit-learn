"""PCA variance/Minka dimensions and probabilistic covariance, precision and scores."""
from pathlib import Path
import argparse,copy,hashlib,json,subprocess,warnings
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.decomposition._pca import _assess_dimension
from scipy.optimize import linear_sum_assignment
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block

REJECTIONS={'mode':'unsupported PCA dimension_mode','fraction_zero':'variance_fraction must be in (0,1)',
 'fraction_one':'variance_fraction must be in (0,1)','fraction_nan':'variance_fraction must be finite',
 'mle_wide':'MLE requires samples >= features','mle_tied':'MLE spectrum is numerically invalid',
 'components':'invalid PCA n_components','large_components':'invalid PCA n_components',
 'one_sample':'PCA requires at least two samples','nan_x':'X contains NaN or infinity',
 'covariance_unfitted':'PCA is not fitted','precision_unfitted':'PCA is not fitted','score_unfitted':'PCA is not fitted',
 'query_features':'X and y sample counts differ','query_nan':'X contains NaN or infinity',
 'singular_precision':'covariance must be positive definite','singular_score':'covariance must be positive definite',
 'wide_density':'covariance must be positive definite','inverse_zero_shape':'X and y sample counts differ'}


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
 parser.add_argument('--executable',type=Path);args=parser.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_pca_probability_driver')
 settings=[]
 for n,p in [(50,7),(80,5),(12,18),(30,1),(40,4)]:
  for seed in [0,42,2026]:
   choices=[('count',0,.95),('count',1,.95),('count',min(3,n,p),.95),('variance',0,.5),
            ('variance',0,.8),('variance',0,.95),('zero',0,.95)]
   if n>=p:choices.append(('mle',0,.95))
   for mode,k,fraction in choices:
    for whiten in [False,True]:settings.append((n,p,seed,mode,k,fraction,whiten,'random'))
 for n,p in [(20,3),(12,1)]:
  for mode in ['count','variance','mle','zero']:
   for whiten in [False,True]:settings.append((n,p,0,mode,0,.9,whiten,'constant'))
 for whiten in [False,True]:settings.append((4,2,0,'variance',0,.5,whiten,'threshold'))
 checks=[];fixtures=0;basis_certificates=[];undefined_ratio=[];singular_models=[];errors={};source_warnings={}
 for n,p,seed,mode,k,fraction,whiten,kind in settings:
  rng=np.random.RandomState(seed);loadings=rng.normal(size=(3,p));x=rng.normal(size=(n,3))@loadings+.3*rng.normal(size=(n,p))
  query=rng.normal(size=(6,3))@loadings+.3*rng.normal(size=(6,p));x+=np.linspace(-.3,.5,p);query+=np.linspace(-.3,.5,p)
  if kind=='constant':x[:]=2.5
  if kind=='threshold':x=np.array([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]]);query=rng.normal(size=(6,2))
  choice=(None if k==0 else k) if mode=='count' else fraction if mode=='variance' else 'mle' if mode=='mle' else 0
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always');model=PCA(n_components=choice,whiten=whiten,svd_solver='full').fit(x)
   full=PCA(svd_solver='full').fit(x)
  for w in caught:
   key=w.category.__name__+': '+str(w.message);source_warnings[key]=source_warnings.get(key,0)+1
  header=[n,p,len(query),['count','variance','mle','zero'].index(mode),k,int(whiten),fraction]
  payload=' '.join(map(str,header))+'\n'+''.join(' '.join(f'{v:.17g}' for v in a.ravel(order='F'))+'\n' for a in [x,query])
  run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
  assert run.returncode==0,(n,p,seed,mode,k,fraction,whiten,kind,run.stdout,run.stderr)
  names=['mean','components','variance','ratio','singular','noise','metadata','covariance','transform','train_transform',
         'inverse','train_inverse','mle_values','mle_kinds','probability','precision','scores','score']
  lines=iter(run.stdout.splitlines());actual={name:block(lines) for name in names};assert next(lines,None) is None
  context=dict(n=n,p=p,seed=seed,mode=mode,k=k,fraction=fraction,whiten=whiten,kind=kind)
  def passed(name):checks.append(dict(**context,name=name,passed=True))
  def close(name,want,atol=2e-6,rtol=2e-6):
   got=actual[name];want=np.asarray(want);assert got.shape==want.shape,(context,name,got.shape,want.shape)
   np.testing.assert_allclose(got,want,atol=atol,rtol=rtol,err_msg=str((context,name)))
   errors[name]=max(errors.get(name,0.),float(np.max(abs(got-want),initial=0.)));passed(name)
  close('mean',model.mean_);close('variance',model.explained_variance_);close('singular',model.singular_values_)
  close('noise',[model.noise_variance_]);close('metadata',[n,model.n_components_],0,0)
  ratio=model.explained_variance_ratio_.copy()
  if np.isnan(ratio).any():
   assert np.all(full.explained_variance_==0) and np.isnan(ratio).all()
   undefined_ratio.append(dict(**context,source_undefined=True,native_zero_convention=True,passed=True));ratio[:]=0
  close('ratio',ratio)
  close('covariance',model.get_covariance())
  components=actual['components'];count=len(components);sample_cov=np.atleast_2d(np.cov(x,rowvar=False))
  np.testing.assert_allclose(components@components.T,np.eye(count),atol=1e-10,rtol=1e-10);passed('orthonormality')
  residual=sample_cov@components.T-components.T*actual['variance']
  bound=256*np.finfo(float).eps*max(n,p)*max(1.,np.linalg.norm(sample_cov,2))
  assert np.max(abs(residual),initial=0.)<=bound,(context,'eigen residual',np.max(abs(residual)),bound)
  passed('spectral_eigen_equation')
  scale=full.explained_variance_.max(initial=0.)
  null=count>0 and np.any(model.explained_variance_<=scale*1e-12)
  tied=count>1 and np.any(abs(np.diff(model.explained_variance_))<=scale*1e-10)
  operator=copy.deepcopy(model)
  if null or tied:
   projected=[]
   positive=full.explained_variance_>scale*1e-12
   null_projector=np.eye(p)-full.components_[positive].T@full.components_[positive]
   for row,eigenvalue in zip(components,actual['variance']):
    if eigenvalue<=scale*1e-12:projected.append(row@null_projector)
    else:
     group=abs(full.explained_variance_-eigenvalue)<=scale*1e-10
     basis=full.components_[group];assert len(basis)>0
     projected.append((row@basis.T)@basis)
   projected=np.asarray(projected).reshape(count,p)
   subspace_error=float(np.max(abs(components-projected),initial=0.))
   close('components',projected)
   basis_certificates.append(dict(**context,zero_variance=bool(null),tied_variance=bool(tied),
        residual=float(np.max(abs(residual),initial=0.)),bound=float(bound),orthonormal=True,
        source_subspace_error=subspace_error,source_subspace_bound=2e-6,passed=True))
   # Evaluate source operators in the independently certified native eigenbasis.
   # Covariance is still compared with the independent original source fit.
   operator.components_=components.copy();operator.explained_variance_=actual['variance'].copy();operator.mean_=actual['mean'].copy()
  else:
   if count:
    inner=components@model.components_.T
    cost=np.sum(components**2,axis=1)[:,None]+np.sum(model.components_**2,axis=1)[None,:]-2*abs(inner)
    rows,cols=linear_sum_assignment(cost);assert np.array_equal(rows,np.arange(count))
    signs=np.where(inner[rows,cols]<0,-1.,1.);operator.components_=model.components_[cols]*signs[:,None]
   close('components',operator.components_)
  close('transform',operator.transform(query))
  if null and whiten:
   # Null whitening divides roundoff by eps. Compare the underlying projection
   # before whitening, with a dot-product roundoff bound, rather than assigning
   # meaningful coordinates to an exactly zero-variance direction.
   got=actual['train_transform']*np.maximum(np.sqrt(actual['variance']),np.finfo(float).eps)
   reference=copy.deepcopy(operator);reference.whiten=False;want=reference.transform(x)
   roundoff=256*np.finfo(float).eps*max(n,p)*max(1.,np.max(abs(x)))
   np.testing.assert_allclose(got,want,atol=roundoff,rtol=2e-6);passed('train_transform')
  else:close('train_transform',operator.transform(x))
  close('inverse',operator.inverse_transform(actual['transform']))
  close('train_inverse',operator.inverse_transform(actual['train_transform']))
  if mode=='mle':
   likelihood=np.full(p,-np.inf)
   for rank in range(1,p):likelihood[rank]=_assess_dimension(full.explained_variance_,rank,n)
   kinds=np.where(np.isfinite(likelihood),0.,-1.)
   close('mle_kinds',kinds,0,0);close('mle_values',np.where(np.isfinite(likelihood),likelihood,0.))
  else:close('mle_kinds',np.empty(0),0,0);close('mle_values',np.empty(0),0,0)
  probability=model.noise_variance_>0 or (count==p and np.all(model.explained_variance_>0))
  close('probability',[int(probability)],0,0)
  if probability:
   close('precision',model.get_precision());close('scores',model.score_samples(query));close('score',[model.score(query)])
   np.testing.assert_allclose(actual['covariance']@actual['precision'],np.eye(p),atol=2e-8,rtol=2e-8)
   passed('covariance_precision_identity')
  else:
   assert np.linalg.matrix_rank(model.get_covariance())<p
   singular_models.append(dict(**context,proper_density=False,source_covariance_rank=int(np.linalg.matrix_rank(model.get_covariance())),passed=True))
   close('precision',np.empty((0,0)),0,0);close('scores',np.empty(0),0,0);close('score',np.empty(0),0,0)
  fixtures+=1
  if fixtures%36==0:print(f'{fixtures} PCA probability fixtures passed',flush=True)
 results=ROOT/'validation/results';results.mkdir(exist_ok=True);source=Path(sklearn.__file__).parent
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 report=dict(profile=args.profile,fixtures=fixtures,checks=checks,basis_certificates=basis_certificates,
    undefined_ratio_certificates=undefined_ratio,singular_models=singular_models,max_absolute_errors=errors,
    source_warnings=source_warnings,sklearn=sklearn.__version__,numpy=np.__version__,native_sha256=digest(exe),
    validator_sha256=digest(Path(__file__)),source_sha256={p:digest(source/p) for p in ['decomposition/_pca.py','decomposition/_base.py']})
 (results/f'pca_probability_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejected=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True)
  assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and message in run.stdout,(case,run.stdout,run.stderr)
  rejected.append(dict(name=case,expected=message,passed=True))
 (results/f'pca_probability_invalid_{args.profile}.json').write_text(json.dumps(dict(profile=args.profile,checks=rejected,
    native_sha256=report['native_sha256'],validator_sha256=report['validator_sha256']),indent=2,allow_nan=False)+'\n')
 print(f'{len(checks)} PCA probability groups / {fixtures} fixtures, {len(basis_certificates)} basis certificates, {len(rejected)} rejections passed')


if __name__=='__main__':main()
