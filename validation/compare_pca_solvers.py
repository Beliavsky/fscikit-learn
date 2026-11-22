"""PCA solver dispatch, randomized controls and shared randomized-SVD contracts."""
from pathlib import Path
import argparse,copy,hashlib,json,subprocess,warnings
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
from scipy.optimize import linear_sum_assignment
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block

REJECTIONS={'solver':'unsupported PCA svd_solver','power':'iterated_power must be >= -1',
 'oversampling':'n_oversamples must be positive','seed':'random_state must be nonnegative',
 'normalizer':'unsupported PCA power_iteration_normalizer',
 'random_fraction':'randomized solver requires a positive integer','random_mle':'randomized solver requires a positive integer',
 'random_zero':'randomized solver requires a positive integer','capacity':'randomized sample dimension exceeds integer capacity',
 'core_oversampling':'oversampling must be nonnegative','core_power':'iteration count must be nonnegative',
 'core_normalizer':'unsupported randomized SVD normalizer','core_capacity':'invalid randomized SVD component count'}
NORMS=['auto','none','LU','QR'];SOLVERS=['full','randomized','covariance_eigh','auto']


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
 parser.add_argument('--executable',type=Path);args=parser.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_pca_solvers_driver')
 checks=[];fixtures=0;core_fixtures=0;basis_certificates=[];undefined_ratios=[];source_warnings={};errors={}
 options=[('full',-1,10,'auto'),('covariance_eigh',-1,10,'auto')]+[
  ('randomized',power,extra,norm) for power,extra,norm in [(-1,10,'auto'),(0,2,'none'),(1,5,'auto'),(2,5,'QR'),(3,5,'LU'),(3,5,'QR')]]
 settings=[]
 for n,p in [(42,9),(18,24),(100,7),(60,28)]:
  for seed in [0,42,2026]:
   for solver,power,extra,norm in options:
    for k in [1,3,0]:
     for white in [False,True]:settings.append((n,p,seed,solver,power,extra,norm,k,white,'count',.8,'random'))
 for n,p in [(100,7),(42,9),(18,24)]:
  for seed in [0,42]:
   for solver in ['full','covariance_eigh','auto']:
    for mode in ['variance','mle','zero']:
     if mode=='mle' and n<p:continue
     for white in [False,True]:settings.append((n,p,seed,solver,-1,10,'auto',0,white,mode,.8,'random'))
 for n,p,k,mode in [(520,60,3,'count'),(520,60,55,'count'),(520,60,0,'variance'),(520,60,0,'mle'),(60,520,3,'count'),(600,30,3,'count')]:
  settings.append((n,p,42,'auto',-1,10,'auto',k,False,mode,.8,'random'))
 for solver in SOLVERS:
  for white in [False,True]:settings.append((40,6,0,solver,3,5,'auto',3,white,'count',.8,'zero'))
 for n,p,seed,solver,power,extra,norm,k,white,mode,fraction,kind in settings:
  rng=np.random.RandomState(seed);loadings=rng.normal(size=(3,p))
  x=rng.normal(size=(n,3))@loadings+.3*rng.normal(size=(n,p))+np.linspace(-.3,.5,p)
  query=rng.normal(size=(6,3))@loadings+.3*rng.normal(size=(6,p))+np.linspace(-.3,.5,p)
  if kind=='zero':x[:]=0
  choice=(None if k==0 else k) if mode=='count' else fraction if mode=='variance' else 'mle' if mode=='mle' else 0
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always')
   source=PCA(n_components=choice,whiten=white,svd_solver=solver,iterated_power='auto' if power==-1 else power,
              n_oversamples=extra,power_iteration_normalizer=norm,random_state=seed)
   source_fit=source.fit_transform(x)
  for w in caught:
   key=w.category.__name__+': '+str(w.message);source_warnings[key]=source_warnings.get(key,0)+1
  header=[n,p,len(query),k,['count','variance','mle','zero'].index(mode),fraction,int(white),SOLVERS.index(solver),power,extra,NORMS.index(norm),seed]
  payload=' '.join(map(str,header))+'\n'+''.join(' '.join(f'{v:.17g}' for v in a.ravel(order='F'))+'\n' for a in [x,query])
  run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
  assert run.returncode==0,(n,p,k,solver,power,extra,norm,seed,white,mode,run.stdout,run.stderr)
  lines=iter(run.stdout.splitlines());names=['components','variance','ratio','singular','mean','noise','metadata',
       'fit_transform','train_transform','transform','inverse','covariance']
  actual={name:block(lines) for name in names};assert next(lines,None) is None
  context=dict(n=n,p=p,seed=seed,solver=solver,power=power,extra=extra,normalizer=norm,k=k,white=white,mode=mode,kind=kind)
  def passed(name):checks.append(dict(**context,name=name,passed=True))
  def close(name,want,tol=3e-6):
   got=actual[name];want=np.asarray(want);assert got.shape==want.shape,(context,name,got.shape,want.shape)
   np.testing.assert_allclose(got,want,atol=tol,rtol=tol,err_msg=str((context,name)))
   errors[name]=max(errors.get(name,0.),float(np.max(abs(got-want),initial=0.)));passed(name)
  close('variance',source.explained_variance_);close('singular',source.singular_values_);close('mean',source.mean_)
  close('noise',[source.noise_variance_]);close('metadata',[n,source.n_components_,SOLVERS.index(source._fit_svd_solver)],0)
  ratio=source.explained_variance_ratio_.copy()
  if np.isnan(ratio).any():
   assert kind=='zero' and np.isnan(ratio).all();ratio[:]=0;undefined_ratios.append(dict(**context,passed=True))
  close('ratio',ratio);close('covariance',source.get_covariance())
  comp=actual['components'];count=len(comp);np.testing.assert_allclose(comp@comp.T,np.eye(count),atol=1e-9,rtol=1e-9);passed('orthogonality')
  inner=comp@source.components_.T
  if count:
   rows,perm=linear_sum_assignment(2-2*abs(inner));assert np.array_equal(rows,np.arange(count))
   signs=np.where(inner[rows,perm]<0,-1.,1.)
  else:perm=np.empty(0,dtype=int);signs=np.empty(0)
  null=source.explained_variance_<=source.explained_variance_.max(initial=0.)*1e-12
  operator=copy.deepcopy(source)
  if null.any():
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');full=PCA(svd_solver='full').fit(x)
   full_positive=full.explained_variance_>full.explained_variance_.max(initial=0.)*1e-12
   null_projector=np.eye(p)-full.components_[full_positive].T@full.components_[full_positive]
   native_null=comp[null]
   err=float(np.max(abs(native_null-native_null@null_projector),initial=0.))
   assert err<=2e-6,(context,'null eigenspace',err)
   basis_certificates.append(dict(**context,source_nullspace_error=err,bound=2e-6,orthonormal=True,passed=True))
   if (~null).any():np.testing.assert_allclose(comp[~null],(source.components_[perm]*signs[:,None])[~null],rtol=3e-6,atol=3e-6)
   passed('components')
   operator.components_=comp.copy();operator.explained_variance_=actual['variance'].copy();operator.mean_=actual['mean'].copy()
  else:
   operator.components_=source.components_[perm]*signs[:,None];close('components',operator.components_)
  close('transform',operator.transform(query));close('inverse',operator.inverse_transform(actual['transform']))
  if null.any() and white:
   scale=np.maximum(np.sqrt(actual['variance']),np.finfo(float).eps);reference=copy.deepcopy(operator);reference.whiten=False
   bound=256*np.finfo(float).eps*max(n,p)*max(1.,np.max(abs(x)))
   np.testing.assert_allclose(actual['train_transform']*scale,reference.transform(x),atol=bound,rtol=3e-6);passed('train_transform')
   if source._fit_svd_solver=='randomized':
    np.testing.assert_allclose(actual['fit_transform'].T@actual['fit_transform']/(n-1),np.eye(count),rtol=1e-8,atol=1e-8)
    if (~null).any():np.testing.assert_allclose(actual['fit_transform'][:,~null],(source_fit[:,perm]*signs)[:,~null],rtol=3e-6,atol=3e-6)
   else:np.testing.assert_array_equal(actual['fit_transform'],actual['train_transform'])
   passed('fit_transform')
  else:
   close('train_transform',operator.transform(x))
   if source._fit_svd_solver=='randomized':close('fit_transform',source_fit[:,perm]*signs)
   else:close('fit_transform',operator.transform(x))
  fixtures+=1
  if fixtures%36==0:print(f'{fixtures} PCA solver fixtures passed',flush=True)
 # Direct full-rank shared-core checks include zero oversampling, explicit sign
 # control, automatic powers, a request larger than matrix rank and RNG continuity.
 core_settings=[(n,p,k,seed,extra,power,norm,flip) for n,p,k in [(21,9,3),(9,21,3),(40,24,4)]
   for seed in [0,42,2026] for extra in [0,2,10] for power,norm in [(0,'none'),(1,'auto'),(2,'QR'),(3,'LU'),(3,'QR'),(4,'none'),(-1,'auto')]
   for flip in [False,True]]
 core_settings +=[(21,9,12,42,0,0,'none',True),(9,21,12,42,0,1,'auto',False)]
 for n,p,k,seed,extra,power,norm,flip in core_settings:
  x=np.random.RandomState(seed+7).normal(size=(n,p));rng=np.random.RandomState(seed)
  u,s,vt=randomized_svd(x,k,n_oversamples=extra,n_iter='auto' if power==-1 else power,power_iteration_normalizer=norm,flip_sign=flip,random_state=rng)
  next_normal=rng.normal();next_uniform=rng.uniform()
  header=[n,p,k,power,extra,NORMS.index(norm),seed,int(flip)]
  payload=' '.join(map(str,header))+'\n'+' '.join(f'{v:.17g}' for v in x.ravel(order='F'))+'\n'
  run=subprocess.run([str(exe),'core'],input=payload,text=True,capture_output=True);assert run.returncode==0,(header,run.stdout,run.stderr)
  lines=iter(run.stdout.splitlines());got_s=block(lines);got_u=block(lines);got_vt=block(lines);after=block(lines);assert next(lines,None) is None
  context=dict(n=n,p=p,k=k,seed=seed,extra=extra,power=power,normalizer=norm,flip=flip)
  def core_check(name,got,want,tol=3e-7):
   np.testing.assert_allclose(got,want,atol=tol,rtol=tol,err_msg=str((context,name)));checks.append(dict(**context,name='core_'+name,passed=True))
  core_check('singular',got_s,s)
  sign=np.where(np.sum(got_u*u,axis=0)<0,-1.,1.)
  core_check('left',got_u,u*sign);core_check('right',got_vt,vt*sign[:,None])
  core_check('rng_after',after,[next_normal,next_uniform],2e-14)
  core_check('left_orthogonality',got_u.T@got_u,np.eye(len(s)))
  core_check('right_orthogonality',got_vt@got_vt.T,np.eye(len(s)))
  if flip:
   pivots=np.argmax(abs(got_u),axis=0);assert np.all(got_u[pivots,np.arange(len(s))]>=0)
  checks.append(dict(**context,name='core_sign_contract',passed=True));core_fixtures+=1
 results=ROOT/'validation/results';results.mkdir(exist_ok=True);source=Path(sklearn.__file__).parent;digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 report=dict(profile=args.profile,fixtures=fixtures,core_fixtures=core_fixtures,checks=checks,basis_certificates=basis_certificates,
  undefined_ratio_certificates=undefined_ratios,source_warnings=source_warnings,max_absolute_errors=errors,
  native_sha256=digest(exe),validator_sha256=digest(Path(__file__)),sklearn=sklearn.__version__,numpy=np.__version__,
  source_sha256={p:digest(source/p) for p in ['decomposition/_pca.py','utils/extmath.py']})
 (results/f'pca_solvers_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejected=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True)
  assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and message in run.stdout,(case,run.stdout,run.stderr)
  rejected.append(dict(name=case,expected=message,passed=True))
 (results/f'pca_solvers_invalid_{args.profile}.json').write_text(json.dumps(dict(profile=args.profile,checks=rejected,
  native_sha256=report['native_sha256'],validator_sha256=report['validator_sha256']),indent=2,allow_nan=False)+'\n')
 print(f'{len(checks)} PCA solver groups / {fixtures} PCA and {core_fixtures} core fixtures, {len(rejected)} rejections passed')


if __name__=='__main__':main()
