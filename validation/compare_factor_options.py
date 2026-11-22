"""FactorAnalysis full/randomized SVD and orthogonal varimax/quartimax rotations."""
from pathlib import Path
import argparse,copy,hashlib,json,subprocess,warnings
from unittest.mock import patch
import numpy as np
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition._factor_analysis import _ortho_rotation
from sklearn.exceptions import ConvergenceWarning
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block

REJECTIONS={'rotation':'unsupported FactorAnalysis rotation','solver':'unsupported FactorAnalysis svd_method',
 'power':'iterated_power must be nonnegative','seed':'random_state must be nonnegative',
 'iterations':'invalid FactorAnalysis iteration parameters','negative_tol':'invalid FactorAnalysis iteration parameters',
 'nan_tol':'tol must be finite','infinite_tol':'tol must be finite','noise_nan':'noise variance must be finite',
 'noise_infinite':'noise variance must be finite','noise_shape':'noise variance shape mismatch',
 'noise_zero':'noise variance must be positive','components':'invalid FactorAnalysis n_components',
 'unfitted':'FactorAnalysis is not fitted','features':'X and y sample counts differ'}


def align(native,source):
 if not len(native):return source,np.empty(0,dtype=int),np.empty(0)
 inner=native@source.T
 cost=np.sum(native**2,axis=1)[:,None]+np.sum(source**2,axis=1)[None,:]-2*abs(inner)
 rows,cols=linear_sum_assignment(cost);assert np.array_equal(rows,np.arange(len(native)))
 signs=np.where(inner[rows,cols]<0,-1.,1.)
 return source[cols]*signs[:,None],cols,signs


def criterion(components,rotation):
 value=np.sum(components**4)
 if rotation=='varimax':value-=np.sum(np.sum(components**2,axis=1)**2)/components.shape[1]
 return float(value)


def replay_rotation(raw,rotation):
 values=[];svd=np.linalg.svd
 def tracked(*a,**kw):
  result=svd(*a,**kw);values.append(float(np.sum(result[1])));return result
 with patch('numpy.linalg.svd',new=tracked):rotated=_ortho_rotation(raw.T,method=rotation)
 converged=len(values)>1 and values[-2]!=0 and values[-1]<values[-2]*(1+1e-6)
 return rotated,len(values),converged


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
 parser.add_argument('--executable',type=Path);args=parser.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_factor_options_driver')
 shapes=[(48,7,0),(48,7,1),(50,9,3),(52,22,3),(12,18,3),(24,4,4),(20,1,-1)]
 modes=[('lapack',3)]+[('randomized',p) for p in [0,1,2,3,7]]
 settings=[(n,p,k,solver,power,seed,rot,False) for n,p,k in shapes for solver,power in modes
           for seed in [0,42,2026] for rot in ['none','varimax','quartimax']]
 settings +=[(40,p,3,solver,power,0,rot,True) for p in [6,18] for solver,power in [modes[0],modes[1],modes[4]]
            for rot in ['none','varimax','quartimax']]
 checks=[];fixtures=0;rotations=[];errors={};warnings_seen={};stop_certificates=[];trivial_rotation_certificates=[]
 for n,p,k,solver,power,seed,rotation,degenerate in settings:
  rng=np.random.RandomState(seed);latent=rng.normal(size=(n,3));loadings=rng.normal(size=(3,p))
  x=latent@loadings+.25*rng.normal(size=(n,p))+np.linspace(-.3,.5,p)
  query=rng.normal(size=(6,3))@loadings+.25*rng.normal(size=(6,p))+np.linspace(-.3,.5,p)
  limit=1 if degenerate else 2 if seed==2026 else 300;tol=0 if seed==2026 else 1e-3
  noise=np.full(p,100.) if degenerate else .35+.01*np.arange(p) if seed==42 else None
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always')
   base=FactorAnalysis(n_components=None if k==-1 else k,svd_method=solver,iterated_power=power,random_state=seed,
                       max_iter=limit,tol=tol,noise_variance_init=noise).fit(x)
  for w in caught:
   key=w.category.__name__+': '+str(w.message);warnings_seen[key]=warnings_seen.get(key,0)+1
  source_converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
  model=copy.deepcopy(base)
  if rotation!='none':model.components_=_ortho_rotation(base.components_.T,method=rotation)
  header=[n,p,len(query),k,int(solver=='randomized'),['none','varimax','quartimax'].index(rotation),power,seed,limit,tol,int(noise is not None)]
  payload=' '.join(map(str,header))+'\n'+''.join(' '.join(f'{v:.17g}' for v in a.ravel(order='F'))+'\n' for a in [x,query])
  if noise is not None:payload+=' '.join(f'{v:.17g}' for v in noise)+'\n'
  run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
  assert run.returncode==0,(n,p,k,solver,power,seed,rotation,degenerate,run.stdout,run.stderr)
  lines=iter(run.stdout.splitlines());names=['components','raw','mean','noise','loglike','fit_metadata','rotation_matrix',
             'rotation_metadata','transform','covariance','precision','score_samples','score','train_transform']
  actual={name:block(lines) for name in names};assert next(lines,None) is None
  context=dict(n=n,p=p,k=k,solver=solver,power=power,seed=seed,rotation=rotation,degenerate=degenerate)
  def compare(name,want,atol=2e-6,rtol=2e-6):
   got=actual[name];want=np.asarray(want);assert got.shape==want.shape,(context,name,got.shape,want.shape)
   np.testing.assert_allclose(got,want,atol=atol,rtol=rtol,err_msg=str((context,name)))
   errors[name]=max(errors.get(name,0.),float(np.max(abs(got-want),initial=0.)))
   checks.append(dict(**context,name=name,passed=True))
  compare('mean',base.mean_);compare('noise',base.noise_variance_);compare('loglike',np.asarray(base.loglike_))
  assert actual['fit_metadata'][0]==base.n_iter_,(context,'iteration count')
  if actual['fit_metadata'][1]==int(source_converged):
   compare('fit_metadata',[base.n_iter_,int(source_converged)],0,0)
  else:
   assert tol==0 and base.n_iter_==limit and len(base.loglike_)>=2,context
   native_delta=float(actual['loglike'][-1]-actual['loglike'][-2])
   source_delta=float(base.loglike_[-1]-base.loglike_[-2])
   bound=128*np.finfo(float).eps*max(n*p,abs(base.loglike_[-1]),abs(base.loglike_[-2]))
   assert abs(native_delta)<=bound and abs(source_delta)<=bound,(context,native_delta,source_delta,bound)
   assert bool(actual['fit_metadata'][1])==(native_delta<0) and source_converged==(source_delta<0)
   stop_certificates.append(dict(**context,native_delta=native_delta,source_delta=source_delta,bound=float(bound),
                                same_iteration=True,passed=True))
   checks.append(dict(**context,name='fit_metadata',passed=True))
  matched,perm,sign=align(actual['components'],model.components_);compare('components',matched)
  raw,_,_=align(actual['raw'],base.components_);compare('raw',raw)
  compare('transform',model.transform(query)[:,perm]*sign)
  compare('train_transform',model.transform(x)[:,perm]*sign)
  compare('covariance',model.get_covariance());compare('precision',model.get_precision())
  compare('score_samples',model.score_samples(query));compare('score',[model.score(query)])
  count=len(model.components_);r=actual['rotation_matrix']
  np.testing.assert_allclose(r.T@r,np.eye(count),rtol=1e-10,atol=1e-10)
  checks.append(dict(**context,name='orthogonality',passed=True))
  np.testing.assert_allclose(actual['components'].T@actual['components'],actual['raw'].T@actual['raw'],rtol=1e-10,atol=1e-10)
  checks.append(dict(**context,name='rotation_preserves_covariance',passed=True))
  if rotation!='none' and count:
   replay,iters,converged=replay_rotation(actual['raw'],rotation)
   aligned,_,_=align(actual['components'],replay)
   np.testing.assert_allclose(actual['components'],aligned,rtol=2e-7,atol=2e-7)
   checks.append(dict(**context,name='independent_rotation_replay',passed=True))
   if np.array_equal(actual['rotation_metadata'],[iters,int(converged)]):
    compare('rotation_metadata',[iters,int(converged)],0,0)
   else:
    # One feature and one factor has identically zero varimax criterion.
    # NumPy power rounding can create a nonzero gradient and stop in two steps;
    # an exact zero gradient reaches the helper limit. Neither changes the model.
    assert p==count==1 and rotation=='varimax',context
    assert actual['rotation_metadata'][0] in [2,100] and iters in [2,100]
    trivial_rotation_certificates.append(dict(**context,native_iterations=int(actual['rotation_metadata'][0]),
        source_iterations=iters,criterion_identically_zero=True,orthogonal_and_components_checked=True,passed=True))
    checks.append(dict(**context,name='rotation_metadata',passed=True))
   initial=criterion(actual['raw'],rotation);final=criterion(actual['components'],rotation)
   source_final=criterion(model.components_,rotation)
   assert final>=initial-1e-8*max(1,abs(initial)),(context,'rotation objective',initial,final)
   np.testing.assert_allclose(final,source_final,rtol=2e-6,atol=2e-6)
   rotations.append(dict(**context,initial=initial,native=final,source=source_final,iterations=iters,converged=bool(converged),passed=True))
   checks.append(dict(**context,name='rotation_criterion',passed=True))
  else:
   compare('rotation_matrix',np.eye(count),0,0);compare('rotation_metadata',[0,1],0,0)
  fixtures+=1
  if fixtures%36==0:print(f'{fixtures} factor options fixtures passed',flush=True)
 results=ROOT/'validation/results';results.mkdir(exist_ok=True);source=Path(sklearn.__file__).parent
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 report=dict(profile=args.profile,fixtures=fixtures,checks=checks,rotations=rotations,max_absolute_errors=errors,
    source_warnings=warnings_seen,roundoff_stop_certificates=stop_certificates,trivial_rotation_certificates=trivial_rotation_certificates,sklearn=sklearn.__version__,numpy=np.__version__,native_sha256=digest(exe),
    validator_sha256=digest(Path(__file__)),source_sha256={p:digest(source/p) for p in ['decomposition/_factor_analysis.py','utils/extmath.py']})
 (results/f'factor_options_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejected=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True)
  assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and message in run.stdout,(case,run.stdout,run.stderr)
  rejected.append(dict(name=case,expected=message,passed=True))
 (results/f'factor_options_invalid_{args.profile}.json').write_text(json.dumps(dict(profile=args.profile,checks=rejected,
    native_sha256=report['native_sha256'],validator_sha256=report['validator_sha256']),indent=2,allow_nan=False)+'\n')
 print(f'{len(checks)} factor options groups / {fixtures} fixtures, {len(rotations)} rotation comparisons, {len(rejected)} rejections passed')


if __name__=='__main__':main()
