"""GP sampling laws and seeded Gaussian structure, independent of SVD orientation."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,WhiteKernel,RBF
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block
from compare_gpr_advanced import make_kernel as base_kernel

REJECTIONS={'negative_count':'sample count must be nonnegative','negative_seed':'sampling random_state must be nonnegative',
 'multiple_targets':'multiple targets require sample_y_multi','capacity':'sample output exceeds integer capacity',
 'invalid_targets':'n_targets must be nonnegative','nan_query':'X contains NaN or infinity',
 'wrong_features':'kernel feature count mismatch','nonfinite_moments':'sampling moments must be finite'}


def kernel_for(code,p):
 if code<9:return base_kernel(code,p)
 return {9:lambda:ConstantKernel(1.7,'fixed'),10:lambda:WhiteKernel(.2,'fixed'),
         11:lambda:ConstantKernel(0.,'fixed'),12:lambda:RBF(.8,'fixed'),
         13:lambda:ConstantKernel(1e-240,'fixed'),14:lambda:ConstantKernel(1e240,'fixed')}[code]()


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
 parser.add_argument('--executable',type=Path);args=parser.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_gp_sampling_driver')
 checks=[];fixtures=0;moment_fixtures=0;rank_counts={};max_errors={};source_warnings={};source_shape_bugs=[]
 settings=[(q,c,s,config,[0,1,7][i]) for q in [1,3] for c in range(13) for i,s in enumerate([0,42,2026]) for config in range(6)]
 settings +=[(q,c,s,config,20000) for q in [1,3] for c in [0,1,3,4,7,9,10,11,12] for s in [0,42,2026] for config in [0,3]]
 settings +=[(q,c,s,0,[0,1,7][i]) for q in [1,3] for c in [13,14] for i,s in enumerate([0,42,2026])]
 for targets,code,seed,config,draws in settings:
  n=9;p=1 if code in [3,8] else 2;m=1 if code>=13 else 5 if code>=9 else 4
  rng=np.random.RandomState(123);x=np.empty((n,p));x[:,0]=np.linspace(-1.4,1.4,n)
  if p>1:x[:,1]=np.cos(x[:,0])+.1*rng.normal(size=n)
  query=np.empty((m,p));query[:,0]=np.linspace(-1.2,1.2,m)
  if p>1:query[:,1]=np.cos(query[:,0])+.1*rng.normal(size=m)
  if code==12:query[:]=.23
  y=np.column_stack([np.sin(1.3*x[:,0])+2,4*np.cos(.7*x[:,0])-3,np.full(n,5.)])[:,:targets]
  kernel=kernel_for(code,p);alpha=.02+.01*np.arange(1,n+1) if config==5 else .04
  model=GaussianProcessRegressor(kernel=kernel,alpha=alpha,optimizer=None,normalize_y=config>=3,n_targets=targets,random_state=99)
  if config>0:model.fit(x,y[:,0] if config==1 and targets==1 else y)
  if config==4:model.kernel=ConstantKernel(0.,'fixed')
  mean,cov=model.predict(query,return_cov=True)
  mean=mean.reshape(m,targets);cov=cov.reshape(m,m,targets)
  header=[code,n,p,m,targets,draws,seed,config]
  payload=' '.join(map(str,header))+'\n'+''.join(' '.join(f'{v:.17g}' for v in a.ravel(order='F'))+'\n' for a in [x,query,y])
  run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
  assert run.returncode==0,(targets,code,seed,config,draws,run.stdout,run.stderr)
  lines=iter(run.stdout.splitlines());actual_mean=block(lines);actual_cov=block(lines)
  if draws<=1000:samples=block(lines)
  else:empirical_mean=block(lines);empirical_cov=block(lines)
  agreement=block(lines);after_mean=block(lines);after_cov=block(lines);assert next(lines,None) is None
  context=dict(targets=targets,code=code,seed=seed,config=config,draws=draws)
  def passed(name):checks.append(dict(**context,name=name,passed=True))
  def close(name,got,want,atol=3e-8,rtol=3e-8):
   assert got.shape==want.shape,(context,name,got.shape,want.shape)
   np.testing.assert_allclose(got,want,atol=atol,rtol=rtol,err_msg=str((context,name)))
   max_errors[name]=max(max_errors.get(name,0.),float(np.max(abs(got-want),initial=0.)));passed(name)
  close('predictive_mean',actual_mean,mean);close('predictive_covariance',actual_cov,cov)
  close('unchanged_mean',after_mean,actual_mean,0,0);close('unchanged_covariance',after_cov,actual_cov,0,0)
  if targets==1:close('single_target_view',agreement,np.array([0.]),0,0)
  if draws<=1000:
   assert samples.shape==(m,targets,draws),(context,samples.shape);assert np.isfinite(samples).all();passed('sample_shape')
   if m==1 and config==0:
    try:model.sample_y(query,draws,random_state=seed)
    except (IndexError,ValueError) as exc:
     assert (targets==1 and isinstance(exc,IndexError) and str(exc)=='tuple index out of range') or (
       targets>1 and isinstance(exc,ValueError) and str(exc)=='cov must be 2 dimensional and square'),str(exc)
     source_shape_bugs.append(dict(**context,error_type=type(exc).__name__,message=str(exc)))
    else:raise AssertionError('Pinned sklearn singleton-prior shape bug changed')
    rng_source=np.random.RandomState(seed)
    source_samples=np.stack([rng_source.multivariate_normal(mean[:,t],cov[:,:,t],draws).T for t in range(targets)],axis=1)
   else:source_samples=model.sample_y(query,draws,random_state=seed).reshape(m,targets,draws)
   assert source_samples.shape==samples.shape;passed('source_sample_shape')
   if m==1:close('scalar_exact_source_draws',samples,source_samples,0,1e-12)
   # Source-seeded normal stream is target-major, then draw-major, then query-major.
   rng=np.random.RandomState(seed)
   for t in range(targets):
    z=rng.normal(size=(draws,m)).T
    _,singular,vh=np.linalg.svd(cov[:,:,t]);rank=int(np.sum(singular>1e-9*singular.max(initial=0.)))
    rank_counts[str(rank)]=rank_counts.get(str(rank),0)+1
    residual=samples[:,t,:]-mean[:,t,None]
    if rank:
     white=(vh[:rank]@residual)/np.sqrt(singular[:rank,None])
     gram=white.T@white
    else:gram=np.zeros((draws,draws))
    # Whitened draw Gram matrices are invariant to SVD signs/rotations, including
    # repeated eigenvalues. Rank-r covariance consumes all m normals, using the
    # first r coordinates; this also checks cached-normal continuity across targets.
    close(f'seeded_gaussian_gram_{t}',gram,z[:rank].T@z[:rank],1e-5,1e-5)
    null=vh[rank:]@residual
    bound=10*np.sqrt(np.sum(singular[rank:])+np.linalg.norm(actual_cov[:,:,t]-cov[:,:,t],2)
                     +np.finfo(float).eps*np.linalg.norm(cov[:,:,t],2))*max(1.,np.linalg.norm(z))+5e-8
    assert np.max(abs(null),initial=0.)<=bound,(context,t,'nullspace',np.max(abs(null)),bound)
    passed(f'covariance_range_{t}')
   # Driver independently repeats the public call and requires exact native equality.
   passed('repeatable_native_calls')
  else:
   moment_fixtures+=1
   expected_mean=mean.ravel(order='F');expected_cov=np.zeros((m*targets,m*targets))
   for t in range(targets):expected_cov[t*m:(t+1)*m,t*m:(t+1)*m]=cov[:,:,t]
   diagonal=np.maximum(np.diag(expected_cov),0)
   mean_bound=8*np.sqrt(diagonal/draws)+5e-9
   covariance_bound=8*np.sqrt((expected_cov**2+np.outer(diagonal,diagonal))/(draws-1))+5e-9
   assert np.all(abs(empirical_mean-expected_mean)<=mean_bound),(context,'native sample mean')
   assert np.all(abs(empirical_cov-expected_cov)<=covariance_bound),(context,'native sample covariance')
   passed('native_gaussian_mean');passed('native_gaussian_covariance')
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always');source_samples=model.sample_y(query,draws,random_state=seed).reshape(m,targets,draws)
   for w in caught:
    key=w.category.__name__+': '+str(w.message);source_warnings[key]=source_warnings.get(key,0)+1
   flat=source_samples.reshape(m*targets,draws,order='F')
   assert np.all(abs(flat.mean(axis=1)-expected_mean)<=mean_bound),(context,'source sample mean')
   assert np.all(abs(np.cov(flat)-expected_cov)<=covariance_bound),(context,'source sample covariance')
   passed('source_gaussian_mean');passed('source_gaussian_covariance')
  fixtures+=1
  if fixtures%36==0:print(f'{fixtures} sampling fixtures passed',flush=True)
 results=ROOT/'validation/results';results.mkdir(exist_ok=True);source=Path(sklearn.__file__).parent
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 report=dict(profile=args.profile,fixtures=fixtures,moment_fixtures=moment_fixtures,checks=checks,
   rank_counts=rank_counts,max_absolute_errors=max_errors,source_warnings=source_warnings,source_shape_bugs=source_shape_bugs,
   orientation_contract='Native Jacobi SVD; seeded paths may differ from NumPy LAPACK; Gaussian law and normal stream verified.',
   native_sha256=digest(exe),validator_sha256=digest(Path(__file__)),sklearn=sklearn.__version__,numpy=np.__version__,
   source_sha256={'gaussian_process/_gpr.py':digest(source/'gaussian_process/_gpr.py')})
 (results/f'gp_sampling_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejected=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True)
  assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and message in run.stdout,(case,run.stdout,run.stderr)
  rejected.append(dict(name=case,expected=message,passed=True))
 (results/f'gp_sampling_invalid_{args.profile}.json').write_text(json.dumps(dict(profile=args.profile,checks=rejected,
   native_sha256=report['native_sha256'],validator_sha256=report['validator_sha256']),indent=2,allow_nan=False)+'\n')
 print(f'{len(checks)} GP sampling groups / {fixtures} fixtures, {moment_fixtures} moment fixtures, {len(rejected)} rejections passed')


if __name__=='__main__':main()
