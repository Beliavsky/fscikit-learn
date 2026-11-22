"""Multi-target/noise/likelihood GPR and bounded L-BFGS-B against sklearn."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import sklearn
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Matern,RationalQuadratic,ExpSineSquared,DotProduct,ConstantKernel,WhiteKernel
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block

REJECTIONS={
 'alpha_empty':'alpha_values sample count mismatch','alpha_count':'alpha_values sample count mismatch',
 'alpha_negative':'alpha_values must be nonnegative','alpha_nan':'alpha_values must be finite',
 'optimizer':'unsupported optimizer','restart':'restart count must be nonnegative','seed':'random_state must be nonnegative',
 'iterations':'iteration and evaluation limits must be positive','evaluations':'iteration and evaluation limits must be positive',
 'tolerance':'tolerances must be nonnegative','nan_tolerance':'tolerances must be finite',
 'target_count':'target count differs from n_targets','negative_targets':'n_targets must be nonnegative',
 'target_nan':'X contains NaN or infinity','bounds_shape':'bound shape mismatch','bounds_order':'lower bounds exceed upper bounds',
 'bounds_nan':'bounds must not contain NaN','bounds_infinity':'bound endpoints have invalid infinity signs',
 'restart_infinite':'restarts require finite bounds','unfitted_likelihood':'Gaussian process is not fitted',
 'gradient_without_theta':'likelihood gradient requires theta','predict_vector_multi':'multiple targets require predict_multi',
 'std_vector_multi':'multiple targets require predict_std_multi','cov_vector_multi':'multiple targets require predict_cov_multi',
 'score_weights':'score weight count mismatch','score_zero_weights':'weights must have positive sum',
 'score_negative_weights':'weights must be nonnegative','score_nan_weights':'weights must be finite',
}


def make_kernel(code,p):
 lengths=np.linspace(.6,1.,p) if p>1 else np.array([.6])
 if code==0:return 1.2*RBF(.8)+WhiteKernel(.07)
 if code==1:return Matern(lengths,nu=1.5)+WhiteKernel(.05)
 if code==2:return RationalQuadratic(.9,.7)+WhiteKernel(.04)
 if code==3:return ExpSineSquared(.9,1.8)+WhiteKernel(.04)
 if code==4:return DotProduct(.6)+WhiteKernel(.07)
 if code==5:return ConstantKernel(1.2,'fixed')*RBF(.8,'fixed')+WhiteKernel(.04,'fixed')
 if code==6:return RBF(lengths)+WhiteKernel(.05)
 if code==7:return Matern(lengths,nu=.7)+WhiteKernel(.05)
 if code==8:return RBF(.8)*ExpSineSquared(.9,1.8)+WhiteKernel(.04)
 raise AssertionError(code)


def apply_bounds(kernel,bounds):
 pos=0
 for h in kernel.hyperparameters:
  if h.fixed:continue
  values=np.exp(bounds[pos:pos+h.n_elements]);pos+=h.n_elements
  kernel.set_params(**{h.name+'_bounds':tuple(values[0]) if h.n_elements==1 else values})
 assert pos==len(bounds)
 return kernel


def tensor3(a,m,q):return np.asarray(a).reshape(m,m,q)
def matrix2(a,m,q):return np.asarray(a).reshape(m,q)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
 parser.add_argument('--executable',type=Path);args=parser.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_gpr_advanced_driver')
 checks=[];fixtures=0;errors={};optimizations=[];warning_counts={}
 for mode in [0,1,2,3]:
  for code in range(9) if mode!=3 else [0]:
   for seed in [0,42,2026]:
    for config in range(6 if mode in [1,2] else 2):
     n=13;p=1 if code in [3,8] else 2;m=5;outputs=1 if mode==1 else 3
     if mode==2 and config==4:outputs=1
     rng=np.random.RandomState(seed+config*13);x=rng.uniform(-1,1,(n,p));q=rng.uniform(-1.1,1.1,(m,p))
     y=np.column_stack([np.sin(2.3*x[:,0])+.05*rng.normal(size=n),np.cos(1.7*x[:,0])+.4, .4*x[:,0]+.1*rng.normal(size=n)])[:,:outputs]
     if config==1:y[:,-1]=3.
     if config==3 and mode==2:y[:,-1]=1.+np.arange(n)*1e-16
     weights=rng.uniform(.1,2,m);weights[0]=0
     alpha=.03;na=-1
     if config==2:alpha=rng.uniform(.01,.08,n);na=n
     if config==4:alpha=np.array([.04]);na=1
     kernel=make_kernel(code,p);bounds=None
     opt=mode in [1,2] and config in [3,5];restarts=1 if opt and config==5 else 0
     if opt:
      bounds=np.tile(np.log([.08,4.]),(kernel.n_dims,1))
      pos=0
      for h in kernel.hyperparameters:
       if h.fixed:continue
       if 'noise_level' in h.name:bounds[pos:pos+h.n_elements]=np.log([.0001,.3])
       pos+=h.n_elements
      kernel=apply_bounds(kernel,bounds)
     runs=[]
     def optimizer(objective,initial,bounds):
      result=minimize(objective,initial,jac=True,bounds=bounds,method='L-BFGS-B',
                      options=dict(maxiter=15000,maxfun=15000,gtol=1e-5,ftol=2.220446049250313e-9,maxls=20,maxcor=10))
      runs.append(dict(point=result.x.copy(),value=float(result.fun)))
      return result.x,result.fun
     source_y=y[:,0] if mode==1 else y
     model=GaussianProcessRegressor(kernel=kernel,alpha=alpha,normalize_y=bool(config%2),
                                    optimizer=optimizer if opt else None,n_restarts_optimizer=restarts,random_state=seed,
                                    n_targets=outputs if mode==0 else None)
     with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      if mode!=0:model.fit(x,source_y)
     for w in caught:
      key=w.category.__name__+': '+str(w.message);warning_counts[key]=warning_counts.get(key,0)+1
     header=[mode,code,n,p,m,outputs,config,seed,int(opt),restarts,na,-1 if bounds is None else len(bounds)]
     payload=' '.join(map(str,header))+'\n'+''.join(' '.join(f'{v:.17g}' for v in np.asarray(a).ravel(order='F'))+'\n' for a in [x,q,y,weights])
     if bounds is not None:payload+=' '.join(f'{v:.17g}' for v in bounds.ravel(order='F'))+'\n'
     if na>=0:payload+=' '.join(f'{v:.17g}' for v in alpha)+'\n'
     run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
     assert run.returncode==0,(mode,code,seed,config,run.stdout,run.stderr)
     lines=iter(run.stdout.splitlines())
     if mode==3:
      model.kernel_=RBF(.8)**(-1);want,grad=model.log_marginal_likelihood(model.kernel_.theta,eval_gradient=True)
      assert np.isneginf(block(lines)[0]) and np.isneginf(want)
      np.testing.assert_array_equal(block(lines),grad);assert next(lines,None) is None
      checks.append(dict(mode=mode,code=code,seed=seed,config=config,name='nonpositive_likelihood',passed=True));fixtures+=1;continue
     actual={}
     names=['mean','std','cov'] if mode==0 else ['mean','std','cov','cached_lml','alpha','cholesky','targets','mean_stats','std_stats',
          'theta','bounds','likelihood','gradient','shift_likelihood','shift_gradient','clone_snapshot','mutate_likelihood','mutate_gradient',
          'mutated_theta','score']+(['vector_mean','vector_std','vector_cov'] if mode==1 else [])+['metadata','run_points','run_values']
     for name in names:actual[name]=block(lines)
     assert next(lines,None) is None
     if opt and kernel.n_dims:
      native_theta=actual['theta'];source_value=model.log_marginal_likelihood_value_
      native_value=float(actual['cached_lml'][0]);gap=abs(native_value-source_value)
      assert gap<=3e-6*max(1,abs(source_value)),(mode,code,seed,config,'optimization likelihood',gap,native_theta,model.kernel_.theta)
      np.testing.assert_allclose(model.predict(q),actual['mean'].squeeze() if outputs==1 else actual['mean'],rtol=3e-3,atol=3e-3,
                                 err_msg=f'optimized predictions {mode} {code} {seed} {config}')
      assert np.all(native_theta>=bounds[:,0]-1e-12) and np.all(native_theta<=bounds[:,1]+1e-12)
      optimizations.append(dict(mode=mode,code=code,seed=seed,config=config,native_lml=native_value,source_lml=float(source_value),
                                native_theta=native_theta.tolist(),source_theta=model.kernel_.theta.tolist(),passed=True))
      # Validate all fitted state at the native optimum independently of optimizer path rounding.
      model=GaussianProcessRegressor(kernel=kernel.clone_with_theta(native_theta),alpha=alpha,normalize_y=bool(config%2),optimizer=None).fit(x,source_y)
     mean,std=model.predict(q,return_std=True);cov=model.predict(q,return_cov=True)[1]
     expected=dict(mean=matrix2(mean,m,outputs),std=matrix2(std,m,outputs),cov=tensor3(cov,m,outputs))
     if mode!=0:
      theta=model.kernel_.theta;value,gradient=model.log_marginal_likelihood(theta,eval_gradient=True)
      shifted,shift_gradient=model.log_marginal_likelihood(theta+.07,eval_gradient=True)
      expected.update(cached_lml=np.array([model.log_marginal_likelihood_value_]),alpha=matrix2(model.alpha_,n,outputs),cholesky=model.L_,
                      targets=matrix2(model.y_train_,n,outputs),mean_stats=np.atleast_1d(model._y_train_mean),std_stats=np.atleast_1d(model._y_train_std),
                      theta=theta,bounds=model.kernel_.bounds.reshape(-1,2),likelihood=np.array([value]),gradient=gradient,
                      shift_likelihood=np.array([shifted]),shift_gradient=shift_gradient,clone_snapshot=theta.copy(),
                      mutate_likelihood=np.array([shifted]),mutate_gradient=shift_gradient,mutated_theta=theta+.07,
                      score=np.array([model.score(q,y[:m,0] if mode==1 else y[:m],sample_weight=weights)]))
      if mode==1:expected.update(vector_mean=mean,vector_std=std,vector_cov=cov)
      expected['metadata']=np.array([outputs,p,len(runs)])
      assert actual['run_points'].shape==(len(runs),kernel.n_dims)
      assert actual['run_values'].shape==(len(runs),)
      for idx,point in enumerate(actual['run_points']):
       f=-model.log_marginal_likelihood(point);np.testing.assert_allclose(actual['run_values'][idx],f,rtol=3e-9,atol=3e-9)
       assert abs(f-runs[idx]['value'])<=3e-6*max(1,abs(f)),(mode,code,seed,config,idx,'restart objective',f,runs[idx]['value'])
      checks.append(dict(mode=mode,code=code,seed=seed,config=config,name='optimization_runs',passed=True))
      if kernel.n_dims:
       central=np.empty(kernel.n_dims);step=1e-5
       for j in range(kernel.n_dims):
        up=theta.copy();down=theta.copy();up[j]+=step;down[j]-=step
        central[j]=(model.log_marginal_likelihood(up)-model.log_marginal_likelihood(down))/(2*step)
       np.testing.assert_allclose(actual['gradient'],central,rtol=3e-5,atol=3e-6,err_msg=f'central LML gradient {mode} {code} {seed} {config}')
       checks.append(dict(mode=mode,code=code,seed=seed,config=config,name='central_gradient',passed=True))
     for name,want in expected.items():
      got=actual[name];assert got.shape==want.shape,(mode,code,seed,config,name,got.shape,want.shape)
      tol=8e-4 if code==7 and 'gradient' in name else 3e-8
      if name=='cached_lml' and opt:tol=3e-6
      np.testing.assert_allclose(got,want,rtol=tol,atol=tol,equal_nan=True,err_msg=f'{mode} {code} {seed} {config} {name}')
      finite=np.isfinite(got)&np.isfinite(want);errors[name]=max(errors.get(name,0.),float(np.max(abs(got[finite]-want[finite]),initial=0)))
      checks.append(dict(mode=mode,code=code,seed=seed,config=config,name=name,passed=True))
     fixtures+=1
   print(f'mode {mode} kernel {code} passed',flush=True)
 results=ROOT/'validation/results';results.mkdir(exist_ok=True);source=Path(sklearn.__file__).parent
 report=dict(profile=args.profile,fixtures=fixtures,checks=checks,max_absolute_errors=errors,optimizations=optimizations,source_warnings=warning_counts,
             sklearn=sklearn.__version__,numpy=np.__version__,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
             validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             source_sha256={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in ['gaussian_process/_gpr.py','gaussian_process/kernels.py']})
 (results/f'gpr_advanced_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejected=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True)
  assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and message in run.stdout,(case,run.stdout,run.stderr)
  rejected.append(dict(name=case,expected=message,passed=True))
 (results/f'gpr_advanced_invalid_{args.profile}.json').write_text(json.dumps(dict(profile=args.profile,checks=rejected,native_sha256=report['native_sha256'],validator_sha256=report['validator_sha256']),indent=2,allow_nan=False)+'\n')
 print(f'{len(checks)} advanced GPR groups / {fixtures} fixtures, {len(optimizations)} optimization comparisons, {len(rejected)} rejections passed')


if __name__=='__main__':main()
