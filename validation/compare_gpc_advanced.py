"""Laplace GPC derivatives, learned OVR kernels, warm caches and theta semantics."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import sklearn
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process._gpc import _BinaryGaussianProcessClassifierLaplace as Binary
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block
from compare_gpr_advanced import make_kernel,apply_bounds

REJECTIONS={
 'optimizer':'unsupported optimizer','restart':'restart count must be nonnegative','seed':'random_state must be nonnegative',
 'iterations':'iteration and evaluation limits must be positive','evaluations':'iteration and evaluation limits must be positive',
 'tolerance':'tolerances must be nonnegative','nan_tolerance':'tolerances must be finite',
 'restart_infinite':'restarts require finite bounds','unfitted_theta':'GPC is not fitted',
 'cached_gradient':'likelihood gradient requires theta','theta_shape':'theta must match one kernel',
 'multiclass_gradient':'multiclass likelihood gradients are not supported','multiclass_shape':'theta must match one kernel',
}


def children(model):return model.base_estimator_.estimators_ if len(model.classes_)>2 else [model.base_estimator_]
def cache(model):return np.column_stack([e.f_cached for e in children(model)])
def thetas(model):return np.column_stack([e.kernel_.theta for e in children(model)])


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
 parser.add_argument('--executable',type=Path);args=parser.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_gpc_advanced_driver')
 checks=[];fixtures=0;errors={};optimizations=[];warning_counts={};restart_comparisons=[]
 for classes in [2,3]:
  for code in range(9):
   for seed in [0,42,2026]:
    for config in range(6):
     n=18;p=1 if code in [3,8] else 2;m=5;warm=config in [1,3,5];opt=config in [2,3]
     restarts=int(config==3);limit=1 if config==4 else 3 if config==5 else 100
     rng=np.random.RandomState(seed);x=np.empty((n,p));x[:,0]=np.linspace(-1.5,1.5,n)
     if p>1:x[:,1]=np.cos(1.8*x[:,0])+.08*rng.normal(size=n)
     q=np.empty((m,p));q[:,0]=np.array([-1.3,-.7,.1,.6,1.3])
     if p>1:q[:,1]=np.cos(1.8*q[:,0])+.08*rng.normal(size=m)
     labels=np.where(x[:,0]>0,9,-4) if classes==2 else np.repeat([-4,9,17],6)
     kernel=make_kernel(code,p);bounds=None
     if opt:
      bounds=np.tile(np.log([.08,4.]),(kernel.n_dims,1));pos=0
      for h in kernel.hyperparameters:
       if h.fixed:continue
       if 'noise_level' in h.name:bounds[pos:pos+h.n_elements]=np.log([.0001,.3])
       pos+=h.n_elements
      kernel=apply_bounds(kernel,bounds)
     runs=[]
     def optimizer(objective,initial,bounds):
      raw_initial=initial.copy()
      result=minimize(objective,initial,jac=True,bounds=bounds,method='L-BFGS-B',
                      options=dict(maxiter=15000,maxfun=15000,gtol=1e-5,ftol=2.220446049250313e-9,maxls=20,maxcor=10))
      runs.append(dict(initial=raw_initial,point=result.x.copy(),value=float(result.fun),converged=bool(result.success),message=str(result.message)));return result.x,result.fun
     with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      model=GaussianProcessClassifier(kernel=kernel,optimizer=optimizer if opt else None,warm_start=warm,
                                      n_restarts_optimizer=restarts,max_iter_predict=limit,random_state=seed).fit(x,labels)
     for w in caught:
      key=w.category.__name__+': '+str(w.message);warning_counts[key]=warning_counts.get(key,0)+1
     header=[code,n,p,m,config,seed,int(opt),restarts,int(warm),limit,-1 if bounds is None else len(bounds)]
     payload=' '.join(map(str,header))+'\n'+''.join(' '.join(f'{v:.17g}' for v in a.ravel(order='F'))+'\n' for a in [x,q,labels])
     if bounds is not None:payload+=' '.join(f'{v:.17g}' for v in bounds.ravel(order='F'))+'\n'
     run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
     assert run.returncode==0,(classes,code,seed,config,run.stdout,run.stderr)
     lines=iter(run.stdout.splitlines());actual={}
     names=['proba','labels','cached_lml','channel_lml','pi','w','cholesky','alpha','fit_mode','fit_cache','theta','fit_initial_cache',
            'run_points','run_values','run_converged','run_eval_cache','run_initial','metadata','likelihood']
     if classes==2:names+=['gradient','cache_after_theta','shift_likelihood','shift_gradient','cache_after_shift']
     else:names+=['cache_after_theta','shift_likelihood','cache_after_shift']
     heads=1 if classes==2 else classes
     names +=[f'clone_theta_{i}' for i in range(heads)]+['clone_proba','mutate_likelihood','cache_after_mutate']
     names +=[f'mutated_theta_{i}' for i in range(heads)]+['mutated_proba','cached_after_mutate']
     for name in names:actual[name]=block(lines)
     assert next(lines,None) is None
     source_children=children(model);source_lml=np.array([e.log_marginal_likelihood_value_ for e in source_children])
     replay_values=None
     if opt and kernel.n_dims:
      np.testing.assert_allclose(actual['channel_lml'],source_lml,rtol=1e-5,atol=1e-5,
                                 err_msg=f'optimized likelihood {classes} {code} {seed} {config}')
      source_probability=model.predict_proba(q)
      np.testing.assert_allclose(actual['proba'],source_probability,rtol=1e-4,atol=1e-4,
                                 err_msg=f'learned classifier predictions {classes} {code} {seed} {config}')
      assert np.all(actual['theta']>=bounds[:,0,None]-1e-12) and np.all(actual['theta']<=bounds[:,1,None]+1e-12)
      optimizations.append(dict(classes=classes,code=code,seed=seed,config=config,native_lml=actual['channel_lml'].tolist(),
                                source_lml=source_lml.tolist(),native_theta=actual['theta'].tolist(),source_theta=thetas(model).tolist(),probability_gap=float(np.max(abs(actual['proba']-source_probability))),passed=True))
      # Independent fixed fits at each native optimum validate the learned model state.
      fixed=[];replay_values=[];native_cold=[];source_cold=[]
      for i in range(heads):
       target=labels if classes==2 else (labels==model.classes_[i]).astype(int)
       fixed.append(Binary(kernel=kernel.clone_with_theta(actual['theta'][:,i]),optimizer=None,warm_start=warm,
                           max_iter_predict=limit,random_state=seed).fit(x,target))
       native_check=Binary(kernel=kernel.clone_with_theta(actual['theta'][:,i]),optimizer=None,warm_start=False,
                           max_iter_predict=limit).fit(x,target)
       source_check=Binary(kernel=source_children[i].kernel_,optimizer=None,warm_start=False,
                           max_iter_predict=limit).fit(x,target)
       native_cold.append(native_check.log_marginal_likelihood_value_);source_cold.append(source_check.log_marginal_likelihood_value_)
       best=int(np.argmin(actual['run_values'][:,i]));point=actual['run_points'][best,:,i]
       np.testing.assert_allclose(point,actual['theta'][:,i],rtol=0,atol=1e-14)
       replay=Binary(kernel=kernel.clone_with_theta(point),optimizer=None,warm_start=False,max_iter_predict=limit).fit(x,target)
       if warm:
        replay.warm_start=True;replay.f_cached=actual['run_eval_cache'][:,best,i].copy()
       replay_values.append(replay.log_marginal_likelihood(point))
       fixed[-1].log_marginal_likelihood_value_=replay_values[-1]
      native_cold=np.array(native_cold);source_cold=np.array(source_cold)
      np.testing.assert_allclose(native_cold,source_cold,rtol=1e-5,atol=1e-5,
                                 err_msg=f'independent model likelihood {classes} {code} {seed} {config}')
      optimizations[-1].update(native_cold_lml=native_cold.tolist(),source_cold_lml=source_cold.tolist())
      if warm:
       for i,e in enumerate(fixed):
        e.f_cached=actual['fit_initial_cache'][:,i].copy()
        _,temporaries=e._posterior_mode(e.kernel_(x),return_temporaries=True)
        e.pi_,e.W_sr_,e.L_=temporaries[:3]
       checks.append(dict(classes=classes,code=code,seed=seed,config=config,name='warm_final_transition',passed=True))
      if classes==2:model.base_estimator_=fixed[0]
      else:model.base_estimator_.estimators_=fixed
      model.log_marginal_likelihood_value_=np.mean([e.log_marginal_likelihood_value_ for e in fixed])
     es=children(model);theta=thetas(model);initial_cache=cache(model)
     expected=dict(proba=model.predict_proba(q),labels=model.predict(q),cached_lml=np.array([model.log_marginal_likelihood_value_]),
                   channel_lml=np.array([e.log_marginal_likelihood_value_ for e in es]),pi=np.column_stack([e.pi_ for e in es]),
                   w=np.column_stack([e.W_sr_ for e in es]),cholesky=np.stack([e.L_ for e in es],axis=2),
                   alpha=np.column_stack([e.y_train_-e.pi_ for e in es]),fit_mode=initial_cache,fit_cache=initial_cache,
                   theta=theta,metadata=np.array([len(runs)//heads,classes,heads]))
     if classes==2:
      value,gradient=model.log_marginal_likelihood(theta[:,0],eval_gradient=True)
      expected.update(likelihood=np.array([value]),gradient=gradient,cache_after_theta=cache(model))
      shifted,shift_gradient=model.log_marginal_likelihood(theta[:,0]+.04,eval_gradient=True)
      expected.update(shift_likelihood=np.array([shifted]),shift_gradient=shift_gradient,cache_after_shift=cache(model))
     else:
      value=model.log_marginal_likelihood(theta[:,0]);expected.update(likelihood=np.array([value]),cache_after_theta=cache(model))
      shifted=model.log_marginal_likelihood(theta.ravel(order='F')+.04)
      expected.update(shift_likelihood=np.array([shifted]),cache_after_shift=cache(model))
     for i,e in enumerate(es):expected[f'clone_theta_{i}']=e.kernel_.theta.copy()
     expected['clone_proba']=model.predict_proba(q)
     value=model.log_marginal_likelihood(theta.ravel(order='F')+.02,clone_kernel=False)
     expected.update(mutate_likelihood=np.array([value]),cache_after_mutate=cache(model))
     for i,e in enumerate(es):expected[f'mutated_theta_{i}']=e.kernel_.theta.copy()
     expected.update(mutated_proba=model.predict_proba(q),cached_after_mutate=np.array([model.log_marginal_likelihood()]))
     for name,want in expected.items():
      got=actual[name];assert got.shape==want.shape,(classes,code,seed,config,name,got.shape,want.shape)
      if name=='labels':np.testing.assert_array_equal(got,want)
      else:
       tol=1e-3 if code==7 and 'gradient' in name else 2e-6 if opt else 3e-8
       if ('lml' in name or name=='cached_after_mutate') and opt:tol=1e-5
       np.testing.assert_allclose(got,want,rtol=tol,atol=tol,equal_nan=False,err_msg=f'{classes} {code} {seed} {config} {name}')
       errors[name]=max(errors.get(name,0.),float(np.max(abs(got-want),initial=0)))
      checks.append(dict(classes=classes,code=code,seed=seed,config=config,name=name,passed=True))
     np.testing.assert_array_equal(actual['cached_after_mutate'],actual['cached_lml'])
     nruns=len(runs)//heads
     assert actual['run_points'].shape==(nruns,kernel.n_dims,heads) and actual['run_values'].shape==(nruns,heads)
     for h in range(heads):
      for j in range(nruns):
       f=actual['run_values'][j,h];want=runs[h*nruns+j]['value']
       np.testing.assert_allclose(actual['run_initial'][j,:,h],runs[h*nruns+j]['initial'],rtol=1e-14,atol=1e-14)
       target=labels if classes==2 else (labels==model.classes_[h]).astype(int)
       point=actual['run_points'][j,:,h]
       cold=Binary(kernel=kernel.clone_with_theta(point),optimizer=None,warm_start=False,max_iter_predict=100).fit(x,target)
       cold_value=-cold.log_marginal_likelihood(point)
       if warm:
        cold.warm_start=True;cold.f_cached=actual['run_eval_cache'][:,j,h].copy()
       independent=-cold.log_marginal_likelihood(point)
       assert abs(f-independent)<=3e-8*max(1,abs(independent)),(classes,code,seed,config,h,j,'independent restart objective',f,independent)
       assert abs(f-want)<=1e-5*max(1,abs(want)),(classes,code,seed,config,h,j,'cold restart objective',f,want)
       restart_comparisons.append(dict(classes=classes,code=code,seed=seed,config=config,head=h,run=j,warm=warm,
                                       native_value=float(f),source_value=float(want),independent_value=float(independent),cold_value=float(cold_value),
                                       native_converged=bool(actual['run_converged'][j,h]),
                                       source_converged=runs[h*nruns+j]['converged'],source_message=runs[h*nruns+j]['message'],passed=True))
     checks.append(dict(classes=classes,code=code,seed=seed,config=config,name='optimizer_runs',passed=True))
     if classes==2 and not warm and limit==100 and kernel.n_dims:
      binary=Binary(kernel=kernel.clone_with_theta(theta[:,0]),optimizer=None,max_iter_predict=100).fit(x,labels)
      central=np.empty(kernel.n_dims);step=1e-5
      for j in range(kernel.n_dims):
       up=theta[:,0].copy();down=theta[:,0].copy();up[j]+=step;down[j]-=step
       central[j]=(binary.log_marginal_likelihood(up)-binary.log_marginal_likelihood(down))/(2*step)
      np.testing.assert_allclose(actual['gradient'],central,rtol=2e-4,atol=2e-5,
                                 err_msg=f'central Laplace gradient {code} {seed} {config}')
      checks.append(dict(classes=classes,code=code,seed=seed,config=config,name='central_gradient',passed=True))
     fixtures+=1
   print(f'{classes} classes kernel {code} passed',flush=True)
 results=ROOT/'validation/results';results.mkdir(exist_ok=True);source=Path(sklearn.__file__).parent
 report=dict(profile=args.profile,fixtures=fixtures,checks=checks,max_absolute_errors=errors,optimizations=optimizations,
             source_warnings=warning_counts,restart_comparisons=restart_comparisons,sklearn=sklearn.__version__,numpy=np.__version__,
             native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             source_sha256={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in ['gaussian_process/_gpc.py','gaussian_process/kernels.py','multiclass.py']})
 (results/f'gpc_advanced_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejected=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True)
  assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and message in run.stdout,(case,run.stdout,run.stderr)
  rejected.append(dict(name=case,expected=message,passed=True))
 (results/f'gpc_advanced_invalid_{args.profile}.json').write_text(json.dumps(dict(profile=args.profile,checks=rejected,native_sha256=report['native_sha256'],validator_sha256=report['validator_sha256']),indent=2,allow_nan=False)+'\n')
 print(f'{len(checks)} advanced GPC groups / {fixtures} fixtures, {len(optimizations)} optimization comparisons, {len(rejected)} rejections passed')


if __name__=='__main__':main()
