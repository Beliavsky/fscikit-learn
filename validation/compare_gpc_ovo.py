"""Gaussian process one-vs-one subsets, votes, likelihoods and learned pair kernels."""
from pathlib import Path
from itertools import combinations
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process._gpc import _BinaryGaussianProcessClassifierLaplace as Binary
from scipy.optimize import minimize
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block
from compare_gpr_advanced import make_kernel,apply_bounds

REJECTIONS={'mode':'unsupported multi_class','binary_mode':'unsupported multi_class','unfitted':'GPC is not fitted',
            'probability':'one_vs_one does not support probability','theta':'theta must match one kernel',
            'gradient':'multiclass likelihood gradients are not supported'}


def children(model):return [model.base_estimator_] if len(model.classes_)==2 else model.base_estimator_.estimators_


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
 parser.add_argument('--executable',type=Path);args=parser.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_gpc_ovo_driver')
 checks=[];optimizations=[];restarts_checked=0;fixtures=0;errors={};source_warnings={}
 voting_ties=0;confidence_changes=0
 for classes in [2,3,4]:
  for code in range(9):
   for seed in [0,42,2026]:
    for config in range(4):
     warm=config in [1,3];opt=config in [2,3];limit=3 if config==1 else 100;restarts=int(config==3)
     rng=np.random.RandomState(seed);labels=np.repeat([-4,9,17,31][:classes],[5,7,4,6][:classes]);n=len(labels)
     p=1 if code in [3,8] else 2;m=7;x=np.empty((n,p));x[:,0]=np.linspace(-1.5,1.5,n)+.02*rng.normal(size=n)
     if p>1:x[:,1]=np.sin(1.4*x[:,0])+.07*rng.normal(size=n)
     order=rng.permutation(n);x=x[order];labels=labels[order]
     if seed==2026:labels=labels[rng.permutation(n)]
     q=np.empty((m,p));q[:,0]=np.linspace(-1.4,1.4,m)
     if p>1:q[:,1]=np.sin(1.4*q[:,0])
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
      result=minimize(objective,initial,jac=True,bounds=bounds,method='L-BFGS-B',
                      options=dict(maxiter=15000,maxfun=15000,gtol=1e-5,ftol=2.220446049250313e-9,maxls=20,maxcor=10))
      runs.append(dict(initial=initial.copy(),point=result.x.copy(),value=float(result.fun)));return result.x,result.fun
     with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      model=GaussianProcessClassifier(kernel=kernel,optimizer=optimizer if opt else None,multi_class='one_vs_one',
             warm_start=warm,max_iter_predict=limit,n_restarts_optimizer=restarts,random_state=seed).fit(x,labels)
     for w in caught:
      key=w.category.__name__+': '+str(w.message);source_warnings[key]=source_warnings.get(key,0)+1
     pairs=list(combinations(range(classes),2));heads=len(pairs);es=children(model)
     indices=[np.flatnonzero((labels==model.classes_[i])|(labels==model.classes_[j])) for i,j in pairs]
     header=[code,n,p,m,seed,int(opt),restarts,int(warm),limit,-1 if bounds is None else len(bounds)]
     payload=' '.join(map(str,header))+'\n'+''.join(' '.join(f'{v:.17g}' for v in a.ravel(order='F'))+'\n' for a in [x,q,labels])
     if bounds is not None:payload+=' '.join(f'{v:.17g}' for v in bounds.ravel(order='F'))+'\n'
     run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
     assert run.returncode==0,(classes,code,seed,config,run.stdout,run.stderr)
     lines=iter(run.stdout.splitlines());actual={}
     for name in ['labels','cached_lml','pairs','counts','theta','channel_lml']:actual[name]=block(lines)
     names=['indices','pi','w','l','alpha','mode','cache','initial_cache','points','values','eval_cache','initial_points']
     for h in range(heads):
      for name in names:actual[f'{h}_{name}']=block(lines)
     actual['shared_lml']=block(lines)
     for h in range(heads):actual[f'{h}_shared_cache']=block(lines)
     actual['clone_labels']=block(lines);actual['packed_lml']=block(lines)
     for h in range(heads):
      actual[f'{h}_mutated_theta']=block(lines);actual[f'{h}_mutated_cache']=block(lines)
     actual['mutated_labels']=block(lines);actual['cached_after']=block(lines);assert next(lines,None) is None
     context=(classes,code,seed,config)
     def compare(name,want,tol=3e-8,exact=False):
      got=actual[name];want=np.asarray(want);assert got.shape==want.shape,(context,name,got.shape,want.shape)
      if exact:np.testing.assert_array_equal(got,want,err_msg=str((context,name)))
      else:
       np.testing.assert_allclose(got,want,atol=tol,rtol=tol,err_msg=str((context,name)))
       errors[name]=max(errors.get(name,0.),float(np.max(abs(got-want),initial=0)))
      checks.append(dict(classes=classes,code=code,seed=seed,config=config,name=name,passed=True))
     # Full independent source fit: exact voting labels, including optimized models.
     compare('labels',model.predict(q),exact=True)
     if classes>2:
      votes=np.zeros((m,classes))
      for (i,j),e in zip(pairs,es):
       positive=e.predict(q)==1;votes[:,i]+=~positive;votes[:,j]+=positive
      voting_ties+=int(np.sum(np.sum(votes==votes.max(axis=1,keepdims=True),axis=1)>1))
      confidence_changes+=int(np.sum(np.argmax(votes,axis=1)!=np.argmax(model.base_estimator_.decision_function(q),axis=1)))
     compare('pairs',np.array(pairs).T+1 if classes>2 else np.zeros((2,1)),exact=True)
     compare('counts',[len(i) for i in indices],exact=True)
     source_lml=np.array([e.log_marginal_likelihood_value_ for e in es])
     compare('channel_lml',source_lml,tol=1e-5 if opt else 3e-8)
     compare('cached_lml',[model.log_marginal_likelihood()],tol=1e-5 if opt else 3e-8)
     if opt and kernel.n_dims:
      assert np.all(actual['theta']>=bounds[:,0,None]-1e-12) and np.all(actual['theta']<=bounds[:,1,None]+1e-12)
      optimizations.append(dict(classes=classes,code=code,seed=seed,config=config,
          native_lml=actual['channel_lml'].tolist(),source_lml=source_lml.tolist(),passed=True))
     replay=[]
     for h,((i,j),ix) in enumerate(zip(pairs,indices)):
      compare(f'{h}_indices',ix+1,exact=True)
      target=labels if classes==2 else (labels[ix]==model.classes_[j]).astype(int)
      if classes==2:target=labels[ix]
      e=es[h];theta=actual['theta'][:,h]
      if opt and kernel.n_dims:
       e=Binary(kernel=kernel.clone_with_theta(theta),optimizer=None,warm_start=warm,max_iter_predict=limit).fit(x[ix],target)
       if warm:
        e.f_cached=actual[f'{h}_initial_cache'].copy()
        _,temporaries=e._posterior_mode(e.kernel_(x[ix]),return_temporaries=True)
        e.pi_,e.W_sr_,e.L_=temporaries[:3]
       nruns=restarts+1
       compare(f'{h}_initial_points',np.array([r['initial'] for r in runs[h*nruns:(h+1)*nruns]]),tol=1e-14)
       compare(f'{h}_values',[r['value'] for r in runs[h*nruns:(h+1)*nruns]],tol=1e-5)
       best=int(np.argmin(actual[f'{h}_values']));np.testing.assert_allclose(theta,actual[f'{h}_points'][best],rtol=0,atol=1e-14)
       for k in range(nruns):
        point=actual[f'{h}_points'][k]
        check=Binary(kernel=kernel.clone_with_theta(point),optimizer=None,max_iter_predict=limit).fit(x[ix],target)
        if warm:check.warm_start=True;check.f_cached=actual[f'{h}_eval_cache'][:,k].copy()
        value=-check.log_marginal_likelihood(point)
        np.testing.assert_allclose(actual[f'{h}_values'][k],value,rtol=3e-8,atol=3e-8)
        restarts_checked+=1
      for name,want in [('pi',e.pi_),('w',e.W_sr_),('l',e.L_),('alpha',e.y_train_-e.pi_),('mode',e.f_cached),('cache',e.f_cached)]:
       compare(f'{h}_{name}',want,tol=2e-6 if opt else 3e-8)
      if not opt or not kernel.n_dims:
       compare(f'{h}_initial_cache',Binary(kernel=e.kernel_,optimizer=None,max_iter_predict=limit).fit(x[ix],target).f_cached)
      replay.append(e)
     # sklearn's outer packed-theta shape check incorrectly uses class count for OVO.
     # Evaluate each independent binary child for the intended pair-major contract.
     if classes==2:model.base_estimator_=replay[0]
     else:model.base_estimator_.estimators_=replay
     shared=actual['theta'][:,0]+.04
     compare('shared_lml',[model.log_marginal_likelihood(shared)],tol=2e-6 if opt else 3e-8)
     for h,e in enumerate(replay):compare(f'{h}_shared_cache',e.f_cached,tol=2e-6 if opt else 3e-8)
     compare('clone_labels',model.predict(q),exact=True)
     values=[]
     for h,e in enumerate(replay):values.append(e.log_marginal_likelihood(actual['theta'][:,h]+.02,clone_kernel=False))
     compare('packed_lml',[np.mean(values)],tol=2e-6 if opt else 3e-8)
     for h,e in enumerate(replay):
      compare(f'{h}_mutated_theta',e.kernel_.theta,tol=1e-14)
      compare(f'{h}_mutated_cache',e.f_cached,tol=2e-6 if opt else 3e-8)
     compare('mutated_labels',model.predict(q),exact=True)
     compare('cached_after',actual['cached_lml'],exact=True)
     fixtures+=1
   print(f'{classes} classes kernel {code} passed',flush=True)
 assert voting_ties>0 and confidence_changes>0,(voting_ties,confidence_changes)
 results=ROOT/'validation/results';results.mkdir(exist_ok=True);source=Path(sklearn.__file__).parent
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 report=dict(profile=args.profile,fixtures=fixtures,checks=checks,max_absolute_errors=errors,
     optimizations=optimizations,restart_comparisons=restarts_checked,voting_ties=voting_ties,
     confidence_changes=confidence_changes,source_warnings=source_warnings,
     native_sha256=digest(exe),validator_sha256=digest(Path(__file__)),sklearn=sklearn.__version__,numpy=np.__version__,
     source_sha256={p:digest(source/p) for p in ['gaussian_process/_gpc.py','multiclass.py','utils/multiclass.py']})
 (results/f'gpc_ovo_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejected=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True)
  assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and message in run.stdout,(case,run.stdout,run.stderr)
  rejected.append(dict(name=case,expected=message,passed=True))
 (results/f'gpc_ovo_invalid_{args.profile}.json').write_text(json.dumps(dict(profile=args.profile,checks=rejected,
    native_sha256=report['native_sha256'],validator_sha256=report['validator_sha256']),indent=2,allow_nan=False)+'\n')
 print(f'{len(checks)} OVO GPC groups / {fixtures} fixtures, {len(optimizations)} optimizations, {restarts_checked} restart checks, {len(rejected)} rejections passed')


if __name__=='__main__':main()
