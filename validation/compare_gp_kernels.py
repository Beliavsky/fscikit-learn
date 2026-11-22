"""Covariances, log gradients, GP regression/classification and SciFort Bessel support."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import sklearn
from scipy.special import kve
from sklearn.gaussian_process import GaussianProcessRegressor,GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF,Matern,RationalQuadratic,ExpSineSquared,DotProduct,ConstantKernel,WhiteKernel
from compare_function_transformer import ROOT,resolve

REJECTIONS={
 'empty':'GP kernel is empty','empty_expression':'empty child','empty_lengths':'lengths must be nonempty',
 'zero_length':'lengths must be positive','negative_length':'lengths must be positive','nan_length':'lengths must be finite',
 'nu_zero':'nu must be positive','nu_nan':'nu must not be NaN','alpha_zero':'alpha must be positive',
 'period_zero':'periodicity must be positive','negative_constant':'value must be nonnegative',
 'nan_constant':'value must be finite','negative_noise':'value must be nonnegative','negative_sigma':'value must be nonnegative',
 'nan_power':'exponent must be finite','length_count':'length count mismatch','diag_length_count':'length count mismatch',
 'feature_count':'feature count mismatch','nan_X':'X contains NaN or infinity','theta_count':'theta length mismatch',
 'theta_nan':'theta must not contain NaN','theta_overflow':'theta exceeds exponential range',
 'theta_zero_length':'lengths must be positive','fractional_negative':'fractional power of negative value',
 'negative_zero':'negative power of zero value','gp_nan_alpha':'invalid Gaussian process alpha',
 'gpc_kernel_length':'length count mismatch',
}


def make_kernel(code,p,config):
 lengths=np.linspace(.4,1.8,p) if p>1 else np.array([.4])
 fixed='fixed' if config%3==1 else (1e-5,1e5);other='fixed' if config%3==2 else (1e-5,1e5)
 if code==0:return RBF(.7,fixed)
 if code==1:return RBF(lengths,fixed)
 if code==2:return Matern(1.2,fixed,nu=.5)
 if code==3:return Matern(lengths,fixed,nu=1.5)
 if code==4:return Matern(1.2,fixed,nu=2.5)
 if code==5:return Matern(lengths,fixed,nu=np.inf)
 if code==6:return Matern(1.2,fixed,nu=.1)
 if code==7:return Matern(lengths,fixed,nu=.7)
 if code==8:return Matern(1.2,fixed,nu=1.2)
 if code==9:return Matern(lengths,fixed,nu=3.2)
 if code==10:return Matern(1.2,fixed,nu=5.3)
 if code==11:return RationalQuadratic(.8,.3,length_scale_bounds=fixed,alpha_bounds=other)
 if code==12:return ExpSineSquared(.9,1.7,length_scale_bounds=fixed,periodicity_bounds=other)
 if code==13:return DotProduct(.4,fixed)
 if code==14:return ConstantKernel(1.8,fixed)
 if code==15:return WhiteKernel(.12,fixed)
 if code==16:return 1.3*RBF(lengths,fixed)+WhiteKernel(.07,other)
 if code==17:return (RBF(.7,fixed)+DotProduct(.4,other))*RationalQuadratic(.8,.3,length_scale_bounds=fixed,alpha_bounds=other)
 if code==18:return (ExpSineSquared(.9,1.7)*Matern(lengths,nu=1.5)+WhiteKernel(.02))**2
 if code==19:return (.8+RBF(1.3,fixed))**1.7
 if code==20:return ConstantKernel(1.2,'fixed')*RBF(.8,'fixed')+WhiteKernel(.03,'fixed')
 if code==21:return RationalQuadratic(.8,.3,alpha_bounds='fixed')+ExpSineSquared(.9,1.7,length_scale_bounds='fixed')+DotProduct(.4,'fixed')
 if code==22:return DotProduct(.4,fixed)**2
 if code==23:return ConstantKernel(0.,fixed)+WhiteKernel(0.,fixed)+DotProduct(0.,fixed)
 if code==24:return RBF(.7,fixed)**(-1.)
 raise AssertionError(code)


def block(lines):
 shape=tuple(map(int,next(lines).split()));data=np.fromstring(next(lines),sep=' ')
 assert data.size==np.prod(shape),shape
 return data.reshape(shape,order='F')


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
 parser.add_argument('--executable',type=Path);args=parser.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_gp_kernels_driver')
 checks=[];fixtures=0;errors={};source_warnings={};certificates=[]
 for mode in [0,1,2]:
  for code in range(25 if mode==0 else 24):
   for seed in [0,42,2026]:
    for config in range(4):
     n=[7,11,9,13][config];p=[1,3,2,3][config];m=5
     if mode>0 and code in [12,18,21]:p=1
     rng=np.random.RandomState(seed+config*11);x=rng.normal(size=(n,p))*.45;q=rng.normal(size=(m,p))*.6
     yr=np.sin(2.3*x[:,0])+.4*x[:,-1]+.1*rng.normal(size=n)
     if config==1:yr[:]=3.2
     yi=np.resize([-4,9,17] if config>=2 else [-4,9],n)
     if config==3:x[1]=x[0];q[0]=x[0];q[1]=x[-1]
     kernel=make_kernel(code,p,config)
     with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      if mode==0:
       value,gradient=kernel(x,eval_gradient=True);theta=kernel.theta
       expected=dict(matrix=value,cross=kernel(x,q),diag=kernel.diag(x),gradient=gradient,theta=theta,
                     roundtrip=kernel.clone_with_theta(theta)(x),shifted=kernel.clone_with_theta(theta+.03)(x),
                     shifted_theta=kernel.clone_with_theta(theta+.03).theta,snapshot=kernel(x))
      elif mode==1:
       model=GaussianProcessRegressor(kernel=kernel,optimizer=None,alpha=1.,normalize_y=config%2==1)
       expected=dict(prior_mean=model.predict(q),prior_std=model.predict(q,return_std=True)[1],prior_cov=model.predict(q,return_cov=True)[1])
       model.fit(x,yr);mean,std=model.predict(q,return_std=True);cov=model.predict(q,return_cov=True)[1]
       expected.update(mean=mean,std=std,cov=cov,lml=np.array([model.log_marginal_likelihood_value_]),alpha=model.alpha_,cholesky=model.L_)
       model.kernel=DotProduct(4.);expected['snapshot']=model.predict(q);model.fit(x,yr)
       expected.update(refit=model.predict(q),refit_lml=np.array([model.log_marginal_likelihood_value_]))
      else:
       model=GaussianProcessClassifier(kernel=kernel,optimizer=None,max_iter_predict=100).fit(x,yi)
       estimators=model.base_estimator_.estimators_ if len(model.classes_)>2 else [model.base_estimator_]
       expected=dict(proba=model.predict_proba(q),labels=model.predict(q),lml=np.array([model.log_marginal_likelihood_value_]),
                     channel_lml=np.array([e.log_marginal_likelihood_value_ for e in estimators]),
                     pi=np.column_stack([e.pi_ for e in estimators]),w=np.column_stack([e.W_sr_ for e in estimators]),
                     alpha=np.column_stack([e.y_train_-e.pi_ for e in estimators]))
       model.kernel=DotProduct(4.);expected['snapshot']=model.predict_proba(q);model.fit(x,yi)
       expected.update(refit=model.predict_proba(q),refit_lml=np.array([model.log_marginal_likelihood_value_]))
     for warning in caught:
      key=warning.category.__name__+': '+str(warning.message);source_warnings[key]=source_warnings.get(key,0)+1
     payload=f'{mode} {code} {n} {p} {m} {config}\n'+''.join(' '.join(f'{v:.17g}' for v in np.asarray(a).ravel(order='F'))+'\n' for a in [x,q,yr,yi])
     run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
     assert run.returncode==0,(mode,code,seed,config,run.stdout,run.stderr)
     lines=iter(run.stdout.splitlines());native_gradient=None;native_proba=None
     for name,want in expected.items():
      got=block(lines)
      if name=='gradient':native_gradient=got
      if name=='proba':native_proba=got
      assert got.shape==want.shape,(mode,code,name,got.shape,want.shape)
      if name=='labels':
       classes,counts=np.unique(yi,return_counts=True)
       if code==14 and len(classes)>2 and np.all(counts==counts[0]):
        # Rank-one constant covariance and equal class counts make every OVR
        # head permutation-equivalent; normalized probabilities are exactly 1/c.
        # LAPACK/reduction rounding can change the reported winning class.
        assert isinstance(kernel,ConstantKernel)
        assert np.all(kernel(x)==kernel.constant_value) and np.all(kernel(x,q)==kernel.constant_value)
        assert all(isinstance(e.kernel_,ConstantKernel) and e.kernel_.constant_value==kernel.constant_value for e in estimators)
        bound=64*np.finfo(float).eps*np.sum(np.abs([-1854.8214151,3516.89893646,221.29346712,128.12323805,-2010.49422654]))
        native_error=float(np.max(abs(native_proba-1/len(classes))))
        source_error=float(np.max(abs(expected['proba']-1/len(classes))))
        assert native_error<=bound and source_error<=bound,(code,seed,config,native_error,source_error,bound)
        assert np.all(np.isin(got,classes)) and np.all(np.isin(want,classes))
        certificates.append(dict(code=code,seed=seed,config=config,classes=classes.tolist(),counts=counts.tolist(),
                                 native_labels=got.tolist(),source_labels=want.tolist(),
                                 native_probability_error=native_error,source_probability_error=source_error,
                                 error_bound=float(bound),oracle='constant_covariance_equal_class_symmetry',passed=True))
        checks.append(dict(mode=mode,code=code,seed=seed,config=config,name='constant_tie_certificate',passed=True))
       else:np.testing.assert_array_equal(got,want,err_msg=f'{mode} {code} {seed} {config} labels')
      else:
       tolerance=3e-4 if name=='gradient' and code in [6,7,8,9,10] else 3e-8 if mode==2 else 3e-9
       np.testing.assert_allclose(got,want,rtol=tolerance,atol=tolerance,equal_nan=False,err_msg=f'{mode} {code} {seed} {config} {name}')
       finite=np.isfinite(got)&np.isfinite(want)
       errors[name]=max(errors.get(name,0.),float(np.max(abs(got[finite]-want[finite]),initial=0)))
      checks.append(dict(mode=mode,code=code,seed=seed,config=config,name=name,passed=True))
     assert next(lines,None) is None
     if mode==0 and code in [6,7,8,9,10] and kernel.n_dims:
      central=np.empty_like(native_gradient);step=1e-5
      for j in range(kernel.n_dims):
       up=kernel.theta.copy();down=kernel.theta.copy();up[j]+=step;down[j]-=step
       central[:,:,j]=(kernel.clone_with_theta(up)(x)-kernel.clone_with_theta(down)(x))/(2*step)
      np.testing.assert_allclose(native_gradient,central,rtol=4e-6,atol=4e-7,
                                 err_msg=f'general Matern central derivative {code} {seed} {config}')
      errors['general_matern_central_gradient']=max(errors.get('general_matern_central_gradient',0.),
                                                   float(np.max(abs(native_gradient-central),initial=0)))
      checks.append(dict(mode=mode,code=code,seed=seed,config=config,name='general_matern_central_gradient',passed=True))
     fixtures+=1
   print(f'mode {mode} kernel {code} passed',flush=True)
 points=[(nu,z) for nu in [.1,.7,1.2,3.2,5.3,8.2] for z in [1e-16,1e-8,.01,.5,3.,12.,30.,100.]]
 payload=f'3 0 {len(points)} 1 1 0\n'+''.join(f'{nu:.17g} {z:.17g}\n' for nu,z in points)
 run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True);assert run.returncode==0,run.stderr
 got=block(iter(run.stdout.splitlines()));want=np.array([np.log(kve(nu,z))-z for nu,z in points])
 np.testing.assert_allclose(got,want,rtol=2e-11,atol=2e-11)
 for nu,z in points:checks.append(dict(mode=3,nu=nu,z=z,name='log_bessel_k',passed=True))
 source=Path(sklearn.__file__).parent;results=ROOT/'validation/results';results.mkdir(exist_ok=True)
 report=dict(profile=args.profile,fixtures=fixtures,checks=checks,max_absolute_errors=errors,source_warnings=source_warnings,roundoff_certificates=certificates,
             sklearn=sklearn.__version__,numpy=np.__version__,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
             validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             source_sha256={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in ['gaussian_process/kernels.py','gaussian_process/_gpr.py','gaussian_process/_gpc.py']})
 (results/f'gp_kernels_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejected=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True)
  assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and message in run.stdout,(case,run.stdout,run.stderr)
  rejected.append(dict(name=case,expected=message,passed=True))
 (results/f'gp_kernels_invalid_{args.profile}.json').write_text(json.dumps(dict(profile=args.profile,checks=rejected,native_sha256=report['native_sha256'],validator_sha256=report['validator_sha256']),indent=2,allow_nan=False)+'\n')
 print(f'{len(checks)} GP kernel groups / {fixtures} fixtures and {len(rejected)} isolated rejections passed')


if __name__=='__main__':main()
