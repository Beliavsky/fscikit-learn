"""KernelPCA kernel-ridge pre-images against installed sklearn and ridge identities."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np
import sklearn
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_kernels
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block

REJECTIONS={'negative_alpha':'alpha must be finite and nonnegative','nan_alpha':'alpha must be finite and nonnegative',
 'nan_gamma':'controls must be finite','nan_coef0':'controls must be finite','precomputed':'does not support precomputed',
 'unfitted':'inverse model is not fitted','disabled':'inverse fitting was not enabled','width':'inverse component count mismatch',
 'query_nan':'X contains NaN or infinity','query_empty':'X must be nonempty','singular':'matrix is not positive definite',
 'zero_components':'X must be nonempty','refit_disabled':'inverse model is not fitted','indefinite':'matrix is not positive definite'}
NAMES=['eigenvalues','fit','train_transform','query','stored_latent','dual','inverse_train','inverse_query','inverse_arbitrary','cleared']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
 ap.add_argument('--executable',type=Path);args=ap.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_kernel_preimage_driver')
 checks=[];certificates=[];fixtures=[]
 for seed in [0,42,2026]:
  rng=np.random.RandomState(seed)
  for n,p in [(9,3),(5,11),(21,4)]:
   x=rng.normal(size=(n,p))*.4+np.linspace(.2,1.4,p);q=rng.normal(size=(4,p))*.4+.6
   for code in range(4):
    for alpha in [.001,.2,3.]:
     for k in [1,min(3,n)]:
      for gamma in [-1.,.13]:
       fixtures.append((seed,x,q,k,code,alpha,gamma,1.,int(k==3),len(fixtures)%2,'random'))
 # Zero/constant latent coordinates remain valid when explicit components retain zero eigenvalues.
 for code in range(4):
  for value in [0.,2.]:
   fixtures.append((0,np.full((6,3),value),np.full((2,3),value+.1),2,code,.2,-1.,1.,0,0,'constant'))
 # Two distinct sample values give an unambiguous one-dimensional positive sigmoid eigenspace.
 for alpha in [.01,.7]:
  x=np.array([[-.4],[-.4],[.7],[.7]])
  fixtures.append((0,x,np.array([[-.2],[.5]]),1,4,alpha,.05,.3,1,1,'sigmoid'))
 # Zero regularization is supported for a nonsingular latent RBF Gram matrix.
 fixtures.append((42,np.random.RandomState(42).normal(size=(7,3)),np.ones((2,3))*.3,3,1,0.,.4,1.,1,0,'zero_alpha'))
 for code in range(4):
  rng=np.random.RandomState(123)
  fixtures.append((123,rng.normal(size=(8,3))*.2,rng.normal(size=(2,3))*.2,0,code,.2,-1.,1.,1,0,'all_components'))
 for code in range(5):
  fixtures.append((0,np.array([[.2,.4,.8]]),np.array([[.3,.5,.9]]),2,code,.2,.05,.3,0,1,'single_sample'))
 for index,(seed,x,q,k,code,alpha,gamma,coef0,remove,operation,kind) in enumerate(fixtures):
  n,p=x.shape;m=len(q);kernel=['linear','rbf','poly','cosine','sigmoid'][code]
  model=KernelPCA(n_components=None if k==0 else k,kernel=kernel,degree=2,gamma=None if gamma<0 else gamma,coef0=coef0,
    alpha=alpha,fit_inverse_transform=True,remove_zero_eig=bool(remove),eigen_solver='dense')
  z=model.fit_transform(x) if operation==0 else model.fit(x).X_transformed_fit_
  tq=model.transform(q)
  expected=[model.eigenvalues_,z,model.transform(x),tq,model.X_transformed_fit_,model.dual_coef_,
    model.inverse_transform(z),model.inverse_transform(tq),model.inverse_transform(.73*tq+.11),np.array([1.])]
  payload=f'{n} {p} {m} {k} {code} {remove} {operation} {alpha:.17g} {gamma:.17g} {coef0:.17g}\n'
  payload+=' '.join(f'{v:.17g}' for v in x.ravel(order='F'))+'\n'
  payload+=' '.join(f'{v:.17g}' for v in q.ravel(order='F'))+'\n'
  run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True,check=True,cwd=ROOT)
  lines=iter(run.stdout.splitlines());actual=[block(lines) for _ in NAMES];assert next(lines,None) is None
  for name,got,want in zip(NAMES,actual,expected):
   np.testing.assert_allclose(got,want,atol=3e-7,rtol=3e-7,err_msg=f'{index} {kernel} {name}')
   checks.append(dict(fixture=index,seed=seed,kernel=kernel,kind=kind,name=name,passed=True))
  latent=actual[4];dual=actual[5]
  gram=pairwise_kernels(latent,metric=kernel,gamma=model.gamma_,degree=2,coef0=coef0,filter_params=True)
  gram.flat[::n+1]+=alpha
  residual=float(np.linalg.norm(gram@dual-x,np.inf))
  bound=2e-10*max(1.,np.linalg.norm(gram,np.inf)*np.linalg.norm(dual,np.inf),np.linalg.norm(x,np.inf))
  assert residual<=bound,(index,residual,bound)
  certificates.append(dict(fixture=index,residual=residual,bound=float(bound),passed=True))
  if (index+1)%36==0:print(index+1,'KernelPCA preimage fixtures passed',flush=True)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 results=ROOT/'validation/results';results.mkdir(exist_ok=True)
 report=dict(profile=args.profile,sklearn_version=sklearn.__version__,fixtures=len(fixtures),checks=checks,
  ridge_certificates=certificates,native_sha256=sha(exe),validator_sha256=sha(Path(__file__)),
  source_sha256={'decomposition/_kernel_pca.py':sha(Path(sklearn.__file__).parent/'decomposition/_kernel_pca.py')})
 (results/f'kernel_preimage_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 # This indefinite sigmoid inverse must also fail in the source SPD solver.
 indefinite=np.vstack([np.full((2,3),-.4),np.full((2,3),.7)])
 try:
  KernelPCA(n_components=1,kernel='sigmoid',gamma=.05,coef0=-.3,alpha=.01,
    fit_inverse_transform=True,eigen_solver='dense').fit(indefinite)
 except np.linalg.LinAlgError:
  pass
 else:
  raise AssertionError('source accepted indefinite preimage system')
 rejects=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True,cwd=ROOT)
  assert run.returncode!=0 and message in run.stdout+run.stderr,(case,run.stdout,run.stderr)
  rejects.append(dict(case=case,passed=True))
 report.update(checks=rejects);report.pop('ridge_certificates');report.pop('fixtures')
 (results/f'kernel_preimage_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(len(checks),'groups,',len(fixtures),'ridge certificates and',len(rejects),'rejections PASS',flush=True)

if __name__=='__main__':main()
