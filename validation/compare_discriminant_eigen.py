"""Generalized-eigen LDA with scalar shrinkage, source eigenspaces and residuals."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
from scipy.linalg import eigh
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,_cov
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block

REJECTIONS={'solver':'unknown LDA solver','tol':'LDA tol must be nonnegative','tol_nan':'LDA controls must be finite',
 'shrink_nan':'LDA controls must be finite','shrink_range':'invalid LDA shrinkage','shrink_negative':'invalid LDA shrinkage',
 'components':'n_components must be nonnegative','components_high':'n_components exceeds class/feature limit',
 'one_class':'requires at least two classes','samples':'needs more samples than classes',
 'singular':'matrix is not positive definite','zero_cov':'matrix is not positive definite',
 'nan_x':'X contains NaN or infinity','labels':'X and y sample counts differ','unfitted':'LDA eigen model is not fitted',
 'width':'X and y sample counts differ','nan_query':'X contains NaN or infinity','empty_query':'X must be nonempty',
 'lsqr_transform':'LDA transform requires SVD solver','refit_lsqr':'LDA eigen model is not fitted'}
NAMES=['classes','means','priors','eigenvalues','variance_ratio','scalings','fit','train_transform','query_transform',
 'coef','intercept','decision','proba','predict','covariance','precision','logdet','components','score']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
 ap.add_argument('--executable',type=Path);args=ap.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_discriminant_eigen_driver')
 fixtures=[];checks=[];certificates=[];basis_certificates=[];undefined=[]
 for seed in [0,42,2026]:
  rng=np.random.RandomState(seed)
  for n,p in [(27,4),(10,9),(30,1),(20,6)]:
   for count in [2,3,4]:
    variants=['plain','singleton']+(['rank_deficient'] if p>=3 else [])
    for kind in variants:
     y=np.arange(n)%count
     if kind=='singleton':y[(y==0)&(np.arange(n)!=0)]=1
     y=y*3-7;x=rng.normal(size=(n,p))+.17*y[:,None]*np.linspace(.3,1.4,p)
     q=rng.normal(size=(4,p))+.2
     if kind=='rank_deficient':x[:,-1]=2;q[:,-1]=2
     alphas=[.15,.7,1.]
     if n>p+count and kind!='rank_deficient':alphas=[0.,*alphas]
     for alpha in alphas:
      for k in [0,1]:fixtures.append((seed,x,y,q,k,alpha,kind))
 for p in [1,2]:
  cloud=np.vstack([np.eye(p),-np.eye(p)]);x=np.tile(cloud,(2,1));y=np.repeat([-7,-4],2*p)
  for alpha in [0.,.4,1.]:fixtures.append((0,x,y,cloud,0,alpha,'coincident_means'))
 for index,(seed,x,y,q,k,alpha,kind) in enumerate(fixtures):
  n,p=x.shape;m=len(q);tol=1e-4 if index%2==0 else .8
  model=LinearDiscriminantAnalysis(solver='eigen',shrinkage=alpha,tol=tol,
    n_components=None if k==0 else k,store_covariance=bool(index%2))
  with warnings.catch_warnings(record=True),np.errstate(invalid='ignore',divide='ignore'):
   z=model.fit_transform(x,y) if index%2==0 else model.fit(x,y).transform(x)
  payload=f'{n} {p} {m} {k} {index%2} {index%2} {tol:.17g} {alpha:.17g}\n'
  payload+=' '.join(f'{v:.17g}' for v in x.ravel(order='F'))+'\n'+' '.join(map(str,y))+'\n'
  payload+=' '.join(f'{v:.17g}' for v in q.ravel(order='F'))+'\n'
  run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True,check=True,cwd=ROOT)
  lines=iter(run.stdout.splitlines());actual=[block(lines) for _ in NAMES];assert next(lines,None) is None
  within=model.covariance_;between=_cov(x,alpha)-within
  values,unused=eigh(between,within);values=values[np.argsort(values)[::-1]]
  basis=actual[5];source=model.scalings_;rotation=np.zeros((p,p));groups=[];start=0
  gap=1e-8*max(1.,float(np.max(np.abs(values))))
  while start<p:
   stop=start+1
   while stop<p and abs(values[stop]-values[start])<=gap:stop+=1
   cross=basis[:,start:stop].T@within@source[:,start:stop]
   u,s,vt=np.linalg.svd(cross);r=u@vt;rotation[start:stop,start:stop]=r
   if stop-start>1:groups.append([start,stop])
   start=stop
  # Align only within source eigenvalue clusters; distinct axes get a sign.
  expected_basis=source@rotation.T
  span_error=float(np.max(np.abs(basis-expected_basis),initial=0.))
  span_bound=3e-7*max(1.,float(np.max(np.abs(source))))
  assert span_error<=span_bound,(index,kind,span_error,span_bound)
  if groups:
   basis_certificates.append(dict(fixture=index,groups=groups,span_error=span_error,span_bound=span_bound,
      crosses_component_cutoff=any(a<z.shape[1]<b for a,b in groups),passed=True))
  coef=model.coef_;intercept=model.intercept_;decision=model.decision_function(q)
  if len(model.classes_)==2:
   coef=np.vstack([np.zeros(p),coef[0]]);intercept=np.array([0.,intercept[0]])
   decision=np.column_stack([np.zeros(m),decision])
   actual[9]=actual[9]-actual[9][:1];actual[10]=actual[10]-actual[10][0];actual[11]=actual[11]-actual[11][:,:1]
  expected=[model.classes_,model.means_,model.priors_,values,model.explained_variance_ratio_,expected_basis,
    (x@expected_basis)[:,:z.shape[1]],(x@expected_basis)[:,:z.shape[1]],(q@expected_basis)[:,:z.shape[1]],
    coef,intercept,decision,model.predict_proba(q),model.predict(q),within,np.linalg.inv(within),
    np.repeat(np.linalg.slogdet(within)[1],len(model.classes_)),np.array([z.shape[1]]),np.array([model.score(x,y)])]
  for name,got,want in zip(NAMES,actual,expected):
   np.testing.assert_allclose(got,want,atol=3e-7,rtol=3e-7,err_msg=f'{index} {kind} {alpha} {name}')
   checks.append(dict(fixture=index,seed=seed,kind=kind,name=name,passed=True))
  if np.any(np.isnan(model.explained_variance_ratio_)):
   assert kind=='coincident_means' and np.all(between==0) and np.all(np.isnan(actual[4]))
   undefined.append(dict(fixture=index,source_undefined=True,native_nan_convention=True,passed=True))
  white=basis.T@within@basis
  white_error=float(np.max(np.abs(white-np.eye(p))))
  residual=between@basis-within@basis*actual[3]
  eigen_error=float(np.max(np.abs(residual)))
  eigen_bound=2e-8*max(1.,np.linalg.norm(between,np.inf)*np.linalg.norm(basis,np.inf),
    np.linalg.norm(within,np.inf)*np.linalg.norm(basis*actual[3],np.inf))
  assert white_error<=2e-8 and eigen_error<=eigen_bound,(index,white_error,eigen_error,eigen_bound)
  certificates.append(dict(fixture=index,whitening_error=white_error,whitening_bound=2e-8,
    eigen_error=eigen_error,eigen_bound=float(eigen_bound),passed=True))
  if (index+1)%36==0:print(index+1,'eigen discriminant fixtures passed',flush=True)
 # Source generalized eigh also rejects exact zero-variance within directions.
 source_rejections=[]
 for kind,data,alpha in [('singular',np.column_stack([np.arange(6),np.arange(6)**2,np.zeros(6)]),0.),
                          ('zero_cov',np.zeros((6,3)),.2)]:
  try:
   LinearDiscriminantAnalysis(solver='eigen',shrinkage=alpha).fit(data,[1,1,1,2,2,2])
  except np.linalg.LinAlgError:
   source_rejections.append(dict(case=kind,source_exception='LinAlgError',passed=True))
  else:raise AssertionError('source accepted singular within covariance')
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();results=ROOT/'validation/results';results.mkdir(exist_ok=True)
 report=dict(profile=args.profile,sklearn_version=sklearn.__version__,fixtures=len(fixtures),checks=checks,
  eigen_certificates=certificates,basis_certificates=basis_certificates,undefined_ratio_certificates=undefined,
  source_singular_rejections=source_rejections,native_sha256=sha(exe),validator_sha256=sha(Path(__file__)),
  source_sha256={'discriminant_analysis.py':sha(Path(sklearn.__file__).parent/'discriminant_analysis.py')})
 (results/f'discriminant_eigen_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejects=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True,cwd=ROOT)
  assert run.returncode!=0 and message in run.stdout+run.stderr,(case,run.stdout,run.stderr)
  rejects.append(dict(case=case,passed=True))
 report.update(checks=rejects)
 for field in ['fixtures','eigen_certificates','basis_certificates','undefined_ratio_certificates','source_singular_rejections']:report.pop(field)
 (results/f'discriminant_eigen_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(len(checks),'groups,',len(fixtures),'eigen certificates,',len(basis_certificates),'basis certificates,',
   len(undefined),'undefined ratios and',len(rejects),'rejections PASS',flush=True)

if __name__=='__main__':main()
