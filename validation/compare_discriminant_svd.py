"""SVD discriminant fitting, prediction and projection against sklearn."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
from scipy.linalg import svdvals
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block

REJECTIONS={'solver':'unknown LDA solver','tol':'LDA tol must be nonnegative','tol_nan':'LDA controls must be finite',
 'shrink_nan':'LDA controls must be finite','shrink_range':'invalid LDA shrinkage','shrink_svd':'does not support shrinkage',
 'components':'n_components must be nonnegative','components_high':'n_components exceeds class/feature limit',
 'one_class':'requires at least two classes','samples':'needs more samples than classes',
 'rank_zero':'no retained within-class directions','rank_tol':'no retained within-class directions',
 'nan_x':'X contains NaN or infinity','labels':'X and y sample counts differ','unfitted':'LDA SVD is not fitted',
 'lsqr_transform':'LDA transform requires SVD solver','width':'X and y sample counts differ',
 'nan_query':'X contains NaN or infinity','empty_query':'X must be nonempty','refit_lsqr':'LDA SVD is not fitted'}
NAMES=['classes','means','priors','xbar','variance_ratio','scalings','fit','train_transform','query_transform',
 'coef','intercept','decision','proba','predict','covariance','ranks','score']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
 ap.add_argument('--executable',type=Path);args=ap.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_discriminant_svd_driver')
 fixtures=[];checks=[];certificates=[];undefined=[];large=[]
 for seed in [0,42,2026]:
  rng=np.random.RandomState(seed)
  for n,p in [(30,4),(9,16),(24,1),(18,6)]:
   for classes in [2,3,4]:
    variants=['plain','singleton','unequal_scale']+(['rank_deficient','rank_one'] if p>=3 else [])
    for kind in variants:
     y=np.arange(n)%classes
     if kind=='singleton':y[(y==0)&(np.arange(n)!=0)]=1
     y=y*3-7
     x=rng.normal(size=(n,p))+.17*y[:,None]*np.linspace(.3,1.4,p)
     q=rng.normal(size=(4,p))+.2
     if kind=='unequal_scale':
      scale=np.geomspace(.001,1000,p);x*=scale;q*=scale
     if kind=='rank_deficient':
      x[:,-1]=x[:,0];q[:,-1]=q[:,0];x[:,1]=2;q[:,1]=2
     if kind=='rank_one':
      x=x[:,:1]*np.arange(1,p+1);q=q[:,:1]*np.arange(1,p+1)
     for tol in [1e-4,.3]:
      for k in [0,1]:fixtures.append((seed,x,y,q,k,tol,kind,int(len(fixtures)%3!=0)))
 for classes in [2,3,4]:
  cloud=np.vstack([np.eye(3),-np.eye(3)])
  x=np.tile(cloud,(classes,1));y=np.repeat(np.arange(classes)*3-7,6)
  fixtures.append((0,x,y,cloud[:2],0,1e-4,'coincident_means',1))
 rng=np.random.RandomState(123);x=rng.normal(size=(12,20000));y=np.arange(12)%3
 fixtures.append((123,x,y,rng.normal(size=(2,20000)),0,1e-4,'large_wide',0))
 for index,(seed,x,y,q,k,tol,kind,store) in enumerate(fixtures):
  n,p=x.shape;m=len(q);model=LinearDiscriminantAnalysis(solver='svd',tol=tol,
    n_components=None if k==0 else k,store_covariance=bool(store))
  with warnings.catch_warnings(record=True),np.errstate(invalid='ignore'):
   z=model.fit_transform(x,y) if index%2==0 else model.fit(x,y).transform(x)
  payload=f'{n} {p} {m} {k} {index%2} {store} {tol:.17g}\n'
  payload+=' '.join(f'{v:.17g}' for v in x.ravel(order='F'))+'\n'+' '.join(map(str,y))+'\n'
  payload+=' '.join(f'{v:.17g}' for v in q.ravel(order='F'))+'\n'
  run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True,check=True,cwd=ROOT)
  lines=iter(run.stdout.splitlines());actual=[block(lines) for _ in NAMES];assert next(lines,None) is None
  classes=model.classes_;r=model.scalings_.shape[1]
  assert actual[5].shape==model.scalings_.shape
  signs=np.sign(np.sum(actual[5]*model.scalings_,axis=0));signs[signs==0]=1
  centered=np.concatenate([x[y==c]-model.means_[j] for j,c in enumerate(classes)])
  std=np.std(centered,axis=0);std[std==0]=1
  within_rank=int(np.sum(svdvals(centered/std/np.sqrt(n-len(classes)))>tol))
  coef=model.coef_;intercept=model.intercept_;decision=model.decision_function(q)
  if len(classes)==2:
   coef=np.vstack([np.zeros(p),coef[0]]);intercept=np.array([0.,intercept[0]])
   decision=np.column_stack([np.zeros(m),decision])
   actual[9]=actual[9]-actual[9][:1];actual[10]=actual[10]-actual[10][0];actual[11]=actual[11]-actual[11][:,:1]
  expected=[classes,model.means_,model.priors_,model.xbar_,model.explained_variance_ratio_,model.scalings_*signs,
    z*signs[:z.shape[1]],model.transform(x)*signs[:z.shape[1]],model.transform(q)*signs[:z.shape[1]],
    coef,intercept,decision,model.predict_proba(q),model.predict(q),model.covariance_ if store else np.empty((0,0)),
    np.array([z.shape[1],within_rank,r]),np.array([model.score(x,y)])]
  for name,got,want in zip(NAMES,actual,expected):
   np.testing.assert_allclose(got,want,atol=3e-7,rtol=3e-7,err_msg=f'{index} {kind} {name}')
   checks.append(dict(fixture=index,seed=seed,kind=kind,name=name,passed=True))
  if np.any(np.isnan(model.explained_variance_ratio_)):
   assert kind=='coincident_means' and np.all(model.means_==0) and np.all(np.isnan(actual[4]))
   undefined.append(dict(fixture=index,source_undefined=True,native_nan_convention=True,passed=True))
  scaling=actual[5];projected=centered@scaling
  white=projected.T@projected/(n-len(classes))
  residual=float(np.max(np.abs(white-np.eye(r)),initial=0.))
  bound=2e-7;assert residual<=bound,(index,residual)
  means=(model.means_-model.xbar_)@scaling
  weighted=means*np.sqrt(n*model.priors_/(len(classes)-1))[:,None]
  between=weighted.T@weighted
  offdiag=float(np.max(np.abs(between-np.diag(np.diag(between))),initial=0.))
  between_bound=2e-7*max(1.,float(np.max(np.abs(between),initial=0.)))
  assert offdiag<=between_bound,(index,offdiag,between_bound)
  certificates.append(dict(fixture=index,whitening_error=residual,whitening_bound=bound,
    between_offdiag=offdiag,between_bound=between_bound,passed=True))
  if kind=='large_wide':
   assert actual[14].shape==(0,0) and store==0
   large.append(dict(fixture=index,n=n,p=p,source_covariance_absent=not hasattr(model,'covariance_'),
     native_covariance_absent=True,one_dense_covariance_bytes=p*p*8,passed=True))
  if (index+1)%36==0:print(index+1,'SVD discriminant fixtures passed',flush=True)
 source_empty_rank_errors=[]
 for name,data,threshold in [('zero',np.zeros((6,3)),1e-4),('tol',np.arange(18).reshape(6,3),1e9)]:
  try:
   LinearDiscriminantAnalysis(solver='svd',tol=threshold).fit(data,[1,1,1,2,2,2])
  except IndexError as error:
   assert 'index 0' in str(error)
   source_empty_rank_errors.append(dict(case=name,source_exception='IndexError',native_diagnostic=True,passed=True))
  else:
   raise AssertionError('source empty-rank behavior changed')
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();results=ROOT/'validation/results';results.mkdir(exist_ok=True)
 report=dict(profile=args.profile,sklearn_version=sklearn.__version__,fixtures=len(fixtures),checks=checks,
  source_empty_rank_errors=source_empty_rank_errors,rank_certificates=certificates,undefined_ratio_certificates=undefined,large_wide_cases=large,native_sha256=sha(exe),
  validator_sha256=sha(Path(__file__)),source_sha256={'discriminant_analysis.py':sha(Path(sklearn.__file__).parent/'discriminant_analysis.py')})
 (results/f'discriminant_svd_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejects=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True,cwd=ROOT)
  assert run.returncode!=0 and message in run.stdout+run.stderr,(case,run.stdout,run.stderr)
  rejects.append(dict(case=case,passed=True))
 report.update(checks=rejects)
 for field in ['fixtures','rank_certificates','undefined_ratio_certificates','large_wide_cases','source_empty_rank_errors']:report.pop(field)
 (results/f'discriminant_svd_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(len(checks),'groups,',len(fixtures),'rank certificates,',len(undefined),'undefined ratios and',len(rejects),'rejections PASS',flush=True)

if __name__=='__main__':main()
