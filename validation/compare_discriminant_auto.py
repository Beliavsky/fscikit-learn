"""Standardized classwise Ledoit-Wolf LDA against sklearn and covariance identities."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import mpmath as mp
from scipy.linalg import eigh,pinv
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import ledoit_wolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block
REJECTIONS={'mode':'unknown LDA shrinkage_mode','solver':'unknown LDA solver','svd':'does not support automatic shrinkage',
 'tol':'LDA tol must be nonnegative','tol_nan':'LDA controls must be finite','shrink_nan':'LDA controls must be finite',
 'shrink_range':'invalid LDA shrinkage','components':'n_components must be nonnegative',
 'components_high':'n_components exceeds class/feature limit','one_class':'requires at least two classes',
 'samples':'needs more samples than classes','eigen_zero':'matrix is not positive definite',
 'eigen_rank_one':'matrix is not positive definite','nan_x':'X contains NaN or infinity',
 'labels':'X and y sample counts differ','unfitted':'LDA is not fitted','width':'X and y sample counts differ',
 'nan_query':'X contains NaN or infinity','empty_query':'X must be nonempty','lsqr_transform':'LDA transform requires SVD solver'}
COMMON=['classes','means','priors','covariance','coef','intercept','decision','proba','predict','score',
 'class_shrinkage','total_shrinkage','precision','logdet']
EXTRA=['eigenvalues','variance_ratio','scalings','fit','train_transform','query_transform','components']

def covariance(x):
 scaler=StandardScaler();z=scaler.fit_transform(x);cov,alpha=ledoit_wolf(z)
 return cov*scaler.scale_[:,None]*scaler.scale_[None,:],alpha

def stable_reference(x,means,priors,within):
 # 80-digit evaluation of the source affine model, with common terms removed.
 with mp.workdps(80):
  cov=mp.matrix(within.tolist());precision=cov**-1
  base=mp.matrix(means[0].tolist());rows=[]
  for row in x:
   query=mp.matrix(row.tolist());scores=[mp.mpf(0)]
   for c in range(1,len(means)):
    delta=mp.matrix(means[c].tolist())-base
    value=(delta.T*precision*(query-base-delta/2))[0]+mp.log(mp.mpf(priors[c])/mp.mpf(priors[0]))
    scores.append(value)
   rows.append([float(value) for value in scores])
 return np.array(rows)

def softmax(scores):
 values=np.exp(scores-scores.max(axis=1,keepdims=True));return values/values.sum(axis=1,keepdims=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
 ap.add_argument('--executable',type=Path);args=ap.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_discriminant_auto_driver')
 fixtures=[];checks=[];certificates=[];basis_certificates=[];undefined=[];singular=[];indefinite=[];roundoff=[];spectral_roundoff=[];isotropic=[]
 for seed in [0,42,2026]:
  rng=np.random.RandomState(seed)
  for n,p in [(30,4),(10,9),(24,1)]:
   for count in [2,3,4]:
    variants=['plain','singleton','scaled']+(['constant','near_constant'] if p>=3 else [])
    for kind in variants:
     y=np.arange(n)%count
     if kind=='singleton':y[(y==0)&(np.arange(n)!=0)]=1
     y=y*3-7;x=rng.normal(size=(n,p))+.17*y[:,None]*np.linspace(.3,1.4,p);q=rng.normal(size=(4,p))+.2
     if kind=='scaled':scale=np.geomspace(.01,10,p);x*=scale;q*=scale
     if kind=='constant':x[:,-1]=2;q[:,-1]=2
     if kind=='near_constant':x[:,-1]=100+x[:,-1]*1e-13;q[:,-1]=100+q[:,-1]*1e-13
     for solver,k in [(0,0),(1,0),(1,1)]:fixtures.append((seed,x,y,q,solver,k,kind))
 for value in [0.,2.]:
  x=np.full((10,4),value);y=np.arange(10)%3*3-7
  fixtures.append((0,x,y,x[:2],0,0,'zero_covariance'))
 x=np.array([[-1.,-2,-3],[1,2,3],[0,0,0],[2,4,6]]);y=np.array([-7,-7,-4,-4])
 fixtures.append((0,x,y,x+.13,0,0,'singular_covariance'))
 cloud=np.array([[-1.],[1.]]);x=np.tile(cloud,(2,1));y=np.repeat([-7,-4],2)
 fixtures.append((0,x,y,cloud,1,0,'zero_between'))
 cloud=np.vstack([np.eye(3),-np.eye(3)]);x=np.concatenate([cloud-1,cloud,cloud+1]);y=np.repeat([-7,-4,-1],6)
 for solver in [0,1]:fixtures.append((0,x,y,x[:3]+.1,solver,0,'repeated_basis'))
 for index,(seed,x,y,q,solver,k,kind) in enumerate(fixtures):
  n,p=x.shape;m=len(q);names=COMMON+(EXTRA if solver else [])
  model=LinearDiscriminantAnalysis(solver='eigen' if solver else 'lsqr',shrinkage='auto',n_components=None if k==0 else k)
  with warnings.catch_warnings(record=True),np.errstate(invalid='ignore',divide='ignore'):
   if solver and index%2==0:z=model.fit_transform(x,y)
   else:
    model.fit(x,y)
    if solver:z=model.transform(x)
   class_cov=[];alphas=[]
   for label in model.classes_:
    cov,alpha=covariance(x[y==label]);class_cov.append(cov);alphas.append(alpha)
   within=sum(prob*cov for prob,cov in zip(model.priors_,class_cov))
   if solver:total,total_alpha=covariance(x)
   else:total_alpha=-1.
  np.testing.assert_allclose(within,model.covariance_,atol=1e-12,rtol=1e-12)
  payload=f'{n} {p} {m} {solver} {k} {index%2}\n'
  payload+=' '.join(f'{v:.17g}' for v in x.ravel(order='F'))+'\n'+' '.join(map(str,y))+'\n'
  payload+=' '.join(f'{v:.17g}' for v in q.ravel(order='F'))+'\n'
  run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True,check=True,cwd=ROOT)
  lines=iter(run.stdout.splitlines());actual=[block(lines) for _ in names];assert next(lines,None) is None
  coef=model.coef_;intercept=model.intercept_;decision=model.decision_function(q)
  if len(model.classes_)==2:
   coef=np.vstack([np.zeros(p),coef[0]]);intercept=np.array([0.,intercept[0]])
   decision=np.column_stack([np.zeros(m),decision])
   actual[4]=actual[4]-actual[4][:1];actual[5]=actual[5]-actual[5][0];actual[6]=actual[6]-actual[6][:,:1]
  if len(model.classes_)>2:decision=decision-decision[:,:1]
  sv=np.linalg.svd(within,compute_uv=False);rank=int(np.sum(sv>np.finfo(float).eps*np.max(sv)))
  logdet=np.linalg.slogdet(within)[1] if solver or rank==p else -np.inf
  expected=[model.classes_,model.means_,model.priors_,within,coef,intercept,decision,model.predict_proba(q),
    model.predict(q),np.array([model.score(x,y)]),np.array(alphas),np.array([total_alpha]),
    np.linalg.inv(within) if solver else pinv(within,rtol=np.finfo(float).eps),np.repeat(logdet,len(model.classes_))]
  if solver:
   between=total-within;values,unused=eigh(between,within);values=values[np.argsort(values)[::-1]]
   basis=actual[16];source=model.scalings_;rotation=np.zeros((p,p));groups=[];start=0
   gap=1e-8*max(1.,float(np.max(np.abs(values))))
   while start<p:
    stop=start+1
    while stop<p and abs(values[stop]-values[start])<=gap:stop+=1
    cross=basis[:,start:stop].T@within@source[:,start:stop];u,s,vt=np.linalg.svd(cross)
    rotation[start:stop,start:stop]=u@vt
    if stop-start>1:groups.append([start,stop])
    start=stop
   expected_basis=source@rotation.T
   span_error=float(np.max(np.abs(basis-expected_basis)));span_bound=3e-7*max(1.,float(np.max(np.abs(source))))
   assert span_error<=span_bound,(index,span_error,span_bound)
   if groups:basis_certificates.append(dict(fixture=index,groups=groups,span_error=span_error,span_bound=span_bound,
     crosses_component_cutoff=any(a<z.shape[1]<b for a,b in groups),passed=True))
   expected +=[values,model.explained_variance_ratio_,expected_basis,(x@expected_basis)[:,:z.shape[1]],
     (x@expected_basis)[:,:z.shape[1]],(q@expected_basis)[:,:z.shape[1]],np.array([z.shape[1]])]
   white_error=float(np.max(np.abs(basis.T@within@basis-np.eye(p))))
   residual=between@basis-within@basis*actual[14]
   eigen_error=float(np.max(np.abs(residual)));eigen_bound=2e-8*max(1.,np.linalg.norm(between,np.inf)*np.linalg.norm(basis,np.inf),
     np.linalg.norm(within,np.inf)*np.linalg.norm(basis*actual[14],np.inf))
   assert white_error<=2e-8 and eigen_error<=eigen_bound
   if np.any(values < -1e-9):indefinite.append(dict(fixture=index,minimum_eigenvalue=float(values.min()),passed=True))
   if np.any(np.isnan(model.explained_variance_ratio_)):
    assert kind=='zero_between' and np.all(between==0) and np.all(np.isnan(actual[15]))
    undefined.append(dict(fixture=index,source_undefined=True,native_nan_convention=True,passed=True))
  else:white_error=0.;eigen_error=0.;eigen_bound=0.
  if solver:
   # Large common terms can destroy every significant digit in source logits.
   # Keep the original fixture and certify the stable mathematical classifier.
   raw_coef=model.means_@model.scalings_@model.scalings_.T
   raw_intercept=-.5*np.sum(model.means_*raw_coef,axis=1)+np.log(model.priors_)
   diagonal=np.sqrt(np.diag(within));normalized=within/diagonal[:,None]/diagonal[None,:]
   condition=float(np.linalg.cond(normalized))
   scale=np.abs(q)@np.abs(raw_coef.T)+np.abs(raw_intercept)
   budget=256*np.finfo(float).eps*max(1.,condition)*(scale+scale[:,:1]+1)
   if np.max(budget)>1e-6:
    assert kind=='near_constant' and condition<1e8
    np.testing.assert_array_equal(actual[1],model.means_)
    normalized_error=float(np.max(np.abs(actual[3]-within)/diagonal[:,None]/diagonal[None,:]))
    assert normalized_error<=3e-7
    precise=stable_reference(q,model.means_,model.priors_,within)
    train_precise=stable_reference(x,model.means_,model.priors_,within)
    discrepancy=float(np.max(np.abs(precise-decision)))
    assert np.all(np.abs(precise-decision)<=budget)
    expected[6]=precise;expected[7]=softmax(precise)
    expected[8]=model.classes_[np.argmax(precise,axis=1)]
    expected[9]=np.array([np.mean(model.classes_[np.argmax(train_precise,axis=1)]==y)])
    native_error=float(np.max(np.abs(actual[6]-precise)))
    native_bound=float(np.max(3e-7+3e-7*np.abs(precise)))
    assert np.all(np.abs(actual[6]-precise)<=3e-7+3e-7*np.abs(precise))
    roundoff.append(dict(fixture=index,normalized_condition=condition,normalized_covariance_error=normalized_error,
      native_margin_error=native_error,native_margin_bound=native_bound,high_precision_predictions_checked=True,
      source_error=discrepancy,source_bound=float(np.max(budget)),high_precision_digits=80,passed=True))
  for c,(got,want) in enumerate(zip(actual[10],alphas)):
   if not np.isclose(got,want,atol=3e-7,rtol=3e-7):
    data=x[y==model.classes_[c]];standard=StandardScaler();normalized=standard.fit_transform(data)
    normalized=normalized-normalized.mean(axis=0)
    empirical=normalized.T@normalized/len(data);mu=np.trace(empirical)/p
    error=float(np.linalg.norm(empirical-mu*np.eye(p),np.inf))
    bound=128*np.finfo(float).eps*p*max(1.,np.linalg.norm(empirical,np.inf))
    assert error<=bound and 0<=got<=1 and 0<=want<=1
    isotropic.append(dict(fixture=index,class_index=c,native=float(got),source=float(want),
      isotropy_error=error,isotropy_bound=float(bound),covariance_insensitive=True,passed=True))
  spectral_bound=None
  if solver and not np.allclose(actual[14],values,atol=3e-7,rtol=3e-7):
   # Absolute eigenvalue uncertainty scales with the whitened matrix norm.
   # This is essential when a 1e26 leading value obscures order-one roots.
   assert kind=='near_constant' and roundoff and roundoff[-1]['fixture']==index
   lower=np.linalg.cholesky(within);inverse_lower=np.linalg.inv(lower)
   whitened=inverse_lower@between@inverse_lower.T
   spectral_bound=float(128*np.finfo(float).eps*p*max(1.,np.linalg.norm(whitened,np.inf)))
   spectral_error=float(np.max(np.abs(actual[14]-values)))
   assert spectral_error<=spectral_bound
   metric_error=float(np.max(np.abs(diagonal[:,None]*(basis-expected_basis))))
   assert metric_error<=3e-7
   spectral_roundoff.append(dict(fixture=index,error=spectral_error,bound=spectral_bound,
     metric_basis_error=metric_error,metric_basis_bound=3e-7,projection_identity_checked=True,passed=True))
  for name,got,want in zip(names,actual,expected):
   if name=='class_shrinkage':
    for c,(a,b) in enumerate(zip(got,want)):
     if not np.isclose(a,b,atol=3e-7,rtol=3e-7):
      assert any(t['fixture']==index and t['class_index']==c for t in isotropic)
   elif name=='eigenvalues' and spectral_bound is not None:
    np.testing.assert_allclose(got,want,atol=spectral_bound,rtol=0)
   elif spectral_bound is not None and name=='scalings':
    np.testing.assert_allclose(diagonal[:,None]*got,diagonal[:,None]*want,atol=3e-7,rtol=3e-7)
   elif spectral_bound is not None and name in ['fit','train_transform','query_transform']:
    data=q if name=='query_transform' else x
    # The source small eigenspace is unresolved; check the API against the
    # retained, source-certified native basis, without deleting any rows.
    np.testing.assert_allclose(got,(data@basis)[:,:z.shape[1]],atol=3e-7,rtol=3e-7)
   else:
    np.testing.assert_allclose(got,want,atol=3e-7,rtol=3e-7,err_msg=f'{index} {kind} {solver} {name}')
   checks.append(dict(fixture=index,seed=seed,kind=kind,solver=solver,name=name,passed=True))
  error=float(np.max(np.abs(within@actual[12]@within-within)));bound=2e-8*max(1.,np.linalg.norm(within,np.inf))
  assert error<=bound,(index,error,bound)
  certificates.append(dict(fixture=index,pseudoinverse_error=error,pseudoinverse_bound=float(bound),
    whitening_error=white_error,whitening_bound=2e-8,eigen_error=eigen_error,eigen_bound=float(eigen_bound),passed=True))
  if rank<p and not solver:singular.append(dict(fixture=index,rank=rank,p=p,solver=solver,passed=True))
  if (index+1)%36==0:print(index+1,'automatic discriminant fixtures passed',flush=True)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();results=ROOT/'validation/results';results.mkdir(exist_ok=True)
 files=['discriminant_analysis.py','covariance/_shrunk_covariance.py','preprocessing/_data.py','utils/extmath.py']
 report=dict(profile=args.profile,sklearn_version=sklearn.__version__,fixtures=len(fixtures),checks=checks,
  isotropic_shrinkage_certificates=isotropic,spectral_roundoff_certificates=spectral_roundoff,roundoff_certificates=roundoff,covariance_certificates=certificates,basis_certificates=basis_certificates,undefined_ratio_certificates=undefined,
  singular_cases=singular,indefinite_between_cases=indefinite,native_sha256=sha(exe),validator_sha256=sha(Path(__file__)),
  source_sha256={rel:sha(Path(sklearn.__file__).parent/rel) for rel in files})
 (results/f'discriminant_auto_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejects=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True,cwd=ROOT)
  assert run.returncode!=0 and message in run.stdout+run.stderr,(case,run.stdout,run.stderr)
  rejects.append(dict(case=case,passed=True))
 report.update(checks=rejects)
 for field in ['fixtures','isotropic_shrinkage_certificates','spectral_roundoff_certificates','roundoff_certificates','covariance_certificates','basis_certificates','undefined_ratio_certificates','singular_cases','indefinite_between_cases']:report.pop(field)
 (results/f'discriminant_auto_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(len(checks),'groups,',len(fixtures),'covariance certificates,',len(basis_certificates),'basis certificates,',
   len(singular),'singular cases,',len(indefinite),'indefinite between cases and',len(rejects),'rejections PASS',flush=True)

if __name__=='__main__':main()
