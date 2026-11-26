"""Custom, zero and nonunit class priors for LDA and QDA against sklearn."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings,copy
import numpy as np
from scipy.linalg import eigh,lstsq
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis,_cov
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block
REJECTIONS={}
for family in ['lda']:
 for case,message in [('length','must match class count'),('empty','must match class count'),
  ('negative','must be nonnegative'),('nan','must be finite'),('inf','must be finite'),('zero','need positive total weight')]:
  REJECTIONS[family+'_'+case]=message
REJECTIONS.update(lda_overflow='prior sum exceeds numeric range',lda_samples='need more samples than classes',
 eigen_singleton='matrix is not positive definite',svd_zero='no retained within-class directions')
# Former QDA rejection fixtures are retained as actual Source fit acceptances.
ACCEPTED_FITS=['qda_'+n for n in ['length','empty','negative','nan','inf','zero','wide_zero']]
COMMON=['classes','means','priors','input','normalized','covariance','decision','proba','log_proba','predict','score']
EXTRA=['coef','intercept','reference']
PROJECT=['scalings','ratio','fit','train','query','components','location_or_eigenvalues']

def full_lda_parameters(model,xbar):
 if model.solver=='svd':
  latent=(model.means_-xbar)@model.scalings_;coef=latent@model.scalings_.T
  intercept=-.5*np.sum(latent*latent,axis=1)+np.log(model.priors_)-xbar@coef.T
 elif model.solver=='eigen':
  coef=model.means_@model.scalings_@model.scalings_.T
  intercept=-.5*np.sum(model.means_*coef,axis=1)+np.log(model.priors_)
 else:
  coef=lstsq(model.covariance_,model.means_.T)[0].T
  intercept=-.5*np.sum(model.means_*coef,axis=1)+np.log(model.priors_)
 return coef,intercept

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
 ap.add_argument('--executable',type=Path);args=ap.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_discriminant_priors_driver')
 fixtures=[];checks=[];zero=[];normalization=[];basis_certificates=[];undefined=[];qda_thin=[]
 for seed in [0,42,2026]:
  rng=np.random.RandomState(seed)
  for n,p in [(30,3),(24,1),(10,7)]:
   for count in [2,3]:
    y=np.arange(n)%count*3-7;x=rng.normal(size=(n,p))+.17*y[:,None]*np.linspace(.3,1.4,p)
    q=rng.normal(size=(4,p))+.2;skew=np.arange(1,count+1,dtype=float);skew/=skew.sum()
    first_zero=skew.copy();first_zero[0]=0;first_zero/=first_zero.sum()
    one=np.zeros(count);one[-1]=1
    options=[('uniform',np.full(count,1/count)),('skew',skew),('scaled',skew*3.7),('near_sum',skew*1.000005),
             ('zero_first',first_zero),('one_hot',one),('tiny',skew*1e-100),('large',skew*1e100)]
    for kind,priors in options:
     for mode in range(6 if n>p+count else 5):fixtures.append((seed,x,y,q,priors,mode,kind))
 rng=np.random.RandomState(51);x=rng.normal(size=(30,3));y=np.arange(30)%3*3-7
 fixtures.append((51,x,y,x[:2]+.1,np.full(3,1e308),5,'qda_huge'))
 for index,(seed,x,y,q,priors,mode,kind) in enumerate(fixtures):
  n,p=x.shape;m=len(q);count=len(priors);components=index%2
  names=COMMON+([] if mode==5 else EXTRA)+(PROJECT if mode in [1,2,4] else [])
  normalized=mode!=5 and abs(priors.sum()-1)>1e-5
  if mode==5:model=QuadraticDiscriminantAnalysis(reg_param=.2,priors=priors.copy(),store_covariance=True)
  else:
   solver=['lsqr','eigen','svd','lsqr','eigen'][mode]
   model=LinearDiscriminantAnalysis(solver=solver,shrinkage=None if mode==2 else ('auto' if mode>=3 else .2),
     priors=priors.copy(),store_covariance=True,n_components=None if components==0 else components)
  with warnings.catch_warnings(record=True) as caught,np.errstate(divide='ignore',invalid='ignore'):
   if mode in [1,2,4] and index%2==0:z=model.fit_transform(x,y)
   else:
    model.fit(x,y)
    if mode in [1,2,4]:z=model.transform(x)
   assert any('Renormalizing' in str(w.message) for w in caught)==normalized
   probability=model.predict_proba(q);logprobability=model.predict_log_proba(q);prediction=model.predict(q);score=model.score(x,y)
   if mode==5:decision=model._decision_function(q)
   else:decision=model.decision_function(q)
  payload=f'{n} {p} {m} {count} {mode} {components} {index%2}\n'
  payload+=' '.join(f'{v:.17g}' for v in x.ravel(order='F'))+'\n'+' '.join(map(str,y))+'\n'
  payload+=' '.join(f'{v:.17g}' for v in q.ravel(order='F'))+'\n'+' '.join(f'{v:.17g}' for v in priors)+'\n'
  run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True,check=True,cwd=ROOT)
  lines=iter(run.stdout.splitlines());a={name:block(lines) for name in names};assert next(lines,None) is None
  if mode==5:
   if min(np.sum(y==c) for c in model.classes_)<p:
    aligned=copy.deepcopy(model)
    for c,label in enumerate(model.classes_):
     data=x[y==label]-model.means_[c];nc=len(data);r=min(nc,p)
     sample=data.T@data/(nc-1);native=a['covariance'][:,:,c]
     projector=(native-.8*sample)/.2
     identity_error=float(np.linalg.norm(projector@projector-projector,np.inf))
     data_error=float(np.linalg.norm(data@projector-data,np.inf))
     rank_error=abs(float(np.trace(projector))-r)
     assert identity_error<=3e-7 and data_error<=3e-7*max(1.,np.linalg.norm(data,np.inf)) and rank_error<=3e-7
     vals,vecs=np.linalg.eigh(native);order=np.argsort(vals)[::-1][:r]
     np.testing.assert_allclose(vals[order],model.scalings_[c],atol=3e-7,rtol=3e-7)
     # Data-informed components and regularized spectrum agree. The retained
     # zero-singular direction is arbitrary within the centered-data nullspace.
     aligned.rotations_[c]=vecs[:,order];aligned.scalings_[c]=model.scalings_[c]
     qda_thin.append(dict(fixture=index,class_index=c,nc=nc,p=p,rank=r,projector_error=identity_error,
       training_span_error=data_error,training_span_bound=float(3e-7*max(1.,np.linalg.norm(data,np.inf))),
       rank_error=rank_error,source_spectrum_checked=True,passed=True))
    with np.errstate(divide='ignore',invalid='ignore'):
     decision=aligned._decision_function(q);probability=aligned.predict_proba(q);logprobability=aligned.predict_log_proba(q)
     prediction=aligned.predict(q);score=aligned.score(x,y)
    cov=np.stack([rot@np.diag(scales)@rot.T for rot,scales in zip(aligned.rotations_,aligned.scalings_)],axis=2)
   else:cov=np.stack(model.covariance_,axis=2)
  else:cov=np.repeat(model.covariance_[:,:,None],count,axis=2)
  expected=dict(classes=model.classes_,means=model.means_,priors=model.priors_,input=priors,
    normalized=np.array([int(normalized)]),covariance=cov,decision=decision,proba=probability,log_proba=logprobability,predict=prediction,score=np.array([score]))
  if mode!=5:
   with np.errstate(divide='ignore',invalid='ignore'):
    fullcoef,fullintercept=full_lda_parameters(model,model.xbar_ if mode==2 else None)
   expected.update(coef=model.coef_,intercept=model.intercept_,reference=np.array([np.flatnonzero(model.priors_>0)[0]+1 if mode>=3 else 0]))
   if count==2:
    a['coef']=(a['coef'][1]-a['coef'][0])[None,:];a['intercept']=np.array([a['intercept'][1]-a['intercept'][0]])
   if mode>=3:
    base=np.flatnonzero(model.priors_>0)[0]
    with np.errstate(invalid='ignore'):raw=q@fullcoef.T+fullintercept
    expected['decision']=raw-raw[:,base,None]
   elif count==2:a['decision']=a['decision'][:,1]-a['decision'][:,0]
  if mode in [1,2,4]:
   source=model.scalings_;native=a['scalings'];assert source.shape==native.shape
   if mode==2:
    signs=np.sign(np.sum(source*native,axis=0));signs[signs==0]=1
    expected_basis=source*signs;extra=model.xbar_
   else:
    total=_cov(x,'auto' if mode==4 else .2);within=model.covariance_;between=total-within
    values,unused=eigh(between,within);values=values[np.argsort(values)[::-1]];rotation=np.zeros((p,p));groups=[];start=0
    gap=1e-8*max(1.,float(np.max(np.abs(values))))
    while start<p:
     stop=start+1
     while stop<p and abs(values[stop]-values[start])<=gap:stop+=1
     cross=native[:,start:stop].T@within@source[:,start:stop];u,s,vt=np.linalg.svd(cross)
     rotation[start:stop,start:stop]=u@vt
     if stop-start>1:groups.append([start,stop])
     start=stop
    expected_basis=source@rotation.T;extra=values
    error=float(np.max(np.abs(native-expected_basis)));bound=3e-7*max(1.,float(np.max(np.abs(source))))
    assert error<=bound
    if groups:basis_certificates.append(dict(fixture=index,error=error,bound=bound,passed=True))
   with np.errstate(invalid='ignore'):
    ratio=model.explained_variance_ratio_
   train=(x-(model.xbar_ if mode==2 else 0))@expected_basis
   query=(q-(model.xbar_ if mode==2 else 0))@expected_basis
   width=z.shape[1]
   expected.update(scalings=expected_basis,ratio=ratio,fit=train[:,:width],train=train[:,:width],query=query[:,:width],
     components=np.array([width]),location_or_eigenvalues=extra)
   if np.any(np.isnan(ratio)):
    assert mode==2 and np.count_nonzero(model.priors_)==1 and width==0 and np.all(np.isnan(a['ratio']))
    undefined.append(dict(fixture=index,source_undefined=True,native_nan_convention=True,passed=True))
  for name,want in expected.items():
   np.testing.assert_allclose(a[name],want,atol=5e-7,rtol=5e-7,err_msg=f'{index} {mode} {kind} {name}')
   checks.append(dict(fixture=index,seed=seed,mode=mode,kind=kind,name=name,passed=True))
  np.testing.assert_array_equal(a['input'],priors)
  normalization.append(dict(fixture=index,normalized=bool(normalized),input_preserved=True,passed=True))
  mask=model.priors_==0
  if np.any(mask):
   assert np.all(a['proba'][:,mask]==0) and not np.any(np.isin(a['predict'],model.classes_[mask]))
   zero.append(dict(fixture=index,excluded_classes=model.classes_[mask].tolist(),zero_posterior=True,passed=True))
  if (index+1)%36==0:print(index+1,'discriminant prior fixtures passed',flush=True)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();results=ROOT/'validation/results';results.mkdir(exist_ok=True)
 report=dict(profile=args.profile,sklearn_version=sklearn.__version__,fixtures=len(fixtures),checks=checks,
  qda_thin_certificates=qda_thin,normalization_certificates=normalization,zero_prior_certificates=zero,basis_certificates=basis_certificates,
  undefined_ratio_certificates=undefined,native_sha256=sha(exe),validator_sha256=sha(Path(__file__)),
  source_sha256={'discriminant_analysis.py':sha(Path(sklearn.__file__).parent/'discriminant_analysis.py')})
 (results/f'discriminant_priors_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejects=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True,cwd=ROOT)
  assert run.returncode!=0 and message in run.stdout+run.stderr,(case,run.stdout,run.stderr)
  rejects.append(dict(case=case,passed=True))
 accepted=[]
 for case in ACCEPTED_FITS:
  labels=np.arange(1,10)%3+1;data=np.c_[np.sin(np.arange(1,10)),np.cos(.7*np.arange(1,10))]
  priors=np.array([.2,.3,.5])
  if case=='qda_length':priors=np.array([.5,.5])
  if case=='qda_empty':priors=np.empty(0)
  if case=='qda_negative':priors[0]=-1
  if case=='qda_nan':priors[0]=np.nan
  if case=='qda_inf':priors[0]=np.inf
  if case=='qda_zero':priors[:]=0
  reg=.2
  if case=='qda_wide_zero':data=np.arange(1,25,dtype=float).reshape(4,6,order='F');labels=np.array([1,1,2,2]);priors=np.array([.5,.5]);reg=0.
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');QuadraticDiscriminantAnalysis(priors=priors,reg_param=reg).fit(data,labels)
  run=subprocess.run([str(exe),case],text=True,capture_output=True,cwd=ROOT)
  assert run.returncode==0,(case,run.stdout,run.stderr)
  accepted.append(dict(case=case,source_fit_accepted=True,native_fit_accepted=True))
 report.update(checks=rejects,source_accepted_fits=accepted)
 for field in ['fixtures','qda_thin_certificates','normalization_certificates','zero_prior_certificates','basis_certificates','undefined_ratio_certificates']:report.pop(field)
 (results/f'discriminant_priors_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(len(checks),'groups,',len(fixtures),'normalization certificates,',len(zero),'zero-prior cases,',
   len(undefined),'undefined ratios,',len(rejects),'rejections and',len(accepted),'Source-accepted fits PASS',flush=True)

if __name__=='__main__':main()
