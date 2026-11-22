"""Weighted batch/streaming Multinomial, Bernoulli and Complement NB against sklearn."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,ComplementNB
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block
from compare_classifier_chains import certify_complement_labels
NAMES=['classes','feature_count','class_count','feature_log_prob','class_log_prior','joint','log_proba','proba','predict','features','score','train_predict']
REJECTIONS={'first_classes':'first partial_fit requires classes','empty_classes':'labels must be nonempty',
 'changed_classes':'classes changed between partial_fit','changed_width':'feature count changed between partial_fit',
 'unknown_label':'label is absent from declared classes','alpha_zero':'alpha must be strictly positive',
 'alpha_nan':'alpha must be finite','negative_x':'requires nonnegative data','nan_x':'X contains NaN or infinity',
 'negative_weight':'sample weights must be nonnegative','nan_weight':'sample weights must be finite',
 'weight_size':'sample weight length mismatch','prior_size':'class_prior must match class count',
 'prior_negative':'class_prior must be nonnegative','prior_nan':'class_prior must be finite','prior_inf':'class_prior must be finite',
 'prior_zero':'class_prior needs positive weight','threshold_negative':'threshold must be nonnegative',
 'threshold_nan':'threshold must be finite','bernoulli_nan':'X contains NaN or infinity',
 'unfitted':'NB is not fitted','query_width':'X and y sample counts differ','query_negative':'requires nonnegative data',
 'query_nan':'X contains NaN or infinity','overflow_product':'weighted count exceeds numeric range'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
 ap.add_argument('--executable',type=Path);args=ap.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_discrete_stream_driver')
 fixtures=[];checks=[];certificates=[];undefined=[];batch_equivalence=[];label_certificates=[];snapshots=0
 configs=[(0,False,0.),(1,False,0.),(1,False,.7),(1,False,1.5),(2,False,0.),(2,True,0.)]
 for seed in [0,42,2026]:
  rng=np.random.RandomState(seed)
  for n,p in [(18,5),(9,1),(7,9)]:
   for count in [1,2,3]:
    y=np.arange(n)%count*5-2;classes=np.unique(y)
    for kind,norm,threshold in configs:
     x=rng.normal(size=(n,p)) if kind==1 else rng.poisson(2,size=(n,p)).astype(float)
     q=rng.normal(size=(4,p)) if kind==1 else rng.poisson(2,size=(4,p)).astype(float)
     w=rng.uniform(.1,3,size=n);w[::4]=0
     for scenario in range(5):
      declared=np.r_[classes,classes[-1]+5] if scenario==1 else classes.copy()
      prior=np.arange(1,len(declared)+1,dtype=float)
      if scenario==3 and len(prior)>1:prior[0]=0
      fixtures.append((seed,x,y,q,w,declared,kind,norm,threshold,scenario,prior))
 for kind in [0,1,2]:
  for count in [1,2]:
   for fitprior in [False,True]:
    x=np.arange(18).reshape(6,3)%4;y=np.arange(6)%count*5-2;q=x[:2]+.1
    # Extra cases retain all-zero weights with empirical and uniform priors.
    fixtures.append((0,x,y,q,np.zeros(6),np.unique(y),kind,False,0.,5+int(fitprior),None))
 x=np.ones((4,3));y=np.array([-2,-2,3,3]);q=np.array([[0.,0,0],[1.,1,1]])
 fixtures.append((0,x,y,q,np.full(4,1e20),np.unique(y),1,False,0.,7,None))
 x=np.vstack([np.ones((3,3)),[[0.,1,0],[1,0,1],[0,0,1]]]);y=np.array([-2,-2,-2,3,3,3])
 fixtures.append((0,x,y,q,np.full(6,1e20),np.unique(y),1,False,0.,7,None))
 for index,(seed,x,y,q,w,declared,kind,norm,threshold,scenario,prior) in enumerate(fixtures):
  n,p=x.shape;m=len(q);stream=scenario in [1,2,3];weighted=scenario in [0,1,3,5,6,7]
  fitprior=scenario not in [2,4,5];fixed=scenario in [3,4];ends=[1,n//2,n] if stream else [n]
  alpha=[.1,.7,3.][index%3]
  kwargs=dict(alpha=alpha,fit_prior=fitprior,class_prior=prior if fixed else None)
  if kind==0:model=MultinomialNB(**kwargs)
  elif kind==1:model=BernoulliNB(binarize=threshold,**kwargs)
  else:model=ComplementNB(norm=norm,**kwargs)
  payload=f'{kind} {n} {p} {m} {len(declared)} {int(stream)} {len(ends)} {int(weighted)} {int(fitprior)} {int(fixed)} {int(norm)} {alpha:.17g} {threshold:.17g}\n'
  payload+=' '.join(f'{v:.17g}' for v in x.ravel(order='F'))+'\n'+' '.join(map(str,y))+'\n'
  payload+=' '.join(f'{v:.17g}' for v in q.ravel(order='F'))+'\n'+' '.join(f'{v:.17g}' for v in w)+'\n'
  payload+=' '.join(map(str,declared[::-1]))+'\n'+' '.join(map(str,ends))+'\n'
  if fixed:payload+=' '.join(f'{v:.17g}' for v in prior)+'\n'
  run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True,check=True,cwd=ROOT)
  lines=iter(run.stdout.splitlines());start=0
  for stage,end in enumerate(ends):
   with warnings.catch_warnings(record=True),np.errstate(divide='ignore',invalid='ignore'):
    if stream:
     model.partial_fit(x[start:end],y[start:end],classes=declared[::-1] if stage==0 or stage%2==1 else None,
       sample_weight=w[start:end] if weighted else None)
    else:model.fit(x,y,sample_weight=w if weighted else None)
    expected=[model.classes_,model.feature_count_,model.class_count_,model.feature_log_prob_,model.class_log_prior_,
      model._joint_log_likelihood(model._check_X(q)),model.predict_log_proba(q),model.predict_proba(q),model.predict(q),
      np.array([model.n_features_in_]),np.array([model.score(x,y)]),model.predict(x)]
    names=NAMES.copy()
    if kind==2:expected.append(model.feature_all_);names.append('feature_all')
   actual=[block(lines) for _ in names]
   for name,got,want in zip(names,actual,expected):
    if kind==2 and norm and name in ['predict','train_predict']:
     data=q if name=='predict' else x
     with warnings.catch_warnings(record=True),np.errstate(divide='ignore',invalid='ignore'):
      witnesses=certify_complement_labels(got.astype(int),model,data)
     for witness in witnesses:
      label_certificates.append(dict(fixture=index,stage=stage,name=name,**witness,passed=True))
    elif kind==2 and norm and name=='score':
     np.testing.assert_allclose(got,[np.mean(actual[11]==y)],atol=1e-14,rtol=1e-14)
    else:
     np.testing.assert_allclose(got,want,atol=3e-8,rtol=3e-8,err_msg=f'{index} stage={stage} kind={kind} {name}')
    checks.append(dict(fixture=index,stage=stage,seed=seed,kind=kind,name=name,passed=True))
    if np.any(np.isnan(want)):
     assert (np.sum(w[:end])==0 and weighted and fitprior and not fixed) or (kind==2 and norm and p==1) or (kind==1 and np.any(np.exp(model.feature_log_prob_)==1))
     undefined.append(dict(fixture=index,stage=stage,name=name,source_nan=True,native_nan=True,passed=True))
   events=(x[:end]>threshold).astype(float) if kind==1 else x[:end]
   weights=w[:end] if weighted else np.ones(end)
   total=events.T@weights
   error=float(np.max(np.abs(actual[1].sum(axis=0)-total)))
   bound=2e-10*max(1.,float(np.max(np.abs(total))))
   assert error<=bound and abs(actual[2].sum()-weights.sum())<=2e-10*max(1.,weights.sum())
   certificates.append(dict(fixture=index,stage=stage,event_total_error=error,event_total_bound=bound,passed=True))
   start=end;snapshots+=1
  assert next(lines,None) is None
  if stream and np.array_equal(declared,np.unique(y)):
   with warnings.catch_warnings(record=True),np.errstate(divide='ignore',invalid='ignore'):
    if kind==0:batch=MultinomialNB(**kwargs)
    elif kind==1:batch=BernoulliNB(binarize=threshold,**kwargs)
    else:batch=ComplementNB(norm=norm,**kwargs)
    batch.fit(x,y,sample_weight=w if weighted else None)
   np.testing.assert_allclose(model.feature_count_,batch.feature_count_,atol=3e-10,rtol=3e-10)
   np.testing.assert_allclose(model.feature_log_prob_,batch.feature_log_prob_,atol=3e-10,rtol=3e-10)
   np.testing.assert_allclose(model.class_log_prior_,batch.class_log_prior_,atol=3e-10,rtol=3e-10)
   batch_equivalence.append(dict(fixture=index,passed=True))
  if (index+1)%36==0:print(index+1,'discrete NB fixtures passed',flush=True)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();results=ROOT/'validation/results';results.mkdir(exist_ok=True)
 report=dict(profile=args.profile,sklearn_version=sklearn.__version__,fixtures=len(fixtures),snapshots=snapshots,checks=checks,
  label_certificates=label_certificates,count_certificates=certificates,undefined_certificates=undefined,batch_equivalence=batch_equivalence,native_sha256=sha(exe),
  validator_sha256=sha(Path(__file__)),helper_sha256=sha(ROOT/'validation/compare_classifier_chains.py'),source_sha256={'naive_bayes.py':sha(Path(sklearn.__file__).parent/'naive_bayes.py')})
 (results/f'discrete_stream_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejects=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True,cwd=ROOT)
  assert run.returncode!=0 and message in run.stdout+run.stderr,(case,run.stdout,run.stderr)
  rejects.append(dict(case=case,passed=True))
 report.update(checks=rejects)
 for field in ['fixtures','snapshots','label_certificates','count_certificates','undefined_certificates','batch_equivalence']:report.pop(field)
 (results/f'discrete_stream_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(len(checks),'groups,',len(fixtures),'fixtures,',snapshots,'snapshots,',len(undefined),'undefined fields,',
   len(batch_equivalence),'batch equivalence cases and',len(rejects),'rejections PASS',flush=True)

if __name__=='__main__':main()
