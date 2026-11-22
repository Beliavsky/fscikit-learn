"""Feature-wise/zero smoothing and dense IEEE boundaries for discrete NB."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT","2")
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,ComplementNB
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block
NAMES=['classes','feature_count','class_count','feature_log_prob','class_log_prior','joint','log_proba','proba','predict','features','score','train_predict']
REJECTIONS={'vector_size':'alpha_per_feature length mismatch','vector_empty':'alpha_per_feature length mismatch',
 'vector_negative':'alpha must be nonnegative','vector_nan':'alpha must be finite','vector_inf':'alpha must be finite',
 'bernoulli_vector':'Bernoulli NB requires scalar alpha','scalar_range':'smoothed count exceeds numeric range',
 'bernoulli_range':'smoothed count exceeds numeric range','complement_range':'smoothed count exceeds numeric range',
 'vector_range':'smoothed count exceeds numeric range','complement_vector_range':'smoothed count exceeds numeric range'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
 ap.add_argument('--executable',type=Path);args=ap.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_nb_smoothing_driver')
 fixtures=[];checks=[];certificates=[];undefined=[];batch_equivalence=[];label_certificates=[];snapshots=0
 for seed in [0,42,2026]:
  rng=np.random.RandomState(seed)
  for n,p in [(9,1),(18,3),(7,8)]:
   for count in [1,2,3]:
    y=np.arange(n)%count*5-2;classes=np.unique(y)
    x=rng.poisson(1.5,size=(n,p)).astype(float);x[y==classes[-1],0]=0
    q=rng.poisson(1.5,size=(3,p)).astype(float);q[0]=0;q[1]=1
    w=rng.randint(0,8,size=n)/4.;w[::4]=0
    for kind,norm in [(0,False),(1,False),(2,False),(2,True)]:
     for config in range(4 if kind==1 else 6):
      for scenario in range(4):
       declared=np.r_[classes,classes[-1]+5] if scenario==1 else classes
       prior=np.zeros(len(declared)) if scenario==3 else None
       alpha=0. if config in [0,1,4,5] else 1e-15
       vector=np.resize([0.,.2,1.],p) if config>=4 else None
       fixtures.append((seed,x,y,q,w,declared,kind,norm,config,scenario,prior,alpha,vector))
 # Default positive smoothing can disappear when source adds it to huge totals.
 x=np.array([[1.,1,1],[1,1,1],[1,1,1],[0,1,1],[0,1,1],[0,1,1]])
 y=np.array([-2,-2,-2,3,3,3]);q=np.array([[1.,1,1],[0,1,1],[1,0,0]])
 for norm in [False,True]:
  fixtures.append((2026,x,y,q,np.full(6,1e20),np.unique(y),2,norm,6,0,None,1.,None))
 for index,(seed,x,y,q,w,declared,kind,norm,config,scenario,prior,alpha,vector) in enumerate(fixtures):
  n,p=x.shape;m=len(q);stream=scenario in [1,2];weighted=scenario in [0,1,2]
  fitprior=scenario!=2;fixed=scenario==3;ends=[1,n//2,n] if stream else [n]
  weights=np.zeros(n) if scenario==2 else w
  threshold=.7 if seed==42 else 0.
  force=config%2==0
  kwargs=dict(alpha=vector if vector is not None else alpha,force_alpha=force,
    fit_prior=fitprior,class_prior=prior if fixed else None)
  if kind==0:model=MultinomialNB(**kwargs)
  elif kind==1:model=BernoulliNB(binarize=threshold,**kwargs)
  else:model=ComplementNB(norm=norm,**kwargs)
  payload=f'{kind} {n} {p} {m} {len(declared)} {int(stream)} {len(ends)} {int(weighted)} {int(fitprior)} {int(fixed)} {int(norm)} {alpha:.17g} {threshold:.17g} {int(force)} {int(vector is not None)}\n'
  for values in [x.ravel(order='F'),y,q.ravel(order='F'),weights,declared[::-1],ends]:
   payload+=' '.join(f'{v:.17g}' for v in values)+'\n'
  if fixed:payload+=' '.join(f'{v:.17g}' for v in prior)+'\n'
  if vector is not None:payload+=' '.join(f'{v:.17g}' for v in vector)+'\n'
  run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True,check=True,cwd=ROOT)
  lines=iter(run.stdout.splitlines());start=0
  for stage,end in enumerate(ends):
   with warnings.catch_warnings(record=True),np.errstate(divide='ignore',invalid='ignore'):
    if stream:
     model.partial_fit(x[start:end],y[start:end],classes=declared[::-1] if stage==0 or stage%2==1 else None,
       sample_weight=weights[start:end] if weighted else None)
    else:model.fit(x,y,sample_weight=weights if weighted else None)
    expected=[model.classes_,model.feature_count_,model.class_count_,model.feature_log_prob_,model.class_log_prior_,
      model._joint_log_likelihood(model._check_X(q)),model.predict_log_proba(q),model.predict_proba(q),model.predict(q),
      np.array([model.n_features_in_]),np.array([model.score(x,y)]),model.predict(x)]
    names=NAMES.copy()
    if kind==2:expected.append(model.feature_all_);names.append('feature_all')
   actual=[block(lines) for _ in names]
   for name,got,want in zip(names,actual,expected):
    if name in ['predict','train_predict']:
     data=q if name=='predict' else x
     with warnings.catch_warnings(record=True),np.errstate(divide='ignore',invalid='ignore'):
      scores=model._joint_log_likelihood(model._check_X(data))
     for row,(a,b) in enumerate(zip(got,want)):
      assert a in model.classes_
      if a==b:continue
      assert not np.isnan(scores[row]).any(),(index,stage,name,row)
      finite=scores[row,np.isfinite(scores[row])]
      assert len(finite),(index,stage,name,row)
      gap=float(np.max(scores[row])-scores[row,np.searchsorted(model.classes_,a)])
      scale=max(1.,float(np.max(np.abs(finite))))
      bound=64*(p+1)*np.finfo(float).eps*scale
      assert np.isfinite(gap) and 0<=gap<=bound,(index,stage,name,row,gap,bound)
      label_certificates.append(dict(fixture=index,stage=stage,name=name,row=row,gap=gap,bound=bound,passed=True))
    elif name=='score':
     np.testing.assert_allclose(got,[np.mean(actual[11]==y)],atol=0,rtol=0)
    else:
     np.testing.assert_allclose(got,want,atol=3e-9,rtol=3e-9,equal_nan=True,err_msg=f'{index} stage={stage} kind={kind} {name}')
    checks.append(dict(fixture=index,stage=stage,seed=seed,kind=kind,name=name,passed=True))
    if np.any(~np.isfinite(want)):
     np.testing.assert_equal(np.isnan(got),np.isnan(want))
     np.testing.assert_equal(np.isposinf(got),np.isposinf(want));np.testing.assert_equal(np.isneginf(got),np.isneginf(want))
     undefined.append(dict(fixture=index,stage=stage,name=name,passed=True))
   events=(x[:end]>threshold).astype(float) if kind==1 else x[:end]
   mass=weights[:end] if weighted else np.ones(end)
   total=events.T@mass
   error=float(np.max(np.abs(actual[1].sum(axis=0)-total)))
   bound=2e-10*max(1.,float(np.max(np.abs(total))))
   assert error<=bound and abs(actual[2].sum()-mass.sum())<=2e-10*max(1.,mass.sum())
   certificates.append(dict(fixture=index,stage=stage,event_total_error=error,event_total_bound=bound,passed=True))
   start=end;snapshots+=1
  assert next(lines,None) is None
  if stream:
   with warnings.catch_warnings(record=True),np.errstate(divide='ignore',invalid='ignore'):
    if kind==0:batch=MultinomialNB(**kwargs)
    elif kind==1:batch=BernoulliNB(binarize=threshold,**kwargs)
    else:batch=ComplementNB(norm=norm,**kwargs)
    batch.partial_fit(x,y,classes=declared,sample_weight=weights if weighted else None)
   np.testing.assert_allclose(model.feature_count_,batch.feature_count_,atol=3e-10,rtol=3e-10)
   np.testing.assert_allclose(model.feature_log_prob_,batch.feature_log_prob_,atol=3e-10,rtol=3e-10)
   np.testing.assert_allclose(model.class_log_prior_,batch.class_log_prior_,atol=3e-10,rtol=3e-10)
   batch_equivalence.append(dict(fixture=index,passed=True))
  if (index+1)%36==0:print(index+1,'NB smoothing fixtures passed',flush=True)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();results=ROOT/'validation/results';results.mkdir(exist_ok=True)
 report=dict(profile=args.profile,sklearn_version=sklearn.__version__,fixtures=len(fixtures),snapshots=snapshots,checks=checks,
  label_certificates=label_certificates,count_certificates=certificates,undefined_certificates=undefined,batch_equivalence=batch_equivalence,native_sha256=sha(exe),
  validator_sha256=sha(Path(__file__)),source_sha256={'naive_bayes.py':sha(Path(sklearn.__file__).parent/'naive_bayes.py')})
 (results/f'nb_smoothing_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejects=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True,cwd=ROOT)
  assert run.returncode!=0 and message in run.stdout+run.stderr,(case,run.stdout,run.stderr)
  rejects.append(dict(case=case,passed=True))
 report.update(checks=rejects)
 for field in ['fixtures','snapshots','label_certificates','count_certificates','undefined_certificates','batch_equivalence']:report.pop(field)
 (results/f'nb_smoothing_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(len(checks),'groups,',len(fixtures),'fixtures,',snapshots,'snapshots,',len(undefined),'undefined fields,',
   len(batch_equivalence),'batch equivalence cases and',len(rejects),'rejections PASS',flush=True)

if __name__=='__main__':main()
