"""Native CSR discrete NB, noncanonical storage and mixed dense/CSR streaming."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT","2")
import numpy as np
import sklearn
import scipy
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,ComplementNB
from compare_function_transformer import ROOT,resolve
from compare_gp_kernels import block
NAMES=['classes','feature_count','class_count','feature_log_prob','class_log_prior','joint','log_proba','proba','predict','features','score','train_predict','dense_joint','dense_logp','dense_proba','dense_predict']
REJECTIONS={'unallocated':'CSR arrays are unallocated','shape':'invalid CSR shape',
 'pointer_size':'CSR pointer length mismatch','pointer_order':'CSR pointers must be nondecreasing',
 'pointer_end':'CSR pointer endpoints are invalid','index_size':'CSR data/index lengths differ',
 'index_range':'CSR column index is out of bounds','empty_rows':'NB sparse X must be nonempty',
 'empty_columns':'NB sparse X must be nonempty','nan_data':'sparse X contains NaN or infinity',
 'inf_data':'sparse X contains NaN or infinity','negative_data':'requires nonnegative data',
 'negative_complement':'requires nonnegative data','bernoulli_nan':'sparse X contains NaN or infinity',
 'first_classes':'first partial_fit requires classes','empty_classes':'labels must be nonempty',
 'unknown_label':'label is absent from declared classes','changed_classes':'classes changed between partial_fit',
 'changed_width':'feature count changed between partial_fit','target_size':'sparse X and y sample counts differ',
 'weight_size':'sample weight length mismatch','negative_weight':'sample weights must be nonnegative',
 'nan_weight':'sample weights must be finite','alpha_size':'alpha_per_feature length mismatch',
 'alpha_negative':'alpha must be nonnegative','threshold_negative':'threshold must be nonnegative',
 'unfitted':'NB is not fitted','query_width':'sparse query feature count mismatch',
 'query_nan':'sparse X contains NaN or infinity','query_negative':'requires nonnegative data',
 'product_range':'weighted count exceeds numeric range'}


def stored(matrix,noncanonical):
 if not noncanonical:return csr_matrix(matrix)
 data=[];columns=[];pointers=[0]
 for row,values in enumerate(matrix):
  entries=[]
  for column,value in enumerate(values):
   if value!=0:entries.extend([(column,value/2),(column,value/2)])
  entries.append((row%len(values),0.))
  entries=entries[::-1]
  for column,value in entries:columns.append(column);data.append(value)
  pointers.append(len(data))
 return csr_matrix((np.array(data),np.array(columns),np.array(pointers)),shape=matrix.shape)


def pack(matrix):
 text=f'{matrix.shape[0]} {matrix.shape[1]} {matrix.nnz}\n'
 for values in [matrix.indptr+1,matrix.indices+1,matrix.data]:
  text+=' '.join(f'{value:.17g}' for value in values)+'\n'
 return text

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
 ap.add_argument('--executable',type=Path);args=ap.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_nb_sparse_driver')
 fixtures=[];checks=[];certificates=[];undefined=[];batch_equivalence=[];label_certificates=[];snapshots=0
 for seed in [0,42,2026]:
  rng=np.random.RandomState(seed)
  for n,p in [(9,1),(12,4),(7,17)]:
   for count in [1,2,3]:
    y=np.arange(n)%count*5-2;classes=np.unique(y)
    base=rng.poisson(1.2,size=(n,p)).astype(float);base[0]=0;base[y==classes[-1],0]=0
    if seed==2026 and n==7 and count==3:base[:]=0
    query=rng.poisson(1.2,size=(3,p)).astype(float);query[0]=0;query[1]=1
    w=rng.randint(0,8,size=n)/4.;w[::3]=0
    for noncanonical in [False,True]:
     x=stored(base,noncanonical);q=stored(query,noncanonical)
     for kind,norm in [(0,False),(1,False),(2,False),(2,True)]:
      for config in range(4):
       for scenario in range(3):
        declared=np.r_[classes,classes[-1]+5] if scenario==1 else classes
        prior=np.zeros(len(declared)) if scenario==2 else None
        alpha=1. if config==0 else (1e-15 if config==2 and kind==1 else 0.)
        vector=np.resize([0.,.2,1.],p) if config==2 and kind!=1 else None
        fixtures.append((seed,x,y,q,w,declared,kind,norm,config,scenario,prior,alpha,vector,noncanonical))
 # Bernoulli keeps signed entries until per-entry thresholding, before duplicate summation.
 x=csr_matrix((np.array([2.,-1,1,.5,.5,-2,-3,2,-4,1]),np.array([0,0,1,0,0,1,0,1,0,1]),
   np.array([0,3,6,8,10])),shape=(4,2))
 q=csr_matrix((np.array([.5,.5,-1.,2,0]),np.array([0,0,0,0,1]),np.array([0,2,5])),shape=(2,2))
 for config in [0,1]:
  fixtures.append((42,x,np.array([-2,-2,3,3]),q,np.ones(4),np.array([-2,3]),1,False,config,0,None,1. if config==0 else 0.,None,True))
 for index,(seed,x,y,q,w,declared,kind,norm,config,scenario,prior,alpha,vector,noncanonical) in enumerate(fixtures):
  n,p=x.shape;m=q.shape[0];stream=scenario in [1,2];weighted=scenario!=2
  fitprior=scenario!=2;fixed=scenario==2;ends=[1,n//2,n] if stream else [n]
  weights=w;threshold=[0.,.7,1.5][[0,42,2026].index(seed)];force=config!=3
  dense_mask=(5 if (index//3)%2==0 else 2) if scenario==2 else 0
  kwargs=dict(alpha=vector if vector is not None else alpha,force_alpha=force,
    fit_prior=fitprior,class_prior=prior if fixed else None)
  if kind==0:model=MultinomialNB(**kwargs)
  elif kind==1:model=BernoulliNB(binarize=threshold,**kwargs)
  else:model=ComplementNB(norm=norm,**kwargs)
  payload=f'{kind} {n} {p} {m} {len(declared)} {int(stream)} {len(ends)} {int(weighted)} {int(fitprior)} {int(fixed)} {int(norm)} {alpha:.17g} {threshold:.17g} {int(force)} {int(vector is not None)} {dense_mask}\n'
  payload+=pack(x)+pack(q)
  for values in [y,weights,declared[::-1],ends]:payload+=' '.join(f'{v:.17g}' for v in values)+'\n'
  if fixed:payload+=' '.join(f'{v:.17g}' for v in prior)+'\n'
  if vector is not None:payload+=' '.join(f'{v:.17g}' for v in vector)+'\n'
  run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True,check=True,cwd=ROOT)
  lines=iter(run.stdout.splitlines());start=0;event_total=np.zeros(p)
  for stage,end in enumerate(ends):
   with warnings.catch_warnings(record=True),np.errstate(divide='ignore',invalid='ignore'):
    if stream:
     training=x[start:end].toarray() if dense_mask&(1<<stage) else x[start:end]
     model.partial_fit(training,y[start:end],classes=declared[::-1] if stage==0 or stage%2==1 else None,
       sample_weight=weights[start:end] if weighted else None)
    else:
     training=x;model.fit(x,y,sample_weight=weights if weighted else None)
    expected=[model.classes_,model.feature_count_,model.class_count_,model.feature_log_prob_,model.class_log_prior_,
      model._joint_log_likelihood(model._check_X(q)),model.predict_log_proba(q),model.predict_proba(q),model.predict(q),
      np.array([model.n_features_in_]),np.array([model.score(x,y)]),model.predict(x),
      model._joint_log_likelihood(model._check_X(q.toarray())),model.predict_log_proba(q.toarray()),
      model.predict_proba(q.toarray()),model.predict(q.toarray())]
    names=NAMES.copy()
    if kind==2:expected.append(model.feature_all_);names.append('feature_all')
   actual=[block(lines) for _ in names]
   for name,got,want in zip(names,actual,expected):
    if name in ['predict','train_predict','dense_predict']:
     data=q if name=='predict' else (q.toarray() if name=='dense_predict' else x)
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
   with warnings.catch_warnings(record=True),np.errstate(divide='ignore',invalid='ignore'):
    events=model._check_X(training) if kind==1 else training
   mass=weights[start:end] if weighted else np.ones(end-start)
   if isinstance(events,np.ndarray):event_total+=events.T@mass
   else:
    for row in range(events.shape[0]):
     begin,stop=events.indptr[row:row+2]
     np.add.at(event_total,events.indices[begin:stop],events.data[begin:stop]*mass[row])
   error=float(np.max(np.abs(actual[1].sum(axis=0)-event_total)))
   bound=2e-10*max(1.,float(np.max(np.abs(event_total))))
   assert error<=bound and abs(actual[2].sum()-(weights[:end].sum() if weighted else end))<=2e-10*max(1.,float(end))
   certificates.append(dict(fixture=index,stage=stage,event_total_error=error,event_total_bound=bound,passed=True))
   start=end;snapshots+=1
  assert next(lines,None) is None
  if stream and dense_mask==0:
   with warnings.catch_warnings(record=True),np.errstate(divide='ignore',invalid='ignore'):
    if kind==0:batch=MultinomialNB(**kwargs)
    elif kind==1:batch=BernoulliNB(binarize=threshold,**kwargs)
    else:batch=ComplementNB(norm=norm,**kwargs)
    batch.partial_fit(x,y,classes=declared,sample_weight=weights if weighted else None)
   np.testing.assert_allclose(model.feature_count_,batch.feature_count_,atol=3e-10,rtol=3e-10)
   np.testing.assert_allclose(model.feature_log_prob_,batch.feature_log_prob_,atol=3e-10,rtol=3e-10)
   np.testing.assert_allclose(model.class_log_prior_,batch.class_log_prior_,atol=3e-10,rtol=3e-10)
   batch_equivalence.append(dict(fixture=index,passed=True))
  if (index+1)%36==0:print(index+1,'CSR NB fixtures passed',flush=True)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();results=ROOT/'validation/results';results.mkdir(exist_ok=True)
 report=dict(profile=args.profile,sklearn_version=sklearn.__version__,scipy_version=scipy.__version__,fixtures=len(fixtures),snapshots=snapshots,checks=checks,
  label_certificates=label_certificates,count_certificates=certificates,undefined_certificates=undefined,batch_equivalence=batch_equivalence,native_sha256=sha(exe),
  validator_sha256=sha(Path(__file__)),source_sha256={rel:sha(Path(sklearn.__file__).parent/rel) for rel in ['naive_bayes.py','preprocessing/_data.py']})
 (results/f'nb_sparse_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejects=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True,cwd=ROOT)
  assert run.returncode!=0 and message in run.stdout+run.stderr,(case,run.stdout,run.stderr)
  rejects.append(dict(case=case,passed=True))
 report.update(checks=rejects)
 for field in ['fixtures','snapshots','label_certificates','count_certificates','undefined_certificates','batch_equivalence']:report.pop(field)
 (results/f'nb_sparse_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(len(checks),'groups,',len(fixtures),'fixtures,',snapshots,'snapshots,',len(undefined),'undefined fields,',
   len(batch_equivalence),'batch equivalence cases and',len(rejects),'rejections PASS',flush=True)

if __name__=='__main__':main()
