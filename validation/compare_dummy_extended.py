"""Weighted, multioutput and seeded Dummy baselines vs installed sklearn."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings,sys
import numpy as np,scipy,sklearn
from scipy.sparse import csr_matrix
from sklearn.dummy import DummyClassifier,DummyRegressor
import sklearn.dummy as source_dummy
import sklearn.utils.stats as source_stats
import sklearn.utils.multiclass as source_distribution
import numpy.random.mtrand as source_random


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True,type=Path)
 parser.add_argument('--reference-cache',type=Path);parser.add_argument('--report',type=Path)
 args=parser.parse_args();root=Path(__file__).resolve().parents[1]
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 provenance=dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,
   files={m.__file__:sha(m.__file__)for m in [source_dummy,source_stats,source_distribution,source_random]})
 cache={};cases=[];calls=hits=warns=0
 if args.reference_cache and args.reference_cache.exists():
  saved=json.loads(args.reference_cache.read_text());assert saved['provenance']==provenance;cache=saved['cases']
 def add(seed,n,p,t,rank,storage,family,strategy,mode,weight_kind,q,binary,y_override=None,weight_override=None):
  nonlocal calls,hits,warns
  rng=np.random.RandomState(seed+17*n+19*t);m=5;qp=2 if p==0 else 0
  x=np.full((n,p),np.nan);query=np.full((m,qp),np.nan)
  xx=csr_matrix(x)if storage else x;qq=csr_matrix(query)if storage else query
  if family==0:
   y=np.column_stack([rng.choice([0,1]if binary else [-7,4,19][:1+(j+n)%3],size=n)for j in range(t)])
   truth=np.column_stack([rng.choice(np.unique(y[:,j]),size=m)for j in range(t)])
   constant=np.array([np.unique(y[:,j])[-1]for j in range(t)])
   score_mode=int(t==1 or binary)
   name=['prior','most_frequent','stratified','uniform','constant'][strategy-1]
   model=DummyClassifier(strategy=name,random_state=seed,constant=constant)
  else:
   y=rng.normal(size=(n,t));y=np.round(y,1);truth=rng.normal(size=(m,t));constant=rng.normal(size=t)
   score_mode=1;name=['mean','median','quantile','constant'][strategy-1]
   model=DummyRegressor(strategy=name,quantile=q,constant=constant)
  if y_override is not None:y=np.asarray(y_override).reshape(n,t)
  w=rng.uniform(.1,3,size=n)
  if weight_kind=='ones':w[:]=1
  elif weight_kind=='zeros':w[:]=0
  elif weight_kind=='signed':w[::2]*=-1
  elif weight_kind=='masked':w[::2]=0
  if mode==2:w[:]=1.75
  if weight_override is not None:w=np.asarray(weight_override)
  qw=np.array([.5,2,1,0,3.])
  target=y[:,0]if rank==0 else y
  weight=None if mode==0 else float(w[0])if mode==2 else w
  block=[f'{n} {p} {m} {qp} {t} {family} {storage} {rank} {mode} {strategy} {seed} {score_mode} {q:.17g}']
  for a in [y,truth]:block+=[' '.join(map(str,row))for row in a]
  for a in [w,qw,constant]:block+=[' '.join(map(str,a))]
  key=hashlib.sha256(('\n'.join(block)+'\n').encode()).hexdigest()
  label=dict(seed=seed,n=n,p=p,t=t,rank=rank,storage=storage,family=family,strategy=name,weight_mode=mode,weights=weight_kind,q=q,binary=binary,
    fixture='signed_ties'if y_override is not None else 'grid')
  if key in cache:expected=cache[key];hits+=1
  else:
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    try:
     model.fit(xx,target,sample_weight=weight)
     pred=model.predict(qq)
     score=model.score(qq,truth[:,0]if rank==0 else truth,sample_weight=qw)if score_mode else None
     if family==0:
      proba=model.predict_proba(qq)
      classes=[model.classes_]if t==1 else model.classes_
      priors=[model.class_prior_]if t==1 else model.class_prior_
      proba=[proba]if t==1 else proba
      expected=dict(meta=[model.n_outputs_,model.n_features_in_,*[len(c)for c in classes]],
        classes=[a.tolist()for a in classes],priors=[a.tolist()for a in priors],
        proba=[a.ravel(order='F').tolist()for a in proba],pred=np.asarray(pred).reshape(m,t).ravel(order='F').tolist(),score=score)
     else:expected=dict(meta=[model.n_outputs_,model.n_features_in_],constants=model.constant_.ravel().tolist(),
       pred=np.asarray(pred).reshape(m,t).ravel(order='F').tolist(),score=score)
    except (TypeError,ValueError,ZeroDivisionError)as error:expected=dict(failure=dict(type=type(error).__name__,message=str(error)))
   expected['warnings']=len(caught);cache[key]=expected;calls+=1
  warns+=expected['warnings'];cases.append((expected,label,block))
 for seed in [0,7,42,2026,2147483647]:
  for n in [1,12,31]:
   for p in [0,4]:
    for rank,t in [(0,1),(1,1),(1,3)]:
     for storage in [0,1]:
      for family in [0,1]:
       for strategy in range(1,6 if family==0 else 5):
        qs=[0.,.25,.5,.99,1.]if family==1 and strategy==3 else [.5]
        for q in qs:
         for mode,kind in [(0,'none'),(1,'positive'),(1,'ones'),(1,'masked'),(2,'scalar')]:
          if family==1 and mode==2:continue
          add(seed,n,p,t,rank,storage,family,strategy,mode,kind,q,seed%2==0)
 for rank,t in [(0,1),(1,3)]:
  for storage in [0,1]:
   add(42,12,3,t,rank,storage,1,1,2,'scalar',.5,False)
 # Signed and zero weights are accepted by the source in several deterministic
 # paths. Keep source failures and verify their native counterparts separately.
 for family in [0,1]:
  for strategy in range(1,6 if family==0 else 5):
   for kind in ['signed','zeros']:
    for storage in [0,1]:
     add(42,12,3,3,1,storage,family,strategy,1,kind,.5,False)
 rng=np.random.RandomState(987)
 for n in [7,12,31,80]:
  for i in range(100):
   y=rng.choice([-1.,0.,1.],n);w=rng.choice([-2.,-1.,0.,1.,2.],n);q=[0.,.25,.5,.75,1.][i%5]
   for rank,t in [(0,1),(1,1),(1,3)]:
    target=y if t==1 else np.column_stack([y,-y,y+1])
    for storage in [0,1]:add(987,n,1,t,rank,storage,1,3,1,'tied_signed',q,False,target,w)
 if args.reference_cache:
  args.reference_cache.parent.mkdir(parents=True,exist_ok=True)
  args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 payload='\n'.join(line for expected,_,block in cases if 'failure'not in expected for line in block)+'\n'
 proc=subprocess.run([args.executable],input=payload,text=True,capture_output=True)
 if proc.returncode:
  raise RuntimeError(proc.stderr+'\nlast output:\n'+proc.stdout[-1400:])
 rows=iter(proc.stdout.splitlines());groups=0;worst=0;rejections=0
 def compare(actual,expected,label,exact=False):
  nonlocal groups,worst
  groups+=1;a=np.asarray(actual);b=np.atleast_1d(expected)
  if exact:np.testing.assert_array_equal(a,b,err_msg=str(label))
  else:
   np.testing.assert_allclose(a,b,rtol=3e-8,atol=3e-8,equal_nan=True,err_msg=str(label))
   finite=np.isfinite(a)&np.isfinite(b)
   if np.any(finite):worst=max(worst,float(np.max(abs(a[finite]-b[finite])/(1+abs(b[finite])))))
 for expected,label,block in cases:
  if 'failure'in expected:
   p=subprocess.run([args.executable],input='\n'.join(block)+'\n',capture_output=True,text=True)
   assert p.returncode and 'ERROR STOP'in p.stderr and 'SIGFPE'not in p.stderr,(label,p.stderr)
   rejections+=1;continue
  vector=lambda dtype=float:np.fromstring(next(rows),sep=' ',dtype=dtype)
  compare(vector(int),expected['meta'],label,True)
  if label['family']==0:
   for j in range(label['t']):
    compare(vector(int),expected['classes'][j],label,True)
    compare(vector(),expected['priors'][j],label)
    compare(vector(),expected['proba'][j],label,True)
   compare(vector(int),expected['pred'],label,True)
  else:
   compare(vector(),expected['constants'],label);compare(vector(),expected['pred'],label)
  score=vector()
  if expected['score']is not None:compare(score,expected['score'],label)
 assert next(rows,None)is None
 report=dict(configurations=len(cases),groups=groups,source_rejections=rejections,max_scaled_error=worst,
   source_warnings=warns,source_calls=calls,reference_cache_hits=hits,reference_provenance=provenance,
   configuration_inventory=[dict(label,source_failure=expected.get('failure'))for expected,label,_ in cases],
   reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,
   source_sha256=sha(root/'src/sklearn_dummy.f90'),validator_sha256=sha(__file__),executable_sha256=sha(args.executable),
   input_sha256=hashlib.sha256(payload.encode()).hexdigest())
 print(json.dumps({k:v for k,v in report.items()if k!='configuration_inventory'},indent=2))
 if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
