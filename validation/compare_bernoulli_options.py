"""BernoulliNB raw features and thresholded dense/CSR streaming vs sklearn."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings,sys
import numpy as np,scipy,sklearn
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import BernoulliNB
import sklearn.naive_bayes as source_nb
import sklearn.preprocessing._data as source_binarize


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path)
 ap.add_argument('--reference-cache',type=Path);ap.add_argument('--report',type=Path)
 args=ap.parse_args();root=Path(__file__).resolve().parents[1]
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 provenance=dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,
   files={m.__file__:sha(m.__file__)for m in [source_nb,source_binarize]})
 cache={};cases=[];source_calls=hits=warns=0
 if args.reference_cache and args.reference_cache.exists():
  saved=json.loads(args.reference_cache.read_text());assert saved['provenance']==provenance;cache=saved['cases']
 def sparse(x,duplicate):
  if not duplicate:return csr_matrix(x)
  n,p=x.shape;data=np.repeat(x.ravel()/2,2);indices=np.tile(np.repeat(np.arange(p),2),n)
  return csr_matrix((data,indices,np.arange(n+1)*2*p),shape=x.shape)
 for seed in [0,7,42,2026]:
  for n,k in [(1,1),(9,3),(23,3),(65,5),(23,1),(65,1)]:
   for p in [1,3,7]:
    for data_kind in ['binary','fractional','signed','zeros','ones','counts']:
     rng=np.random.RandomState(seed+31*n+13*p);m=5
     x=rng.randint(0,2,(n,p)).astype(float);q=rng.randint(0,2,(m,p)).astype(float)
     if data_kind=='fractional':x=rng.uniform(0,1,(n,p));q=rng.uniform(-1,2,(m,p))
     elif data_kind=='signed':x=rng.uniform(-1,1,(n,p));q=rng.uniform(-2,2,(m,p))
     elif data_kind=='zeros':x[:]=0;q[:]=0
     elif data_kind=='ones':x[:]=1;q=np.resize(np.array([-1.,0.,1.,2.,-.5]),(m,p))
     elif data_kind=='counts':x=rng.randint(0,4,(n,p)).astype(float);q=rng.uniform(-2,3,(m,p))
     classes=np.array([-7,4,19,33,71])[:k];y=classes[np.arange(n)%k];rng.shuffle(y)
     truth=np.resize(classes,m);w=rng.uniform(.1,2,n)
     if n>1:w[1]=0
     for storage in range(4):
      for duplicate in [0,1]:
       for stream in [0,1]:
        for bin_mode in [0,1,2]:
         index=len(cases);alpha=[0.,1e-12,.2,1.][index%4];force=int((index//4)%2==0)
         weighted=int((index//8)%2==0);prior=(index//16)%3
         ww=np.zeros(n)if index%19==0 else w
         threshold=0. if bin_mode!=2 else .7
         label=dict(seed=seed,n=n,p=p,data=data_kind,storage=storage,duplicates=duplicate,stream=stream,
                    binarize=bin_mode,alpha=alpha,force_alpha=force,weighted=weighted,prior=prior)
         block=[f'{n} {p} {m} {k} {storage} {duplicate} {stream} {weighted} {int(bin_mode!=0)} {force} {prior} {alpha:.17g} {threshold:.17g}']
         for a in [x,q]:block+=[' '.join(map(str,row))for row in a]
         for a in [y,truth,classes,ww]:block+=[' '.join(map(str,a))]
         key=hashlib.sha256(('\n'.join(block)+'\n').encode()).hexdigest()
         if key in cache:expected=cache[key];hits+=1
         else:
          priors=None if prior!=2 else np.full(k,1/k)
          if prior==2 and k>1:priors[0]=0;priors[1:]=1/(k-1)
          model=BernoulliNB(alpha=alpha,force_alpha=bool(force),fit_prior=prior!=1,class_prior=priors,
                            binarize=None if bin_mode==0 else threshold)
          batches=1 if not stream else min(3,n);expected=[]
          with warnings.catch_warnings(record=True)as caught:
           warnings.simplefilter('always')
           for batch in range(batches):
            first=batch*n//batches;last=(batch+1)*n//batches
            xx=sparse(x[first:last],duplicate)if storage in [1,3]else x[first:last]
            qq=sparse(q,duplicate)if storage in [1,2]else q
            weight=ww[first:last]if weighted else None
            if stream:model.partial_fit(xx,y[first:last],classes=classes,sample_weight=weight)
            else:model.fit(xx,y,sample_weight=weight)
            joint=model._joint_log_likelihood(model._check_X(qq))
            expected.append(dict(meta=[model.n_features_in_,len(model.classes_)],classes=model.classes_.tolist(),
             class_count=model.class_count_.tolist(),feature_count=model.feature_count_.ravel(order='F').tolist(),
             feature_log_prob=model.feature_log_prob_.ravel(order='F').tolist(),class_log_prior=model.class_log_prior_.tolist(),
             joint=joint.ravel(order='F').tolist(),log_proba=model.predict_log_proba(qq).ravel(order='F').tolist(),
             proba=model.predict_proba(qq).ravel(order='F').tolist(),predict=model.predict(qq).tolist()))
          expected=dict(snapshots=expected,warnings=len(caught));cache[key]=expected;source_calls+=1
         warns+=expected['warnings'];cases.append((label,block,expected))
 if args.reference_cache:
  args.reference_cache.parent.mkdir(parents=True,exist_ok=True)
  args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 payload='\n'.join(line for _,block,_ in cases for line in block)+'\n'
 proc=subprocess.run([args.executable],input=payload,text=True,capture_output=True)
 if proc.returncode:raise RuntimeError(proc.stderr+'\nlast output:\n'+proc.stdout[-1200:])
 rows=iter(proc.stdout.splitlines());groups=snapshots=0;worst=0
 for label,_,expected in cases:
  for snapshot in expected['snapshots']:
   snapshots+=1
   fields=['meta','classes','class_count','feature_count','feature_log_prob','class_log_prior','joint','log_proba','proba','predict']
   values={field:np.fromstring(next(rows),sep=' ')for field in fields}
   if any(not np.allclose(values[f],snapshot[f],rtol=3e-8,atol=3e-8,equal_nan=True)for f in fields):
    (root/'validation/results/bernoulli_mismatch.json').write_text(json.dumps(dict(label=label,block=cases[[c[0]for c in cases].index(label)][1],expected=snapshot,actual={f:a.tolist()for f,a in values.items()}),indent=2)+'\n')
   for field in fields:
    actual=values[field];want=np.asarray(snapshot[field]);groups+=1
    if field in ['meta','classes','predict']:np.testing.assert_array_equal(actual,want,err_msg=str((label,field)))
    else:
     np.testing.assert_allclose(actual,want,rtol=3e-8,atol=3e-8,equal_nan=True,err_msg=str((label,field)))
     finite=np.isfinite(actual)&np.isfinite(want)
     if np.any(finite):worst=max(worst,float(np.max(abs(actual[finite]-want[finite])/(1+abs(want[finite])))))
 assert next(rows,None)is None
 report=dict(configurations=len(cases),snapshots=snapshots,groups=groups,max_scaled_error=worst,
   source_warnings=warns,source_calls=source_calls,reference_cache_hits=hits,reference_provenance=provenance,
   reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,
   source_sha256=sha(root/'src/sklearn_naive_bayes.f90'),validator_sha256=sha(__file__),executable_sha256=sha(args.executable),
   input_sha256=hashlib.sha256(payload.encode()).hexdigest(),configuration_inventory=[c[0]for c in cases])
 print(json.dumps({k:v for k,v in report.items()if k!='configuration_inventory'},indent=2))
 if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
