"""Public covariance functions and arbitrary-rank square-matrix shrinkage."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings,sys
import numpy as np,sklearn,scipy
from sklearn.covariance import empirical_covariance,shrunk_covariance,ledoit_wolf,oas
import sklearn.covariance._empirical_covariance as source_empirical
import sklearn.covariance._shrunk_covariance as source_shrinkage


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path)
 ap.add_argument('--reference-cache',type=Path);ap.add_argument('--report',type=Path)
 args=ap.parse_args();root=Path(__file__).resolve().parents[1]
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 provenance=dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,
   files={m.__file__:sha(m.__file__)for m in [source_empirical,source_shrinkage]})
 cache={};cases=[];calls=hits=warns=0
 if args.reference_cache and args.reference_cache.exists():
  old=json.loads(args.reference_cache.read_text());assert old['provenance']==provenance;cache=old['cases']
 def add(mode,x,center=False,block=1000,rank=2,amount=.1,label=None):
  nonlocal calls,hits,warns
  n,p=(x.shape[0],x.shape[-1]);lines=[f'{mode} {n} {p} {int(center)} {block} {rank} {amount:.17g}']
  if mode<3:
   lines+=[' '.join(map(str,row))for row in x]
  else:
   data=x.reshape(n,p,p)
   lines+=[' '.join(map(str,row))for batch in data for row in batch]
  key=hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    if mode==0:
     cov=empirical_covariance(x[0]if rank==1 else x,assume_centered=center);shrink=-1
    elif mode==1:cov,shrink=ledoit_wolf(x,assume_centered=center,block_size=block)
    elif mode==2:cov,shrink=oas(x,assume_centered=center)
    else:
     if rank==2:a=x[0]
     else:a=x.reshape((n,)+(1,)*(rank-3)+(p,p))
     cov=shrunk_covariance(a,shrinkage=amount);shrink=amount
    meta=list(cov.shape)if mode<3 else [n,p,p]
    expected=dict(meta=meta,covariance=np.asarray(cov).ravel(order='F').tolist(),shrinkage=float(shrink),warnings=len(caught))
   cache[key]=expected;calls+=1
  label=dict(label or {},mode=mode,n=n,p=p,center=center,block_size=block,rank=rank,amount=amount)
  cases.append((label,lines,expected));warns+=expected['warnings']
 for seed in [0,7,42,2026,314159]:
  rng=np.random.RandomState(seed)
  for n in [1,2,7,33,129]:
   for p in [1,2,5,9]:
    for data in ['normal','zeros','constant','duplicates','rank_one']:
     x=rng.normal(size=(n,p))
     if data=='zeros':x[:]=0
     elif data=='constant':x[:]=rng.normal(size=p)
     elif data=='duplicates'and n>1:x[1]=x[0]
     elif data=='rank_one':x=np.outer(rng.normal(size=n),rng.normal(size=p))
     for center in [False,True]:
      for mode in [0,1,2]:
       for block in ([1,3,1000]if mode==1 else [1000]):
        add(mode,x,center,block,label=dict(seed=seed,data=data))
      if n==1:add(0,x,center,rank=1,label=dict(seed=seed,data=data))
  for rank in range(2,16):
   for p in [1,2,5]:
    for batches in ([1]if rank==2 else [1,3]):
     for data in ['asymmetric','zero','signed_diagonal','overflow_trace']:
      a=rng.normal(size=(batches,p,p))
      if data=='zero':a[:]=0
      elif data=='signed_diagonal':
       a[:]=0
       for b in range(batches):a[b]=np.diag(rng.normal(size=p))
      elif data=='overflow_trace':
       a[:]=0
       for i in range(p):a[:,i,i]=np.finfo(float).max*.75
      for amount in [0.,.1,.5,1.]:add(3,a,rank=rank,amount=amount,label=dict(seed=seed,data=data))
 # Empirical covariance alone permits nonfinite input, unlike estimator fits.
 for n in [1,4]:
  for p in [1,3]:
   for value in [np.nan,np.inf,-np.inf]:
    x=np.arange(n*p,dtype=float).reshape(n,p);x[0,0]=value
    for center in [False,True]:
     add(0,x,center,label=dict(data='nonfinite'))
     if n==1:add(0,x,center,rank=1,label=dict(data='nonfinite_vector'))
 if args.reference_cache:
  args.reference_cache.parent.mkdir(parents=True,exist_ok=True)
  args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 payload='\n'.join(line for _,block,_ in cases for line in block)+'\n'
 run=subprocess.run([args.executable],input=payload,text=True,capture_output=True)
 if run.returncode:raise RuntimeError(run.stderr+'\nlast output:\n'+run.stdout[-1200:])
 rows=iter(run.stdout.splitlines());groups=0;worst=0
 for label,_,expected in cases:
  meta=np.fromstring(next(rows),dtype=int,sep=' ');np.testing.assert_array_equal(meta,expected['meta'],err_msg=str(label));groups+=1
  for field in ['covariance','shrinkage']:
   actual=np.fromstring(next(rows),sep=' ');want=np.atleast_1d(expected[field]);groups+=1
   np.testing.assert_allclose(actual,want,rtol=3e-8,atol=3e-8,equal_nan=True,err_msg=str((label,field)))
   finite=np.isfinite(actual)&np.isfinite(want)
   if np.any(finite):worst=max(worst,float(np.max(abs(actual[finite]-want[finite])/(1+abs(want[finite])))))
 assert next(rows,None)is None
 report=dict(configurations=len(cases),groups=groups,max_scaled_error=worst,source_warnings=warns,source_calls=calls,
   reference_cache_hits=hits,reference_provenance=provenance,reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,
   source_sha256=sha(root/'src/sklearn_covariance_functions.f90'),validator_sha256=sha(__file__),executable_sha256=sha(args.executable),
   input_sha256=hashlib.sha256(payload.encode()).hexdigest(),configuration_inventory=[c[0]for c in cases])
 print(json.dumps({k:v for k,v in report.items()if k!='configuration_inventory'},indent=2))
 if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
