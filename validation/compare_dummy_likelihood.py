"""Dense/CSR add_dummy_feature and general covariance log_likelihood."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings,sys
import numpy as np,scipy,sklearn
from scipy.sparse import csr_matrix
from sklearn.preprocessing import add_dummy_feature
from sklearn.covariance import log_likelihood
import sklearn.preprocessing._data as source_dummy
import sklearn.covariance._empirical_covariance as source_cov
import scipy.sparse._sparsetools as source_sparse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True)
 ap.add_argument('--reference-cache',type=Path);ap.add_argument('--report',type=Path);args=ap.parse_args()
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 provenance=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,files={m.__name__:sha(m.__file__)for m in [source_dummy,source_cov,source_sparse]})
 cache={};cases=[];calls=hits=warns=0
 if args.reference_cache and args.reference_cache.exists():
  old=json.loads(args.reference_cache.read_text());assert old['provenance']==provenance;cache=old['cases']
 def add(mode,x,value=1.,precision=None,label=None):
  nonlocal calls,hits,warns
  n,p=x.shape;lines=[f'{mode} {n} {p} {value:.17g}']
  if mode==1:lines += [str(x.nnz),' '.join(map(str,x.indptr+1)),' '.join(map(str,x.indices+1)),' '.join(map(str,x.data))]
  else:
   lines+=[' '.join(map(str,row))for row in x]
   if mode==2:lines+=[' '.join(map(str,row))for row in precision]
  request='\n'.join(lines)+'\n';key=hashlib.sha256(request.encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    if mode==0:
     z=add_dummy_feature(x,value=value);expected=dict(meta=list(z.shape),data=z.ravel(order='F').tolist())
    elif mode==1:
     z=add_dummy_feature(x,value=value);expected=dict(meta=[*z.shape,z.nnz],indptr=(z.indptr+1).tolist(),indices=(z.indices+1).tolist(),data=z.data.tolist())
    else:expected=dict(likelihood=[float(log_likelihood(x,precision))])
   expected['warnings']=len(caught);cache[key]=expected;calls+=1
  warns+=expected['warnings'];cases.append((dict(mode=mode,n=n,p=p,value=value,**(label or {})),request,expected))
 for seed in [0,7,42,2026]:
  rng=np.random.default_rng(seed)
  for n in [1,3,17]:
   for p in [1,3,9]:
    for kind in ['normal','zero','huge']:
     x=rng.normal(size=(n,p))
     if kind=='zero':x[:]=0
     if kind=='huge':x=np.clip(x,-1,1)*1e308
     # Keep finite inputs even when multiplication by the largest scale overflows.
     if kind=='huge':x=np.copysign(np.minimum(abs(np.nan_to_num(x)),1e308),x)
     for value in [1.,0.,-0.,-2.]:add(0,x,value,label=dict(seed=seed,kind=kind))
    for count in [0,3,17,33,129]:
     for kind in ['normal','cancellation','overflow']:
      indices=rng.integers(0,p,n*count);data=rng.normal(size=n*count)
      if kind=='cancellation':data=rng.choice([1e16,-1e16,1.,-1.,0.],n*count)
      if kind=='overflow':data=rng.choice([1e308,-1e308,0.],n*count)
      x=csr_matrix((data,indices,np.arange(n+1)*count),shape=(n,p))
      for value in [1.,0.,-0.,-2.]:add(1,x,value,label=dict(seed=seed,kind=kind,count=count))
  for p in [0,1,2,5,9]:
   identity=np.eye(p);random=rng.normal(size=(p,p));spd=random@random.T+identity
   precisions={'random':random,'spd':spd,'negative':-identity,'singular':np.ones((p,p)),'zero':np.zeros((p,p)),'tiny':spd*1e-160,'huge':spd*1e160}
   if p:
    q=spd.copy();q[[0,-1]]=q[[-1,0]];precisions['row_swapped']=q
    for value,name in [(np.nan,'nan'),(np.inf,'inf'),(-np.inf,'neginf')]:
     q=identity.copy();q[0,0]=value;precisions[name]=q
   for kind,precision in precisions.items():
    for emp_kind in ['identity','random','zero','nonfinite']:
     emp=rng.normal(size=(p,p))if emp_kind=='random'else identity.copy()
     if emp_kind=='zero':emp[:]=0
     if emp_kind=='nonfinite'and p:emp[0,0]=np.inf
     add(2,emp,precision=precision,label=dict(seed=seed,precision=kind,empirical=emp_kind))
 # Larger row sorts exercise introsort partitioning around duplicate columns.
 rng=np.random.default_rng(123)
 for count in [15,16,17,31,32,33,65,257,513]:
  for layout in ['random','ascending','descending','organ_pipe']:
   columns=rng.integers(0,17,count)
   if layout=='ascending':columns.sort()
   elif layout=='descending':columns=np.sort(columns)[::-1]
   elif layout=='organ_pipe':columns=np.minimum(np.arange(count),np.arange(count)[::-1])%17
   data=rng.choice([1e16,-1e16,1.,-1.],count)
   add(1,csr_matrix((data,columns,[0,count]),shape=(1,17)),label=dict(count=count,layout=layout))
 if args.reference_cache:
  args.reference_cache.parent.mkdir(parents=True,exist_ok=True);args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 request=''.join(c[1]for c in cases);run=subprocess.run([args.executable],input=request,text=True,capture_output=True)
 if run.returncode:raise RuntimeError(run.stderr+'\n'+run.stdout[-1200:])
 rows=iter(run.stdout.splitlines());groups=0;worst=0
 for label,_,expected in cases:
  fields=['meta','data']if label['mode']==0 else ['meta','indptr','indices','data']if label['mode']==1 else ['likelihood']
  for field in fields:
   want=np.asarray(expected[field]);actual=np.fromstring(next(rows),sep=' ');groups+=1
   if field in ['meta','indptr','indices']:np.testing.assert_array_equal(actual,want,err_msg=str((label,field)))
   else:
    np.testing.assert_allclose(actual,want,rtol=3e-8,atol=3e-8,equal_nan=True,err_msg=str((label,field)))
    if field=='data':np.testing.assert_array_equal(np.signbit(actual[want==0]),np.signbit(want[want==0]),err_msg=str((label,'signed_zero')))
    finite=np.isfinite(actual)&np.isfinite(want)
    if finite.any():worst=max(worst,float(np.max(abs(actual[finite]-want[finite])/(1+abs(want[finite])))))
 assert next(rows,None)is None
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),groups=groups,max_scaled_error=worst,source_warnings=warns,source_calls=calls,reference_cache_hits=hits,reference_provenance=provenance,reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,
  source_sha256={name:sha(root/'src'/name)for name in ['sklearn_stateless_preprocessing.f90','sklearn_covariance_functions.f90']},validator_sha256=sha(__file__),executable_sha256=sha(args.executable),input_sha256=hashlib.sha256(request.encode()).hexdigest(),configuration_inventory=[c[0]for c in cases])
 print(json.dumps({k:v for k,v in report.items()if k!='configuration_inventory'},indent=2))
 if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
