"""Weighted/multioutput R2 and explained variance against sklearn 1.6.1.

The numerical oracle uses C-contiguous float64 target matrices. NumPy's
slow-axis summation can round differently for other array memory layouts.
"""
from pathlib import Path
import argparse, hashlib, itertools, json, os, subprocess, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.metrics import r2_score, explained_variance_score
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fixtures():
 rng=np.random.default_rng(613701)
 for n,q in itertools.product([1,2,7,129],[1,3]):
  y=rng.normal(size=(n,q));p=y+rng.normal(size=(n,q))
  for kind in range(4):
   a=y.copy();b=p.copy()
   if kind==1:a[:]=2;b[:]=3
   if kind==2:b=a+1
   if kind==3:b=a.copy()
   weights=[None,np.arange(1,n+1,dtype=float),np.resize([1.,-1.,2.],n),np.zeros(n)]
   for w,mode,force in itertools.product(weights,range(4),[True,False]):
    for ky,kp in ([(1,1),(1,2),(2,1),(2,2)] if q==1 else [(2,2)]):
     yield dict(family='ordinary',kind=kind),a,b,w,mode,force,ky,kp,None
 for n,q in itertools.product([2,17,131],[1,3]):
  for kind in range(5):
   h=np.finfo(float).max
   y=np.full((n,q),[h,1e-300,1e16,0.,2.][kind]);p=y.copy()
   if kind==0:y[::2]*=-1;p=-y
   if kind==1:y[::2]*=-1;p[:]=0
   if kind==2:y[::2]+=2;p=y-1
   if kind==3:p[:]=1
   for w in [None,np.full(n,np.nan),np.full(n,np.inf),np.full(n,h),np.resize([h,-h,1.],n),np.resize([np.nextafter(0.,1.),0.],n)]:
    for mode,force in itertools.product(range(4),[True,False]):
     yield dict(family='extreme',kind=kind),y,p,w,mode,force,1 if q==1 else 2,2,None
 for n,q in itertools.product([8191,8192,8193,16385],[1,3]):
  y=np.resize([1e16,1.,-1e16,3.,2.,-3.,0.],(n,q));p=np.resize([1.,-1.,0.],(n,q))
  for w in [None,np.resize([1.,-1.,2.,3.],n)]:
   for mode in [0,1,3]:
    yield dict(family='buffered_reductions'),y,p,w,mode,False,2,2,None
 for q in [2,8,129]:
  y=rng.normal(size=(17,q));p=rng.normal(size=(17,q))
  for ow in [np.arange(1,q+1,dtype=float),np.resize([1.,-1.],q),np.zeros(q),np.full(q,1e308)]:
   yield dict(family='custom_outputs'),y,p,None,2,True,2,2,ow

def evaluate(executable,fixture,metric):
 cfg,y,p,w,mode,force,ky,kp,ow=fixture
 q=y.shape[1]
 if ow is None:ow=np.arange(1,q+1,dtype=float)
 source=r2_score if metric==0 else explained_variance_score
 output=['uniform_average','variance_weighted',ow,'raw_values'][mode]
 error=None
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore')
  try:
   expected=np.atleast_1d(source(y[:,0] if ky==1 else np.ascontiguousarray(y),p[:,0] if kp==1 else np.ascontiguousarray(p),sample_weight=w,multioutput=output,force_finite=force))
  except (ValueError,ZeroDivisionError,TypeError) as exc:error=exc
 rows=[f'{metric} {mode} {"T" if force else "F"} {len(y)} {q} {len(p)} {p.shape[1]} {ky} {kp} {-1 if w is None else len(w)} {len(ow)}']
 for x in [y,p]+([] if w is None else [w])+[ow]:rows.append(' '.join(format(float(v),'.17g') for v in x.ravel(order='F')))
 result=subprocess.run([str(executable)],input='\n'.join(rows)+'\n',text=True,capture_output=True)
 if error is not None:
  assert result.returncode and 'ERROR STOP' in result.stderr and not any(t in result.stderr for t in ['SIGFPE','Segmentation fault','Fortran runtime error']),(cfg,error,result.stdout,result.stderr)
  return dict(rejected=True,source_type=type(error).__name__)
 assert result.returncode==0,(cfg,result.stdout,result.stderr)
 actual={r.split()[0]:np.array([float(v) for v in r.split()[1:]]) for r in result.stdout.splitlines()}
 assert actual.keys()=={'state','identity','values'},(cfg,actual)
 assert actual['state'].tolist()==[1.] and actual['identity'].tolist()==[1.],(cfg,actual)
 v=actual['values']
 assert v.shape==expected.shape,(cfg,v,expected)
 for check in [np.isnan,np.isposinf,np.isneginf]:assert np.array_equal(check(v),check(expected)),(cfg,v,expected)
 good=np.isfinite(expected)
 assert np.allclose(v[good],expected[good],rtol=3e-12,atol=3e-12),(cfg,v,expected)
 return dict(groups=len(expected)+2,nonfinite=int(np.count_nonzero(~good)),max_absolute_error=float(np.max(np.abs(v[good]-expected[good]),initial=0)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',default='checked');ap.add_argument('--report',type=Path,required=True);ap.add_argument('--limit',type=int);args=ap.parse_args()
 paths=[args.executable,Path(__file__),ROOT/'src/sklearn_variance_scores.f90',ROOT/'src/sklearn_metrics.f90',ROOT/'app/reference_variance_scores_driver.f90',Path(sklearn.__file__).parent/'metrics/_regression.py']
 before={str(p):sha(p) for p in paths};records=[]
 for fixture in fixtures():
  for metric in range(2):
   if args.limit is not None and len(records)>=args.limit:break
   record=evaluate(args.executable,fixture,metric)
   cfg,y,p,w,mode,force,ky,kp,ow=fixture
   records.append(dict(configuration=len(records),metric=metric,mode=mode,force_finite=force,ranks=[ky,kp],shape=list(y.shape),family=cfg['family'],data_hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in [y,p]+([] if w is None else [w])+([] if ow is None else [ow])],**record))
  if args.limit is not None and len(records)>=args.limit:break
 assert before=={str(p):sha(p) for p in paths}
 report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,source_layout='C-contiguous',configurations=len(records),groups=sum(r.get('groups',0) for r in records),rejections=sum(r.get('rejected',False) for r in records),hashes=before,records=records)
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
 print({k:v for k,v in report.items() if k not in ['hashes','records']})

if __name__=='__main__':main()
