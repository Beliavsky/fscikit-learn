"""MSLE/RMSLE/Poisson/Gamma against sklearn 1.6.1, using batched accepted cases.

Targets use C-contiguous float64 matrices. Expected rejections run separately
because the native require() contract terminates the process.
"""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_log_error,root_mean_squared_log_error,mean_poisson_deviance,mean_gamma_deviance
ROOT=Path(__file__).resolve().parents[1]
SOURCES=[mean_squared_log_error,root_mean_squared_log_error,mean_poisson_deviance,mean_gamma_deviance]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fixtures():
 rng=np.random.default_rng(821571);h=np.finfo(float).max
 for n,q in itertools.product([1,2,7,129],[1,3]):
  for kind in range(4):
   y=np.expm1(rng.normal(size=(n,q)));p=np.expm1(rng.normal(size=(n,q)))
   if kind==1:p=y.copy()
   if kind==2:y[:]=-.5;p[:]=-.25
   if kind==3:y[:]=0;p[:]=0
   for w in [None,np.arange(1,n+1,dtype=float),np.r_[-1.,np.full(n-1,2.)],np.full(n,np.nan),np.full(n,np.inf)]:
    for mode in ([0,1] if q==1 else [0,1,2,6]):
     for ky,kp in ([(1,1),(1,2),(2,1),(2,2)] if q==1 else [(2,2)]):
      yield dict(family='log_ordinary',metrics=[1] if mode==6 else [0,1],kind=kind,y=y,p=p,w=w,ow=None,mode=mode,ky=ky,kp=kp)
 vals=[np.nextafter(-1.,0.),-.5,-np.nextafter(0.,1.),0.,np.nextafter(0.,1.),1e-300,1e-18,1.,1e16,h]
 for yv,pv in itertools.product(vals,vals):
  for w in [None,np.full(3,h),np.array([-h,h,1.])]:
   yield dict(family='log_extreme',metrics=[0,1],y=np.full((3,1),yv),p=np.full((3,1),pv),w=w,ow=None,mode=1,ky=1,kp=2)
 for n,q in itertools.product([8191,8192,8193,16385],[1,3]):
  y=np.resize([-.99,-.5,0.,1.,2.,3.,1e16],(n,q));p=np.resize([1.,0.,-.75],(n,q))
  for w,mode in itertools.product([None,np.resize([1.,-1.,2.,3.],n)],[0,1]):
   yield dict(family='log_buffered',metrics=[0,1],y=y,p=p,w=w,ow=None,mode=mode,ky=2,kp=2)
 for q in [1,2,8,129]:
  y=np.expm1(rng.normal(size=(7,q)));p=np.expm1(rng.normal(size=(7,q)))
  for ow,mode in itertools.product([np.arange(1,q+1,dtype=float),np.r_[-1.,np.full(q-1,2.)],np.full(q,1e308),np.full(q,np.nan),np.full(q,np.inf),np.zeros(q)],[2,6]):
   yield dict(family='log_custom',metrics=[1] if mode==6 else [0,1],y=y,p=p,w=None,ow=ow,mode=mode,ky=2,kp=2)
 for n,kind in itertools.product([1,2,7,129],range(3)):
  y=np.exp(rng.normal(size=(n,1)));p=np.exp(rng.normal(size=(n,1)))
  if kind==1:p=y.copy()
  if kind==2:p=y*(1+np.finfo(float).eps)
  for w,ky,kp in itertools.product([None,np.arange(1,n+1,dtype=float),np.r_[-1.,np.full(n-1,2.)],np.full(n,np.nan),np.full(n,np.inf)],[1,2],[1,2]):
   yield dict(family='deviance_ordinary',metrics=[2,3],kind=kind,y=y,p=p,w=w,ow=None,mode=0,ky=ky,kp=kp)
 for n,wkind in itertools.product([1,2,7],range(3)):
  w=[None,np.ones(n),np.resize([-1.,2.,3.],n)][wkind]
  yield dict(family='poisson_zero',metrics=[2],y=np.zeros((n,1)),p=np.arange(1,n+1,dtype=float)[:,None],w=w,ow=None,mode=0,ky=2,kp=1)
 vals=[np.nextafter(0.,1.),1e-300,1e-16,1.,1e16,1e300,h]
 for yv,pv in itertools.product(vals,vals):
  for w in [None,np.ones(3),np.full(3,h),np.array([-h,h,1.]),np.full(3,np.nan),np.full(3,np.inf)]:
   yield dict(family='deviance_extreme',metrics=[2,3],y=np.full((3,1),yv),p=np.full((3,1),pv),w=w,ow=None,mode=0,ky=1,kp=2)
 for n in [8191,8192,8193,16385]:
  y=np.resize([.1,1.,2.,3.,1e16],n)[:,None];p=np.resize([1.,.5,3.],n)[:,None]
  for w in [None,np.resize([1.,-1.,2.,3.],n)]:
   yield dict(family='deviance_buffered',metrics=[2,3],y=y,p=p,w=w,ow=None,mode=0,ky=2,kp=2)

def source(c,metric):
 y,p=c['y'],c['p'];ow=c['ow'];mode=c['mode']
 if ow is None:ow=np.arange(1,y.shape[1]+1,dtype=float)
 args=[y[:,0] if c['ky']==1 else np.ascontiguousarray(y),p[:,0] if c['kp']==1 else np.ascontiguousarray(p)]
 if metric>=2:return np.atleast_1d(SOURCES[metric](*args,sample_weight=c['w']))
 output={0:'uniform_average',1:'raw_values',2:ow,3:'raw_values',4:'variance_weighted',5:'bad',6:ow,7:'uniform_average'}[mode]
 return np.atleast_1d(SOURCES[metric](*args,sample_weight=c['w'],multioutput=output))

def encode(c,metric,ident):
 y,p,w,ow=c['y'],c['p'],c['w'],c['ow']
 if ow is None:ow=np.arange(1,y.shape[1]+1,dtype=float)
 rows=[f"{ident} {metric} {c['mode']} {len(y)} {y.shape[1]} {len(p)} {p.shape[1]} {c['ky']} {c['kp']} {-1 if w is None else len(w)} {len(ow)}"]
 for a in [y,p]+([] if w is None else [w])+([] if not len(ow) else [ow]):rows.append(' '.join(format(float(v),'.17g') for v in a.ravel(order='F')))
 return '\n'.join(rows)+'\n'

def controlled(result):
 text=result.stdout+result.stderr
 assert result.returncode and 'ERROR STOP' in text and not any(s in text for s in ['SIGFPE','Segmentation fault','Fortran runtime error']),text

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 paths=[a.executable,Path(__file__),ROOT/'src/sklearn_distribution_errors.f90',ROOT/'src/sklearn_metrics.f90',ROOT/'app/reference_distribution_errors_driver.f90',Path(sklearn.__file__).parent/'metrics/_regression.py'];before={str(p):sha(p) for p in paths}
 records=[];accepted=[];inputs=[]
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore')
  for c in fixtures():
   for metric in c['metrics']:
    ident=len(records);record=dict(configuration=ident,metric=metric,mode=c['mode'],family=c['family'],ranks=[c['ky'],c['kp']],shape=list(c['y'].shape),data_hashes=[hashlib.sha256(v.tobytes()).hexdigest() for v in [c['y'],c['p']]+([] if c['w'] is None else [c['w']])+([] if c['ow'] is None else [c['ow']])])
    try:expected=source(c,metric)
    except (ValueError,ZeroDivisionError,TypeError) as exc:
     controlled(subprocess.run([str(a.executable)],input='1\n'+encode(c,metric,ident),text=True,capture_output=True));record.update(rejected=True,source_type=type(exc).__name__)
    else:
     inputs.append(encode(c,metric,ident));accepted.append((ident,expected));record.update(groups=len(expected)+2,nonfinite=int(np.count_nonzero(~np.isfinite(expected))))
    records.append(record)
  print(len(records),'source configurations;',len(accepted),'accepted; running batched native comparison',flush=True)
  result=subprocess.run([str(a.executable)],input=str(len(inputs))+'\n'+''.join(inputs),text=True,capture_output=True)
  assert result.returncode==0,(result.stdout[-2000:],result.stderr)
  lines=result.stdout.splitlines();assert len(lines)==len(accepted),(len(lines),len(accepted))
  for line,(ident,want) in zip(lines,accepted):
   parts=line.split();assert [int(v) for v in parts[:3]]==[ident,1,1],(ident,line)
   got=np.array([float(v) for v in parts[3:]]);assert got.shape==want.shape,(ident,got,want)
   for check in [np.isnan,np.isposinf,np.isneginf]:assert np.array_equal(check(got),check(want)),(records[ident],got,want)
   finite=np.isfinite(want);assert np.allclose(got[finite],want[finite],rtol=3e-12,atol=3e-12),(records[ident],got,want)
   records[ident]['max_absolute_error']=float(np.max(np.abs(got[finite]-want[finite]),initial=0))
 assert before=={str(p):sha(p) for p in paths}
 report=dict(profile=a.profile,sklearn=sklearn.__version__,numpy=np.__version__,numpy_buffer_size=np.getbufsize(),source_layout='C-contiguous',configurations=len(records),groups=sum(r.get('groups',0) for r in records),rejections=sum(r.get('rejected',False) for r in records),hashes=before,records=records)
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['hashes','records']},flush=True)

if __name__=='__main__':main()
