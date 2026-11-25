"""MSE/RMSE/MAE/MAPE against sklearn 1.6.1, using batched accepted cases.

Targets use C-contiguous float64 matrices. Expected rejections run separately
because the native require() contract terminates the process.
"""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
ROOT=Path(__file__).resolve().parents[1]
SOURCES=[mean_squared_error,root_mean_squared_error,mean_absolute_error,mean_absolute_percentage_error]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fixtures():
 rng=np.random.default_rng(812477)
 for n,q in itertools.product([1,2,7,129],[1,3]):
  for kind in range(4):
   y=rng.normal(size=(n,q));p=y+rng.normal(size=(n,q))
   if kind==1:p=y.copy()
   if kind==2:y[:]=0
   if kind==3:y[:]=2;p[:]=3
   for w in [None,np.arange(1,n+1,dtype=float),np.r_[-1.,np.full(n-1,2.)],np.full(n,np.nan),np.full(n,np.inf)]:
    for mode in ([0,1] if q==1 else [0,1,2]):
     for ky,kp in ([(1,1),(1,2),(2,1),(2,2)] if q==1 else [(2,2)]):
      yield dict(family='ordinary',kind=kind,y=y,p=p,w=w,ow=None,mode=mode,ky=ky,kp=kp)
 h=np.finfo(float).max
 for n,q,kind in itertools.product([2,17,131],[1,3],range(5)):
  y=np.full((n,q),[h,1e-300,1e16,0.,2.][kind]);p=y.copy()
  if kind==0:y[::2]*=-1;p=-y
  if kind==1:y[::2]*=-1;p[:]=0
  if kind==2:y[::2]+=2;p=y-1
  if kind==3:p[:]=1
  for w,mode in itertools.product([None,np.full(n,h),np.resize([h,-h,1.],n),np.full(n,np.nextafter(0.,1.))],[0,1]):
   yield dict(family='extreme',kind=kind,y=y,p=p,w=w,ow=None,mode=mode,ky=1 if q==1 else 2,kp=2)
 for n,q in itertools.product([8191,8192,8193,16385],[1,3]):
  y=np.resize([1e16,1.,-1e16,3.,2.,-3.,0.],(n,q));p=np.resize([1.,-1.,0.],(n,q))
  for w,mode in itertools.product([None,np.resize([1.,-1.,2.,3.],n)],[0,1]):
   yield dict(family='buffered_reductions',y=y,p=p,w=w,ow=None,mode=mode,ky=2,kp=2)
 for q in [1,2,8,129]:
  y=rng.normal(size=(7,q));p=rng.normal(size=(7,q))
  for ow in [np.arange(1,q+1,dtype=float),np.r_[-1.,np.full(q-1,2.)],np.full(q,1e308),np.full(q,np.nan),np.full(q,np.inf)]:
   yield dict(family='custom_outputs',y=y,p=p,w=None,ow=ow,mode=2,ky=2,kp=2)

def source(c,metric):
 y,p=c['y'],c['p'];ow=c['ow']
 if ow is None:ow=np.arange(1,y.shape[1]+1,dtype=float)
 mode=c['mode'];output=['uniform_average','raw_values',ow][mode] if mode<3 else ['raw_values','variance_weighted','bad'][mode-3]
 return np.atleast_1d(SOURCES[metric](y[:,0] if c['ky']==1 else np.ascontiguousarray(y),p[:,0] if c['kp']==1 else np.ascontiguousarray(p),sample_weight=c['w'],multioutput=output))

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
 paths=[a.executable,Path(__file__),ROOT/'src/sklearn_error_metrics.f90',ROOT/'src/sklearn_metrics.f90',ROOT/'app/reference_error_metrics_driver.f90',Path(sklearn.__file__).parent/'metrics/_regression.py'];before={str(p):sha(p) for p in paths}
 records=[];accepted=[];inputs=[]
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore')
  for c in fixtures():
   for metric in range(4):
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
