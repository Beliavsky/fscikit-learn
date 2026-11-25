"""Tweedie deviance/D2 against sklearn 1.6.1, using batched accepted cases.

Targets use C-contiguous float64 matrices. Expected rejections run separately
because the native require() contract terminates the process.
"""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.metrics import mean_tweedie_deviance,d2_tweedie_score
ROOT=Path(__file__).resolve().parents[1]
SOURCES=[mean_tweedie_deviance,d2_tweedie_score]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fixtures():
 rng=np.random.default_rng(918631);h=np.finfo(float).max
 powers=[-3.5,-3.,-1.,-.1,0.,1.,np.nextafter(1.,2.),1.1,1.5,np.nextafter(2.,1.),2.,np.nextafter(2.,3.),2.5,3.,4.5,10.]
 for n,power,kind in itertools.product([1,2,7,131],powers,range(2)):
  y=np.exp(rng.normal(size=(n,1)));p=np.exp(rng.normal(size=(n,1)))
  if kind==1:p=y.copy()
  for w,ky,kp in itertools.product([None,np.arange(1,n+1,dtype=float),np.r_[-1.,np.full(n-1,2.)],np.full(n,np.nan),np.full(n,np.inf)],[1,2],[1,2]):
   yield dict(family='ordinary',metrics=[0,1],power=power,y=y,p=p,w=w,ky=ky,kp=kp)
 for n,power in itertools.product([1,2,7,131],[-3.,-1.5,-.5,0.]):
  y=np.resize([-2.,0.,1.,3.],n)[:,None];p=np.resize([1.,2.,.5],n)[:,None]
  for w in [None,np.resize([-1.,2.,3.],n)]:
   yield dict(family='negative_truth',metrics=[0,1],power=power,y=y,p=p,w=w,ky=2,kp=1)
 for power in [-1e308,-100.,-20.,-1e-7,np.nextafter(0.,-1.),20.,100.,1e308]:
  for w in [None,np.array([-1.,2.,3.]),np.full(3,np.nan)]:
   yield dict(family='wide_power',metrics=[0,1],power=power,y=np.array([[1.],[2.],[3.]]),p=np.array([[2.],[1.],[4.]]),w=w,ky=1,kp=2)
 vals=[np.nextafter(0.,1.),1e-300,1e-16,1.,1e16,1e300,h]
 for power,yv,pv in itertools.product([-3.,-.5,0.,1.,1.5,2.,3.,4.5],vals,vals):
  for w in [None,np.full(3,h),np.array([-h,h,1.])]:
   yield dict(family='extreme',metrics=[0,1],power=power,y=np.array([[yv],[1.],[2.]]),p=np.array([[pv],[1.5],[2.5]]),w=w,ky=1,kp=2)
 for n,power in itertools.product([8191,8192,8193,16385],[-3.,0.,1.,1.5,2.,3.]):
  y=np.resize([.1,1.,2.,3.,4.],n)[:,None];p=np.resize([1.,.5,3.],n)[:,None]
  for w in [None,np.resize([1.,-1.,2.,3.],n)]:
   yield dict(family='buffered',metrics=[0,1],power=power,y=y,p=p,w=w,ky=2,kp=2)
 for power,w in itertools.product([-3.,0.,1.,1.5,2.,3.],[None,np.zeros(1),np.ones(0),np.ones(2),np.full(2,np.nan)]):
  yield dict(family='singleton_shortcut',metrics=[1],power=power,y=np.array([[0.]]),p=np.array([[-1.]]),w=w,ky=1,kp=2)
 for power,w in itertools.product([0.,1.,1.5,2.,3.],[None,np.array([1.,-1.,1.]),np.array([-2.,1.,2.])]):
  yield dict(family='zero_null',metrics=[0,1] if power<2 else [1],power=power,y=np.zeros((3,1)) if power<2 else np.array([[1.],[2.],[3.]]),p=np.ones((3,1)),w=w,ky=2,kp=1)

def source(c,metric):
 y,p=c['y'],c['p']
 return np.atleast_1d(SOURCES[metric](y[:,0] if c['ky']==1 else np.ascontiguousarray(y),p[:,0] if c['kp']==1 else np.ascontiguousarray(p),sample_weight=c['w'],power=c['power']))

def encode(c,metric,ident):
 y,p,w=c['y'],c['p'],c['w']
 rows=[f"{ident} {metric} {c['power']:.17g} {len(y)} {y.shape[1]} {len(p)} {p.shape[1]} {c['ky']} {c['kp']} {-1 if w is None else len(w)}"]
 for a in [y,p]+([] if w is None else [w]):rows.append(' '.join(format(float(v),'.17g') for v in a.ravel(order='F')))
 return '\n'.join(rows)+'\n'

def controlled(result):
 text=result.stdout+result.stderr
 assert result.returncode and 'ERROR STOP' in text and not any(s in text for s in ['SIGFPE','Segmentation fault','Fortran runtime error']),text

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 paths=[a.executable,Path(__file__),ROOT/'src/sklearn_tweedie_scores.f90',ROOT/'src/sklearn_metrics.f90',ROOT/'app/reference_tweedie_scores_driver.f90',Path(sklearn.__file__).parent/'metrics/_regression.py'];before={str(p):sha(p) for p in paths}
 records=[];accepted=[];inputs=[]
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore')
  for c in fixtures():
   for metric in c['metrics']:
    ident=len(records);record=dict(configuration=ident,metric=metric,power=c['power'],family=c['family'],ranks=[c['ky'],c['kp']],shape=list(c['y'].shape),data_hashes=[hashlib.sha256(v.tobytes()).hexdigest() for v in [c['y'],c['p']]+([] if c['w'] is None else [c['w']])])
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
