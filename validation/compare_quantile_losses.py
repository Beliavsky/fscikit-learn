"""Pinball/D2 quantile losses against sklearn 1.6.1, using batched accepted cases.

Targets use C-contiguous float64 matrices. Expected rejections run separately
because the native require() contract terminates the process.
"""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.metrics import mean_pinball_loss,d2_pinball_score,d2_absolute_error_score
ROOT=Path(__file__).resolve().parents[1]
SOURCES=[mean_pinball_loss,d2_pinball_score,d2_absolute_error_score]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fixtures():
 rng=np.random.default_rng(912765);h=np.finfo(float).max
 for n,q,alpha,kind in itertools.product([1,2,7,129],[1,3],[0.,.1,.5,.9,1.],range(3)):
  y=rng.normal(size=(n,q));p=y+rng.normal(size=(n,q))
  if kind==1:y=rng.integers(0,4,(n,q)).astype(float);p=rng.integers(0,4,(n,q)).astype(float)
  if kind==2:p=y.copy()
  for w,mode in itertools.product([None,np.arange(1,n+1,dtype=float),np.r_[-1.,np.full(n-1,2.)]],[0,1] if q==1 else [0,1,2,6]):
   for ky,kp in ([(1,1),(1,2),(2,1),(2,2)] if q==1 else [(2,2)]):
    yield dict(family='ordinary',metrics=[0,1]+([2] if alpha==.5 else []),alpha=alpha,y=y,p=p,w=w,ow=None,mode=mode,ky=ky,kp=kp)
 for n,q,alpha in itertools.product([1,2,7],[1,3],[0.,.5,1.]):
  y=np.ones((n,q));p=2*y
  for w,mode in itertools.product([np.full(n,np.nan),np.full(n,np.inf)],[0,1]):
   yield dict(family='nonfinite_weights',metrics=([0,1,2] if n==1 and alpha==.5 else [0,1] if n==1 else [0]),alpha=alpha,y=y,p=p,w=w,ow=None,mode=mode,ky=1 if q==1 else 2,kp=2)
 for n,q,alpha,kind in itertools.product([2,3,17,129],[1,3],[0.,.1,.5,.9,1.],range(3)):
  y=np.full((n,q),h);p=-y
  if kind==1:y[::2]*=-1;p=np.zeros((n,q))
  if kind==2:y=np.zeros((n,q));p=np.full((n,q),np.nextafter(0.,1.))
  for w,mode in itertools.product([None,np.ones(n),np.resize([h,-h,1.],n)],[0,1]):
   yield dict(family='extreme',metrics=[0,1]+([2] if alpha==.5 else []),alpha=alpha,y=y,p=p,w=w,ow=None,mode=mode,ky=2,kp=2)
 for n,q,alpha in itertools.product([8191,8192,8193,16385],[1,3],[0.,.1,.5,1.]):
  y=np.resize([0.,1.,2.,3.,4.],(n,q));p=np.resize([1.,.5,3.],(n,q))
  for w in [None,np.resize([1e16,1.,-1e16,3.],n)]:
   yield dict(family='buffered',metrics=[0,1]+([2] if alpha==.5 else []),alpha=alpha,y=y,p=p,w=w,ow=None,mode=1,ky=2,kp=2)
 for q in [1,2,8,129]:
  y=rng.normal(size=(7,q));p=rng.normal(size=(7,q))
  for ow,mode in itertools.product([np.arange(1,q+1,dtype=float),np.r_[-1.,np.full(q-1,2.)],np.full(q,1e308),np.full(q,np.nan),np.full(q,np.inf),np.zeros(q)],[2,6]):
   yield dict(family='custom_outputs',metrics=[0,1,2],alpha=.5,y=y,p=p,w=None,ow=ow,mode=mode,ky=2,kp=2)
 for w in [np.zeros(1),np.full(1,np.nan),np.ones(1)]:
  yield dict(family='singleton_aggregation_shortcut',metrics=[1,2],alpha=.5,y=np.ones((1,3)),p=np.zeros((1,3)),w=w,ow=np.zeros(3),mode=2,ky=2,kp=2)

def source(c,metric):
 y,p=c['y'],c['p'];ow=c['ow'];mode=c['mode']
 if ow is None:ow=np.arange(1,y.shape[1]+1,dtype=float)
 output={0:'uniform_average',1:'raw_values',2:ow,3:'raw_values',4:'variance_weighted',5:'bad',6:ow,7:'uniform_average'}[mode]
 args=[y[:,0] if c['ky']==1 else np.ascontiguousarray(y),p[:,0] if c['kp']==1 else np.ascontiguousarray(p)]
 options=dict(sample_weight=c['w'],multioutput=output)
 if metric<2:options['alpha']=c['alpha']
 return np.atleast_1d(SOURCES[metric](*args,**options))

def encode(c,metric,ident):
 y,p,w,ow=c['y'],c['p'],c['w'],c['ow']
 if ow is None:ow=np.arange(1,y.shape[1]+1,dtype=float)
 rows=[f"{ident} {metric} {c['mode']} {c['alpha']:.17g} {len(y)} {y.shape[1]} {len(p)} {p.shape[1]} {c['ky']} {c['kp']} {-1 if w is None else len(w)} {len(ow)}"]
 for a in [y,p]+([] if w is None else [w])+([] if not len(ow) else [ow]):rows.append(' '.join(format(float(v),'.17g') for v in a.ravel(order='F')))
 return '\n'.join(rows)+'\n'

def controlled(result):
 text=result.stdout+result.stderr
 assert result.returncode and 'ERROR STOP' in text and not any(s in text for s in ['SIGFPE','Segmentation fault','Fortran runtime error']),text

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 paths=[a.executable,Path(__file__),ROOT/'src/sklearn_quantile_losses.f90',ROOT/'src/sklearn_metrics.f90',ROOT/'src/sklearn_numpy_ordering.f90',ROOT/'LICENSES/x86-simd-sort-BSD',ROOT/'app/reference_quantile_losses_driver.f90',Path(sklearn.__file__).parent/'metrics/_regression.py',Path(sklearn.__file__).parent/'utils/stats.py',Path(sklearn.__file__).parent/'utils/extmath.py'];before={str(p):sha(p) for p in paths}
 records=[];accepted=[];inputs=[]
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore')
  for c in fixtures():
   for metric in c['metrics']:
    ident=len(records);record=dict(configuration=ident,metric=metric,mode=c['mode'],alpha=c['alpha'],family=c['family'],ranks=[c['ky'],c['kp']],shape=list(c['y'].shape),data_hashes=[hashlib.sha256(v.tobytes()).hexdigest() for v in [c['y'],c['p']]+([] if c['w'] is None else [c['w']])+([] if c['ow'] is None else [c['ow']])])
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
