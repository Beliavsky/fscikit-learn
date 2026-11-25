"""Confusion matrices: numeric adapters, weighted/normalized cells and storage."""
from pathlib import Path
import argparse,hashlib,io,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import scipy,sklearn
from sklearn.metrics import confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
NORMALIZATIONS=[None,'true','pred','all',None]

def form(x,k):
 z=np.asarray(x,dtype=[np.int32,np.float64,bool][k%3])
 return z if k<3 else z.reshape(-1,1)

def fixtures():
 rng=np.random.default_rng(713845)
 for n in [0,1,7,131]:
  y=rng.integers(0,4,n);p=rng.integers(0,4,n)
  for ky,kp in itertools.product(range(6),repeat=2):
   for wm in range(3):
    w=None if wm==0 else rng.uniform(.1,4,n) if wm==1 else rng.integers(-3,4,n).astype(float)
    for norm in range(5):yield dict(family='adapters',ky=ky,kp=kp,norm=norm,weight=wm),form(y,ky),form(p,kp),None,w
 for counts in itertools.product(range(2),repeat=4):
  cells=np.repeat(np.arange(4),counts);y=cells//2;p=cells%2
  selections=[None,np.array([1,0]),np.array([False,True]),np.array([0]),np.array([0,0]),np.array([1,1]),np.array([],int),np.array([2,3]),np.array([0,1,2]),np.array([0.,np.nan]),np.array([0.,np.inf]),np.array([.5,1.])]
  for labels in selections:
   for weighted in [False,True]:
    w=rng.integers(-3,4,len(y)).astype(float) if weighted else None
    for norm in range(4):yield dict(family='selected',ky=0,kp=0,norm=norm,counts=list(counts)),y,p,labels,w
 for k in [2,3,7,8,16,31,32,65,91,127,128,129]:
  y=np.repeat(np.arange(k),4);p=(y+np.tile([0,0,1,2],k))%k;n=len(y)
  patterns=[None,rng.normal(size=n),np.resize([1e16,1.,-1e16,3.],n),np.resize([np.finfo(float).max]*2+[-np.finfo(float).max]*2,n),np.resize([np.inf,1.,-np.inf,0.],n),np.resize([np.nan,1.,0.,2.],n),np.resize([np.nextafter(0.,1.),-0.,0.,np.nextafter(0.,1.)],n)]
  for wm,w in enumerate(patterns):
   for norm in range(4):yield dict(family='reductions',ky=0,kp=0,norm=norm,weight=wm),y,p,None,w
 for values in [[-7,42],[-2147483648,2147483647],[-2.**63,0.],[np.nextafter(2.**63,0.),0.]]:
  y=np.resize(values,19).astype(float);p=y.copy();p[::3]=values[0]
  for norm in range(4):yield dict(family='label_values',ky=1,kp=1,norm=norm),y,p,np.array(values[::-1],float),np.resize([1.,-2.,3.,.5],19)
 boundaries=[([0,1],[0],None,None),([0.,.5],[0.,1.],None,None),([np.nan,1.],[0.,1.],None,None),([np.inf,1.],[0.,1.],None,None),([2.**63,0.],[0.,0.],None,None),([0,1],[1,0],None,[1.]),([],[],[0,1],[1.])]
 for y,p,labels,w in boundaries:
  for norm in range(5):yield dict(family='boundary',ky=1,kp=1,norm=norm),np.array(y,float),np.array(p,float),None if labels is None else np.array(labels),None if w is None else np.array(w,float)
 for shape in [(0,1),(2,0),(2,2)]:
  for norm in range(5):yield dict(family='matrix_boundary',ky=4,kp=4,norm=norm),np.zeros(shape),np.zeros(shape),None,None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--start',type=int,default=0);ap.add_argument('--limit',type=int);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 paths=[Path(__file__),ROOT/'src/sklearn_confusion_matrix.f90',ROOT/'src/sklearn_categorical_scores.f90',ROOT/'app/reference_confusion_matrix_driver.f90',a.executable,Path(sklearn.__file__).parent/'metrics/_classification.py']
 hashes={str(p):sha(p) for p in paths};configs=[];checks=[];rejections=[];nonfinite_weight_cases=0
 for ident,(cfg,y,p,labels,w) in enumerate(fixtures()):
  if ident<a.start:continue
  if a.limit is not None and len(configs)>=a.limit:break
  kl=-1 if labels is None else 2 if labels.dtype.kind=='b' else 0 if labels.dtype.kind in 'iu' else 1
  cfg=dict(id=ident,**cfg,shapes=[list(y.shape),list(p.shape)],label_kind=kl,data_hashes=[None if x is None else hashlib.sha256(x.tobytes()).hexdigest() for x in [y,p,labels,w]])
  configs.append(cfg);error=None
  with warnings.catch_warnings(),np.errstate(all='ignore'):
   warnings.simplefilter('ignore')
   try:expected=confusion_matrix(y,p,labels=labels,sample_weight=w,normalize=NORMALIZATIONS[cfg['norm']])
   except (ValueError,TypeError,IndexError,ZeroDivisionError) as e:error=e
  stream=io.StringIO();stream.write(f"{len(y)} {1 if y.ndim==1 else y.shape[1]} {cfg['ky']} {len(p)} {1 if p.ndim==1 else p.shape[1]} {cfg['kp']} {kl} {0 if labels is None else len(labels)} {int(w is not None)} {0 if w is None else len(w)} {cfg['norm']}\n")
  for x in [y,p,labels,w]:
   if x is None:continue
   v=x.ravel(order='F');v=v.astype(int) if v.dtype.kind=='b' else v
   stream.write(' '.join(format(vv,'.17g') for vv in v)+'\n')
  r=subprocess.run([str(a.executable)],input=stream.getvalue(),text=True,capture_output=True)
  if error is not None:
   assert r.returncode and 'ERROR STOP' in r.stderr and not any(t in r.stderr for t in ['SIGFPE','Segmentation fault','Fortran runtime error']),(cfg,error,r.stdout,r.stderr)
   rejections.append(dict(configuration=ident,source_type=type(error).__name__,message=str(error),passed=True));continue
  assert r.returncode==0,(cfg,r.stdout,r.stderr)
  actual={name:np.array(values,float) for name,*values in [line.split() for line in r.stdout.splitlines()]}
  integer=w is None and cfg['norm']==0
  wanted=dict(shape=np.array(expected.shape),storage=np.array([1,32] if integer else [0,64]),matrix=expected.ravel(order='F'),identity=np.array([1]),ieee_state=np.array([1]))
  assert actual.keys()==wanted.keys(),(cfg,actual.keys())
  for name,want in wanted.items():
   got=actual[name];assert got.shape==want.shape,(cfg,name,got.shape,want.shape)
   assert np.all(np.isfinite(want)) and np.all(np.isfinite(got)),(cfg,name,'sanitized matrices must be finite')
   if name=='matrix' and not integer:np.testing.assert_allclose(got,want,rtol=3e-13,atol=0,err_msg=str((cfg,name)))
   else:np.testing.assert_array_equal(got,want,err_msg=str((cfg,name)))
   zeros=want==0;np.testing.assert_array_equal(np.signbit(got[zeros]),np.signbit(want[zeros]),err_msg=str((cfg,name,'zero sign')))
   checks.append(dict(configuration=ident,field=name,passed=True,actual=[float(v).hex() for v in got],expected=[float(v).hex() for v in want]))
  nonfinite_weight_cases+=int(w is not None and np.any(~np.isfinite(w)))
  if ident%300==0:print(ident,'configurations checked',flush=True)
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,numpy_buffer_size=np.getbufsize(),hashes=hashes,configurations=configs,checks=checks,rejections=rejections,nonfinite_weight_cases=nonfinite_weight_cases),indent=2)+'\n')
 print(len(configs),'configurations',len(checks),'groups',len(rejections),'source rejections',nonfinite_weight_cases,'nonfinite-weight cases PASS')
if __name__=='__main__':main()
