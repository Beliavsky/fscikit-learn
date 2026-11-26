"""Source constructor/input errors and the explicit native -1 error adapter."""
from pathlib import Path
import argparse,json,subprocess,hashlib,os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.metrics import DistanceMetric
ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
x=np.ones((2,2));y=np.ones((1,3))
cases=dict(unknown=('cosine',{},None,None),unused_p=('euclidean',dict(p=2),None,None),mink_zero=('minkowski',dict(p=0),None,None),mink_negative=('minkowski',dict(p=-1),None,None),weighted_inf=('minkowski',dict(p=np.inf,w=[1.,1.]),None,None),weights_empty=('minkowski',dict(p=2,w=[]),None,None),weights_negative=('minkowski',dict(w=[-1.,1.]),None,None),weights_nan=('minkowski',dict(w=[np.nan,1.]),None,None),weights_inf=('minkowski',dict(w=[np.inf,1.]),None,None),weights_features=('minkowski',dict(w=[1.]),x,None),variance_missing=('seuclidean',{},None,None),variance_rank=('seuclidean',dict(V=x),None,None),variance_features=('seuclidean',dict(V=[1.]),x,None),precision_missing=('mahalanobis',{},None,None),precision_square=('mahalanobis',dict(VI=y),None,None),precision_features=('mahalanobis',dict(VI=np.ones((1,1))),x,None),covariance_singular=('mahalanobis',dict(V=x),None,None),haversine_features=('haversine',{},y,None),input_nan=('euclidean',{},np.array([[np.nan,1.],[1.,1.]]),None),input_inf=('euclidean',{},np.array([[np.inf,1.],[1.,1.]]),None),input_empty=('euclidean',{},x[:0],None),input_features=('euclidean',{},x,y))
checks=[]
for name,(metric,kwargs,xx,yy)in cases.items():
 try:
  obj=DistanceMetric.get_metric(metric,**kwargs)
  if xx is not None:obj.pairwise(xx,yy)
 except (ValueError,TypeError,np.linalg.LinAlgError)as error:source_error=type(error).__name__
 else:raise AssertionError(('Source accepted',name))
 p=subprocess.run([str(a.executable),name],capture_output=True,text=True);out=p.stdout+p.stderr
 passed=p.returncode!=0 and 'fscikit-learn:'in out and 'invalid DistanceMetric input was accepted'not in out and not any(t in out for t in ['SIGFPE','SIGSEGV','Fortran runtime error'])
 checks.append(dict(case=name,source_error=source_error,native_adapter=False,passed=passed))
p=subprocess.run([str(a.executable),'callback_minus_one'],capture_output=True,text=True);out=p.stdout+p.stderr
checks.append(dict(case='callback_minus_one',source_error='reserved Cython -1 return can access-violate',native_adapter=True,passed=p.returncode!=0 and 'reserved Cython error value -1'in out and not any(t in out for t in ['SIGFPE','SIGSEGV','Fortran runtime error'])))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=dict(count=len(checks),checks=checks,passed=all(c['passed']for c in checks),validator_sha256=sha(__file__),executable_sha256=sha(a.executable))
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n');print(report)
assert report['passed']
