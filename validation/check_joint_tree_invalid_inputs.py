"""Joint regression tree input errors and native scalar prediction rank adapter."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
a=ap.parse_args();x=np.ones((4,2));y=np.ones((4,2))
cases=dict(empty_outputs=(False,{},y[:,:0],None),extra_empty_outputs=(True,{},y[:,:0],None),
 target_rows=(False,{},y[:3],None),target_nan=(False,{},np.array([[1,np.nan],[1,1],[1,1],[1,1]]),None),
 target_inf=(False,{},np.array([[1,np.inf],[1,1],[1,1],[1,1]]),None),
 weights_length=(False,{},y,[1]),weights_nan=(False,{},y,[np.nan,1,1,1]),weights_inf=(False,{},y,[np.inf,1,1,1]),
 poisson_negative=(False,dict(criterion='poisson'),np.array([[1,-1],[1,1],[1,1],[1,1]]),None),
 poisson_zero=(False,dict(criterion='poisson'),np.zeros_like(y),None),score_outputs=(False,{},y,None))
checks=[]
for name,(extra,kwargs,target,w)in cases.items():
 try:
  model=(ExtraTreeRegressor if extra else DecisionTreeRegressor)(**kwargs).fit(x,target,sample_weight=w)
  if name=='score_outputs':model.score(x,y[:,:1])
 except (ValueError,TypeError)as error:source_error=dict(type=type(error).__name__,message=str(error))
 else:raise AssertionError(('Source accepted',name))
 p=subprocess.run([str(a.executable),name],capture_output=True,text=True);out=p.stdout+p.stderr
 passed=p.returncode!=0 and 'fscikit-learn:'in out and 'invalid joint tree input was accepted'not in out and not any(s in out for s in ['SIGFPE','SIGSEGV','Fortran runtime error'])
 checks.append(dict(case=name,source_error=source_error,passed=passed,native_output=out,native_adapter=False))
p=subprocess.run([str(a.executable),'scalar_predict'],capture_output=True,text=True);out=p.stdout+p.stderr
checks.append(dict(case='scalar_predict',native_adapter=True,native_output=out,passed=p.returncode!=0 and 'joint tree prediction requires predict_multi'in out))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=dict(count=len(checks),checks=checks,passed=all(c['passed']for c in checks),sha256={str(p):sha(p)for p in [Path(__file__),a.executable]})
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
print(len(checks),'joint tree guards passed:',report['passed']);assert report['passed']
