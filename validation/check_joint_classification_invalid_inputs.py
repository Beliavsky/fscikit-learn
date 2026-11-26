"""Actual joint-classifier errors and explicit native result-rank adapters."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
a=ap.parse_args();x=np.ones((4,2));y=np.column_stack([[-7,-7,11,11],[0,1,2,0]])
indicator=np.column_stack([[0,0,1,1],[0,1,0,1]])
cases=dict(empty_outputs=(False,{},x,y[:,:0],None),extra_empty_outputs=(True,{},x,y[:,:0],None),
 target_rows=(False,{},x,y[:3],None),input_inf=(False,{},np.array([[np.inf,1],[1,1],[1,1],[1,1]]),y,None),
 weights_length=(False,{},x,y,[1]),weights_nan=(False,{},x,y,[np.nan,1,1,1]),weights_inf=(False,{},x,y,[np.inf,1,1,1]),
 criterion=(False,dict(criterion='unknown'),x,y,None),score_multiclass=(False,{},x,y,None),
 score_zero_weights=(False,{},x,indicator,np.zeros(4)),score_outputs=(False,{},x,y,None),
 query_features=(False,{},x,y,None),query_inf=(False,{},x,y,None))
checks=[]
for name,(extra,kwargs,xx,target,w)in cases.items():
 try:
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');model=(ExtraTreeClassifier if extra else DecisionTreeClassifier)(**kwargs).fit(xx,target,sample_weight=w)
   if name in ['score_multiclass','score_zero_weights']:model.score(xx,target,sample_weight=w)
   if name=='score_outputs':model.score(xx,y[:,:1])
   if name=='query_features':model.predict(xx[:,:1])
   if name=='query_inf':model.predict(np.full_like(xx,np.inf))
 except (ValueError,TypeError,ZeroDivisionError)as error:source_error=dict(type=type(error).__name__,message=str(error))
 else:raise AssertionError(('Source accepted',name))
 p=subprocess.run([str(a.executable),name],capture_output=True,text=True);out=p.stdout+p.stderr
 passed=p.returncode!=0 and 'fscikit-learn:'in out and 'invalid joint classification input was accepted'not in out and not any(s in out for s in ['SIGFPE','SIGSEGV','Fortran runtime error'])
 checks.append(dict(case=name,source_error=source_error,passed=passed,native_output=out,native_adapter=False))
for name,message in [('scalar_predict','joint classification prediction requires predict_multi'),('scalar_proba','joint classification probabilities require predict_proba_multi')]:
 p=subprocess.run([str(a.executable),name],capture_output=True,text=True);out=p.stdout+p.stderr
 checks.append(dict(case=name,native_adapter=True,native_output=out,passed=p.returncode!=0 and message in out))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=dict(count=len(checks),checks=checks,passed=all(c['passed']for c in checks),sha256={str(p):sha(p)for p in [Path(__file__),a.executable]})
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
print(len(checks),'joint classification guards passed:',report['passed']);assert report['passed']
