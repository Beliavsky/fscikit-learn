"""Actual Source QDA fit and query rejection probes."""
from pathlib import Path
import argparse,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def source_rejects(case):
 i=np.arange(1,13);x=np.c_[np.sin(i),np.cos(.7*i)];y=np.where(i<=6,-3,7);kw=dict(reg_param=.2)
 if case==1:x=x[:0];y=y[:0]
 if case==2:x=x[:,:0]
 if case==3:x[0,0]=np.nan
 if case==4:y=y[:11]
 if case==5:y[:]=1
 if case==6:y[:]=7;y[0]=-3
 changes={7:dict(reg_param=-.1),8:dict(reg_param=1.1),9:dict(reg_param=np.nan),10:dict(reg_param=np.inf),11:dict(tol=-1.),12:dict(tol=np.nan),13:dict(tol=np.inf),18:dict(priors=[.2,.3,.5]),19:dict(priors=[])};kw.update(changes.get(case,{}));model=QuadraticDiscriminantAnalysis(**kw)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  try:
   if case==14:model.predict_log_proba(x);return False
   model.fit(x,y)
   if case==15:model.decision_function(x[:,:1])
   if case==16:x[0,0]=np.nan;model.predict_proba(x)
   if case==17:model.predict_log_proba(x[:0])
   if case in [18,19]:model.predict_log_proba(x)
  except (ValueError,TypeError,AttributeError):return True
 return False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();checks=[]
 for case in range(1,20):
  assert source_rejects(case),('Source accepted',case)
  p=subprocess.run([a.executable,str(case)],text=True,capture_output=True);text=p.stdout+p.stderr
  assert p.returncode and 'fscikit-learn:'in text and not any(s in text for s in ['SIGFPE','SIGSEGV','Fortran runtime error']),(case,text)
  checks.append(dict(case=case,diagnostic=next(s.strip()for s in text.splitlines()if 'fscikit-learn:'in s)))
 report=dict(count=19,source_rejections=19,checks=checks);a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
