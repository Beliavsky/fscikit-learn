"""Actual Source/native BayesianRidge control and weight rejections."""
from pathlib import Path
import os,argparse,json,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
from sklearn.linear_model import BayesianRidge

def source(which):
 x=np.array([[.1,.4],[.2,.1],[.3,.3],[.4,.2]]);y=np.arange(1.,5.);w=np.ones(4);m=BayesianRidge()
 if which==0:m.tol=0
 if which==1:m.tol=np.nan
 if which==2:m.max_iter=0
 for i,k in enumerate(['alpha_1','alpha_2','lambda_1','lambda_2'],3):
  if which==i:setattr(m,k,-1.)
 if which==7:m.alpha_init=-2
 if which==8:m.lambda_init=-2
 if which==9:m.alpha_init=np.inf
 if which==10:m.lambda_2=np.nan
 if which==11:w[0]=-1
 if which==12:w[0]=np.nan
 if which==13:w[0]=np.inf
 if which==14:w=w[:3]
 if which==15:w[:]=0
 if which==16:y=np.column_stack([y,y])
 if which==17:y[0]=np.nan
 if which==18:x[0,0]=np.inf
 if which==19:return m.predict(x,return_std=True)
 return m.fit(x,y,sample_weight=w)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 for which in range(20):
  error=None
  try:
   with warnings.catch_warnings():warnings.simplefilter('ignore');source(which)
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  p=subprocess.run([str(a.executable),str(which)],capture_output=True,text=True);records.append(dict(case=which,source_error=error,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,passed=error is not None and p.returncode==1 and 'fscikit-learn:' in p.stdout))
 a.report.write_text(json.dumps(records,indent=2)+'\n');print('Bayesian guards',sum(r['passed'] for r in records),'/',len(records));raise SystemExit(not all(r['passed'] for r in records))
if __name__=='__main__':main()
