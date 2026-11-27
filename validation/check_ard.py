"""Actual sklearn/native ARDRegression input rejections."""
from pathlib import Path
import os,argparse,json,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
from sklearn.linear_model import ARDRegression

def source(which):
 x=np.array([[.1,.4],[.2,.1],[.3,.3],[.4,.2]]);y=np.arange(1.,5.);m=ARDRegression()
 for i,k,v in [(0,'tol',-1),(1,'tol',np.nan),(2,'max_iter',0),(3,'alpha_1',-1),(4,'alpha_2',-1),(5,'lambda_1',-1),(6,'lambda_2',-1),(7,'threshold_lambda',-1),(8,'threshold_lambda',np.inf),(9,'lambda_2',np.nan)]:
  if which==i:setattr(m,k,v)
 if which==10:return m.fit(x[:1],y[:1])
 if which==11:y=np.column_stack([y,y])
 if which==12:return m.fit(x,y,sample_weight=np.ones(4))
 if which==13:x[0,0]=np.inf
 if which==14:y[0]=np.nan
 if which==15:return m.predict(x,return_std=True)
 return m.fit(x,y)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 for which in range(16):
  error=None
  try:
   with warnings.catch_warnings():warnings.simplefilter('ignore');source(which)
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  p=subprocess.run([str(a.executable),str(which)],capture_output=True,text=True);records.append(dict(case=which,source_error=error,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,passed=error is not None and p.returncode==1 and 'fscikit-learn:' in p.stdout))
 a.report.write_text(json.dumps(records,indent=2)+'\n');print('ARD guards',sum(r['passed'] for r in records),'/',len(records));raise SystemExit(not all(r['passed'] for r in records))
if __name__=='__main__':main()
