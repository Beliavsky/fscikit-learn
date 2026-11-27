"""Actual Source/native rejection checks for CSR QuantileRegressor."""
from pathlib import Path
import argparse,os,json,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import QuantileRegressor

def source(which):
 x=csr_matrix([[.1,.4],[.2,.1],[.3,.3],[.4,.2]]);y=np.arange(1.,5.);w=np.ones(4);m=QuantileRegressor()
 for i,k,v in [(0,'quantile',0),(1,'quantile',1),(2,'quantile',np.nan),(3,'alpha',-1),(4,'alpha',np.inf)]:
  if which==i:setattr(m,k,v)
 if which==5:w=w[:3]
 if which==6:w[0]=-1
 if which==7:w[0]=np.nan
 if which==8:w[0]=np.inf
 if which==9:y=np.column_stack([y,y])
 if which==10:x.data[0]=np.nan
 if which==11:x.data[0]=np.inf
 if which==12:return m.predict(x)
 if which==13:m.fit(x,y);return m.predict(x[:,:1])
 if which==14:x=csr_matrix((0,2));y=np.array([]);w=np.array([])
 if which==15:x=csr_matrix((4,0))
 return m.fit(x,y,sample_weight=w)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 for which in range(16):
  error=None
  try:
   with warnings.catch_warnings():warnings.simplefilter('ignore');source(which)
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  p=subprocess.run([str(a.executable),str(which)],capture_output=True,text=True);records.append(dict(case=which,source_error=error,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,passed=error is not None and p.returncode==1 and 'fscikit-learn:' in p.stdout))
 a.report.write_text(json.dumps(records,indent=2)+'\n');print('CSR quantile guards',sum(r['passed'] for r in records),'/',len(records));raise SystemExit(not all(r['passed'] for r in records))
if __name__=='__main__':main()
