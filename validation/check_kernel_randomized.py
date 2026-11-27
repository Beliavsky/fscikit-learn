"""Actual Source/native randomized KernelPCA control and input rejections."""
from pathlib import Path
import argparse,os,json,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.decomposition import KernelPCA

def source(which):
 x=np.array([[.1,.4],[.2,.1],[.3,.3],[.4,.2]]);m=KernelPCA(n_components=2,eigen_solver='randomized',random_state=0)
 if which==0:m.eigen_solver='unknown'
 if which==1:m.iterated_power=-2
 if which==2:m.random_state=-1
 if which==3:m.n_components=-1
 if which==4:m.gamma=np.nan
 if which==5:return m.transform(x)
 if which==6:x[0,0]=np.nan
 if which==7:x[0,0]=np.inf
 if which==8:m.degree=-1
 if which==9:m.fit_inverse_transform=True;m.kernel='precomputed';return m.fit(x@x.T)
 if which==10:m.fit(x);return m.transform(x[:,:1])
 if which==11:m.fit(x);x[0,0]=np.nan;return m.transform(x)
 return m.fit(x)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 for which in range(12):
  error=None
  try:
   with warnings.catch_warnings():warnings.simplefilter('ignore');source(which)
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  p=subprocess.run([str(a.executable),str(which)],capture_output=True,text=True);records.append(dict(case=which,source_error=error,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,passed=error is not None and p.returncode==1 and 'fscikit-learn:' in p.stdout))
 a.report.write_text(json.dumps(records,indent=2)+'\n');print('KernelPCA guards',sum(x['passed'] for x in records),'/',len(records));raise SystemExit(not all(x['passed'] for x in records))
if __name__=='__main__':main()
