"""Source/native sparse KernelPCA restrictions and input validation."""
from pathlib import Path
import argparse,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import KernelPCA

def source(which):
 x=np.array([[.1,.4],[.2,.1],[.3,.3],[.4,.2]]);s=csr_matrix(x);q=s.copy();m=KernelPCA(n_components=2,eigen_solver='dense')
 if which==0:m.fit_inverse_transform=True
 if which==1:m.kernel='precomputed';s=csr_matrix(x@x.T)
 if which==2:m.kernel='precomputed';m.fit(x@x.T);return m.transform(csr_matrix(x@x.T))
 if which==3:s.data[0]=np.nan
 if which==4:s.data[0]=np.inf
 if which==5:m.fit(s);return m.transform(s[:,:1])
 if which==6:m.fit(s);q.data[0]=np.nan;return m.transform(q)
 if which==7:return m.transform(s)
 if which==8:s=csr_matrix((0,2))
 if which==9:s=csr_matrix((4,0))
 if which==10:m.fit(s);return m.transform(csr_matrix((0,2)))
 if which==11:m.fit(s);q.data[0]=np.inf;return m.transform(q)
 return m.fit(s)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 for which in range(12):
  error=None
  try:
   with warnings.catch_warnings():warnings.simplefilter('ignore');source(which)
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  p=subprocess.run([str(a.executable),str(which)],capture_output=True,text=True);records.append(dict(case=which,source_error=error,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,passed=error is not None and p.returncode==1 and 'fscikit-learn:' in p.stdout))
 a.report.write_text(json.dumps(records,indent=2)+'\n');print('Sparse KernelPCA guards',sum(x['passed'] for x in records),'/',len(records));raise SystemExit(not all(x['passed'] for x in records))
if __name__=='__main__':main()
