"""Source/native rejection checks for IncrementalPCA CSR entry points."""
from pathlib import Path
import argparse,json,os,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import IncrementalPCA

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 x=np.array([[np.sin(i),np.cos(.3*i),float(i)] for i in range(1,7)])
 for case in range(1,13):
  m=IncrementalPCA(n_components=2,batch_size=3);s=csr_matrix(x);error=None
  try:
   if case==1:m.partial_fit(s)
   if case==2:m.fit(csr_matrix(x[:0]))
   if case==3:m.fit(csr_matrix(x[:,:0]))
   if case in [4,5]:s.data[0]=np.nan if case==4 else np.inf;m.fit(s)
   if case==6:m.set_params(batch_size=-1).fit(s)
   if case==7:m.set_params(n_components=4).fit(s)
   if case==8:m.set_params(batch_size=1).fit(s)
   if case==9:m.transform(s)
   if case==10:m.fit(x);m.transform(csr_matrix(x[:,:2]))
   if case==11:m.fit(x);s.data[0]=np.inf;m.transform(s)
   if case==12:m.partial_fit(x);m.transform(s)
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  p=subprocess.run([str(a.executable),str(case)],capture_output=True,text=True)
  records.append(dict(case=case,source_error=error,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,passed=error is not None and p.returncode==1 and 'fscikit-learn:' in p.stdout))
 a.report.write_text(json.dumps(records,indent=2)+'\n');print('IncrementalPCA sparse guards',sum(x['passed'] for x in records),'/',len(records));raise SystemExit(not all(x['passed'] for x in records))
if __name__=='__main__':main()
