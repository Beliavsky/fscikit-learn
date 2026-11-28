"""Source/native spectral sparse validation rejections."""
from pathlib import Path
import argparse,json,os,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import SpectralClustering

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[];x=np.abs(np.arange(6)[:,None]-np.arange(6)[None,:]).astype(float)
 for case in range(1,15):
  m=SpectralEmbedding(n_components=2,affinity='precomputed_nearest_neighbors',n_neighbors=3,random_state=21);s=csr_matrix((x.ravel(),np.tile(np.arange(6),6),np.arange(7)*6),shape=(6,6));error=None
  try:
   if case==1:m.set_params(n_neighbors=None).fit(s)
   if case==2:m.set_params(n_neighbors=7).fit(s)
   if case==3:m.fit(csr_matrix(x[:,:5]))
   if case==4:s.data[0]=-1;m.fit(s)
   if case in [5,6]:s.data[0]=np.nan if case==5 else np.inf;m.fit(s)
   if case==7:m.fit(csr_matrix(x[:0]))
   if case==8:m.set_params(affinity='rbf').fit(csr_matrix(x[:1]))
   if case==9:m.set_params(affinity='precomputed').fit(csr_matrix(x[:,:5]))
   if case==10:m.set_params(affinity='rbf',gamma=-2).fit(s)
   if case==11:m.set_params(affinity='rbf').fit(csr_matrix(x[:,:0]))
   if case==12:m.fit(csr_matrix((np.zeros(6),np.arange(6),np.arange(7)),shape=(6,6)))
   if case==13:SpectralClustering(n_clusters=2,n_neighbors=7,affinity='precomputed_nearest_neighbors').fit(s)
   if case==14:SpectralClustering(n_clusters=2,affinity='precomputed').fit(csr_matrix(x[:,:5]))
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  p=subprocess.run([str(a.executable),str(case)],capture_output=True,text=True);records.append(dict(case=case,source_error=error,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,passed=error is not None and p.returncode==1 and 'fscikit-learn:' in p.stdout))
 a.report.write_text(json.dumps(records,indent=2)+'\n');print('Spectral sparse guards',sum(x['passed'] for x in records),'/',len(records));raise SystemExit(not all(x['passed'] for x in records))
if __name__=='__main__':main()
