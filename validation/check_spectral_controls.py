"""Global spectral parameter validation and dense/sparse shape-boundary parity."""
from pathlib import Path
import argparse,contextlib,io,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 for case in range(1,22):
  x=np.ones((6,6));s=csr_matrix(x);m=SpectralClustering(n_clusters=2,n_components=2,affinity='precomputed',assign_labels='cluster_qr',n_neighbors=3,random_state=0);error=None;capture=io.StringIO()
  try:
   with warnings.catch_warnings(),contextlib.redirect_stdout(capture):
    warnings.simplefilter('ignore')
    if case==1:m.set_params(n_init=0).fit(x)
    if case==2:m.set_params(n_init=0,assign_labels='discretize').fit(s)
    if case==3:m.set_params(gamma=-1).fit(x)
    if case==4:m.set_params(degree=-.5).fit(s)
    if case==5:m.set_params(coef0=np.nan).fit(x)
    if case==6:m.set_params(degree=np.inf).fit(s)
    if case==7:m.set_params(n_neighbors=0).fit(x)
    if case==8:m.set_params(random_state=-1).fit(s)
    if case==9:m.set_params(n_clusters=0).fit(x)
    if case==10:m.set_params(n_components=-1).fit(s)
    if case==11:m.set_params(n_components=6).fit(s)
    if case==12:m.set_params(n_components=8).fit(s)
    if case==13:m.fit(-x)
    if case==14:m.set_params(affinity='chi2').fit(s)
    if case==15:x[0,0]=-1;m.set_params(affinity='chi2').fit(x)
    if case==16:m.set_params(affinity='linear',degree=np.nan).fit(x)
    if case==17:SpectralEmbedding(n_components=6,affinity='precomputed').fit(s)
    if case==18:SpectralEmbedding(affinity='linear').fit(x)
    if case==19:m.set_params(n_clusters=20,n_components=1).fit(x)
    if case==20:m.set_params(n_components=8).fit(x)
    if case==21:SpectralEmbedding(n_components=8,affinity='rbf').fit(s)
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  p=subprocess.run([str(a.executable),str(case)],capture_output=True,text=True)
  passed=(p.returncode==0)==(error is None) and (p.returncode==0 and 'ACCEPTED' in p.stdout or p.returncode==1 and 'fscikit-learn:' in p.stdout)
  records.append(dict(case=case,source_error=error,source_stdout=capture.getvalue(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,passed=passed))
 a.report.write_text(json.dumps(records,indent=2)+'\n');print('Spectral control validation',sum(r['passed'] for r in records),'/',len(records));raise SystemExit(not all(r['passed'] for r in records))
if __name__=='__main__':main()
