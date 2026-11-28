"""Actual sklearn/Fortran CSR NMF validation boundaries, including cancellation."""
from pathlib import Path
import argparse,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

def source(which):
 a=np.array([[1.,.2,.8],[.5,0,.9],[0,.7,.3],[1.,.4,0.]])
 x=csr_matrix(a);q=x.copy();w=np.full((4,2),.3);h=np.full((2,3),.4)
 m=NMF(n_components=2,solver='cd',init='random',max_iter=5,random_state=0,tol=1e-6)
 if which==0:m.max_iter=0
 if which==1:m.tol=np.nan
 if which==2:m.beta_loss=np.inf
 if which==3:m.solver='bad'
 if which==4:m.init='bad'
 if which==5:m.random_state=-1
 if which==6:m.n_components=-1
 if which==7:m.alpha_W=-.1
 if which==8:m.alpha_H=-.2
 if which==9:m.l1_ratio=2.
 if which==10:x.data[0]=np.nan
 if which==11:x.data[0]=np.inf
 if which==12:x.data[0]=-.5
 if which==13:x=csr_matrix((0,3))
 if which==14:x=csr_matrix((4,0))
 if which==15:m.init='custom';h[:]=0
 if which==16:m.init='custom';h[0,0]=-1
 if which==17:m.init='custom';h=h[:,:2]
 if which==18:q=csr_matrix((q.data,q.indices,q.indptr),shape=(4,4))
 if which==19:q.data[0]=np.nan
 if which==20:q=csr_matrix((0,3))
 if which==22:m.solver='mu';m.beta_loss=0
 if which==23:m.solver='mu';m.beta_loss=1;x=csr_matrix((a.ravel(),np.tile(np.arange(3),4),np.arange(0,13,3)),shape=(4,3))
 if which==24:m.n_components=5;m.init='nndsvd'
 if which==25:m.beta_loss=1
 if which==26:m.init='custom';w[:]=0
 if which==27:x=csr_matrix(([1.,-.5,1.,-.2,1.,-.3,2.,-.5],[0,0,1,1,2,2,0,0],[0,2,4,6,8]),shape=(4,3))
 stage='query' if which==21 else 'fit';error=None
 try:
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   if which!=21:m.fit(x,W=w,H=h)
   stage='query';m.transform(q)
 except Exception as exc:error=f'{type(exc).__name__}: {exc}'
 return error,stage

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 for which in range(28):
  error,stage=source(which);p=subprocess.run([str(a.executable),str(which)],capture_output=True,text=True);stages=[s.split()[-1] for s in p.stdout.splitlines() if s.startswith('nmf_stage ')];native_stage=stages[-1] if stages else None
  passed=(p.returncode==0)==(error is None) and native_stage==stage and p.returncode in [0,1] and (p.returncode==0 or 'fscikit-learn:' in p.stdout)
  records.append(dict(case=which,source_error=error,source_stage=stage,native_stage=native_stage,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,passed=passed))
 a.report.write_text(json.dumps(records,indent=2)+'\n');print('CSR NMF controls',sum(r['passed'] for r in records),'/',len(records));raise SystemExit(not all(r['passed'] for r in records))
if __name__=='__main__':main()
