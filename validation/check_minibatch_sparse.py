"""MiniBatchNMF Source/native sparse validation and permissive streaming cases."""
from pathlib import Path
import argparse,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import MiniBatchNMF

def source(which,dense=False):
 a=np.array([[1.,.2,.8],[.5,0,.9],[0,.7,.3],[1.,.4,0.]])
 x=csr_matrix(a);q=x.copy();w=np.full((4,2),.3);h=np.full((2,3),.4);partial=False
 m=MiniBatchNMF(n_components=2,init='random',max_iter=3,batch_size=2,random_state=0)
 if which==0:m.max_iter=0
 if which==1:m.batch_size=0
 if which==2:m.tol=-.1
 if which==3:m.tol=np.nan
 if which==4:m.alpha_W=-.1
 if which==5:m.alpha_H=-.2
 if which==6:m.l1_ratio=2.
 if which==7:m.forget_factor=-.1
 if which==8:m.forget_factor=1.1
 if which==9:m.fresh_restarts_max_iter=0
 if which==10:m.transform_max_iter=-1
 if which==11:m.max_no_improvement=0
 if which==12:m.max_no_improvement=-2
 if which==13:m.random_state=-1
 if which==14:m.n_components=-1
 if which==15:m.init='bad'
 if which==16:m.beta_loss=np.nan
 if which==17:x.data[0]=np.nan
 if which==18:x.data[0]=np.inf
 if which==19:x.data[0]=-.5
 if which==20:x=csr_matrix((0,3))
 if which==21:x=csr_matrix((4,0))
 if which==22:q=csr_matrix((q.data,q.indices,q.indptr),shape=(4,4))
 if which==23:q.data[0]=np.nan
 if which==24:q=csr_matrix((0,3))
 if which==26:m.init='custom';h[:]=0
 if which==27:m.init='custom';w[0,0]=-1
 if which==28:m.init='custom';h=h[:,:2]
 if which in [29,37,38]:
  x=csr_matrix(([1.,-.5,1.,-.2,1.,-.3,2.,-.5],[0,0,1,1,2,2,0,0],[0,2,4,6,8]),shape=(4,3));partial=which!=29
  if which==38:m.init='custom'
 if which==30:m.beta_loss=0
 if which==31:m.beta_loss=1;x=csr_matrix((a.ravel(),np.tile(np.arange(3),4),np.arange(0,13,3)),shape=(4,3))
 if which==32:m.beta_loss=1;q=csr_matrix(([0.],[0],[0,1]),shape=(1,3))
 if which==33:m.beta_loss=0;partial=True
 if which==34:m.init='custom';x.data[0]=-.1;partial=True
 if which==35:m.beta_loss=0;x=csr_matrix(a+1)
 if which in [39,40]:q.data=-q.data
 if which==41:m.init='nndsvd';m.n_components=5
 if which==42:m.beta_loss=1;partial=True;x=csr_matrix(([0.],[0],[0,1]),shape=(1,3));w=w[:1]
 if dense:x=x.toarray();q=q.toarray()
 stage='query' if which==25 else 'partial' if partial else 'fit';error=None
 try:
  with warnings.catch_warnings(),np.errstate(all='ignore'):
   warnings.simplefilter('ignore')
   if which!=25:
    if partial:m.partial_fit(x,W=w.copy(),H=h.copy())
    else:m.fit(x,W=w.copy(),H=h.copy())
   if which in [36,39]:
    if which==36:m.set_params(beta_loss=.5,n_components=1,batch_size=1,forget_factor=0.,transform_max_iter=1)
    stage='partial';m.partial_fit(q)
   stage='query';m.transform(q)
 except Exception as exc:error=f'{type(exc).__name__}: {exc}'
 return error,stage

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--dense-input',action='store_true');a=ap.parse_args();records=[]
 for which in range(43):
  error,stage=source(which,a.dense_input);p=subprocess.run([str(a.executable),str(which)]+(['dense'] if a.dense_input else []),capture_output=True,text=True);stages=[s.split()[-1] for s in p.stdout.splitlines() if s.startswith('minibatch_guard ')];ns=stages[-1] if stages else None
  passed=(p.returncode==0)==(error is None) and ns==stage and p.returncode in [0,1] and (p.returncode==0 or 'fscikit-learn:' in p.stdout)
  records.append(dict(case=which,source_error=error,source_stage=stage,native_stage=ns,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,passed=passed))
 a.report.write_text(json.dumps(records,indent=2)+'\n');print('MiniBatch CSR controls',sum(x['passed'] for x in records),'/',len(records));raise SystemExit(not all(x['passed'] for x in records))
if __name__=='__main__':main()
