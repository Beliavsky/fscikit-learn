"""GaussianMixture Source rejections and explicit native storage contracts."""
from pathlib import Path
import argparse,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.mixture import GaussianMixture

def source_rejects(case):
 i=np.arange(1,13);x=np.c_[np.sin(i),np.cos(2*i)];kw=dict(n_components=2,max_iter=3)
 if case in [29,32]:return False
 changes={1:dict(init_params='invalid'),2:dict(n_components=0),3:dict(n_components=13),4:dict(max_iter=-1),5:dict(n_init=0),6:dict(tol=-1.),7:dict(tol=np.nan),8:dict(reg_covar=-1.),9:dict(reg_covar=np.nan),10:dict(random_state=-1),11:dict(weights_init=[1.]),12:dict(weights_init=[-.1,1.1]),13:dict(weights_init=[.2,.2]),14:dict(weights_init=[np.nan,.5]),15:dict(means_init=np.zeros((1,2))),16:dict(means_init=np.full((2,2),np.nan)),17:dict(precisions_init=np.ones((2,1,1))),20:dict(covariance_type='diag',precisions_init=[[0.,1.],[1.,1.]]),31:dict(tol=np.inf)}
 kw.update(changes.get(case,{}))
 if case in [18,19,21,28]:
  a=np.repeat(np.eye(2)[None,:,:],2,axis=0)
  if case==18:a[0,0,0]=-1
  if case==19:a[0,0,1]=.5
  if case==21:a[0,0,0]=np.nan
  kw['precisions_init']=a
  if case==28:kw['max_iter']=0
 model=GaussianMixture(**kw)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  try:
   if case==22:x=x[:1]
   if case==23:x=x[:,:0]
   if case==24:model.score_samples(x);return False
   model.fit(x)
   if case==25:model.score_samples(x[:,:1])
   if case==26:x[0,0]=np.nan;model.score_samples(x)
   if case==27:model.sample(0)
   if case==30:model.warm_start=True;model.covariance_type='diag';model.fit(x)
  except (ValueError,TypeError,ArithmeticError,AttributeError):return True
 return False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();checks=[]
 for case in range(1,33):
  if case not in [29,32]:assert source_rejects(case),('Source accepted',case)
  proc=subprocess.run([a.executable,str(case)],text=True,capture_output=True);out=proc.stdout+proc.stderr
  assert proc.returncode and 'fscikit-learn:'in out and 'SIGFPE'not in out and 'Fortran runtime error'not in out,(case,out)
  checks.append(dict(case=case,source_rejection=case not in [29,32],diagnostic=next(row.strip()for row in out.splitlines()if 'fscikit-learn:'in row)))
 report=dict(count=len(checks),source_rejections=30,native_contracts=2,checks=checks);a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
