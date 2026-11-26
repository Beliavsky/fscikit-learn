"""Actual sklearn rejections and typed joint-label/probability rank guards."""
from pathlib import Path
import argparse,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def source_rejects(case):
 i=np.arange(1,13);x=np.c_[np.sin(i),np.cos(.7*i)];y=np.c_[i%2,(i+1)%2,i%2]
 options=dict(n_neighbors=3,algorithm='brute');options.update({1:dict(n_neighbors=0),2:dict(weights='invalid'),3:dict(metric='invalid'),4:dict(p=0),5:dict(p=np.nan),6:dict(metric='precomputed'),14:dict(n_neighbors=13),15:dict(n_neighbors=12)}.get(case,{}));model=KNeighborsClassifier(**options)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  try:
   if case==7:KNeighborsClassifier(metric='precomputed').fit([[-1.,0.],[0.,1.]],y[:2]);return False
   if case==8:y=y[:11]
   if case==9:y=y[:,:0]
   if case==10:x[0,0]=np.nan
   if case==11:model.predict(x);return False
   if case==18:y[:,1]=i%3
   model.fit(x,y if case in[8,9,18]else y[:,0])
   if case==12:model.predict(x[:,:1])
   if case==13:x[0,0]=np.nan;model.predict(x)
   if case==14:model.predict(x)
   if case==15:model.predict(None)
   if case==18:model.score(x,y)
   if case==19:model.score(x,y[:,0],sample_weight=[1.])
   if case==20:model.predict(x[:0])
   if case==21:KNeighborsClassifier(n_neighbors=1,metric='precomputed').fit([[0.,1.],[1.,0.]],y[:2,0]).predict([[-1.,1.]])
  except (ValueError,TypeError,AttributeError):return True
 return False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();checks=[]
 for case in range(1,22):
  if case not in[16,17]:assert source_rejects(case),('Source accepted',case)
  p=subprocess.run([a.executable,str(case)],text=True,capture_output=True);text=p.stdout+p.stderr
  assert p.returncode and 'fscikit-learn:'in text and not any(s in text for s in['SIGFPE','SIGSEGV','Fortran runtime error']),(case,text)
  checks.append(dict(case=case,source_rejection=case not in[16,17],diagnostic=next(s.strip()for s in text.splitlines()if 'fscikit-learn:'in s)))
 report=dict(count=21,source_rejections=19,native_contracts=2,checks=checks);a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
