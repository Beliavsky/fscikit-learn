"""Actual sklearn neighbor regression input rejections and one typed rank contract."""
from pathlib import Path
import argparse,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor

def source_rejects(case):
 i=np.arange(1,13);x=np.c_[np.sin(i),np.cos(.7*i)];y=np.c_[i,-i].astype(float);options=dict(n_neighbors=3,algorithm='brute')
 changes={1:dict(n_neighbors=0),2:dict(weights='invalid'),3:dict(metric='invalid'),4:dict(p=0),5:dict(p=np.nan),6:dict(metric='precomputed'),16:dict(n_neighbors=13),17:dict(n_neighbors=12),29:dict(p=-1,metric='euclidean')};options.update(changes.get(case,{}));model=KNeighborsRegressor(**options)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  try:
   if case==7:KNeighborsRegressor(metric='precomputed').fit([[-1.,0.],[0.,1.]],y[:2]);return False
   if case==8:y=y[:11]
   if case==9:y=y[:,:0]
   if case==10:y[0,0]=np.nan
   if case==11:x[0,0]=np.nan
   if case==12:x=x[:,:0]
   if case==13:model.predict(x);return False
   if case in [21,22]:RadiusNeighborsRegressor(radius=-1. if case==21 else np.nan).fit(x,y);return False
   if case==24:x=x[:0];y=y[:0]
   model.fit(x,y)
   if case==14:model.predict(x[:,:1])
   if case==15:x[0,0]=np.nan;model.predict(x)
   if case==16:model.predict(x)
   if case==17:model.predict(None)
   if case==19:model.score(x,y[:,0])
   if case==20:RadiusNeighborsRegressor(radius=0.).fit(x,y).score(x+10,y)
   if case==23:KNeighborsRegressor(n_neighbors=1,metric='precomputed').fit([[0.,1.],[1.,0.]],y[:2]).predict([[-1.,1.]])
   if case==25:model.score(x,y,sample_weight=[1.])
   if case==26:model.predict(x[:0])
   if case==27:RadiusNeighborsRegressor().fit(x,y).predict(x[:0])
   if case==28:RadiusNeighborsRegressor(radius=np.inf).fit(x,y).score(x,y,sample_weight=[1.])
  except (ValueError,TypeError,AttributeError):return True
 return False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();checks=[]
 for case in range(1,30):
  if case!=18:assert source_rejects(case),('Source accepted',case)
  p=subprocess.run([a.executable,str(case)],text=True,capture_output=True);text=p.stdout+p.stderr
  assert p.returncode and 'fscikit-learn:'in text and not any(s in text for s in ['SIGFPE','SIGSEGV','Fortran runtime error']),(case,text)
  checks.append(dict(case=case,source_rejection=case!=18,diagnostic=next(s.strip()for s in text.splitlines()if 'fscikit-learn:'in s)))
 report=dict(count=29,source_rejections=28,native_contracts=1,checks=checks);a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
