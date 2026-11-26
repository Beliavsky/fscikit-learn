"""Source radius-classifier rejections, rank adapters and unrepresented dtype rules."""
from pathlib import Path
import argparse,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
from sklearn.neighbors import RadiusNeighborsClassifier

def source_rejects(case):
 i=np.arange(1,13);x=np.c_[np.sin(i),np.cos(.7*i)];y=np.asarray(np.c_[i%2,(i+1)%2,i%2],dtype=np.int64)
 options=dict(radius=1,algorithm='brute');options.update({1:dict(radius=-1),2:dict(radius=np.nan),3:dict(weights='invalid'),4:dict(metric='invalid'),5:dict(p=0),6:dict(p=np.nan),7:dict(metric='precomputed'),21:dict(outlier_label=[0,1]),22:dict(radius=0),26:dict(p=-1,metric='euclidean')}.get(case,{}));model=RadiusNeighborsClassifier(**options)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  try:
   if case==8:RadiusNeighborsClassifier(metric='precomputed').fit([[-1.,0.],[0.,1.]],y[:2]);return False
   if case==9:y=y[:11]
   if case==10:y=y[:,:0]
   if case==11:x[0,0]=np.nan
   if case==12:model.predict(x);return False
   if case==18:y[:,1]=i%3
   if case==23:x=x[:0];y=y[:0]
   if case==24:x=x[:,:0]
   model.fit(x,y if case in[9,10,18,21,23]else y[:,0])
   if case==13:model.predict(x[:,:1])
   if case==14:x[0,0]=np.nan;model.predict(x)
   if case==15:model.predict(x[:0])
   if case==18:model.score(x,y)
   if case==19:model.score(x,y[:,0],sample_weight=[1.])
   if case==20:RadiusNeighborsClassifier(metric='precomputed').fit([[0.,1.],[1.,0.]],y[:2,0]).predict([[-1.,1.]])
   if case==22:model.predict(x+100)
  except (ValueError,TypeError,AttributeError):return True
 return False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();checks=[]
 for case in range(1,27):
  if case not in[16,17,25]:assert source_rejects(case),('Source accepted',case)
  p=subprocess.run([a.executable,str(case)],text=True,capture_output=True);text=p.stdout+p.stderr
  assert p.returncode and 'fscikit-learn:'in text and not any(s in text for s in['SIGFPE','SIGSEGV','Fortran runtime error']),(case,text)
  checks.append(dict(case=case,source_rejection=case not in[16,17,25],diagnostic=next(s.strip()for s in text.splitlines()if 'fscikit-learn:'in s)))
 dtype_checks=[]
 for label in[-99,np.int32(-99)]:
  try:RadiusNeighborsClassifier(outlier_label=label).fit([[0.],[1.]],np.array([0,1],dtype=np.int32))
  except TypeError as error:dtype_checks.append(dict(parameter_type=type(label).__name__,source_rejection=str(error),native_status='not represented by fixed-kind integer fields'))
  else:raise AssertionError('Source dtype rejection changed')
 report=dict(count=26,source_rejections=23,native_contracts=3,checks=checks,unrepresented_source_dtype_rejections=dtype_checks);a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
