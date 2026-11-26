"""LinearRegression Source rejections and native target-rank contract."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression,HuberRegressor

def source_rejects(case):
 x=np.array([[1.,0.],[2.,1.],[3.,0.]]);y=np.array([1.,3.,2.]);w=np.ones(3);kw={}
 model=LinearRegression()
 try:
  if case==1:w=np.ones(1)
  if case==2:w[:]=-1
  if case==3:w[:]=np.nan
  if case==4:w[:]=0
  if case==5:y=y[:2]
  if case==6:y[:]=np.nan
  if case==7:x[0,0]=np.nan
  if case==8:x=x[:0];y=y[:0];w=w[:0]
  if case==9:model.positive=True;x=csr_matrix(x)
  if case==10:model.predict(csr_matrix(x));return False
  if case==14:y=np.empty((3,0))
  if case==15:w=np.nan
  if case==16:x=csr_matrix(x[:,:0])
  if case==17:HuberRegressor().fit(x,np.column_stack([y,-y]));return False
  model.fit(x,y,sample_weight=w)
  if case==12:model.predict(csr_matrix(x[:,:1]))
  if case==13:x[0,0]=np.nan;model.predict(csr_matrix(x))
 except (ValueError,TypeError,ArithmeticError):return True
 return False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True);ap.add_argument('--report',type=Path);args=ap.parse_args();checks=[]
 for case in range(1,18):
  if case!=11:assert source_rejects(case),('Source accepted',case)
  proc=subprocess.run([args.executable,str(case)],capture_output=True,text=True);out=proc.stdout+proc.stderr
  assert proc.returncode and 'fscikit-learn:'in out and 'SIGFPE'not in out and 'Fortran runtime error'not in out,(case,out)
  checks.append(dict(case=case,diagnostic=next(row.strip()for row in out.splitlines()if 'fscikit-learn:'in row)))
 report=dict(count=len(checks),source_rejections=16,native_rank_contracts=1,checks=checks);print(json.dumps(report,indent=2))
 if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
