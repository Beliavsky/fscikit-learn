"""Actual sklearn graphical_lasso rejection probes with native diagnostics."""
import argparse,json,subprocess,warnings
from pathlib import Path
import numpy as np
from sklearn.covariance import graphical_lasso

def source_rejects(case):
 c=np.array([[2.,.8],[.8,1.]]);kw=dict(alpha=.3)
 if case==1:kw['alpha']=-1
 if case==2:kw['alpha']=np.nan
 if case==3:kw['mode']='unknown'
 if case==4:kw['tol']=0
 if case==5:kw['tol']=np.nan
 if case==6:kw['enet_tol']=-1
 if case==7:kw['enet_tol']=np.nan
 if case==8:kw['eps']=-1
 if case==9:kw['eps']=np.nan
 if case==10:kw['max_iter']=-1
 if case==11:c=c[:,:1]
 if case==12:c=c[:1,:1]
 if case==13:c=c[:0,:0]
 if case==14:c[0,0]=np.nan
 if case==15:c[:]=0;kw['alpha']=0
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  try:graphical_lasso(c,**kw)
  except (ValueError,ArithmeticError,np.linalg.LinAlgError):return True
 return False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True);ap.add_argument('--report',type=Path);args=ap.parse_args();checks=[]
 for case in range(1,16):
  assert source_rejects(case),('Source accepted',case)
  proc=subprocess.run([args.executable,str(case)],capture_output=True,text=True);out=proc.stdout+proc.stderr
  assert proc.returncode and 'fscikit-learn:'in out,(case,out)
  assert 'SIGFPE'not in out and 'Fortran runtime error'not in out,(case,out)
  checks.append(dict(case=case,diagnostic=next(row.strip()for row in out.splitlines()if 'fscikit-learn:'in row)))
 report=dict(count=len(checks),source_rejections=len(checks),checks=checks);print(json.dumps(report,indent=2))
 if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
