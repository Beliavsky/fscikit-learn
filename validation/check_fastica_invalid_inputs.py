"""Source functional FastICA parameter/numerical rejections and native diagnostics."""
import argparse,json,subprocess,warnings
from pathlib import Path
import numpy as np
from sklearn.decomposition import fastica

def source_rejects(case):
 x=np.column_stack([np.arange(1,6),np.sin(np.arange(1,6))]);w=np.zeros((2,2));kw={}
 if case==1:kw['n_components']=-1
 if case==2:kw['max_iter']=0
 if case==3:kw['tol']=-1
 if case==4:kw['tol']=np.nan
 if case==5:kw['fun_args']={'alpha':.5}
 if case==6:kw['fun_args']={'alpha':np.nan}
 if case==7:kw['algorithm']='unknown'
 if case==8:kw['whiten']='unknown'
 if case==9:kw['fun']='unknown'
 if case==10:kw['whiten_solver']='unknown'
 if case==11:kw['random_state']=-1
 if case==12:x[0,0]=np.nan
 if case==13:x=x[:1]
 if case==14:kw['w_init']=w[:1,:1]
 if case==15:w[0,0]=np.nan;kw['w_init']=w
 if case==16:kw['w_init']=w
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  try:fastica(x,**kw)
  except (ValueError,ArithmeticError,np.linalg.LinAlgError):return True
 return False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True);ap.add_argument('--report',type=Path);args=ap.parse_args();checks=[]
 for case in range(1,17):
  assert source_rejects(case),('Source accepted',case)
  proc=subprocess.run([args.executable,str(case)],capture_output=True,text=True);out=proc.stdout+proc.stderr
  assert proc.returncode and 'fscikit-learn:'in out and 'SIGFPE'not in out and 'Fortran runtime error'not in out,(case,out)
  checks.append(dict(case=case,diagnostic=next(row.strip()for row in out.splitlines()if 'fscikit-learn:'in row)))
 report=dict(count=len(checks),source_rejections=len(checks),checks=checks);print(json.dumps(report,indent=2))
 if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
