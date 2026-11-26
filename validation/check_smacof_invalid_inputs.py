"""Actual sklearn SMACOF rejections and isolated native diagnostics."""
import argparse,json,subprocess,warnings
from pathlib import Path
import numpy as np
from sklearn.manifold import smacof

def source_rejects(case):
 d=np.array([[0.,1.,3.],[1.,0.,2.],[3.,2.,0.]]);initial=np.ones((3,2));kw={}
 if case==1:kw['n_components']=0
 if case==2:kw['n_init']=0
 if case==3:kw['max_iter']=0
 if case==4:kw['eps']=-1
 if case==5:kw['eps']=np.nan
 if case==6:kw['random_state']=-1
 if case==7:kw['normalized_stress']='unknown'
 if case==8:kw['normalized_stress']=True
 if case==9:d[0,1]=4
 if case==10:d=d[:,:2]
 if case==11:d=d[:0,:0]
 if case==12:d[0,0]=np.nan
 if case==13:kw['init']=initial[:2]
 if case==14:initial[0,0]=np.nan;kw['init']=initial
 if case==15:d[:]=0;kw['metric']=False
 if case==16:kw.update(metric=False,init=initial*0,max_iter=3)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  try:smacof(d,**kw)
  except (ValueError,ArithmeticError):return True
 return False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True);ap.add_argument('--report',type=Path);args=ap.parse_args()
 checks=[]
 for case in range(1,17):
  assert source_rejects(case),('Source accepted',case)
  proc=subprocess.run([args.executable,str(case)],capture_output=True,text=True);output=proc.stdout+proc.stderr
  assert proc.returncode and 'fscikit-learn:'in output,(case,output)
  assert 'SIGFPE'not in output and 'Fortran runtime error'not in output,(case,output)
  checks.append(dict(case=case,diagnostic=next(line.strip()for line in output.splitlines()if 'fscikit-learn:'in line)))
 report=dict(count=len(checks),source_rejections=len(checks),checks=checks)
 print(json.dumps(report,indent=2))
 if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
