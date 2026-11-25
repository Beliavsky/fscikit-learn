"""Ridge estimator guards, including native rank and fitted-state contracts."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

def source_rejects(case):
 x=np.array([[1.,0.],[2.,1.],[3.,0.]]);y=np.array([1.,3.,2.]);w=np.ones(3);options={}
 if case==1:options['solver']='unknown'
 if case==2:options['max_iter']=0
 if case==3:options['tol']=-1
 if case==4:options['tol']=np.nan
 if case==5:options['solver']='lbfgs'
 if case==6:options.update(positive=True,solver='lsqr')
 if case==7:options['alpha']=-1
 if case==8:options['alpha']=np.array([1.,2.])
 if case==9:w[:]=np.nan
 if case==10:w=np.ones(1)
 if case in [11,12]:options['solver']='cholesky' if case==11 else 'saga';x=csr_matrix(x)
 if case==16:options['random_state']=-1
 if case==18:w[:]=0
 model=Ridge(**options)
 try:
  if case==13:model.predict(csr_matrix(x))
  else:
   model.fit(x,y,sample_weight=w)
   if case==15:model.predict(csr_matrix(x[:,:1]))
   if case==17:x[0,0]=np.nan;model.predict(csr_matrix(x))
 except (ValueError,ArithmeticError):return True
 return False

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True);parser.add_argument('--report',type=Path)
 args=parser.parse_args();cases=[]
 for case in range(1,19):
  if case!=14:assert source_rejects(case),('Source accepted',case)
  result=subprocess.run([args.executable,str(case)],capture_output=True,text=True)
  output=result.stdout+result.stderr
  assert result.returncode and 'fscikit-learn:' in output,(case,output)
  assert 'SIGFPE' not in output and 'Fortran runtime error' not in output,(case,output)
  cases.append(dict(case=case,diagnostic=next(line.strip() for line in output.splitlines() if 'fscikit-learn:' in line)))
 report=dict(cases=cases,count=len(cases),source_rejections=17,native_rank_contracts=1);print(json.dumps(report,indent=2))
 if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
