"""Classification report input guards, checked against actual sklearn rejections."""
from pathlib import Path
import argparse,subprocess,warnings
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);args=ap.parse_args()
 y=[0,1,1];p=[1,1,0];ym=np.ones((3,2),int);pm=np.zeros((3,2),int)
 sparse=csr_matrix(ym);bad_value=sparse.copy();bad_value.data[0]=2
 duplicate=sparse.copy();duplicate.indices[1]=duplicate.indices[0]
 cases=dict(digits=(y,p,dict(digits=-1),'digits must'),dict_digits=(y,p,dict(digits=-1,output_dict=True),'digits must'),names=(y,p,dict(target_names=['a']),'target_names count'),length=(y,p[:2],{},'lengths differ'),shape=(ym,pm[:2],{},'shapes differ'),columns=(ym[:,:0],pm[:,:0],{},'need columns'),weight=(y,p,dict(sample_weight=[1.]),'weight length'),zero=(y,p,dict(zero_division=2.),'zero_division'),labels_low=(ym,pm,dict(labels=[-1]),'[0,n_labels)'),labels_high=(ym,pm,dict(labels=[2]),'[0,n_labels)'),indicator=(np.array([[2,3],[1,1],[1,1]]),pm,{},'at most two'),csr_value=(bad_value,sparse,{},'store ones'),csr_duplicate=(duplicate,sparse,{},'sorted and unique'),empty_labels=(y,p,dict(labels=[]),'text needs target names'),empty_names=(y,p,dict(labels=[0,1],target_names=[]),'text needs target names'))
 for name,(a,b,kw,message)in cases.items():
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   try:classification_report(a,b,**kw)
   except (ValueError,TypeError):
    assert name!='csr_duplicate',('Source unexpectedly rejected documented native contract',name)
   else:assert name=='csr_duplicate',('Source accepted',name)
  r=subprocess.run([args.executable,name],capture_output=True,text=True);output=r.stdout+r.stderr
  primary=next((line for line in output.splitlines()if line.startswith('fscikit-learn:')),'')
  assert r.returncode and message in primary,(name,output)
  assert 'SIGFPE'not in output and 'Fortran runtime error'not in output,(name,output)
 print(len(cases),'classification report guards passed: 14 Source rejections and 1 inherited native canonical-CSR contract')
if __name__=='__main__':main()
