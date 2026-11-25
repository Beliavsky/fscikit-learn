"""Source-confirmed chunk validation and one native tagged-output contract."""
from pathlib import Path
import argparse,subprocess,warnings
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances_chunked

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);args=ap.parse_args()
 x=np.ones((3,2));y=np.full((2,2),2.);negative=x.copy();negative[0,0]=-1
 bad=dict(memory_negative=(x,None,dict(working_memory=-1.),'nonnegative'),memory_nan=(x,None,dict(working_memory=np.nan),'not be NaN'),memory_inf=(x,None,dict(working_memory=np.inf),'finite'),memory_precomputed_inf=(x,x[:2],dict(metric='precomputed',working_memory=np.inf),'finite'),metric=(x,None,dict(metric='invalid'),'unsupported'),features=(x,y[:,:1],{},'feature counts'),empty=(x[:0],None,{},'nonempty'),precomputed_negative=(negative,y,dict(metric='precomputed'),'nonnegative'),precomputed_shape=(x,None,dict(metric='precomputed'),'columns'),sparse_metric=(csr_matrix(x),None,dict(metric='dice'),'dense inputs'),variance_missing=(x,y,dict(metric='seuclidean'),'requires V'),precision_missing=(x,y,dict(metric='mahalanobis'),'requires VI'),variance_shape=(x,None,dict(metric='seuclidean',V=[1.]),'V length'),precision_auto_scalar=(x[:,:1],None,dict(metric='mahalanobis'),'multiple features'),weights_shape=(x,None,dict(metric='minkowski',w=[1.]),'weight count'),weights_negative=(x,None,dict(metric='minkowski',w=[1.,-1.]),'nonnegative'),reduction_rows=(x,None,dict(reduce_func=lambda d,s:np.zeros(len(d)+1)),'row count'),reduction_none_member=(x,None,dict(reduce_func=lambda d,s:(None,)),'must be sequences'),reduction_nested=(x,None,dict(reduce_func=lambda d,s:((),)),'nested tuples'),reduction_multiple=(x,None,dict(reduce_func=lambda d,s:(np.zeros(len(d)),np.zeros(len(d)))),'multiple payload'))
 for name,(a,b,kw,message)in bad.items():
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   try:list(pairwise_distances_chunked(a,b,**kw))
   except (ValueError,TypeError,ArithmeticError,np.linalg.LinAlgError):assert name!='reduction_multiple',name
   else:assert name=='reduction_multiple',('Source accepted',name)
  r=subprocess.run([args.executable,name],capture_output=True,text=True);output=r.stdout+r.stderr
  primary=next((s for s in output.splitlines()if s.startswith('fscikit-learn:')),'')
  assert r.returncode and message in primary,(name,output)
  assert 'SIGFPE'not in output and 'Fortran runtime error'not in output,(name,output)
 print(len(bad),'chunked distance guards passed: 19 Source rejections and 1 native payload-tag contract')
if __name__=='__main__':main()
