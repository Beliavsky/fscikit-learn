"""Source diagnostics and identified native shape/NaN-power guards."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np
from compare_parameter_distances import fixture,expected,wire

def cases():
 x=[[0.,1.],[1.,2.]];y=[[1.,0.],[2.,1.]]
 for name,w,message in [('empty',[],'length'),('short',[1.],'length'),('long',[1.,2.,3.],'length'),('negative',[-1.,1.],'nonnegative'),('nan',[np.nan,1.],'finite'),('inf',[np.inf,1.],'finite'),('negative_inf',[-np.inf,1.],'finite')]:
  yield 'w_'+name,fixture('minkowski',x,y,w=w),message,None
 for p,name,message,adapter in [(0.,'zero','positive',None),(-2.,'negative','positive',None),(np.inf,'inf','finite p',None),(np.nan,'nan','NaN','source_accepts_nan_power')]:
  yield 'p_'+name,fixture('minkowski',x,y,w=[1.,1.],p=p),message,adapter
 for name,v in [('missing',None),('empty',[]),('short',[1.]),('long',[1.,2.,3.])]:yield 'V_'+name,fixture('seuclidean',x,y,v=v),'requires V'if v is None else'length',None
 yield 'VI_missing',fixture('mahalanobis',x,y),'requires VI',None
 yield 'VI_auto_scalar',fixture('mahalanobis',[[1.],[2.],[3.]],[[1.],[2.],[3.]],self=True),'at least two',None
 yield 'VI_auto_singular',fixture('mahalanobis',np.zeros((3,2)),np.zeros((3,2)),self=True),'singular',None
 for ni,nj in [(0,0),(1,1),(2,1),(1,2),(2,3),(3,2)]:yield f'VI_shape_{ni}_{nj}',fixture('mahalanobis',x,y,vi=np.ones((ni,nj))),'shape','unsafe_source_shape'
 yield 'V_two_sparse',fixture('seuclidean',x,y,storage=1,v=[1.,1.]),'two CSR',None
 for storage in [2,3]:
  a=np.zeros((2,2))if storage==2 else x;b=np.zeros((2,2))if storage==3 else y
  yield f'V_empty_csr_{storage}',fixture('seuclidean',a,b,storage=storage,v=[1.,1.]),'nonempty',None
 for storage in [1,2,3]:
  for precision in [None,np.eye(2)]:yield f'VI_sparse_{storage}_{precision is None}',fixture('mahalanobis',x,y,storage=storage,vi=precision),'dense',None

def run(executable,profile,report):
 records=[]
 for i,(name,c,message,adapter)in enumerate(cases()):
  error=None;status='not_called_unsafe_source_shape'if adapter=='unsafe_source_shape'else'accepted'
  if adapter!='unsafe_source_shape':
   try:expected(c)
   except(ValueError,TypeError,AttributeError,np.linalg.LinAlgError)as e:error=type(e).__name__;status='rejected'
   if adapter is None:assert error is not None,name
   else:assert error is None,name
  p=subprocess.run([str(executable)],input='1\n'+wire(c,i),capture_output=True,text=True,timeout=15)
  output=p.stdout+p.stderr;primary=next((line for line in output.splitlines()if line.startswith('fscikit-learn:')),'')
  passed=p.returncode!=0 and message.lower()in primary.lower()and not any(s in output for s in ['SIGFPE','Fortran runtime error'])
  records.append(dict(case=name,source_error=error,source_status=status,native_adapter=adapter is not None,adapter_reason=adapter,passed=passed))
 result=dict(profile=profile,checks=records,passed=all(r['passed']for r in records),validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest())
 report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--profile',default='checked');p.add_argument('--report',type=Path,required=True);a=p.parse_args();run(a.executable,a.profile,a.report)
