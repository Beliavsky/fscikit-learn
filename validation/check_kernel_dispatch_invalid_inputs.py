"""Controlled rejections at the typed pairwise-kernel dispatch boundary."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np
from compare_kernel_dispatch import fixture,expected,wire

def cases():
 x=[[0.,1.],[1.,2.]];y=[[1.,0.],[2.,1.]]
 yield 'empty',fixture('linear',np.empty((0,2)),y),'nonempty'
 yield 'features',fixture('linear',x,[[1.,2.,3.]]),'feature counts'
 yield 'nan_x',fixture('linear',[[np.nan,0.]],y),'finite'
 yield 'inf_y',fixture('linear',x,[[np.inf,0.]]),'finite'
 yield 'unknown',fixture('unknown',x,y),'unknown'
 yield 'precomputed_square',fixture('precomputed',[[1.,2.,3.]],mode=1),'column count'
 yield 'precomputed_reference',fixture('precomputed',x,[[1.,2.]]),'column count'
 yield 'precomputed_nan_y',fixture('precomputed',x,[[np.nan,2.],[1.,2.]]),'finite'
 for metric in ['chi2','additive_chi2']:
  for storage in [1,2,3]:yield f'{metric}_sparse_{storage}',fixture(metric,x,y,storage=storage),'dense'
  yield metric+'_negative_x',fixture(metric,[[-1.,0.]],y),'nonnegative'
  yield metric+'_negative_y',fixture(metric,x,[[-1.,0.]]),'nonnegative'
 applicable=dict(linear=[],cosine=[],rbf=['gamma'],laplacian=['gamma'],sigmoid=['gamma','coef0'],additive_chi2=[],chi2=['gamma'])
 for metric,keys in applicable.items():
  for param in ['gamma','degree','coef0']:
   if param not in keys:yield f'{metric}_{param}',fixture(metric,x,y,**{param:2.}),'not applicable'

def run(executable,profile,report):
 checks=[]
 for i,(name,c,message)in enumerate(cases()):
  error=None
  try:expected(c)
  except(ValueError,TypeError)as e:error=type(e).__name__
  assert error is not None,name
  p=subprocess.run([str(executable)],input='1\n'+wire(c,i),text=True,capture_output=True,timeout=15)
  out=p.stdout+p.stderr;primary=next((line for line in out.splitlines()if line.startswith('fscikit-learn:')),'')
  passed=p.returncode!=0 and message in primary and not any(s in out for s in ['SIGFPE','Fortran runtime error'])
  checks.append(dict(case=name,source_error=error,native_adapter=False,passed=passed))
 result=dict(profile=profile,checks=checks,passed=all(r['passed']for r in checks),validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest())
 report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--profile',default='checked');p.add_argument('--report',type=Path,required=True);a=p.parse_args();run(a.executable,a.profile,a.report)
