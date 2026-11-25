"""Isolated source rejections and explicitly identified native adapter guards."""
from pathlib import Path
import argparse,json,subprocess
import numpy as np
from compare_pairwise_reductions import fixture,expected,wire

def cases():
 x=[[0.,1.],[1.,2.]];y=[[1.,0.],[2.,1.]]
 for axis in [-1,2]:yield f'axis_{axis}',fixture('euclidean',x,y,axis=axis),'axis',False
 yield 'features',fixture('euclidean',x,[[1.,0.,2.]]),'feature counts',False
 for target in ['x','y']:
  for value,name in [(np.nan,'nan'),(np.inf,'inf'),(-np.inf,'negative_inf')]:
   a=np.array(x);b=np.array(y);(a if target=='x' else b)[0,0]=value
   yield target+'_'+name,fixture('euclidean',a,b),'finite',False
 for metric in ['notametric','precomputed']:
  yield metric,fixture(metric,x,y),'unsupported',False
 for p in [0.,-2.]:yield f'power_{p}',fixture('minkowski',x,y,p=p),'positive',False
 yield 'nan_power_adapter',fixture('minkowski',x,y,p=np.nan),'NaN',True
 for storage in [1,2,3]:yield f'nan_sparse_{storage}',fixture('nan_euclidean',x,y,storage=storage),'requires dense',False
 yield 'squared_sparse',fixture('sqeuclidean',x,y,storage=1),'two CSR',False
 for metric in ['sqeuclidean','chebyshev','minkowski']:
  yield metric+'_empty_storage',fixture(metric,[[0.,0.]],y,storage=2),'nonempty CSR storage',False
 yield 'haversine_width',fixture('haversine',[[1.,2.,3.]],[[4.,5.,6.]]),'latitude',False
 yield 'no_queries',fixture('euclidean',np.empty((0,2)),y),'nonempty',False
 yield 'no_candidates',fixture('euclidean',x,np.empty((0,2))),'nonempty',False
 yield 'no_features',fixture('euclidean',np.empty((2,0)),np.empty((3,0))),'nonempty',False
 yield 'nan_euclidean_inf',fixture('nan_euclidean',[[np.inf,0.]],y),'infinity',False

def run(executable,profile,report):
 records=[]
 for i,(name,c,diagnostic,adapter) in enumerate(cases()):
  error=None
  try:expected(c)
  except (ValueError,TypeError) as e:error=type(e).__name__
  if not adapter:assert error is not None,name
  else:assert error is None,name
  p=subprocess.run([str(executable)],input='1\n'+wire(c,i),text=True,capture_output=True,timeout=15)
  passed=p.returncode!=0 and diagnostic.lower() in (p.stdout+p.stderr).lower()
  records.append(dict(case=name,source_error=error,native_adapter=adapter,passed=passed))
 result=dict(profile=profile,checks=records,passed=all(r['passed'] for r in records))
 report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--profile',default='checked');p.add_argument('--report',type=Path,required=True);a=p.parse_args();run(a.executable,a.profile,a.report)
