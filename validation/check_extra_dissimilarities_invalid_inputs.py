"""Controlled source rejections for added nearest-pair dissimilarities."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np
from compare_pairwise_reductions import fixture,expected,wire
from compare_extra_dissimilarities import METRICS
DENSE_ONLY=['correlation','hamming','dice','jaccard','rogerstanimoto','russellrao','sokalmichener','sokalsneath','yule']
def cases():
 x=[[1.,0.],[0.,2.]];y=[[2.,1.],[1.,0.]]
 for metric in DENSE_ONLY:
  for storage in [1,2,3]:yield f'{metric}_sparse_{storage}',fixture(metric,x,y,storage=storage),'requires dense'
 for metric in ['braycurtis','canberra','matching']:
  for storage in [1,2,3]:
   a=[[0.,0.]] if storage in [1,2] else x;b=[[0.,0.]] if storage in [1,3] else y
   yield f'{metric}_empty_storage_{storage}',fixture(metric,a,b,storage=storage),'nonempty CSR storage'
 for metric in METRICS:
  for value,name in [(np.nan,'nan'),(np.inf,'inf')]:yield f'{metric}_{name}',fixture(metric,[[value,0.]],y),'finite'
def run(executable,profile,report):
 records=[]
 for i,(name,c,message) in enumerate(cases()):
  error=None
  try:expected(c)
  except (ValueError,TypeError) as e:error=type(e).__name__
  assert error is not None,name
  p=subprocess.run([str(executable)],input='1\n'+wire(c,i),text=True,capture_output=True,timeout=15)
  text=p.stdout+p.stderr;primary=next((line for line in text.splitlines()if line.startswith('fscikit-learn:')),'')
  passed=p.returncode!=0 and message.lower()in primary.lower()and not any(s in text for s in ['SIGFPE','Fortran runtime error'])
  records.append(dict(case=name,source_error=error,native_adapter=False,passed=passed))
 result=dict(profile=profile,checks=records,passed=all(r['passed']for r in records),validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest())
 report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--profile',default='checked');p.add_argument('--report',type=Path,required=True);a=p.parse_args();run(a.executable,a.profile,a.report)
