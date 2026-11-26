"""Actual NumPy full index sorts/partitions with NaNs and equal-key cutoffs."""
from pathlib import Path
import argparse,hashlib,json,subprocess,os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import numpy._core._multiarray_umath as oracle

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--record-differences',action='store_true');a=ap.parse_args()
 cases=[];requests=[]
 for seed in [7,42]:
  rng=np.random.RandomState(seed)
  for n in [1,2,7,8,9,16,31,32,33,40,41,65,129,257,1025]:
   original=rng.normal(size=n)
   for kind in ['first','last','mixed','constant','integer','infinite']:
    x=original.copy()
    if kind=='first':x[0]=np.nan
    if kind=='last':x[-1]=np.nan
    if kind=='mixed':x[::3]=np.nan
    if kind=='constant':x[:]=np.nan
    if kind=='integer':x=np.round(x);x[::4]=np.nan
    if kind=='infinite':x[::3]=np.inf;x[1::3]=-np.inf;x[2::3]=np.nan;x[-1]=np.nan
    for k in sorted(set([1,min(3,n),max(1,n//2),n])):
     cases.append((dict(seed=seed,n=n,kind=kind,k=k),(np.argsort(x)+1,np.argpartition(x,k-1)+1)))
     requests.append(f'{n} {k}\n'+' '.join(map(str,x))+'\n')
 text=''.join(requests);proc=subprocess.run([str(a.executable)],input=text,text=True,capture_output=True);lines=proc.stdout.splitlines()
 if proc.returncode or len(lines)!=2*len(cases):
  for suffix,data in [('in',text),('stdout',proc.stdout),('stderr',proc.stderr)]:a.report.with_suffix('.'+suffix).write_text(data)
 assert proc.returncode==0,proc.stderr
 assert len(lines)==2*len(cases),(len(lines),len(cases))
 differences=[]
 for i,(label,expected)in enumerate(cases):
  for j,name in enumerate(['argsort','argpartition']):
   actual=np.fromstring(lines[2*i+j],sep=' ',dtype=int)
   if not np.array_equal(actual,expected[j]):differences.append(dict(case=label,field=name,actual=actual.tolist(),expected=expected[j].tolist()))
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),groups=2*len(cases),strict_parity=not differences,failed_comparison_groups=len(differences),differences=differences,numpy=np.__version__,oracle_sha256=sha(oracle.__file__),source_sha256=sha(root/'src/sklearn_numpy_ordering.f90'),validator_sha256=sha(__file__),driver_sha256=sha(root/'app/reference_numpy_nan_ordering_driver.f90'),executable_sha256=sha(a.executable),input_sha256=hashlib.sha256(text.encode()).hexdigest())
 a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items()if k!='differences'})
 assert not differences or a.record_differences,differences[:1]
if __name__=='__main__':main()
