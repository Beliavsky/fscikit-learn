"""Blocked Ledoit-Wolf shrinkage against the installed sklearn function."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,sklearn
from sklearn.covariance import ledoit_wolf_shrinkage,LedoitWolf


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True);parser.add_argument('--report')
 args=parser.parse_args();cases=[];lines=[];warning_count=0
 for seed in [7,42,2026]:
  rng=np.random.default_rng(seed)
  for n,p in [(31,7),(7,19),(1,5),(17,1),(10,8)]:
   for kind in ['normal','shifted','rank_deficient','zeros','constant','isotropic']:
    x=rng.normal(size=(n,p))
    if kind=='shifted':x+=np.arange(p)*3+1e3
    if kind=='rank_deficient'and p>1:x[:,-1]=x[:,0]
    if kind=='zeros':x[:]=0
    if kind=='constant':x[:]=np.arange(p)+1
    if kind=='isotropic':x[:]=0;np.fill_diagonal(x,1)
    for centered in [False,True]:
     for block in sorted(set([1,2,3,p,p+1,1000])):
      with warnings.catch_warnings(record=True)as caught:
       warnings.simplefilter('always');expected=ledoit_wolf_shrinkage(x,assume_centered=centered,block_size=block)
       model=LedoitWolf(assume_centered=centered,block_size=block).fit(x)
      warning_count+=len(caught)
      cases.append((expected,model,dict(seed=seed,n=n,p=p,kind=kind,centered=centered,block=block)))
      lines += [f'{n} {p} {block} {int(centered)} 1']
      lines += [' '.join(map(str,row))for row in x]
 for n,p in [(1,1),(1,4),(3,4)]:
  for value in [1e-160,1e80,1e160,1e300]:
   x=np.full((n,p),value)
   for centered in [False,True]:
    for block in [1,3,1000]:
     with warnings.catch_warnings(record=True)as caught:
      warnings.simplefilter('always');expected=ledoit_wolf_shrinkage(x,assume_centered=centered,block_size=block)
     warning_count+=len(caught)
     cases.append((expected,None,dict(n=n,p=p,value=value,centered=centered,block=block)))
     lines += [f'{n} {p} {block} {int(centered)} 0']
     lines += [' '.join(map(str,row))for row in x]
 proc=subprocess.run([args.executable],input='\n'.join(lines)+'\n',capture_output=True,text=True)
 if proc.returncode:raise RuntimeError(proc.stderr+'\nLast output:\n'+proc.stdout[-1500:])
 rows=iter(proc.stdout.splitlines());worst=0;groups=0
 for want,model,label in cases:
  got=float(next(rows));groups+=1
  np.testing.assert_allclose(got,want,rtol=3e-8,atol=3e-8,equal_nan=True,err_msg=str(label))
  if np.isfinite(want):worst=max(worst,float(abs(got-want)/(1+abs(want))))
  if model is not None:
   p=label['p']
   values=[float(next(rows)),np.fromstring(next(rows),sep=' '),
     np.array([np.fromstring(next(rows),sep=' ')for _ in range(p)]),
     np.array([np.fromstring(next(rows),sep=' ')for _ in range(p)])]
   for got,want in zip(values,[model.shrinkage_,model.location_,model.covariance_,model.precision_]):
    np.testing.assert_allclose(got,want,rtol=3e-8,atol=3e-8,equal_nan=True,err_msg=str(label))
    worst=max(worst,float(np.max(np.abs(got-want)/(1+np.abs(want)))))
    groups+=1
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),groups=groups,max_scaled_error=worst,source_warnings=warning_count,
  sklearn=sklearn.__version__,numpy=np.__version__,executable_sha256=hashlib.sha256(Path(args.executable).read_bytes()).hexdigest(),
  source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()for name in ['src/sklearn_shrinkage.f90','src/sklearn_covariance.f90']},
  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  input_sha256=hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest(),
  configuration_inventory=[c[2]for c in cases])
 print(json.dumps({k:v for k,v in report.items()if k!='configuration_inventory'},indent=2))
 if args.report:Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
