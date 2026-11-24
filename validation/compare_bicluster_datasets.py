"""Sklearn block-diagonal/checkerboard generators and integer-seeded shuffle streams."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.datasets import make_biclusters,make_checkerboard
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SEEDS=[0,7,42,2026,2147483647]
def fixtures():
 shapes=[(0,0),(0,7),(7,0),(1,1),(7,3),(31,17),(61,62),(150,130)]
 for seed,(n,p),(kr,kc),kind,scenario,shuf in itertools.product(SEEDS,shapes,[(1,1),(2,2),(3,4),(7,2)],[0,2],range(3),[False,True]):
  noise,low,high=[(0.,10.,100.),(.25,-3.,7.),(1.e308,2.,2.)][scenario]
  yield dict(kind=kind,n=n,p=p,kr=kr,kc=kc if kind==2 else kr,seed=seed,shuf=shuf,noise=noise,low=low,high=high)
 for seed,n,k,kind,shuf in itertools.product(SEEDS,[60,61,62,1000,4096],[2,3],[0,1,2],[False,True]):
  yield dict(kind=kind,n=n,p=3,kr=k,kc=k,seed=seed,shuf=shuf,noise=0.,low=10.,high=100.)
def expected(c):
 options=dict(shape=(c['n'],c['p']),n_clusters=c['kr'] if c['kind']!=2 else (c['kr'],c['kc']),noise=c['noise'],minval=c['low'],maxval=c['high'],shuffle=c['shuf'],random_state=c['seed'])
 with np.errstate(over='ignore',invalid='ignore'):
  x,rows,cols=(make_biclusters if c['kind']==0 else make_checkerboard)(**options)
 return dict(data=x.ravel(),rows=rows.ravel().astype(float),cols=cols.ravel().astype(float),metadata=np.array([c['n'],c['p'],len(rows)],dtype=float),repeat_identity=np.ones(1))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--report',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--case',type=int)
 args=ap.parse_args();cases=list(fixtures())
 if args.case is not None:cases=[cases[args.case]]
 inputs=[str(len(cases))];references=[]
 for i,c in enumerate(cases):
  inputs.append(' '.join(map(str,[c['kind'],c['n'],c['p'],c['kr'],c['kc'],c['seed'],int(c['shuf']),c['noise'],c['low'],c['high']])));references.append(expected(c))
  if (i+1)%500==0:print('Prepared',i+1,'source configurations',flush=True)
 exe=args.executable.resolve();run=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,cwd=ROOT);assert run.returncode==0,(run.stdout[-2500:],run.stderr[-2500:])
 checks=[];inventories=[];index=-1;fields={};nonfinite=0
 for line in run.stdout.splitlines():
  parts=line.split()
  if parts[0]=='BEGIN':index+=1;fields={}
  elif parts[0]=='END':
   want=references[index];assert fields.keys()==want.keys(),(index,fields.keys()^want.keys());inventories.append(sorted(fields))
   for field,value in fields.items():
    oracle=want[field];tolerance=3e-12 if field=='data' else 0
    np.testing.assert_allclose(value,oracle,rtol=tolerance,atol=tolerance,equal_nan=True,err_msg=str((index,cases[index],field)))
    finite=np.isfinite(value)&np.isfinite(oracle);nonfinite+=int(np.count_nonzero(~np.isfinite(oracle)))
    checks.append(dict(fixture=index,field=field,passed=True,max_abs_error=float(np.max(np.abs(value[finite]-oracle[finite]),initial=0))))
  else:
   name=parts[0];assert name not in fields;fields[name]=np.array(list(map(float,parts[2:])));assert int(parts[1])==len(fields[name])
 assert index+1==len(cases)
 invalid=dict(shape='two entries',negative_shape='nonnegative',clusters='positive',pair_shape='two entries',pair_clusters='positive',block_overflow='index range',noise='nonnegative',nan_noise='finite',inf_noise='finite',nan_min='finite',inf_max='finite',bounds='at least',range='range must be finite',seed='nonnegative')
 rejections=[]
 for mode,message in invalid.items():
  result=subprocess.run([str(exe),mode],capture_output=True,text=True,cwd=ROOT);output=result.stdout+result.stderr;diagnostic=next((line for line in output.splitlines() if line.startswith('fscikit-learn:')), '')
  assert result.returncode!=0 and message.lower() in diagnostic.lower(),(mode,output)
  assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(mode,output)
  rejections.append(dict(case=mode,passed=True))
 import sklearn.datasets._samples_generator as source
 report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,rejections=rejections,source_nonfinite_values=nonfinite,source_sha256=sha(source.__file__),validator_sha256=sha(__file__),executable_sha256=sha(exe),native_source_sha256={name:sha(ROOT/'src'/name) for name in ['sklearn_bicluster_datasets.f90','sklearn_datasets.f90','sklearn_legacy_rng.f90','sklearn_ieee_math.f90']},numpy_source_sha256={name:sha(ROOT/'validation/results/tools'/name) for name in ['numpy-2.2.6-distributions.c','numpy-2.2.6-legacy-distributions.c']})
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(len(checks),'groups /',len(cases),'fixtures /',len(rejections),'diagnostics passed;',nonfinite,'nonfinite source values',flush=True)
if __name__=='__main__':main()
