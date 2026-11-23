"""Seeded regression and low-rank generator arrays against sklearn 1.6.1."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy
import sklearn
from sklearn.datasets import make_low_rank_matrix,make_regression
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SEEDS=[0,7,42,2026,2147483647]
SHAPES=[(1,1),(1,7),(7,1),(3,7),(7,3),(8,8),(31,12),(70,5),(75,70)]
def fixtures():
 for seed,(n,p),rank,tail in itertools.product(SEEDS,SHAPES,range(4),[0.,.2,.5,1.]):
  yield dict(mode=0,n=n,p=p,rank=[1,3,10,min(n,p)+5][rank],tail=tail,seed=seed)
 yield dict(mode=2,n=100,p=100,rank=10,tail=.5,seed=0)
 for seed,(n,p),nt,rank,scenario in itertools.product(SEEDS,SHAPES,[1,2,4],range(4),range(4)):
  info,noise,bias,shuf,tail=[(0,0.,2.,False,0.),(1,.2,-3.,True,.5),(p+5,1.5,0.,False,1.),(p//2,0.,2.,True,.3)][scenario]
  yield dict(mode=1,n=n,p=p,nt=nt,rank=[0,1,3,min(n,p)+5][rank],tail=tail,seed=seed,info=info,noise=noise,bias=bias,shuf=shuf,scenario=scenario)
 for seed,nt,rank,bias,shuf in itertools.product(SEEDS,[1,2,4],[0,3],[-1.e308,1.e308],[False,True]):
  yield dict(mode=1,n=31,p=5,nt=nt,rank=rank,tail=.5,seed=seed,info=7,noise=1.e308,bias=bias,shuf=shuf,scenario=4)
def expected(c):
 if c['mode'] in [0,2]:
  x=make_low_rank_matrix(n_samples=c['n'],n_features=c['p'],effective_rank=c['rank'],tail_strength=c['tail'],random_state=c['seed'])
  result=dict(x=x)
  if c['mode']==0:result['repeated_x']=x
 else:
  options=dict(n_samples=c['n'],n_features=c['p'],n_targets=c['nt'],n_informative=c['info'],noise=c['noise'],bias=c['bias'],shuffle=c['shuf'],tail_strength=c['tail'],effective_rank=c['rank'] or None,random_state=c['seed'])
  with np.errstate(over='ignore',invalid='ignore'):
   x,y,coef=make_regression(**options,coef=True);other,without=make_regression(**options,coef=False)
  np.testing.assert_array_equal(x,other);np.testing.assert_array_equal(y,without)
  result=dict(x=x,y=np.asarray(y).reshape(c['n'],c['nt']),coef=np.asarray(coef).reshape(c['p'],c['nt']),without_coef_x=other,without_coef_y=np.asarray(without).reshape(c['n'],c['nt']))
  if c['nt']==1:result.update(vector_x=x,vector_y=np.asarray(y).ravel(),vector_coef=np.asarray(coef).ravel())
 return {k:np.asarray(v,dtype=float).ravel() for k,v in result.items()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path,required=True);ap.add_argument('--case',type=int)
 args=ap.parse_args();cases=list(fixtures())
 if args.case is not None:cases=[cases[args.case]]
 inputs=[str(len(cases))];references=[]
 for index,c in enumerate(cases):
  inputs.append(' '.join(map(str,[c['mode'],c['n'],c['p'],c.get('nt',1),c.get('info',10),c['rank'],c['seed'],int(c.get('shuf',True)),c['tail'],c.get('noise',0.),c.get('bias',0.)])))
  references.append(expected(c))
  if (index+1)%500==0:print('Prepared',index+1,'source configurations',flush=True)
 exe=args.executable.resolve();run=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,cwd=ROOT)
 assert run.returncode==0,(run.stdout[-2500:],run.stderr[-2500:])
 checks=[];index=-1;fields={};inventories=[];source_nonfinite=0
 for line in run.stdout.splitlines():
  words=line.split()
  if words[0]=='BEGIN':index+=1;fields={}
  elif words[0]=='END':
   want=references[index];c=cases[index];assert set(fields)==set(want),(index,set(fields)^set(want));inventories.append(sorted(fields))
   for name,got in fields.items():
    oracle=want[name];tolerance=0 if name in ['coef','vector_coef'] else 3e-12
    np.testing.assert_allclose(got,oracle,rtol=tolerance,atol=tolerance,equal_nan=True,err_msg=str((index,c,name)))
    finite=np.isfinite(got)&np.isfinite(oracle);source_nonfinite+=int(np.count_nonzero(~np.isfinite(oracle)))
    checks.append(dict(fixture=index,field=name,passed=True,max_abs_error=float(np.max(np.abs(got[finite]-oracle[finite]),initial=0))))
   if c['mode'] in [0,2]:
    singular=np.linalg.svd(fields['x'].reshape(c['n'],c['p']),compute_uv=False)
    i=np.arange(min(c['n'],c['p']))/c['rank'];profile=(1-c['tail'])*np.exp(-i*i)+c['tail']*np.exp(-.1*i)
    np.testing.assert_allclose(singular,profile,atol=3e-12,rtol=3e-12)
    checks.append(dict(fixture=index,field='singular_profile',passed=True,max_abs_error=float(np.max(np.abs(singular-profile),initial=0))))
  else:
   name=words[0];assert name not in fields;fields[name]=np.array(list(map(float,words[2:])))
   assert int(words[1])==len(fields[name])
 assert index+1==len(cases)
 invalid=dict(zero_samples='dimensions must be positive',zero_features='dimensions must be positive',zero_rank='effective_rank',nan_tail='finite',infinite_tail='finite',large_tail='[0,1]',negative_tail='[0,1]',negative_seed='nonnegative',regression_samples='dimensions must be positive',regression_features='dimensions must be positive',negative_informative='nonnegative',zero_targets='positive',vector_zero_targets='positive',vector_targets='matrix y',negative_noise='nonnegative',nan_noise='finite',infinite_noise='finite',nan_bias='finite',infinite_bias='finite',regression_rank='effective_rank',unused_tail='[0,1]',regression_seed='nonnegative')
 rejections=[]
 for mode,message in invalid.items():
  result=subprocess.run([str(exe),mode],capture_output=True,text=True,cwd=ROOT);output=result.stdout+result.stderr
  diagnostic=next((line for line in output.splitlines() if line.startswith('fscikit-learn:')), '')
  assert result.returncode!=0 and message.lower() in diagnostic.lower(),(mode,output)
  assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(mode,output)
  rejections.append(dict(case=mode,passed=True))
 import sklearn.datasets._samples_generator as source
 import scipy.linalg._decomp_qr as qr
 report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,fixtures=len(cases),checks=checks,configurations=cases,inventories=inventories,rejections=rejections,source_nonfinite_values=source_nonfinite,
   source_sha256=sha(source.__file__),qr_source_sha256=sha(qr.__file__),validator_sha256=sha(__file__),executable_sha256=sha(exe),native_source_sha256={name:sha(ROOT/'src'/name) for name in ['sklearn_regression_datasets.f90','sklearn_datasets.f90','sklearn_legacy_rng.f90','sklearn_ieee_math.f90']})
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(len(checks),'groups /',len(cases),'fixtures /',len(rejections),'diagnostics passed;',source_nonfinite,'nonfinite source values',flush=True)
if __name__=='__main__':main()
