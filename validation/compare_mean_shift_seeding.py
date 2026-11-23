from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.cluster import MeanShift,estimate_bandwidth,get_bin_seeds
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fixtures():
 for seed,shape,width in itertools.product([0,7,42,2026],[(1,1),(6,1),(17,2),(31,4),(100,3)],[0.,.1,.3,1.,2.,10.]):
  rng=np.random.RandomState(seed);x=rng.normal(size=shape);x[3::4]=x[0]
  for freq in [1,2,3,len(x)+1]:yield dict(mode=0,x=x,width=width,freq=freq,seed=seed)
 for width in [.1,.3,1.,2.5]:
  ties=np.array([-4.5,-3.5,-2.5,-1.5,-.5,.5,1.5,2.5,3.5,4.5])*width
  for direction in [0,np.inf,-np.inf]:
   x=ties if direction==0 else np.nextafter(ties,direction)
   for freq in [1,2,3]:yield dict(mode=0,x=x[:,None],width=width,freq=freq)
 for x in [np.array([0.,-0.,1.,-1.,0.]),np.array([2**24,2**24+1,2**24+2]*2,dtype=float),np.arange(10000,dtype=float),np.zeros(19)]:
  for width in [.3,1.,4.]:yield dict(mode=0,x=x[:,None],width=width,freq=1)
 for seed,p,width,freq,mode,all_points,max_iter in itertools.product([0,7,42,2026],[2,3],[.25,.6,1.3,-1.],[1,2,4],[1,2,3],[False,True],[0,1,20]):
  rng=np.random.RandomState(seed);centers=np.arange(3)[:,None]*np.arange(1,p+1)[None,:]*3.
  x=np.repeat(centers,12,axis=0)+rng.normal(scale=.06,size=(36,p));x=np.vstack([x,np.full((1,p),20.)])
  q=np.vstack([x[::3]+.017,np.zeros((1,p)),np.full((1,p),40.)]);given=x[[0,12]] if mode==2 else np.empty((0,p))
  yield dict(mode=mode,x=x,q=q,width=width,freq=freq,max_iter=max_iter,all_points=all_points,given=given,seed=seed)
 for seed,n,p,q,ns in itertools.product([0,7,42,2026],[1,5,19,53],[1,3],[0.,.2,.3,.7,1.],[None,1,3,17,'n','extra']):
  rng=np.random.RandomState(seed);x=rng.normal(size=(n,p));x[4::5]=x[0]
  ns=n if ns=='n' else n+5 if ns=='extra' else ns
  yield dict(mode=4,x=x,quantile=q,n_samples=ns,seed=seed)
 for width,freq,large in itertools.product([.1,.3,1.,2.5],[1,2],[False,True]):
  x=np.array([0.,.5,1.5,2.5,-.5,-1.5,1.5,2.5]) if not large else np.array([2**24,2**24+1,2**24+2]*2,dtype=float)
  yield dict(mode=5,x=x[:,None],width=width,freq=freq)
 for ns in [None,19,506]:
  x=np.random.RandomState(7).normal(size=(501,2))
  yield dict(mode=4,x=x,quantile=.3,n_samples=ns,seed=42)
def expected(c):
 x=c['x'];mode=c['mode'];width=c.get('width',1.);freq=c.get('freq',1);fallback=False
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  if mode in [0,5]:
   starts=get_bin_seeds(x,np.float64(width) if mode==5 else width,freq);result=dict(seeds=np.asarray(starts).reshape(-1,x.shape[1]))
  elif mode==4:
   result=dict(bandwidth=[estimate_bandwidth(x,quantile=c['quantile'],n_samples=c['n_samples'],random_state=c['seed'])])
  else:
   bw=width if width!=-1 else estimate_bandwidth(x)
   if mode==1:starts=get_bin_seeds(x,bw,freq)
   elif mode==2:starts=c['given']
   else:starts=x
   model=MeanShift(bandwidth=None if width==-1 else width,bin_seeding=mode!=3,min_bin_freq=freq,
                   max_iter=c['max_iter'],cluster_all=c['all_points'],seeds=c['given'] if mode==2 else None).fit(x)
   result=dict(seeds=starts,centers=model.cluster_centers_,labels=model.labels_,prediction=model.predict(c['q']),iterations=[model.n_iter_],bandwidth=[bw])
  fallback=any(str(w.message).startswith('Binning data failed') for w in caught)
 if mode!=4:result['fallback']=[int(fallback)]
 result['input_preserved']=np.zeros_like(x)
 return {k:np.asarray(v,dtype=float).ravel() for k,v in result.items()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path,required=True);ap.add_argument('--case',type=int)
 args=ap.parse_args();cases=list(fixtures())
 if args.case is not None:cases=[cases[args.case]]
 inputs=[str(len(cases))];references=[];configs=[]
 for index,c in enumerate(cases):
  x=c['x'];p=x.shape[1];q=c.get('q',x[:1]);given=c.get('given',np.empty((0,p)))
  inputs.append(' '.join(map(str,[c['mode'],len(x),p,len(q),c.get('freq',1),c.get('max_iter',300),int(c.get('all_points',True)),len(given),c.get('n_samples') or 0,c.get('seed',0),c.get('width',1.),c.get('quantile',.3)])))
  for a in [x,q,given]:inputs.extend(' '.join(format(v,'.17g') for v in row) for row in a)
  original=x.copy();references.append(expected(c));np.testing.assert_array_equal(x,original)
  configs.append({k:v for k,v in c.items() if k not in ['x','q','given'] }|dict(samples=len(x),features=p,query_samples=len(q),explicit_seeds=len(given)))
 exe=args.executable.resolve();run=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,cwd=ROOT)
 assert run.returncode==0,(run.stdout[-2000:],run.stderr[-2000:])
 checks=[];index=-1;fields={};inventories=[]
 for line in run.stdout.splitlines():
  tokens=line.split()
  if tokens[0]=='BEGIN':index+=1;fields={}
  elif tokens[0]=='END':
   want=references[index];assert set(fields)==set(want),(index,set(fields)^set(want));inventories.append(sorted(fields))
   for name,value in fields.items():
    oracle=want[name];rtol=0 if name in ['seeds','fallback','labels','prediction','iterations','input_preserved'] else 3e-12
    if name=='seeds' and configs[index]['mode']==1 and configs[index]['width']==-1:rtol=3e-12
    np.testing.assert_allclose(value,oracle,rtol=rtol,atol=0 if rtol==0 else 3e-12,err_msg=str((index,configs[index],name)))
    checks.append(dict(fixture=index,field=name,passed=True,max_abs_error=float(np.max(np.abs(value-oracle),initial=0))))
  else:
   name=tokens[0];assert name not in fields;fields[name]=np.array(list(map(float,tokens[2:])))
   assert int(tokens[1])==len(fields[name])
 assert index+1==len(cases)
 invalid=dict(negative_bin='nonnegative',nan_bin='finite',zero_frequency='positive',nan_data='NaN or infinity',bin_overflow='overflow',seed_range='float32 seed range',empty_data='nonempty',zero_sample='positive',negative_seed='nonnegative',nan_quantile='finite',large_quantile='[0,1]',empty_bins='no MeanShift seeds',invalid_frequency='positive',nan_bandwidth='finite')
 rejections=[]
 for mode,message in invalid.items():
  result=subprocess.run([str(exe),mode],text=True,capture_output=True,cwd=ROOT);output=result.stdout+result.stderr
  diagnostic=next((s for s in output.splitlines() if s.startswith('fscikit-learn:')), '')
  assert result.returncode!=0 and message.lower() in diagnostic.lower(),(mode,output)
  assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(mode,output)
  rejections.append(dict(case=mode,passed=True))
 import sklearn.cluster._mean_shift as source
 report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,fixtures=len(cases),checks=checks,configurations=configs,inventories=inventories,rejections=rejections,
             source_sha256=sha(source.__file__),native_source_sha256=sha(ROOT/'src/sklearn_cluster.f90'),validator_sha256=sha(__file__),executable_sha256=sha(exe))
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(len(checks),'groups /',len(cases),'fixtures /',len(rejections),'diagnostics passed',flush=True)
if __name__=='__main__':main()
