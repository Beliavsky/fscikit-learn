"""Pairwise distance/kernel contracts, CSR normalization and missing-data algebra."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.sparse import csr_matrix
import sklearn
from sklearn.metrics import pairwise as pw
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SEEDS=[0,7,42,2026,2147483647]
SHAPES=[(1,1,1),(2,3,2),(3,3,7),(7,5,17),(17,17,31)]
def fixtures():
 for seed,(n,m,p),scenario,gamma in itertools.product(SEEDS,SHAPES,range(6),[.1,1.,3.]):
  rng=np.random.RandomState(seed);x=rng.normal(size=(n,p));y=rng.normal(size=(m,p))
  if scenario==1:x[:]=0;y[:]=0
  if scenario==2:x=np.round(x);y=np.round(y)
  if scenario==3:x*=1e-16;y*=1e-16
  if scenario==4:x*=1e200;y*=1e200
  if scenario==5:x=np.where(x>0,1e308,-1e308);y=np.where(y>0,1e308,-1e308)
  yield dict(kind=0,seed=seed,scenario=scenario,x=x,y=y,gamma=gamma,marker=np.nan,mode='complete')
 for seed,(n,m,p),scenario,marker in itertools.product(SEEDS,SHAPES[:4],range(4),[np.nan,-99.]):
  rng=np.random.RandomState(seed);x=rng.normal(size=(n,p));y=rng.normal(size=(m,p));mx=rng.uniform(size=x.shape)<.3;my=rng.uniform(size=y.shape)<.3
  if scenario==1:mx[:]=True;my[:]=True
  if scenario==2:mx[:,:p//2]=True;my[:,p//2:]=True
  if scenario==3:mx[:]=False;my[:]=False
  x[mx]=marker;y[my]=marker
  yield dict(kind=1,seed=seed,scenario=scenario,x=x,y=y,gamma=1.,marker=marker,mode='missing')
 for seed,(n,m,p),marker in itertools.product(SEEDS,SHAPES[:4],[np.inf,-np.inf]):
  rng=np.random.RandomState(seed);yield dict(kind=1,seed=seed,x=rng.normal(size=(n,p)),y=rng.normal(size=(m,p)),gamma=1.,marker=marker,mode='infinite_marker')
 for seed,offset in itertools.product([7,42],[1e4,1e8]):
  rng=np.random.RandomState(seed);x=offset+rng.normal(size=(3,7));y=offset+rng.normal(size=(3,7));x[0,2]=np.nan;y[1,4]=np.nan
  yield dict(kind=1,seed=seed,x=x,y=y,gamma=1.,marker=np.nan,mode='offset')
 for seed,(n,m,p),scenario,gamma in itertools.product(SEEDS,SHAPES,range(5),[.1,1.,3.]):
  rng=np.random.RandomState(seed);x=np.exp(rng.normal(size=(n,p)));y=np.exp(rng.normal(size=(m,p)))
  if scenario==1:x[:]=0;y[:]=0
  if scenario==2:x=np.round(x);y=np.round(y)
  if scenario==3:x*=1e-300;y*=1e-300
  if scenario==4:x*=1e300;y*=1e300
  yield dict(kind=2,seed=seed,scenario=scenario,x=x,y=y,gamma=gamma,marker=np.nan,mode='histogram')
 for seed,(n,m),scenario in itertools.product(SEEDS,[(1,1),(2,2),(7,5),(31,11)],range(5)):
  rng=np.random.RandomState(seed);x=rng.uniform(-1,1,(n,2))*np.array([np.pi/2,np.pi]);y=rng.uniform(-1,1,(m,2))*np.array([np.pi/2,np.pi])
  if scenario==1:x[:]=0;y[:]=0
  if scenario==2:y=-x[np.arange(m)%n];y[:,1]+=np.pi
  if scenario==3:x[:,0]=np.pi/2;y[:,0]=-np.pi/2
  if scenario==4:x=np.where(x>0,1e308,-1e308);y=np.where(y>0,1e308,-1e308)
  yield dict(kind=3,seed=seed,scenario=scenario,x=x,y=y,gamma=1.,marker=np.nan,mode='angular')
 for seed,p in itertools.product(SEEDS,[129,1025,100000]):
  rng=np.random.RandomState(seed);x=rng.randint(-2,3,(1,p)).astype(float);y=rng.randint(-2,3,(1,p)).astype(float)
  yield dict(kind=0,seed=seed,x=x,y=y,gamma=.1,marker=np.nan,mode='wide')
  mx=rng.uniform(size=x.shape)<.3;my=rng.uniform(size=y.shape)<.3;x=x.copy();y=y.copy();x[mx]=np.nan;y[my]=np.nan
  yield dict(kind=1,seed=seed,x=x,y=y,gamma=1.,marker=np.nan,mode='wide_missing')
def expected(c):
 x,y,gamma,marker=c['x'],c['y'],c['gamma'],c['marker'];kind=c['kind'];out={}
 if kind==0:
  sx,sy=csr_matrix(x),csr_matrix(y)
  for name,a,b in [('cosine',x,y),('cosine_ss',sx,sy),('cosine_sd',sx,y),('cosine_ds',x,sy)]:out[name]=pw.cosine_distances(a,b)
  out['cosine_self']=pw.cosine_distances(x);out['cosine_sparse_self']=pw.cosine_distances(sx)
  out['laplacian_default']=pw.laplacian_kernel(x,y);out['laplacian']=pw.laplacian_kernel(x,y,gamma=gamma)
  out['laplacian_self']=pw.laplacian_kernel(x);out['laplacian_sparse_self']=pw.laplacian_kernel(sx)
  for name,a,b in [('laplacian_ss',sx,sy),('laplacian_sd',sx,y),('laplacian_ds',x,sy)]:out[name]=pw.laplacian_kernel(a,b,gamma=gamma)
  if len(x)==len(y):
   for name,f in [('euclidean',pw.paired_euclidean_distances),('manhattan',pw.paired_manhattan_distances),('paired_cosine',pw.paired_cosine_distances)]:
    for suffix,a,b in [('',x,y),('_ss',sx,sy),('_sd',sx,y),('_ds',x,sy)]:out[name+suffix]=np.asarray(f(a,b)).ravel()
   for name,metric in [('default_paired','euclidean'),('alias_l2','l2'),('alias_l1','l1'),('alias_cityblock','cityblock'),('dispatch_euclidean','euclidean'),('dispatch_manhattan','manhattan'),('dispatch_cosine','cosine')]:out[name]=pw.paired_distances(x,y,metric=metric)
   out['callback']=pw.paired_distances(x,y,metric=lambda a,b:np.max(np.abs(a-b)))
 elif kind==1:
  out['nan_squared']=pw.nan_euclidean_distances(x,y,squared=True,missing_values=marker)
  out['nan_root']=pw.nan_euclidean_distances(x,y,missing_values=marker)
  out['nan_self_squared']=pw.nan_euclidean_distances(x,squared=True,missing_values=marker)
  out['nan_self_root']=pw.nan_euclidean_distances(x,missing_values=marker)
  if np.isnan(marker):out['nan_default']=pw.nan_euclidean_distances(x,y)
 elif kind==2:
  out.update(additive_chi2=pw.additive_chi2_kernel(x,y),additive_chi2_self=pw.additive_chi2_kernel(x),chi2_default=pw.chi2_kernel(x,y),chi2=pw.chi2_kernel(x,y,gamma=gamma),chi2_self=pw.chi2_kernel(x),chi2_self_gamma=pw.chi2_kernel(x,gamma=gamma))
 else:
  sx,sy=csr_matrix(x),csr_matrix(y)
  out.update(haversine=pw.haversine_distances(x,y),haversine_self=pw.haversine_distances(x),haversine_ss=pw.haversine_distances(sx,sy),haversine_sparse_self=pw.haversine_distances(sx),haversine_sd=pw.haversine_distances(sx,y),haversine_ds=pw.haversine_distances(x,sy))
 out={name:np.asarray(value).ravel() for name,value in out.items()};out['input_preserved']=np.ones(1);return out
def roundoff_oracle(c,field):
 x,y=c['x'],c['y']
 if 'self' in field:y=x
 mx=np.isnan(x);my=np.isnan(y);p=x.shape[1];truth=np.empty((len(x),len(y)));bound=np.empty_like(truth)
 for i,a in enumerate(x):
  for j,b in enumerate(y):
   present=~mx[i]&~my[j];count=present.sum()
   if count==0:truth[i,j]=np.nan;bound[i,j]=0;continue
   truth[i,j]=np.sum((a[present]-b[present])**2)*p/count
   magnitude=np.sum(np.nan_to_num(a)**2)+np.sum(np.nan_to_num(b)**2)
   bound[i,j]=64*np.finfo(float).eps*p*magnitude*p/count
 return truth.ravel(),bound.ravel()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--report',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--case',type=int);args=ap.parse_args();cases=list(fixtures())
 if args.case is not None:cases=[cases[args.case]]
 inputs=[str(len(cases))];refs=[];configs=[]
 for i,c in enumerate(cases):
  x,y=c['x'],c['y']
  with warnings.catch_warnings(),np.errstate(all='ignore'):
   warnings.simplefilter('ignore');refs.append(expected(c))
  inputs.append(' '.join(map(str,[c['kind'],len(x),len(y),x.shape[1],c['gamma'],c['marker']])))
  for matrix in [x,y]:inputs.extend(' '.join(map(repr,map(float,row))) for row in matrix)
  configs.append(dict(kind=c['kind'],mode=c['mode'],seed=c['seed'],scenario=c.get('scenario'),n=len(x),m=len(y),p=x.shape[1],gamma=c['gamma'],marker='nan' if np.isnan(c['marker']) else 'inf' if np.isposinf(c['marker']) else '-inf' if np.isneginf(c['marker']) else c['marker']))
  if (i+1)%200==0:print('Prepared',i+1,'source fixtures',flush=True)
 exe=args.executable.resolve();r=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True);assert r.returncode==0,(r.stdout[-2200:],r.stderr[-2200:])
 checks=[];inventories=[];index=-1;nonfinite=0
 for line in r.stdout.splitlines():
  parts=line.split()
  if parts[0]=='BEGIN':index+=1;fields={}
  elif parts[0]=='END':
   assert fields.keys()==refs[index].keys(),(index,fields.keys(),refs[index].keys());inventories.append(sorted(fields))
   for name,value in fields.items():
    want=refs[index][name];method='sklearn';bound_ratio=0.
    if cases[index]['mode']=='offset' and name!='input_preserved':
     truth,bound=roundoff_oracle(cases[index],name);squared='squared' in name;nv=value if squared else value**2;sv=want if squared else want**2
     np.testing.assert_array_equal(np.isfinite(nv),np.isfinite(sv))
     mask=np.isfinite(truth);ne=np.abs(nv[mask]-truth[mask]);se=np.abs(sv[mask]-truth[mask]);assert np.all(ne<=bound[mask]) and np.all(se<=bound[mask]),(index,name,ne,se,bound)
     method='source_algebra_roundoff_bound';bound_ratio=float(np.max(np.maximum(ne,se)/np.maximum(bound[mask],np.finfo(float).tiny),initial=0))
    else:np.testing.assert_allclose(value,want,rtol=3e-12,atol=3e-12,equal_nan=True,err_msg=str((index,configs[index],name)))
    finite=np.isfinite(value)&np.isfinite(want);nonfinite+=int(np.count_nonzero(~np.isfinite(want)))
    checks.append(dict(fixture=index,field=name,method=method,passed=True,max_abs_error=float(np.max(np.abs(value[finite]-want[finite]),initial=0)),max_scaled_error=float(np.max(np.abs(value[finite]-want[finite])/(1+np.abs(want[finite])),initial=0)),roundoff_bound_ratio=bound_ratio))
  else:
   assert parts[0] not in fields;fields[parts[0]]=np.array(list(map(float,parts[2:])));assert len(fields[parts[0]])==int(parts[1])
 assert index+1==len(cases)
 invalid=dict(empty='nonempty',features='feature counts differ',paired_samples='equal sample counts',nan_input='finite',inf_input='finite',paired_metric='unknown paired',haversine_shape='latitude and longitude',lap_gamma_zero='positive',lap_gamma_negative='positive',lap_gamma_nan='finite',lap_gamma_inf='finite',chi_gamma_zero='positive',chi_gamma_negative='positive',chi_gamma_nan='finite',chi_gamma_inf='finite',chi_negative_x='nonnegative',chi_negative_y='nonnegative',nan_inf='infinity',marker_nan_input='finite input',nan_features='feature counts differ',nan_empty='nonempty',csr_nan='finite',csr_inf='finite',csr_duplicate='sorted and unique',csr_unsorted='sorted and unique',csr_index='out of bounds');rejections=[]
 for mode,message in invalid.items():
  r=subprocess.run([str(exe),mode],text=True,capture_output=True);output=r.stdout+r.stderr;primary=next((s for s in output.splitlines() if s.startswith('fscikit-learn:')),'')
  assert r.returncode!=0 and message in primary,(mode,output);assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(mode,output)
  rejections.append(dict(case=mode,passed=True))
 source_dir=Path(sklearn.__file__).parent/'metrics'
 report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=configs,rejections=rejections,source_nonfinite_values=nonfinite,executable_sha256=sha(exe),validator_sha256=sha(__file__),source_sha256=sha(pw.__file__),cython_source_sha256={n:sha(source_dir/n) for n in ['_pairwise_fast.pyx','_dist_metrics.pyx.tp']},native_source_sha256={n:sha(ROOT/'src'/n) for n in ['sklearn_pairwise_measures.f90','sklearn_metrics.f90','sklearn_ieee_math.f90','sklearn_sparse.f90']})
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(len(checks),'groups /',len(cases),'fixtures /',len(rejections),'diagnostics PASS;',nonfinite,'nonfinite source values',flush=True)
if __name__=='__main__':main()
