"""Weighted quantile losses, D2 scores, RMSLE and Tweedie powers against sklearn."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.metrics import mean_pinball_loss,mean_tweedie_deviance,d2_tweedie_score,d2_pinball_score,d2_absolute_error_score,root_mean_squared_log_error
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def weight_vector(n,mode):
 if mode==0:return None
 w=np.arange(1,n+1,dtype=float)
 if mode==2 and n>1:w[0]=0
 if mode==3:w[0]=-1
 return w
def fixtures():
 for seed,n,q,alpha,scenario,weight in itertools.product([0,7,42,2026,2147483647],[1,2,17],[1,3],[0.,.1,.5,.9,1.],range(4),range(4)):
  rng=np.random.RandomState(seed);y=rng.normal(size=(n,q));p=y+rng.normal(size=(n,q))
  if scenario==1:
   y[:]=2;p[:]=2
   if q>1:p[:,-1]=3
  if scenario==2:y=np.exp(y);p=np.exp(rng.normal(size=(n,q)))
  if scenario==3:y=rng.randint(0,3,(n,q)).astype(float);p=rng.randint(0,3,(n,q)).astype(float)
  ow=np.arange(1,q+1,dtype=float) if q>1 and (weight%2) else None
  yield dict(kind=0,seed=seed,scenario=scenario,weight=weight,y=y,p=p,w=weight_vector(n,weight),ow=ow,alpha=alpha,power=0.)
 for seed,n,power,scenario,weight in itertools.product([0,7,42,2026,2147483647],[1,2,7,31],[-3.,-1.,0.,1.,1.5,2.,3.,4.5],range(3),range(4)):
  rng=np.random.RandomState(seed);y=np.exp(rng.normal(size=n));p=np.exp(rng.normal(size=n))
  if power<=0:y=rng.normal(size=n)
  elif power<2:y[::3]=0
  if scenario==1:y[:]=2;p[:]=2
  if scenario==2:y[:]=2;p[:]=3
  yield dict(kind=1,seed=seed,scenario=scenario,weight=weight,y=y[:,None],p=p[:,None],w=weight_vector(n,weight),ow=None,alpha=.5,power=power)
 for seed,alpha,weight in itertools.product([0,7,42,2026,2147483647],[0.,.1,.5,.9,1.],[0,1,2,3]):
  rng=np.random.RandomState(seed);y=rng.normal(size=(129,3));p=rng.normal(size=(129,3));yield dict(kind=0,mode='long',seed=seed,y=y,p=p,w=weight_vector(129,weight),ow=np.array([1.,0.,3.]),alpha=alpha,power=0.)
 for y,p in [(np.array([-1e308,1e308]),np.array([1e308,-1e308])),(np.array([1e308,1e308]),np.array([1e308,1e308])),(np.array([-1e-300,1e-300]),np.zeros(2)),(np.array([-.999999999999,-.5]),np.array([-.99999999999,.5]))]:
  for a in [0.,.5,1.]:yield dict(kind=0,mode='extreme',y=y[:,None],p=p[:,None],w=None,ow=None,alpha=a,power=0.)
 for power,y,p in itertools.product([-3.,0.,1.,1.5,2.,3.,4.5],[0.,1e-300,1.,1e300],[1e-300,1.,1e300]):
  if power>=2 and y==0:continue
  yield dict(kind=1,mode='extreme',y=np.full((2,1),y),p=np.full((2,1),p),w=None,ow=None,alpha=.5,power=power)
def source_expected(c):
 y,p,w,ow=c['y'],c['p'],c['w'],c['ow'];out={}
 if c['kind']==1:
  out['tweedie']=np.array([mean_tweedie_deviance(y.ravel(),p.ravel(),sample_weight=w,power=c['power'])])
  try:out['d2_tweedie']=np.array([d2_tweedie_score(y.ravel(),p.ravel(),sample_weight=w,power=c['power'])])
  except ZeroDivisionError:pass
  if c['power']==0:
   out['default_tweedie']=out['tweedie']
   if 'd2_tweedie' in out:out['default_d2_tweedie']=out['d2_tweedie']
  return out
 for name,f,options in [('pinball',mean_pinball_loss,dict(alpha=c['alpha'])),('d2_pinball',d2_pinball_score,dict(alpha=c['alpha'])),('d2_absolute',d2_absolute_error_score,{})]:
  try:out[name]=np.array([f(y,p,sample_weight=w,multioutput='uniform_average' if ow is None else ow,**options)])
  except ValueError as error:
   assert 'Input contains' in str(error),str(error)
   continue
  raw=np.asarray(f(y,p,sample_weight=w,multioutput='raw_values',**options));out[name+'_raw']=np.full(y.shape[1],np.nan) if raw.ndim==0 else raw
  out['vector_'+name]=np.array([f(y[:,j],p[:,j],sample_weight=w,**options) for j in range(y.shape[1])])
 if np.all(y>-1) and np.all(p>-1):
  out['rmsle']=np.array([root_mean_squared_log_error(y,p,sample_weight=w,multioutput='uniform_average' if ow is None else ow)])
  out['rmsle_raw']=root_mean_squared_log_error(y,p,sample_weight=w,multioutput='raw_values')
  out['vector_rmsle']=np.array([root_mean_squared_log_error(y[:,j],p[:,j],sample_weight=w) for j in range(y.shape[1])])
 if c['alpha']==.5:
  out['default_pinball']=out['pinball']
  if 'd2_pinball' in out:out['default_d2_pinball']=out['d2_pinball']
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--report',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--case',type=int);args=ap.parse_args();cases=list(fixtures())
 if args.case is not None:cases=[cases[args.case]]
 inputs=[str(len(cases))];refs=[];configs=[]
 for i,c in enumerate(cases):
  y,p,w,ow=c['y'],c['p'],c['w'],c['ow'];n,q=y.shape;rmsle=int(np.all(y>-1) and np.all(p>-1))
  with warnings.catch_warnings(),np.errstate(all='ignore'):
   warnings.simplefilter('ignore');refs.append(source_expected(c))
  inputs.append(' '.join(map(str,[c['kind'],n,q,int(w is not None),int(ow is not None),rmsle,int('d2_tweedie' in refs[-1]),int('d2_pinball' in refs[-1]),int('d2_absolute' in refs[-1]),c['alpha'],c['power']])))
  for a,b in zip(y,p):inputs.extend([' '.join(map(repr,map(float,a))),' '.join(map(repr,map(float,b)))])
  if w is not None:inputs.append(' '.join(map(repr,map(float,w))))
  if ow is not None:inputs.append(' '.join(map(repr,map(float,ow))))
  config={k:v for k,v in c.items() if k not in ['y','p','w','ow']};config.update(n=n,q=q,weighted=w is not None,output_weighted=ow is not None,rmsle=rmsle,d2_valid='d2_tweedie' in refs[-1] if c['kind']==1 else True,pin_valid='d2_pinball' in refs[-1] if c['kind']==0 else True,absolute_valid='d2_absolute' in refs[-1] if c['kind']==0 else True);configs.append(config)
  if (i+1)%500==0:print('Prepared',i+1,'source fixtures',flush=True)
 exe=args.executable.resolve();r=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True);assert r.returncode==0,(r.stdout[-2000:],r.stderr[-2000:])
 checks=[];inventories=[];index=-1;nonfinite=0
 for line in r.stdout.splitlines():
  parts=line.split()
  if parts[0]=='BEGIN':index+=1;fields={}
  elif parts[0]=='END':
   assert fields.keys()==refs[index].keys(),(index,fields.keys(),refs[index].keys());inventories.append(sorted(fields))
   for name,value in fields.items():
    want=np.asarray(refs[index][name]);np.testing.assert_allclose(value,want,rtol=3e-11,atol=3e-11,equal_nan=True,err_msg=str((index,configs[index],name)))
    finite=np.isfinite(value)&np.isfinite(want);nonfinite+=int(np.count_nonzero(~np.isfinite(want)))
    checks.append(dict(fixture=index,field=name,passed=True,max_abs_error=float(np.max(np.abs(value[finite]-want[finite]),initial=0)),max_scaled_error=float(np.max(np.abs(value[finite]-want[finite])/(1+np.abs(want[finite])),initial=0))))
  else:
   assert parts[0] not in fields;fields[parts[0]]=np.array(list(map(float,parts[2:])));assert len(fields[parts[0]])==int(parts[1])
 assert index+1==len(cases)
 source_rejections=[];unique={}
 for i,c in enumerate(cases):
  if c['kind']!=1 or 'd2_tweedie' in refs[i]:continue
  lines=[f"{len(c['y'])} {int(c['w'] is not None)} {c['power']}"]+[f'{float(y[0])!r} {float(p[0])!r}' for y,p in zip(c['y'],c['p'])]
  if c['w'] is not None:lines.append(' '.join(map(repr,map(float,c['w']))))
  payload='\n'.join(lines)+'\n'
  if payload not in unique:
   r=subprocess.run([str(exe),'d2_stdin'],input=payload,text=True,capture_output=True);output=r.stdout+r.stderr
   assert r.returncode!=0 and 'fscikit-learn: D2 Tweedie null deviance is zero' in output,(i,configs[i],output)
   assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(i,output)
   unique[payload]=len(unique)
  source_rejections.append(dict(fixture=i,probe=unique[payload],passed=True))
 for i,c in enumerate(cases):
  if c['kind']!=0:continue
  for metric,alpha in [('d2_pinball',c['alpha']),('d2_absolute',.5)]:
   if metric in refs[i]:continue
   y,p,w=c['y'],c['p'],c['w'];lines=[f'{len(y)} {y.shape[1]} {int(w is not None)} {alpha}']
   for a,b in zip(y,p):lines.extend([' '.join(map(repr,map(float,a))),' '.join(map(repr,map(float,b)))])
   if w is not None:lines.append(' '.join(map(repr,map(float,w))))
   payload='\n'.join(lines)+'\n';key='pinball:'+payload
   if key not in unique:
    r=subprocess.run([str(exe),'pinball_stdin'],input=payload,text=True,capture_output=True);output=r.stdout+r.stderr
    assert r.returncode!=0 and 'fscikit-learn: pinball quantile baseline must be finite' in output,(i,configs[i],output)
    assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(i,output)
    unique[key]=len(unique)
   source_rejections.append(dict(fixture=i,metric=metric,probe=unique[key],passed=True))
 invalid=dict(tweedie_constant='null deviance is zero',empty='nonempty',shape='shapes differ',target_nan='finite',pred_inf='finite',alpha_low='[0,1]',alpha_high='[0,1]',alpha_nan='finite',alpha_inf='finite',weight_length='length mismatch',weight_nan='finite',weight_inf='finite',weight_zero='nonzero total',weight_cancel='nonzero total',output_length='length mismatch',output_nan='finite',output_inf='finite',output_zero='nonzero total',rmsle_target='greater than -1',rmsle_prediction='greater than -1',power_gap='at most zero',power_nan='finite',power_inf='finite',positive_prediction='strictly positive',nonnegative_target='nonnegative',positive_target='strictly positive');rejections=[]
 for mode,message in invalid.items():
  r=subprocess.run([str(exe),mode],text=True,capture_output=True);output=r.stdout+r.stderr;primary=next((s for s in output.splitlines() if s.startswith('fscikit-learn:')),'')
  assert r.returncode!=0 and message in primary,(mode,output);assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(mode,output)
  rejections.append(dict(case=mode,passed=True))
 import sklearn.metrics._regression as source
 import sklearn.utils.stats as stats
 report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=configs,rejections=rejections,source_nonfinite_values=nonfinite,source_rejections=source_rejections,unique_source_rejection_probes=len(unique),executable_sha256=sha(exe),validator_sha256=sha(__file__),source_sha256=sha(source.__file__),stats_source_sha256=sha(stats.__file__),native_source_sha256={n:sha(ROOT/'src'/n) for n in ['sklearn_regression_metrics.f90','sklearn_metrics.f90','sklearn_ieee_math.f90']})
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(len(checks),'groups /',len(cases),'fixtures /',len(rejections),'diagnostics PASS;',nonfinite,'nonfinite source values',flush=True)
if __name__=='__main__':main()
