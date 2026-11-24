"""Source MinMaxScaler streaming, NaNs, clipping and rank-one/free maps."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler,minmax_scale
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def config(**kw):return dict(n=7,p=3,nb=4,m=5,seed=0,stream=0,axis=0,clip=0,qclip=0,mode=0,action=0,lo=0.,hi=1.,blo=0.,bhi=1.,qlo=0.,qhi=1.,scenario='random',qwidth=0,bwidth=0)|kw
def fixtures():
 for seed,(n,p),stream,axis,(lo,hi),clip in itertools.product([0,7,42,2026,2147483647],[(7,1),(9,3),(5,7)],[0,1],[0,1],[(0.,1.),(-2.,3.),(1.,2.)],[0,1]):
  yield config(seed=seed,n=n,p=p,stream=stream,axis=axis,lo=lo,hi=hi,blo=lo,bhi=hi,qlo=lo,qhi=hi,clip=clip,qclip=clip)
 for stream,axis,clip,scenario in itertools.product([0,1],[0,1],[0,1],['zero','constant','missing','allnan','batchnan','recover_nan','tiny','threshold_at','threshold_above','huge','huge_span']):
  yield config(stream=stream,axis=axis,clip=clip,qclip=clip,scenario=scenario,lo=-1.,hi=2.,blo=-1.,bhi=2.,qlo=-1.,qhi=2.)
 for axis,stream,(lo,hi),clip in itertools.product([0,1],[0,1],[(np.nan,1.),(0.,np.nan),(-np.inf,np.inf),(-np.inf,1.),(0.,np.inf),(-1e308,1e308)],[0,1]):
  yield config(axis=axis,stream=stream,clip=clip,qclip=clip,lo=lo,hi=hi,blo=lo,bhi=hi,qlo=lo,qhi=hi)
 for axis,stream,(lo,hi),action,clip in itertools.product([0,1],[0,1],[(-3.,4.),(2.,1.),(np.nan,1.)],[0,2,3,4],[0,1]):
  yield config(axis=axis,stream=stream,qlo=lo,qhi=hi,action=action,qclip=clip,blo=-2.,bhi=3.)
 for stream,axis in itertools.product([0,1],[0,1]):yield config(stream=stream,axis=axis,action=1,p=1,qwidth=5)
def inputs(c):
 rng=np.random.RandomState(c['seed']%10000+17);arrays=[rng.normal(size=(n,p)) for n,p in [(c['n'],c['p']),(c['nb'],c['bwidth'] or c['p']),(c['m'],c['qwidth'] or c['p'])]]
 for a in arrays:
  if c['scenario']=='zero':a.fill(0)
  if c['scenario']=='constant':a[:]=np.arange(a.shape[1])
  if c['scenario']=='allnan':a.fill(np.nan)
  if c['scenario']=='missing':a[::2,::2]=np.nan
  if c['scenario'] in ['tiny','threshold_at','threshold_above']:
   value=dict(tiny=1e-310,threshold_at=10*np.finfo(float).eps,threshold_above=np.nextafter(10*np.finfo(float).eps,np.inf))[c['scenario']];a[:]=np.linspace(0,value,len(a))[:,None]
  if c['scenario']=='huge':a[:]=np.linspace(.5e308,1e308,len(a))[:,None]
  if c['scenario']=='huge_span':a[:]=np.linspace(-1,1,len(a))[:,None]*1e308
 if c['scenario']=='batchnan':arrays[1].fill(np.nan)
 if c['scenario']=='recover_nan':arrays[0].fill(np.nan)
 return arrays
def oracle(c,arrays):
 x,b,q=arrays;original=[a.copy() for a in arrays];state=np.random.get_state();m=MinMaxScaler(feature_range=(c['lo'],c['hi']),clip=bool(c['clip']))
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore');m.fit(x)
  if c['stream']:m.feature_range=(c['blo'],c['bhi']);m.partial_fit(b)
  if c['action']==2:m.scale_.fill(0)
  if c['action']==3:m.scale_=-m.scale_
  if c['action']==4:m.scale_.fill(1e308);m.min_.fill(-1e308)
  f=dict(data_min=m.data_min_.copy(),data_max=m.data_max_.copy(),data_range=m.data_range_.copy(),scale=m.scale_.copy(),offset=m.min_.copy(),metadata=[m.n_features_in_,m.n_samples_seen_])
  m.feature_range=(c['qlo'],c['qhi']);m.clip=bool(c['qclip']);f['train']=m.transform(x)
  if c['action']!=1:f['query']=m.transform(q)
  back=m.inverse_transform(q);f.update(inverse_shape=back.shape,inverse=back,free=minmax_scale(x,feature_range=(c['lo'],c['hi']),axis=c['axis']),vector=minmax_scale(x.ravel(order='F'),feature_range=(c['lo'],c['hi']),axis=c['axis']))
  assert all(np.array_equal(a,b,equal_nan=True) for a,b in zip(arrays,original)) and all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state()))
  f['preserved']=[1]*4;m.feature_range=(c['lo'],c['hi']);m.fit(q);f.update(refit=[m.n_features_in_,m.n_samples_seen_],refit_min=m.data_min_,refit_max=m.data_max_)
 return f
def payload(c,arrays):
 return ' '.join(str(c[k]) for k in ['stream','axis','clip','qclip','mode','action','lo','hi','blo','bhi','qlo','qhi'])+'\n'+''.join(f'{a.shape[0]} {a.shape[1]}\n'+' '.join(format(v,'.17g') for v in a.ravel(order='F'))+'\n' for a in arrays)
def reject_cases(executable):
 result=[]
 variants=[('unfitted',dict(mode=1)),('inf_fit',dict(mode=2)),('inf_query',dict(mode=3)),('inf_batch',dict(mode=4,stream=1)),('empty_fit',dict(n=0)),('empty_features',dict(p=0)),('empty_query',dict(m=0)),('empty_batch',dict(nb=0,stream=1)),('batch_width',dict(bwidth=5,stream=1)),('query_width',dict(qwidth=5)),('inverse_width',dict(qwidth=1,action=1)),('zero_range',dict(lo=1.,hi=1.)),('reverse_range',dict(lo=2.,hi=1.)),('bad_batch_range',dict(stream=1,blo=2.,bhi=1.)),('bad_axis',dict(axis=2))]
 for name,change in variants:
  c=config()|change;arrays=inputs(c);wire=payload(c,arrays);x,b,q=arrays;error=None
  if c['mode']==2:x[0,0]=np.inf
  if c['mode']==3:q[0,0]=np.inf
  if c['mode']==4:b[0,0]=np.inf
  m=MinMaxScaler(feature_range=(c['lo'],c['hi']))
  try:
   if c['mode']==1:m.transform(q)
   m.fit(x)
   if c['stream']:m.feature_range=(c['blo'],c['bhi']);m.partial_fit(b)
   if c['action']!=1:m.transform(q)
   m.inverse_transform(q);minmax_scale(x,feature_range=(c['lo'],c['hi']),axis=c['axis'])
  except ValueError as e:error=type(e).__name__
  assert error is not None,('source accepts invalid',name)
  reject(executable,wire,name);result.append(dict(name=name,passed=True,oracle='sklearn_estimator',source_error=error))
 c=config(mode=5,stream=1);reject(executable,payload(c,inputs(c)),'sample_count','overflow');result.append(dict(name='sample_count',passed=True,oracle='native_int64_capacity'))
 return result
def reject(executable,wire,name,message='fscikit-learn:'):
 r=subprocess.run([str(executable)],input='1\n'+wire,text=True,capture_output=True);out=r.stdout+r.stderr
 assert r.returncode and 'fscikit-learn:' in out and message in out and not any(v in out for v in ['SIGFPE','Fortran runtime error']),(name,out)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
 paths=[Path(__file__),args.executable,ROOT/'src/sklearn_minmax_scaling.f90',ROOT/'src/sklearn_preprocessing.f90',ROOT/'src/sklearn_ieee_math.f90',ROOT/'app/reference_minmax_driver.f90',Path(sklearn.__file__).parent/'preprocessing/_data.py'];hashes={str(p):sha(p) for p in paths}
 cases=list(fixtures());assert len({json.dumps(c,sort_keys=True) for c in cases})==len(cases);arrays=[inputs(c) for c in cases];expected=[oracle(c,a) for c,a in zip(cases,arrays)]
 r=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,arrays)),text=True,capture_output=True);assert r.returncode==0,(r.stdout[-800:],r.stderr[-1400:])
 lines=iter(r.stdout.splitlines());index=-1;checks=[];inventories=[];nonfinite=0
 for line in lines:
  p=line.split()
  if not p:continue
  if p[0]=='BEGIN':index+=1;fields=set();assert int(p[1])==index+1
  elif p[0]=='END':assert fields==expected[index].keys();inventories.append(sorted(fields))
  else:
   field=p[0];assert field not in fields;fields.add(field);values=np.fromstring(next(lines),sep=' ');want=np.asarray(expected[index][field]).ravel(order='F');assert len(values)==int(p[1]) and values.shape==want.shape,(index,field)
   if field in ['metadata','preserved','refit','inverse_shape']:np.testing.assert_array_equal(values,want,err_msg=str((index,cases[index],field)))
   else:np.testing.assert_allclose(values,want,rtol=3e-8,atol=3e-8,equal_nan=True,err_msg=str((index,cases[index],field)))
   finite=np.isfinite(want);nonfinite+=int(np.count_nonzero(~finite));checks.append(dict(fixture=index,field=field,passed=True,oracle='native_input_rng_identity' if field=='preserved' else 'sklearn',max_abs_error=float(np.max(np.abs(values[finite]-want[finite]),initial=0))))
 assert index+1==len(cases) and hashes=={str(p):sha(p) for p in paths}
 rejections=reject_cases(args.executable)
 d=dict(rejections=rejections,profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=[{k:('NaN' if np.isnan(v) else 'Infinity' if v>0 else '-Infinity') if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in c.items()} for c in cases],source_nonfinite_values=nonfinite,hashes=hashes)
 (args.report or ROOT/'validation/results'/f'minmax_{args.profile}.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(len(cases),len(checks),len(rejections),nonfinite,'PASS')
if __name__=='__main__':main()
