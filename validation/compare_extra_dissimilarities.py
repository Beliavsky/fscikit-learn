"""Additional sklearn numeric/Boolean nearest-pair metrics, including fallbacks."""
from pathlib import Path
import argparse,hashlib,itertools,json,subprocess
import numpy as np
import scipy
import compare_pairwise_reductions as base
METRICS=['braycurtis','canberra','correlation','hamming','dice','jaccard','rogerstanimoto','russellrao','sokalmichener','sokalsneath','yule','matching']
def fixtures():
 for seed,(n,m,q),scenario,metric,axis,storage in itertools.product([7,42],[(1,1,1),(2,3,2),(7,5,7),(13,11,31),(3,4,129)],range(6),METRICS,[0,1],range(4)):
  rng=np.random.RandomState(seed);x=rng.normal(size=(n,q));y=rng.normal(size=(m,q))
  if scenario==0:x=np.abs(x);y=np.abs(y)
  if scenario==2:x[rng.uniform(size=x.shape)<.7]=0;y[rng.uniform(size=y.shape)<.7]=0
  if scenario==3:x=np.round(x);y=np.round(y)
  if scenario==4:x[:]=0;y[:]=0;x[-1,:]=1;y[-1,:]=2
  if scenario==5:x[:]=1;y[:]=-1
  yield base.fixture(metric,x,y,axis,storage,tag=f'seed{seed}-scenario{scenario}')
 for metric,scale,axis,storage in itertools.product(METRICS,[1e-150,1e150,1e200,1e308],[0,1],range(4)):
  rng=np.random.RandomState(2026);yield base.fixture(metric,scale*rng.uniform(-1,1,(2,7)),scale*rng.uniform(-1,1,(3,7)),axis,storage,tag=f'scale{scale}')
 for metric,axis,storage in itertools.product(METRICS,[0,1],range(4)):
  yield base.fixture(metric,[[0.,0.,0.],[1.,-1.,2.],[0.,1.,0.]],[[0.,0.,0.],[-2.,2.,-4.],[1.,0.,1.]],axis,storage,tag='zero-and-opposite')
 for metric,axis in itertools.product(METRICS,[0,1]):
  rng=np.random.RandomState(23);yield base.fixture(metric,rng.normal(size=(3,8193)),rng.normal(size=(4,8193)),axis,tag='wide-buffer')

source_expected=base.expected
methods=[]
def expected(c):
 # The installed CSR Bray-Curtis tail loops read from an exhausted opposite
 # buffer. Incomplete CSR rows therefore use the independently specified
 # sklearn dense numerical contract, not an unstable out-of-bounds oracle.
 adapter=False
 if c['metric']=='braycurtis' and c['storage']!=0:
  matrices=[c['x']] if c['storage']==2 else [c['y']] if c['storage']==3 else [c['x'],c['y']]
  if all(np.count_nonzero(a)>0 for a in matrices):
   adapter=any(np.count_nonzero(a)!=a.size for a in matrices)
 if adapter:
  original=c;c=dict(c,storage=0)
 result=source_expected(c)
 methods.append('sklearn_dense_contract_for_sparse_braycurtis' if adapter else 'strict_sklearn')
 return result


ROOT=base.ROOT
wire=base.wire
digest=base.digest
upstream_hashes=base.upstream_hashes
def run(executable,profile,report):
 cases=[];rejections=[]
 for c in fixtures():
  try:idx,d=expected(c)
  except (ValueError,TypeError) as e:rejections.append((c,type(e).__name__));continue
  cases.append((c,idx,d))
 proc=subprocess.run([str(executable)],input=str(len(cases))+'\n'+''.join(wire(c,i) for i,(c,_,_) in enumerate(cases)),text=True,capture_output=True,timeout=240)
 if proc.returncode:raise RuntimeError(proc.stdout[-2000:]+proc.stderr[-4000:])
 lines=proc.stdout.strip().splitlines();assert len(lines)==3*len(cases),(len(lines),len(cases))
 records=[];fail=[];maxerr=0;nonfinite=0;maxscaled=0
 for i,(c,idx,d) in enumerate(cases):
  header=list(map(int,lines[3*i].split()));gotidx=np.fromstring(lines[3*i+1],sep=' ',dtype=int);got=np.fromstring(lines[3*i+2],sep=' ')
  states=header==[i,1,1,1,len(idx)];indices=np.array_equal(idx,gotidx);masks=all(np.array_equal(f(got),f(d)) for f in [np.isnan,np.isposinf,np.isneginf]);finite=np.isfinite(d)
  values=bool(np.allclose(got[finite],d[finite],rtol=3e-12,atol=3e-12));err=float(np.max(np.abs(got[finite]-d[finite]),initial=0));maxerr=max(maxerr,err);maxscaled=max(maxscaled,float(np.max(np.abs(got[finite]-d[finite])/np.maximum(1,np.abs(d[finite])),initial=0)));nonfinite+=int(np.sum(~finite))
  rec={k:v for k,v in c.items() if k not in ['x','y']};rec.update(id=i,data_sha256=digest(c),shape_x=list(c['x'].shape),shape_y=list(c['y'].shape),indices=indices,values=values,masks=masks,state_identity=states,max_error=err,source_value_sha256=hashlib.sha256(d.tobytes()).hexdigest(),source_indices=idx.tolist(),native_indices=gotidx.tolist())
  records.append(rec)
  if not all([states,indices,masks,values]):fail.append(dict(rec,expected=d.tolist(),actual=got.tolist()))
 rejected=[];probe_cache={}
 for i,(c,error) in enumerate(rejections):
  key=(c['metric'],c['storage'],c['axis'],tuple(c['x'].shape),tuple(c['y'].shape),
       np.count_nonzero(c['x'])==0,np.count_nonzero(c['y'])==0,c['p'],c['squared'],i%2)
  if key not in probe_cache:
   p=subprocess.run([str(executable)],input='1\n'+wire(c,i),text=True,capture_output=True,timeout=15)
   text=p.stdout+p.stderr
   primary=next((line for line in text.splitlines()if line.startswith('fscikit-learn:')),'')
   message='requires dense' if c['metric'] not in ['braycurtis','canberra','matching'] else 'nonempty CSR storage'
   assert p.returncode!=0 and message.lower()in primary.lower(),(c,primary)
   assert not any(token in text for token in ['SIGFPE','Fortran runtime error'])
   probe_cache[key]=(len(probe_cache),p)
  probe_id,p=probe_cache[key]
  rejected.append(dict(id=i,metric=c['metric'],storage=c['storage'],tag=c['tag'],source_error=error,native_probe=probe_id,passed=p.returncode!=0,data_sha256=digest(c)))
 result=dict(profile=profile,sklearn=sklearn.__version__,numpy=np.__version__,configurations=len(cases),groups=4*len(cases),source_rejections=len(rejections),unique_rejection_probes=len(probe_cache),nonfinite=nonfinite,max_error=maxerr,max_scaled_error=maxscaled,records=records,rejections=rejected,failures=fail,passed=not fail and all(c['passed'] for c in rejected),sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'validation/compare_pairwise_reductions.py',ROOT/'LICENSES/scipy-distance-BSD',ROOT/'src/sklearn_pairwise_reductions.f90',ROOT/'src/sklearn_metrics.f90',ROOT/'src/sklearn_sparse.f90',ROOT/'src/sklearn_pairwise_measures.f90',ROOT/'src/sklearn_ieee_math.f90',ROOT/'src/sklearn_common.f90',ROOT/'app/reference_pairwise_reductions_driver.f90',Path(__file__)]},executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),upstream_sha256=upstream_hashes())
 report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ['records','rejections','failures','sha256']}));print('failures',len(fail));print(json.dumps(fail[:12],indent=2))
 if not result['passed']:raise SystemExit(1)

def scipy_hashes():
 import scipy.spatial.distance as distance
 import scipy.spatial._distance_wrap as wrap
 import scipy.spatial._distance_pybind as pybind
 return {Path(module.__file__).name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
         for module in [distance,wrap,pybind]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--profile',default='checked');args=p.parse_args()
 methods.clear()
 try:run(args.executable,args.profile,args.report)
 finally:
  if args.report.exists():
   d=json.loads(args.report.read_text());assert len(methods)==len(d['records']);
   for r,method in zip(d['records'],methods,strict=True):r['comparison_method']=method
   d['rejection_probe_policy']='Representative guards share metric, storage, axis, shape, empty-storage state, native keywords and incoming-flag parity; every Source rejection is retained.';d['strict_source_configurations']=methods.count('strict_sklearn');d['sparse_bray_adapter_configurations']=len(methods)-d['strict_source_configurations'];d['scipy']=scipy.__version__;d['scipy_source_sha256']=scipy_hashes();d['sha256'][str(Path(__file__).resolve().relative_to(base.ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();d['sha256']['LICENSES/scipy-distance-BSD']=hashlib.sha256((base.ROOT/'LICENSES/scipy-distance-BSD').read_bytes()).hexdigest();args.report.write_text(json.dumps(d,indent=2))
if __name__=='__main__':main()
