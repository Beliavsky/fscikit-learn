"""Additional sklearn numeric/Boolean nearest-pair metrics, including fallbacks."""
from pathlib import Path
import argparse,hashlib,itertools,json
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

def main():
 p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--profile',default='checked');args=p.parse_args()
 base.fixtures=fixtures; base.expected=expected; methods.clear()
 try:base.run(args.executable,args.profile,args.report)
 finally:
  if args.report.exists():
   d=json.loads(args.report.read_text());assert len(methods)==len(d['records']);
   for r,method in zip(d['records'],methods,strict=True):r['comparison_method']=method
   d['strict_source_configurations']=methods.count('strict_sklearn');d['sparse_bray_adapter_configurations']=len(methods)-d['strict_source_configurations'];d['scipy']=scipy.__version__;d['sha256'][str(Path(__file__).resolve().relative_to(base.ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();d['sha256']['LICENSES/scipy-distance-BSD']=hashlib.sha256((base.ROOT/'LICENSES/scipy-distance-BSD').read_bytes()).hexdigest();args.report.write_text(json.dumps(d,indent=2))
if __name__=='__main__':main()
