"""Same-input cluster_qr partitions/pivots against scipy/sklearn."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn,scipy
from scipy.linalg import qr
from sklearn.cluster._spectral import cluster_qr
from sklearn.metrics import adjusted_rand_score

def cases():
 records=[]
 saved=Path(__file__).resolve().parent/'fixtures/spectral_qr_replay.json'
 d=json.loads(saved.read_text())
 for r in d['records']:
  n,k=r['shape'];x=np.array(r['vectors']).reshape((n,k),order='F')
  records.append((dict(group='saved',index=r['index']),x,r['baseline_labels']))
 for n,k in [(13,2),(23,3),(31,5),(3,7),(1,3)]:
  for seed in [0,21,42]:
   rng=np.random.RandomState(seed);gaussian=rng.normal(size=(n,k))
   for pattern in ['gaussian','onehot','duplicates','rankdef','zeros']:
    x=gaussian.copy()
    if pattern=='onehot':x=np.eye(k)[np.arange(n)%k]
    if pattern=='duplicates':x[::2]=x[0]
    if pattern=='rankdef':x=rng.normal(size=(n,1))@rng.normal(size=(1,k))
    if pattern=='zeros':x[:]=0
    records.append((dict(group='synthetic',n=n,k=k,seed=seed,pattern=pattern),x,None))
 return records

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 for index,(config,x,baseline) in enumerate(cases()):
  n,k=x.shape;text=f'{n} {k}\n'+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in x)+'\n';inp=a.report.with_suffix('.in');out=a.report.with_suffix('.out');inp.write_text(text);p=subprocess.run([str(a.executable),str(inp),str(out)],capture_output=True,text=True);error=None;source={};native={};differences=[]
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');source=dict(labels=cluster_qr(x).tolist(),pivots=qr(x.T,pivoting=True)[2].tolist())
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  if p.returncode==0 and error is None:
   for line in out.read_text().splitlines():
    key,*values=line.split();native[key]=list(map(int,values))
   if adjusted_rand_score(native['labels'],source['labels'])!=1:differences.append('partition')
   native['pivots']=[i-1 for i in native['pivots']]
   native['pivot_match']=native['pivots']==source['pivots']
  elif (p.returncode==0)!=(error is None):differences.append('acceptance')
  if p.returncode not in [0,1] or p.returncode==1 and 'fscikit-learn:' not in p.stdout:differences.append('abnormal_native_exit')
  baseline_match=None if baseline is None or error is not None else bool(adjusted_rand_score(baseline,source['labels'])==1)
  records.append(dict(index=index,config=config,input=text,source=source,source_error=error,native=native,baseline_partition_match=baseline_match,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,differences=differences))
 root=Path(sklearn.__file__).parent;fixture=Path(__file__).resolve().parent/'fixtures/spectral_qr_replay.json'
 source_files={'sklearn/cluster/_spectral.py':root/'cluster/_spectral.py','scipy/linalg/_decomp_qr.py':Path(scipy.__file__).parent/'linalg/_decomp_qr.py'}
 report=dict(sklearn=sklearn.__version__,scipy=scipy.__version__,configurations=len(records),failures=[r['index'] for r in records if r['differences']],baseline_saved_failures=[r['config']['index'] for r in records if r['baseline_partition_match'] is False],fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),source_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in source_files.items()},records=records)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('QR',len(records),'strict failures',report['failures'],'saved baseline failures',report['baseline_saved_failures']);raise SystemExit(bool(report['failures']))
if __name__=='__main__':main()
