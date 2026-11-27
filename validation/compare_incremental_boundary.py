"""IncrementalPCA singleton/constant boundaries and projection arithmetic."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.decomposition import IncrementalPCA
from compare_incremental_sparse import storage

def dataset(name):
 rng=np.random.RandomState(417)
 if name=='single':return np.array([[2.,-3.,7.]])
 if name=='single_feature':return np.array([[2.]])
 if name=='zeros':return np.zeros((9,4))
 if name=='constant':return np.tile([2.,-3.,7.,.25],(11,1))
 if name=='regular':return rng.normal(size=(23,4))
 if name=='offset':return rng.normal(size=(23,4))*.03+np.array([1.e7,-3.e7,7.e7,2.e7])
 raise ValueError(name)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 configs=[dict(dataset=name,k=k,b=b,w=w,train=train,mode=mode) for name in ['single','single_feature','zeros','constant','regular','offset'] for k in [0,1] for b in [0,3] for w in [0,1] for train in [0,1,2] for mode in [0,1] if mode==0 or train==0]
 for index,c in enumerate(configs):
  x=dataset(c['dataset']);n,p=x.shape;q=np.tile(x[0],(3,1))+np.linspace(.01,.1,p);query=c['train']
  text=f"{n} {p} 3 {c['k']} {c['b']} {c['w']} {c['train']} {query} {c['mode']}\n"+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in np.vstack([x,q]))+'\n'
  inp=a.report.with_suffix('.in');out=a.report.with_suffix('.out');inp.write_text(text);proc=subprocess.run([str(a.executable),str(inp),str(out)],capture_output=True,text=True)
  source={};native={};error=None;differences=[]
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');m=IncrementalPCA(n_components=c['k'] or None,batch_size=c['b'] or None,whiten=bool(c['w']))
    if c['mode']==0:m.fit(storage(x,c['train']))
    else:
     m.partial_fit(x[:1])
     if n>1:m.partial_fit(x[1:])
    z=m.transform(x);t=m.transform(storage(q,query) if c['mode']==0 else q)
   source=dict(components=m.components_,mean=m.mean_,var=m.var_,singular=m.singular_values_,variance=m.explained_variance_,ratio=m.explained_variance_ratio_,noise=[m.noise_variance_],train=z,query=t,shape=[m.n_samples_seen_,m.n_features_in_,m.n_components_,getattr(m,'batch_size_',0)]);source={key:np.asarray(v).ravel(order='F').tolist() for key,v in source.items()}
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  if proc.returncode==0 and error is None:
   for line in out.read_text().splitlines():
    key,*values=line.split();native[key]=list(map(float,values))
   if native.keys()!=source.keys():differences.append('fields')
   for key in source:
    if key not in native or np.shape(native[key])!=np.shape(source[key]) or not np.allclose(native[key],source[key],rtol=3e-8,atol=3e-8,equal_nan=True):differences.append(key)
  elif (proc.returncode==0)!=(error is None):differences.append('acceptance')
  if proc.returncode not in [0,1] or proc.returncode==1 and 'fscikit-learn:' not in proc.stdout:differences.append('abnormal_native_exit')
  records.append(dict(index=index,config=c,input=text,source=source,source_error=error,native=native,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,differences=differences))
 root=Path(sklearn.__file__).parent;report=dict(sklearn=sklearn.__version__,configurations=len(records),failures=[r['index'] for r in records if r['differences']],source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['decomposition/_incremental_pca.py','decomposition/_base.py','utils/extmath.py']},records=records)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('IncrementalPCA boundaries',len(records),'strict failures',report['failures']);raise SystemExit(bool(report['failures']))
if __name__=='__main__':main()
