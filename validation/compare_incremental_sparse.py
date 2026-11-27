"""IncrementalPCA CSR batching and mixed queries against installed sklearn."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from scipy.sparse import csr_matrix
from sklearn.decomposition import IncrementalPCA

def storage(x,kind):
 if kind==0:return x
 if kind==1:return csr_matrix(x)
 data=[];indices=[];ptr=[0]
 for row in x:
  for j in reversed(range(x.shape[1])):data.extend([.6*row[j],.4*row[j],0.]);indices.extend([j]*3)
  ptr.append(len(data))
 return csr_matrix((data,indices,ptr),shape=x.shape)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 configs=[dict(n=n,p=p,k=k,b=b,w=w,train=train,query=query) for n,p in [(37,5),(29,8)] for k in [0,2,p] for b in [0,7,p+2] for w in [0,1] for train in range(3) for query in range(3)]
 for index,c in enumerate(configs):
  n,p=c['n'],c['p'];rng=np.random.RandomState(814+n+p);x=rng.normal(size=(n,p))*np.arange(1,p+1)+np.arange(p)*3;q=rng.normal(size=(11,p))+np.arange(p)*3;x[abs(x)<.2]=0;q[abs(q)<.2]=0
  text=f"{n} {p} 11 {c['k']} {c['b']} {c['w']} {c['train']} {c['query']}\n"+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in np.vstack([x,q]))+'\n'
  inp=a.report.with_suffix('.in');out=a.report.with_suffix('.out');inp.write_text(text);proc=subprocess.run([str(a.executable),str(inp),str(out)],capture_output=True,text=True)
  source={};native={};error=None;differences=[]
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');m=IncrementalPCA(n_components=c['k'] or None,batch_size=c['b'] or None,whiten=bool(c['w']));m.fit(x);z=m.fit_transform(storage(x,c['train']));t=m.transform(storage(q,c['query']))
   source=dict(components=m.components_,mean=m.mean_,var=m.var_,singular=m.singular_values_,variance=m.explained_variance_,ratio=m.explained_variance_ratio_,noise=[m.noise_variance_],train=z,inverse=m.inverse_transform(z),query=t,shape=[m.n_samples_seen_,m.n_features_in_,m.n_components_,m.batch_size_]);source={key:np.asarray(v).ravel(order='F').tolist() for key,v in source.items()}
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
 root=Path(sklearn.__file__).parent;report=dict(sklearn=sklearn.__version__,configurations=len(records),failures=[r['index'] for r in records if r['differences']],source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['decomposition/_incremental_pca.py','decomposition/_base.py']},records=records)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('IncrementalPCA CSR',len(records),'strict failures',report['failures']);raise SystemExit(bool(report['failures']))
if __name__=='__main__':main()
