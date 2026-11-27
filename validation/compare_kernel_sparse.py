"""KernelPCA dense/CSR fitting, mixed queries and noncanonical storage versus sklearn."""
from pathlib import Path
import argparse,json,os,subprocess,warnings,hashlib
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from scipy.sparse import csr_matrix
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import rbf_kernel

def raw(a):
 data=[];indices=[];ptr=[0]
 for row in a:
  for j in reversed(range(a.shape[1])):data.extend([.6*row[j],.4*row[j],0.]);indices.extend([j]*3)
  ptr.append(len(data))
 return csr_matrix((data,indices,ptr),shape=a.shape)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[];names=['linear','rbf','poly','cosine','precomputed','sigmoid']
 configs=[dict(n=n,p=p,kernel=kernel,solver=solver,train=train,query=query) for n,p in [(23,3),(11,4)] for kernel in range(6) for solver in [0,1] for train in range(3) for query in range(3)]
 for index,c in enumerate(configs):
  n,p=c['n'],c['p'];rng=np.random.RandomState(814+n+p);x=rng.normal(size=(n,p))*.4;q=rng.normal(size=(3,p))*.4;x[abs(x)<.2]=0;q[abs(q)<.2]=0;x[0]=0
  if c['kernel']==4:q=rbf_kernel(q,x,gamma=.4);x=rbf_kernel(x,gamma=.4);p=n
  train=x if c['train']==0 else csr_matrix(x) if c['train']==1 else raw(x)
  query=q if c['query']==0 else csr_matrix(q) if c['query']==1 else raw(q)
  text=f"{n} {p} 3 3 {c['kernel']} {c['solver']} {c['train']} {c['query']}\n"+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in np.vstack([x,q]))+'\n';inp=a.report.with_suffix('.in');out=a.report.with_suffix('.out');inp.write_text(text)
  proc=subprocess.run([str(a.executable),str(inp),str(out)],capture_output=True,text=True);source={};native={};error=None;differences=[]
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');m=KernelPCA(n_components=3,kernel=names[c['kernel']],gamma=.4,degree=2,eigen_solver='randomized' if c['solver'] else 'dense',iterated_power=1,random_state=42);z=m.fit_transform(train);t=m.transform(query)
   source=dict(values=m.eigenvalues_.tolist(),fit=z.ravel(order='F').tolist(),query=t.ravel(order='F').tolist())
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  if proc.returncode==0 and error is None:
   lines=out.read_text().splitlines();assert len(lines)==5,(c,lines);native['count']=int(lines[0])
   for key,line in zip(['values','fit','query'],lines[1:4]):
    native[key]=np.fromstring(line,sep=' ').tolist()
    if np.shape(native[key])!=np.shape(source[key]) or not np.allclose(native[key],source[key],atol=3e-7,rtol=3e-7,equal_nan=True):differences.append(key)
   native['metadata']=np.fromstring(lines[4],sep=' ',dtype=int).tolist()
   if native['metadata']!=[p,n,int(c['train']>0),int(c['train']==0),int(c['train']>0)]:differences.append('metadata')
  elif (proc.returncode==0)!=(error is None):differences.append('acceptance')
  if proc.returncode not in [0,1] or proc.returncode==1 and 'fscikit-learn:' not in proc.stdout:differences.append('abnormal_native_exit')
  records.append(dict(index=index,config=c,input=text,source=source,source_error=error,native=native,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,differences=differences))
 root=Path(sklearn.__file__).parent
 report=dict(sklearn=sklearn.__version__,configurations=len(records),failures=[i for i,r in enumerate(records) if r['differences']],source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['decomposition/_kernel_pca.py','metrics/pairwise.py']},records=records)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('CSR KernelPCA',len(records),'strict failures',report['failures']);raise SystemExit(bool(report['failures']))
if __name__=='__main__':main()
