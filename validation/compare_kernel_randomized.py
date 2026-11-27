"""KernelPCA dense/randomized signed spectra, transforms and inverse versus sklearn."""
from pathlib import Path
import argparse,json,os,subprocess,warnings,hashlib
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
import sklearn
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import rbf_kernel

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 configs=[dict(n=n,p=p,k=k,kind=kind,kernel=kernel,solver=solver,power=power,seed=seed) for n,p,k,kind in [(24,3,3,'normal'),(64,4,3,'normal'),(9,5,0,'all'),(7,3,20,'clipped'),(8,3,3,'zero')] for kernel in range(6) for solver,power,seed in [(0,-1,0)]+[(1,power,seed) for power in [-1,0,1,3] for seed in [0,42]]]
 configs += [dict(n=7,p=7,k=2,kind=kind,kernel=4,solver=solver,power=0,seed=42) for kind in ['indefinite','negative'] for solver in [0,1]]
 names=['linear','rbf','poly','cosine','precomputed','sigmoid']
 for index,c in enumerate(configs):
  n,p,k=c['n'],c['p'],c['k'];rng=np.random.RandomState(842+n+p);x=rng.normal(size=(n,p))*.4;q=rng.normal(size=(3,p))*.4
  if c['kind']=='zero':x[:]=0
  if c['kernel']==4:
   if c['kind'] in ['indefinite','negative']:
    basis=np.linalg.qr(np.column_stack([np.ones(n),rng.normal(size=(n,n-1))]))[0];spectrum=np.array([0.,-8.,5.,3.,1.,.5,.1]) if c['kind']=='indefinite' else np.array([0.,-1.,-2.,-3.,-4.,-5.,-6.]);x=(basis*spectrum)@basis.T;q=rng.normal(size=(3,n));p=n
   else:q=rbf_kernel(q,x,gamma=.4);x=rbf_kernel(x,gamma=.4);p=n
  inverse=int(c['kernel']!=4);remove=0;header=[n,p,3,k,c['kernel'],c['solver'],c['power'],c['seed'],inverse,remove,.4,1.]
  text=' '.join(str(v) for v in header)+'\n'+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in np.vstack([x,q]))+'\n';inp=a.report.with_suffix('.in');out=a.report.with_suffix('.out');inp.write_text(text)
  proc=subprocess.run([str(a.executable),str(inp),str(out)],capture_output=True,text=True);source={};native={};error=None;differences=[]
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');m=KernelPCA(n_components=None if k==0 else k,kernel=names[c['kernel']],degree=2,gamma=.4,coef0=1.,fit_inverse_transform=bool(inverse),eigen_solver='randomized' if c['solver'] else 'dense',iterated_power='auto' if c['power']==-1 else c['power'],random_state=c['seed']);z=m.fit_transform(x);t=m.transform(q)
   source=dict(values=m.eigenvalues_.tolist(),vectors=m.eigenvectors_.ravel(order='F').tolist(),fit=z.ravel(order='F').tolist(),train=m.transform(x).ravel(order='F').tolist(),query=t.ravel(order='F').tolist(),inverse=m.inverse_transform(t).ravel(order='F').tolist() if inverse else [])
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  if proc.returncode==0 and error is None:
   lines=out.read_text().splitlines();assert len(lines)==7,(c,len(lines));count=int(lines[0]);native['count']=count
   for key,line in zip(['values','vectors','fit','train','query','inverse'],lines[1:]):native[key]=np.fromstring(line,sep=' ').tolist()
   for key in ['values','fit','train','query','inverse']:
    if np.shape(native[key])!=np.shape(source[key]) or not np.allclose(native[key],source[key],atol=3e-7,rtol=3e-7,equal_nan=True):differences.append(key)
   if count==len(source['values']):
    positive=np.array(source['values'])>1e-10
    nv=np.array(native['vectors']).reshape(n,count,order='F')[:,positive];sv=m.eigenvectors_[:,positive]
    if not np.allclose(nv,sv,atol=3e-7,rtol=3e-7):differences.append('positive_vectors')
   else:differences.append('count')
  elif (proc.returncode==0)!=(error is None):differences.append('acceptance')
  if proc.returncode not in [0,1] or proc.returncode==1 and 'fscikit-learn:' not in proc.stdout:differences.append('abnormal_native_exit')
  records.append(dict(index=index,config=c,input=text,source=source,source_error=error,native=native,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,differences=differences))
 root=Path(sklearn.__file__).parent
 report=dict(sklearn=sklearn.__version__,configurations=len(records),failures=[i for i,r in enumerate(records) if r['differences']],source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['decomposition/_kernel_pca.py','utils/extmath.py','utils/validation.py']},records=records)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('KernelPCA randomized',len(records),'strict failures',report['failures']);raise SystemExit(bool(report['failures']))
if __name__=='__main__':main()
