"""CSR NMF models/functions versus installed sklearn, retaining strict failures."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
import scipy,sklearn
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF,non_negative_factorization,_nmf

def configurations():
 configs=[]
 solvers=[('cd',2.)]+[('mu',b) for b in [-1.,0.,.5,1.,1.5,2.,3.]]
 for n,p,k in [(13,7,3),(6,11,2)]:
  for method in ['auto','random','nndsvd','nndsvda','nndsvdar','custom']:
   for solver,beta in solvers:
    for seed in [0,42]:
     i=len(configs);configs.append(dict(n=n,p=p,k=k,k0=k,method=method,solver=solver,beta=beta,seed=seed,route=i%4,steps=80 if seed else 12,tol=1e-4 if seed else 0.,aw=.01*(i%3),ah=-1. if i%2 else .03,l1=[0.,.4,1.][i%3],shuffle=i%2,storage='canonical'))
 for storage in ['unsorted','duplicates','explicit_zero','empty']:
  for solver,beta in solvers:
   for route in [0,3]:
    configs.append(dict(n=11,p=6,k=2,k0=2,method='custom',solver=solver,beta=beta,seed=21,route=route,steps=11,tol=0.,aw=.01,ah=-1.,l1=.4,shuffle=1,storage=storage))
 for n,p,k,k0 in [(9,4,6,6),(8,4,0,4),(8,4,0,2)]:
  for method in ['random','custom','nndsvda']:
   for solver,beta in [('cd',2.),('mu',1.),('mu',2.)]:
    configs.append(dict(n=n,p=p,k=k,k0=k0,method=method,solver=solver,beta=beta,seed=42,route=2,steps=20,tol=0.,aw=0.,ah=-1.,l1=0.,shuffle=0,storage='canonical'))
 extra=[dict(c,beta=b) for b in [2.5,4.] for c in configs if c['solver']=='mu' and c['beta']==3.]
 return configs+extra

def storage(a,kind):
 if kind=='empty':return csr_matrix(a.shape)
 x=csr_matrix(a)
 if kind=='explicit_zero':
  x=x.tolil();x[0,0]=0;x=x.tocsr();x=csr_matrix((np.r_[0.,x.data],np.r_[0,x.indices],np.r_[0,x.indptr[1:]+1]),shape=x.shape)
 if kind in ['unsorted','duplicates']:
  ptr=[0];indices=[];data=[]
  for i in range(x.shape[0]):
   for e in range(x.indptr[i+1]-1,x.indptr[i]-1,-1):
    col=x.indices[e];value=x.data[e]
    if kind=='duplicates':indices.extend([col,col]);data.extend([value*.25,value*.75])
    else:indices.append(col);data.append(value)
   ptr.append(len(data))
  x=csr_matrix((data,indices,ptr),shape=x.shape)
 return x

def line(a,integer=False):return ' '.join(str(int(v)) if integer else format(float(v),'.17g') for v in np.asarray(a).ravel(order='F'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();binary=hashlib.sha256(a.executable.read_bytes()).hexdigest();records=[]
 for index,c in enumerate(configurations()):
  rng=np.random.RandomState(100+c['seed']);n,p=c['n'],c['p'];data=.1+rng.random_sample((n,p));query=.1+rng.random_sample((5,p))
  if c['beta']>0:data[rng.random_sample(data.shape)<.4]=0;query[rng.random_sample(query.shape)<.3]=0
  x=storage(data,c['storage']);q=storage(query,c['storage'] if c['storage']!='empty' else 'canonical');w0=.1+rng.random_sample((n,c['k0']));h0=.1+rng.random_sample((c['k0'],p))
  header=[n,p,x.nnz,q.shape[0],q.nnz,c['k'],c['k0'],c['steps'],c['seed'],c['shuffle'],c['route']]
  text=line(header,True)+'\n'+f"{c['solver']} {c['method']} {c['beta']} {c['tol']} {c['aw']} {c['ah']} {c['l1']}\n"+'\n'.join([line(x.indptr+1,True),line(x.indices+1,True),line(x.data),line(q.indptr+1,True),line(q.indices+1,True),line(q.data),line(w0),line(h0)])+'\n'
  inp=a.report.with_suffix('.in');out=a.report.with_suffix('.out');inp.write_text(text);proc=subprocess.run([str(a.executable),str(inp),str(out)],capture_output=True,text=True)
  source={};native={};error=None;stage='fit';differences=[]
  options=dict(n_components='auto' if c['k']==0 else c['k'],init=None if c['method']=='auto' else c['method'],solver=c['solver'],beta_loss=c['beta'],tol=c['tol'],max_iter=c['steps'],random_state=c['seed'],alpha_W=c['aw'],alpha_H='same' if c['ah']==-1 else c['ah'],l1_ratio=c['l1'],shuffle=bool(c['shuffle']))
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    if c['route']<2:
     model=NMF(**options)
     if c['route']==0:w=model.fit_transform(x,W=w0.copy(),H=h0.copy())
     else:model.fit(x,W=w0.copy(),H=h0.copy());w=model.transform(x)
     h=model.components_;it=int(model.n_iter_);err=float(model.reconstruction_err_);stage='query';z=model.transform(q)
    else:
     w,h,it=non_negative_factorization(x,W=w0.copy(),H=h0.copy(),update_H=c['route']==2,**options);err=float(_nmf._beta_divergence(x,w,h,c['beta'],square_root=True));stage='query'
     qo=options.copy();qo['n_components']=h.shape[0];z,_,_=non_negative_factorization(q,H=h.copy(),update_H=False,**qo)
   source=dict(w=w.ravel(order='F').tolist(),h=h.ravel(order='F').tolist(),query=z.ravel(order='F').tolist(),error=err,n_iter=it,k=h.shape[0],p=p)
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  if proc.returncode==0 and error is None:
   lines=out.read_text().splitlines();assert len(lines)==5,(index,lines)
   for key,raw in zip(['w','h','query','error'],lines[:4]):
    values=np.fromstring(raw,sep=' ');native[key]=float(values[0]) if key=='error' else values.tolist()
    if np.shape(native[key])!=np.shape(source[key]) or not np.allclose(native[key],source[key],atol=3e-8,rtol=3e-8,equal_nan=True):differences.append(key)
   meta=np.fromstring(lines[4],sep=' ',dtype=int);native.update(n_iter=int(meta[0]),k=int(meta[1]),p=int(meta[2]),input_preserved=bool(meta[3]))
   if any(native[key]!=source[key] for key in ['n_iter','k','p']):differences.append('metadata')
   if not native['input_preserved']:differences.append('input_mutation')
  elif (proc.returncode==0)!=(error is None):differences.append('acceptance')
  stages=[s.split()[-1] for s in proc.stdout.splitlines() if s.startswith('nmf_stage ')];native_stage=stages[-1] if stages else None
  if error is not None and proc.returncode==1 and native_stage!=stage:differences.append('error_stage')
  if proc.returncode not in [0,1] or proc.returncode==1 and 'fscikit-learn:' not in proc.stdout:differences.append('abnormal_native_exit')
  records.append(dict(index=index,config=c,input=text,source=source,source_error=error,source_stage=stage,native=native,native_stage=native_stage,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,differences=differences))
 assert binary==hashlib.sha256(a.executable.read_bytes()).hexdigest()
 root=Path(sklearn.__file__).parent;report=dict(sklearn=sklearn.__version__,scipy=scipy.__version__,numpy=np.__version__,native_sha256=binary,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256={'decomposition/_nmf.py':hashlib.sha256((root/'decomposition/_nmf.py').read_bytes()).hexdigest()},configurations=len(records),failures=[r['index'] for r in records if r['differences']],records=records)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('CSR NMF',len(records),'strict failures',report['failures']);raise SystemExit(bool(report['failures']))
if __name__=='__main__':main()
