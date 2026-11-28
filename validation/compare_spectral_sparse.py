"""SpectralEmbedding/Clustering CSR and precomputed nearest-neighbor affinities."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from scipy.sparse import csr_matrix
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel,euclidean_distances
from compare_incremental_sparse import storage as raw_storage

def storage(x,kind):
 if kind==0:return x
 if kind==1:
  n,p=x.shape
  return csr_matrix((x.ravel(),np.tile(np.arange(p),n),np.arange(n+1)*p),shape=x.shape)
 return raw_storage(x,2)

def certificate(a,z,drop):
 a=.5*(a+a.T);a=a.copy();np.fill_diagonal(a,0);degree=np.sqrt(a.sum(axis=1));degree[degree==0]=1
 lap=-a/degree[:,None]/degree[None,:];np.fill_diagonal(lap,a.sum(axis=1)/degree**2)
 values=np.linalg.eigvalsh(lap)[drop:drop+z.shape[1]];u=degree[:,None]*z
 residual=float(np.max(np.abs(lap@u-u*values)));orthogonal=float(np.max(np.abs(u.T@u-np.eye(z.shape[1]))))
 return dict(residual=residual,orthogonality=orthogonal,eigenvalues=values.tolist(),passed=residual<3e-8 and orthogonal<3e-8)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[];names=['rbf','nearest_neighbors','precomputed','precomputed_nearest_neighbors']
 configs=[dict(n=n,algorithm=algorithm,affinity=affinity,kind=kind,neighbors=neighbors) for n in [13,23] for algorithm in [0,1,2] for affinity in range(4) for kind in range(3) for neighbors in [3,7]]
 for index,c in enumerate(configs):
  n=c['n'];rng=np.random.RandomState(814+n);x=rng.normal(size=(n,4));x[abs(x)<.4]=0
  if c['affinity']==2:x=rbf_kernel(x,gamma=.4)
  if c['affinity']==3:x=euclidean_distances(x)
  p=x.shape[1];text=f"{n} {p} 2 {c['algorithm']} {c['affinity']} {c['kind']} {c['neighbors']}\n"+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in x)+'\n';inp=a.report.with_suffix('.in');out=a.report.with_suffix('.out');inp.write_text(text);proc=subprocess.run([str(a.executable),str(inp),str(out)],capture_output=True,text=True)
  source={};native={};error=None;differences=[]
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    if c['algorithm']==0:
     m=SpectralEmbedding(n_components=2,affinity=names[c['affinity']],gamma=.4,n_neighbors=c['neighbors'],random_state=21);z=m.fit_transform(storage(x,c['kind']))
    else:
     m=SpectralClustering(n_clusters=2,affinity=names[c['affinity']],gamma=.4,n_neighbors=c['neighbors'],random_state=21,n_init=3,assign_labels='cluster_qr' if c['algorithm']==1 else 'kmeans');m.fit(storage(x,c['kind']));z=None
    affinity=m.affinity_matrix_;affinity=affinity.toarray() if hasattr(affinity,'toarray') else affinity
   source=dict(affinity=affinity.ravel(order='F').tolist(),effective_affinity=m.affinity)
   if z is not None:source['embedding']=z.ravel(order='F').tolist();source['certificate']=certificate(affinity,z,1)
   else:source['labels']=m.labels_.tolist()
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  if proc.returncode==0 and error is None:
   for line in out.read_text().splitlines():
    key,*values=line.split();native[key]=values[0] if key=='effective_affinity' else list(map(float,values))
   if not np.allclose(native['affinity'],source['affinity'],atol=3e-8,rtol=3e-8):differences.append('affinity')
   if native['effective_affinity']!=source['effective_affinity']:differences.append('effective_affinity')
   graph=np.array(source['affinity']).reshape((n,n),order='F');z=np.array(native['embedding']).reshape((n,2),order='F');native['certificate']=certificate(graph,z,int(c['algorithm']==0))
   if not native['certificate']['passed']:differences.append('spectral_certificate')
   if 'labels' in source:
    native['adjusted_rand']=float(adjusted_rand_score(source['labels'],native['labels']))
    if native['adjusted_rand']!=1: differences.append('labels')
  elif (proc.returncode==0)!=(error is None):differences.append('acceptance')
  if proc.returncode not in [0,1] or proc.returncode==1 and 'fscikit-learn:' not in proc.stdout:differences.append('abnormal_native_exit')
  records.append(dict(index=index,config=c,input=text,source=source,source_error=error,native=native,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,differences=differences))
 root=Path(sklearn.__file__).parent;report=dict(sklearn=sklearn.__version__,configurations=len(records),failures=[r['index'] for r in records if r['differences']],source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['manifold/_spectral_embedding.py','cluster/_spectral.py','neighbors/_base.py']},records=records)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('Spectral CSR',len(records),'strict failures',report['failures']);raise SystemExit(bool(report['failures']))
if __name__=='__main__':main()
