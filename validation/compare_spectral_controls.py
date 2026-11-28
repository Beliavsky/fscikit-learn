"""Spectral kernel controls, independent embedding dimensions and signed graphs."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering
from sklearn.cluster._spectral import cluster_qr,discretize
from sklearn.cluster._kmeans import k_means
from sklearn.manifold import SpectralEmbedding,spectral_embedding
from sklearn.manifold._spectral_embedding import csgraph_laplacian,_set_diag
from sklearn.utils.validation import check_symmetric
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel,euclidean_distances
from compare_spectral_sparse import storage
NAMES=['rbf','nearest_neighbors','precomputed','precomputed_nearest_neighbors','linear','poly','polynomial','sigmoid','cosine','laplacian','chi2','additive_chi2']

def configs():
 result=[dict(pattern=pattern,kernel=kernel,kind=kind,assignment=assignment,c=c,k=2,estimator=0,gamma=.4,degree=2.5,coef0=.7) for pattern in ['positive','centered'] for kernel in range(12) for kind in range(3) for assignment in range(3) for c in [0,1,3]]
 result += [dict(pattern=pattern,kernel=2,kind=kind,assignment=assignment,c=c,k=2,estimator=0,gamma=.4,degree=2.5,coef0=.7) for pattern in ['asym','signed','strong_signed'] for kind in range(3) for assignment in range(3) for c in [1,3,15]]
 result += [dict(pattern='positive',kernel=kernel,kind=kind,assignment=assignment,c=c,k=2,estimator=0,gamma=.4,degree=2.5,coef0=.7) for kernel in [0,1,2,3] for kind in range(3) for assignment in range(3) for c in [13,18]]
 result += [dict(pattern=pattern,kernel=kernel,kind=kind,assignment=0,c=c,k=2,estimator=1,gamma=.4,degree=2.5,coef0=.7) for pattern in ['positive','asym','signed','zero_row'] for kernel in [0,2] for kind in range(3) for c in [1,13,18]]
 result += [dict(pattern='zero_row',kernel=kernel,kind=kind,assignment=assignment,c=c,k=2,estimator=0,gamma=.4,degree=2.5,coef0=.7) for kernel in [4,8] for kind in range(3) for assignment in range(3) for c in [1,3]]
 result += [dict(pattern='positive',kernel=kernel,kind=kind,assignment=1,c=3,k=2,estimator=0,gamma=g,degree=d,coef0=b) for kernel in [5,7] for kind in [0,1] for g,d,b in [(0.,0.,.7),(.2,1.,-.2),(.4,2.5,-.2)]]
 return result

def data(c):
 n=13;rng=np.random.RandomState(947);x=np.abs(rng.normal(size=(n,4)))*.3+.1
 if c['pattern']=='centered':x=rng.normal(size=(n,4))*.3
 if c['pattern']=='zero_row':x[0]=0
 if c['kernel']==2:
  x=rbf_kernel(x,gamma=.4)
  if c['pattern']=='asym':x[0,1]+=.3;x[1,0]+=.01
  if c['pattern'] in ['signed','strong_signed']:
   x[:]=1;x[0,1]=x[1,0]=-.25 if c['pattern']=='signed' else -10.
 if c['kernel']==3:x=euclidean_distances(x)
 return x

def certificate(graph,z,source_z):
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');adj=check_symmetric(graph);lap,dd=csgraph_laplacian(adj,normed=True,return_diag=True);lap=_set_diag(lap,1,True)
 lap=lap.toarray() if hasattr(lap,'toarray') else lap
 source_u=dd[:,None]*source_z;values=np.sum(source_u*(lap@source_u),axis=0)
 u=dd[:,None]*z;residual=float(np.max(np.abs(lap@u-u*values)));orthogonal=float(np.max(np.abs(u.T@u-np.eye(z.shape[1]))))
 return dict(eigenvalues=values.tolist(),residual=residual,orthogonality=orthogonal,passed=residual<3e-8 and orthogonal<3e-8)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 for index,c in enumerate(configs()):
  x=data(c);n,p=x.shape;text=f"{n} {p} {c['k']} {c['c']} {c['assignment']} {c['kernel']} {c['kind']} 3 {c['estimator']}\n{c['gamma']} {c['degree']} {c['coef0']}\n"+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in x)+'\n';inp=a.report.with_suffix('.in');out=a.report.with_suffix('.out');inp.write_text(text)
  proc=subprocess.run([str(a.executable),str(inp),str(out)],capture_output=True,text=True);source={};native={};error=None;differences=[];graph=None;source_z=None
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    if c['estimator']==0:
     m=SpectralClustering(n_clusters=c['k'],n_components=c['c'] or None,affinity=NAMES[c['kernel']],assign_labels=['kmeans','cluster_qr','discretize'][c['assignment']],gamma=c['gamma'],degree=c['degree'],coef0=c['coef0'],n_neighbors=3,random_state=21,n_init=3).fit(storage(x,c['kind']))
     graph=m.affinity_matrix_;source_z=spectral_embedding(graph,n_components=c['c'] or c['k'],drop_first=False,random_state=21);source['labels']=m.labels_.tolist()
    else:
     m=SpectralEmbedding(n_components=c['c'],affinity=NAMES[c['kernel']],gamma=c['gamma'],n_neighbors=3,random_state=21).fit(storage(x,c['kind']));graph=m.affinity_matrix_;source_z=m.embedding_
    dense=graph.toarray() if hasattr(graph,'toarray') else graph;adj=check_symmetric(dense);connected=int(connected_components(adj,directed=False,return_labels=False))
   source.update(affinity=dense.ravel(order='F').tolist(),shape=[n,source_z.shape[1],connected],embedding=source_z.ravel(order='F').tolist())
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  if proc.returncode==0 and error is None:
   for line in out.read_text().splitlines():
    key,*values=line.split();native[key]=list(map(float,values))
   if np.shape(native['affinity'])!=np.shape(source['affinity']) or not np.allclose(native['affinity'],source['affinity'],atol=3e-8,rtol=3e-8,equal_nan=True):differences.append('affinity')
   if native['shape']!=source['shape']:differences.append('shape')
   z=np.array(native['embedding']).reshape((n,int(native['shape'][1])),order='F')
   if z.shape==source_z.shape:
    native['certificate']=certificate(graph,z,source_z)
    if not native['certificate']['passed']:differences.append('spectral_certificate')
   if 'labels' in source:
    if adjusted_rand_score(source['labels'],native['labels'])!=1:differences.append('partition')
    try:
     with warnings.catch_warnings():
      warnings.simplefilter('ignore');state=np.random.RandomState(21);state.uniform(-1,1,n)
      if c['assignment']==0:_,replay,_=k_means(z,c['k'],n_init=3,random_state=state)
      elif c['assignment']==1:replay=cluster_qr(z)
      else:replay=discretize(z,random_state=state)
     native['same_embedding_source_labels']=replay.tolist()
     if adjusted_rand_score(native['labels'],replay)!=1:differences.append('same_embedding_partition')
    except Exception as exc:native['replay_error']=str(exc);differences.append('same_embedding_error')
  elif (proc.returncode==0)!=(error is None):differences.append('acceptance')
  if proc.returncode not in [0,1] or proc.returncode==1 and 'fscikit-learn:' not in proc.stdout:differences.append('abnormal_native_exit')
  records.append(dict(index=index,config=c,input=text,source=source,source_error=error,native=native,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,differences=differences))
 root=Path(sklearn.__file__).parent;report=dict(sklearn=sklearn.__version__,configurations=len(records),failures=[r['index'] for r in records if r['differences']],source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['cluster/_spectral.py','manifold/_spectral_embedding.py','utils/validation.py','metrics/pairwise.py']},records=records)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('Spectral controls',len(records),'strict failures',report['failures']);raise SystemExit(bool(report['failures']))
if __name__=='__main__':main()
