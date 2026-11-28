"""Discrete spectral rotation: same-input algorithm and estimator comparisons."""
from pathlib import Path
import argparse,contextlib,hashlib,io,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.cluster import SpectralClustering
from sklearn.cluster._spectral import discretize
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel,euclidean_distances
from compare_spectral_sparse import storage,certificate

def run(executable,report,text):
 inp=report.with_suffix('.in');out=report.with_suffix('.out');inp.write_text(text)
 p=subprocess.run([str(executable),str(inp),str(out)],capture_output=True,text=True)
 return p,out.read_text() if p.returncode==0 else ''

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--spectral-executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 configs=[dict(pattern=pattern,n=n,k=k,seed=seed,iterations=it) for pattern in ['gaussian','clustered','onehot','rankdef','huge'] for n,k in [(17,2),(29,3),(43,4)] for seed in [0,1,21,42] for it in [-1,0,1,20]]
 for index,c in enumerate(configs):
  n,k=c['n'],c['k'];rng=np.random.RandomState(712+n+k);x=rng.normal(size=(n,k))
  if c['pattern'] in ['clustered','onehot']:
   x=np.eye(k)[np.arange(n)%k]
   if c['pattern']=='clustered':x=x+.08*rng.normal(size=(n,k))
  if c['pattern']=='rankdef':x=rng.normal(size=(n,2))@rng.normal(size=(2,k))
  if c['pattern']=='huge':x[:,0]=np.linspace(.3,.7,n)*1e308
  text=f"{n} {k} {c['seed']} {c['iterations']} 30\n"+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in x)+'\n';p,output=run(a.executable,a.report,text);error=None;source=[];differences=[]
  try:
   with warnings.catch_warnings():warnings.simplefilter('ignore');source=discretize(x,random_state=c['seed'],n_iter_max=c['iterations']).tolist()
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  native=np.fromstring(output,sep=' ',dtype=int).tolist() if p.returncode==0 else []
  if p.returncode==0 and error is None:
   if len(native)!=len(source) or adjusted_rand_score(source,native)!=1:differences.append('partition')
  elif (p.returncode==0)!=(error is None):differences.append('acceptance')
  if p.returncode not in [0,1] or p.returncode==1 and 'fscikit-learn:' not in p.stdout:differences.append('abnormal_native_exit')
  records.append(dict(index=index,config=c,input=text,source=source,source_error=error,native=native,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,differences=differences))
 integrations=[];names=['rbf','nearest_neighbors','precomputed','precomputed_nearest_neighbors']
 for n in [13,23]:
  for affinity in range(4):
   for kind in range(3):
    for neighbors in [3,7]:
     rng=np.random.RandomState(814+n);x=rng.normal(size=(n,4));x[abs(x)<.4]=0
     if affinity==2:x=rbf_kernel(x,gamma=.4)
     if affinity==3:x=euclidean_distances(x)
     text=f"{n} {x.shape[1]} 2 3 {affinity} {kind} {neighbors}\n"+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in x)+'\n';p,output=run(a.spectral_executable,a.report,text);source={};native={};error=None;differences=[]
     try:
      with warnings.catch_warnings():
       warnings.simplefilter('ignore');m=SpectralClustering(n_clusters=2,affinity=names[affinity],gamma=.4,n_neighbors=neighbors,random_state=21,n_init=3,assign_labels='discretize').fit(storage(x,kind))
      graph=m.affinity_matrix_;graph=graph.toarray() if hasattr(graph,'toarray') else graph;source=dict(affinity=graph.ravel(order='F').tolist(),labels=m.labels_.tolist())
     except Exception as exc:error=f'{type(exc).__name__}: {exc}'
     if p.returncode==0 and error is None:
      for line in output.splitlines():
       key,*values=line.split();native[key]=values[0] if key=='effective_affinity' else list(map(float,values))
      if not np.allclose(native['affinity'],source['affinity'],atol=3e-8,rtol=3e-8):differences.append('affinity')
      if adjusted_rand_score(source['labels'],native['labels'])!=1:differences.append('partition')
      z=np.array(native['embedding']).reshape((n,2),order='F');native['certificate']=certificate(graph,z,0)
      if not native['certificate']['passed']:differences.append('spectral_certificate')
      state=np.random.RandomState(21);state.uniform(-1,1,n);replay=discretize(z,random_state=state);native['same_embedding_source_labels']=replay.tolist()
      if adjusted_rand_score(native['labels'],replay)!=1:differences.append('same_embedding_partition')
     elif (p.returncode==0)!=(error is None):differences.append('acceptance')
     if p.returncode not in [0,1] or p.returncode==1 and 'fscikit-learn:' not in p.stdout:differences.append('abnormal_native_exit')
     integrations.append(dict(index=len(integrations),config=dict(n=n,affinity=affinity,kind=kind,neighbors=neighbors),input=text,source=source,source_error=error,native=native,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,differences=differences))
 root=Path(sklearn.__file__).parent;report=dict(sklearn=sklearn.__version__,configurations=len(records),failures=[r['index'] for r in records if r['differences']],integration_configurations=len(integrations),integration_failures=[r['index'] for r in integrations if r['differences']],source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['cluster/_spectral.py','manifold/_spectral_embedding.py','utils/_arpack.py']},records=records,integration=integrations)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('Spectral discretize',len(records),'strict failures',report['failures'],'integration',len(integrations),'strict failures',report['integration_failures']);raise SystemExit(bool(report['failures'] or report['integration_failures']))
if __name__=='__main__':main()
