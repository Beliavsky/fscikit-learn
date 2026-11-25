"""Fitted complete/average trees, threshold cuts, sparse forests and feature pooling.

Cluster labels/pooled columns are canonicalized by first sample/feature, the
native package convention. Tree children and metadata are compared directly.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings,heapq,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,sklearn,scipy
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering,FeatureAgglomeration
from sklearn.cluster._agglomerative import linkage_tree
import sklearn.cluster._agglomerative as source_ward
import sklearn.cluster._hierarchical_fast as source_fast
import scipy.cluster._hierarchy as source_hierarchy


def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--executable',required=True)
 parser.add_argument('--report')
 parser.add_argument('--reference-cache',type=Path)
 args=parser.parse_args();root=Path(__file__).resolve().parents[1]
 provenance=dict(python=sys.version,heapq_sha256=hashlib.sha256(Path(heapq.__file__).read_bytes()).hexdigest(),sklearn=sklearn.__version__,scipy=scipy.__version__,numpy=np.__version__,
   files={m.__file__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
          for m in [source_ward,source_fast,source_hierarchy]})
 cache={};cases=[];lines=[];source_calls=cache_hits=warning_count=0
 if args.reference_cache and args.reference_cache.exists():
  saved=json.loads(args.reference_cache.read_text())
  assert saved['provenance']==provenance,'reference cache provenance changed'
  cache=saved['cases']

 metrics=["euclidean","manhattan","chebyshev","cosine","sqeuclidean","precomputed","l1","l2","cityblock","minkowski"]
 methods=["complete","average"]
 def add(x,g,mode,clusters,full,distances,threshold,feature,label):
  nonlocal source_calls,cache_hits,warning_count
  method=len(cases)%2;allowed=[0,1,2,3,4,5]if mode==0 else [0,1,3,5]
  metric=allowed[(len(cases)//2)%len(allowed)]
  if metric==3 and np.any(~np.any(x,axis=1)):metric=0
  if metric==5:
   x=np.sqrt(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2))
  n,p=x.shape
  block=[f'{n} {p} {mode} {clusters} {full} {int(distances)} {threshold:.17g} {int(feature>0)} {max(0,feature-1)} {method+1} {metric+1}']
  block+=[' '.join(map(str,row))for row in x]
  if mode:block+=[' '.join(map(str,row))for row in g]
  key=hashlib.sha256(('\n'.join(block)+'\n').encode()).hexdigest()
  if key not in cache:
   connectivity=None if mode==0 else (g!=0 if mode==2 else g)
   if mode in [3,5]:
    connectivity=csr_matrix(g)if mode==3 else csr_matrix((g.ravel(),np.tile(np.arange(n),n),np.arange(n+1)*n),shape=(n,n))
   if mode==4:
    data=[];columns=[];ptr=[0]
    for row in g:
     for j in range(n-1,-1,-1):
      if row[j]:data.extend([.4*row[j],.6*row[j]]);columns.extend([j,j])
     ptr.append(len(data))
    connectivity=csr_matrix((data,columns,ptr),shape=(n,n))
   options=dict(n_clusters=clusters or None,linkage=methods[method],metric=metrics[metric],connectivity=connectivity,
     compute_full_tree=['auto',True,False][full],compute_distances=distances,
     distance_threshold=threshold if threshold>=0 else None)
   estimator=AgglomerativeClustering if feature==0 else FeatureAgglomeration
   if feature:options['pooling_func']=[np.mean,np.median,np.max][feature-1]
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    model=estimator(**options).fit(x if feature==0 else x.T)
   order=sorted(range(model.n_clusters_),key=lambda c:np.flatnonzero(model.labels_==c)[0])
   labels=np.empty(n,int)
   for canonical,original in enumerate(order):labels[model.labels_==original]=canonical
   expected=dict(labels=labels.tolist(),children=model.children_.tolist(),
     distances=model.distances_.tolist()if distances or threshold>=0 else [],
     clusters=int(model.n_clusters_),leaves=int(model.n_leaves_),features=int(model.n_features_in_),
     components=int(model.n_connected_components_),warnings=len(caught))
   if feature:
    z=model.transform(x.T)
    expected.update(pooled=z[:,order].ravel(order='F').tolist(),inverse=model.inverse_transform(z).ravel(order='F').tolist())
   cache[key]=expected;source_calls+=1
  else:cache_hits+=1
  expected=cache[key];warning_count+=expected['warnings']
  actual_full=mode==0 or full==1 or(full==0 and(threshold>=0 or clusters<max(100,.02*n)))
  case=dict(label,n=n,p=p,mode=mode,clusters=clusters,full=full,distances=distances,threshold=threshold,feature=feature,linkage=methods[method],metric=metrics[metric])
  cases.append((expected,case,actual_full));lines.extend(block)

 for seed in [7,42,2026]:
  rng=np.random.default_rng(seed)
  for n,p in [(13,3),(23,5),(2,1)]:
   for data in ['normal','grid','duplicates','zeros']:
    x=rng.normal(size=(n,p))
    if data=='grid':x=rng.integers(-3,4,size=(n,p)).astype(float)
    if data=='duplicates':x[1]=x[0]
    if data=='zeros':x[:]=0
    # Threshold equality/adjacency is independently selected from source
    # heights, never from native outputs.
    source_heights=linkage_tree(x,linkage="complete",return_distance=True)[4]
    thresholds=[0.,float(source_heights[len(source_heights)//2]),float(np.nextafter(source_heights[-1],np.inf))]
    for topology in ['none','chain','directed','random','split','empty','cancellation','diagonal']:
     g=np.zeros((n,n))
     if topology in ['chain','directed','split','cancellation']:
      for i in range(n-1):
       if topology=='split'and i==n//2-1:continue
       g[i,i+1]=1
       if topology!='directed':g[i+1,i]=-1 if topology=='cancellation'else 1
     if topology=='random':g=(rng.random((n,n))<.2).astype(float)
     if topology=='diagonal':np.fill_diagonal(g,1)
     for mode in ([0]if topology=='none'else [1,2,3,4,5]):
      label=dict(seed=seed,data=data,topology=topology)
      for full in [0,1,2]:
       for distances in [False,True]:
        index=len(cases)
        clusters=[1,min(2,n-1),n-1][(index//2+mode)%3]
        add(x,g,mode,clusters,full,distances,-1,(index//3+mode)%4,label)
      for j,threshold in enumerate(thresholds):
       add(x,g,mode,0,j%2,j%2==0,threshold,(len(cases)+j)%4,label)

 # Exercise the 100-cluster auto boundary with both forced settings as controls.
 for seed in [7,42,2026]:
  rng=np.random.default_rng(seed);n,p=110,2;x=rng.normal(size=(n,p));g=np.eye(n,k=1)
  for mode in [0,1,3]:
   for clusters in [99,100,n-1]:
    for full in [0,1,2]:
     for distances in [False,True]:
      add(x,g,mode,clusters,full,distances,-1,(len(cases)//2)%4,dict(seed=seed,data='normal',topology='auto_boundary'))

 if args.reference_cache:
  args.reference_cache.parent.mkdir(parents=True,exist_ok=True)
  args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 payload='\n'.join(lines)+'\n'
 proc=subprocess.run([args.executable],input=payload,text=True,capture_output=True)
 if proc.returncode:raise RuntimeError(proc.stderr+'\nlast output:\n'+proc.stdout[-1500:])
 rows=iter(proc.stdout.splitlines());worst=0;groups=0
 for expected,label,full in cases:
  meta=next(rows).split();count=int(meta[4])
  labels=np.fromstring(next(rows),sep=' ',dtype=int)
  children=np.array([np.fromstring(next(rows),sep=' ',dtype=int)for _ in range(count)]).reshape(-1,2)
  distance_count=int(next(rows));distances=np.fromstring(next(rows),sep=' ')
  assert list(map(int,meta[:4]))==[expected[k]for k in ['clusters','leaves','features','components']],label
  assert(meta[5]=='T')==full and(meta[6]=='T')==(expected['components']>1),label
  np.testing.assert_array_equal(labels,expected['labels'],err_msg=str(label))
  np.testing.assert_array_equal(children,np.array(expected['children']).reshape(-1,2),err_msg=str(label))
  assert len(distances)==distance_count,label
  want=np.asarray(expected['distances'])
  np.testing.assert_allclose(distances,want,rtol=3e-8,atol=3e-8,err_msg=str(label))
  worst=max(worst,float(np.max(abs(distances-want)/(1+abs(want)),initial=0)));groups+=4
  if label['feature']:
   for name in ['pooled','inverse']:
    actual=np.fromstring(next(rows),sep=' ');want=np.asarray(expected[name])
    np.testing.assert_allclose(actual,want,rtol=3e-8,atol=3e-8,err_msg=str((label,name)))
    worst=max(worst,float(np.max(abs(actual-want)/(1+abs(want)),initial=0)));groups+=1
 assert next(rows,None)is None,'unexpected trailing driver output'
 report=dict(configurations=len(cases),groups=groups,max_scaled_error=worst,source_warnings=warning_count,
   reference_provenance=provenance,source_calls=source_calls,reference_cache_hits=cache_hits,
   reference_cache_sha256=hashlib.sha256(args.reference_cache.read_bytes()).hexdigest()if args.reference_cache else None,
   input_sha256=hashlib.sha256(payload.encode()).hexdigest(),
   source_sha256=hashlib.sha256((root/'src/sklearn_cluster.f90').read_bytes()).hexdigest(),
   executable_sha256=hashlib.sha256(Path(args.executable).read_bytes()).hexdigest(),
   validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   configuration_inventory=[case[1]for case in cases])
 print(json.dumps({k:v for k,v in report.items()if k!='configuration_inventory'},indent=2))
 if args.report:Path(args.report).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
