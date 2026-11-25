"""Complete/average trees: chains, sparse map merges and cost-only heap ties."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings,os,sys,heapq
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,sklearn,scipy
from scipy.sparse import csr_matrix
from sklearn.cluster._agglomerative import linkage_tree
import sklearn.cluster._agglomerative as source_ward
import sklearn.cluster._hierarchical_fast as source_fast
import scipy.cluster._hierarchy as source_hierarchy


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True);parser.add_argument('--report');parser.add_argument('--reference-cache',type=Path)
 args=parser.parse_args();cases=[];lines=[];warning_count=0;source_calls=0;cache_hits=0
 provenance=dict(python=sys.version,heapq_sha256=hashlib.sha256(Path(heapq.__file__).read_bytes()).hexdigest(),sklearn=sklearn.__version__,scipy=scipy.__version__,numpy=np.__version__,
   files={m.__file__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()for m in [source_ward,source_fast,source_hierarchy]})
 cache={}
 if args.reference_cache and args.reference_cache.exists():
  saved=json.loads(args.reference_cache.read_text());assert saved['provenance']==provenance,'reference cache provenance changed'
  cache=saved['cases']
 metrics=['euclidean','manhattan','chebyshev','cosine','sqeuclidean','precomputed','l1','l2','cityblock','minkowski']
 methods=['complete','average']
 def add(x,g,mode,clusters,distances,method,metric,label):
  nonlocal source_calls,cache_hits,warning_count
  n,p=x.shape;source_g=None if mode==0 else(g!=0 if mode==2 else g)
  if mode in [3,5]:
   source_g=csr_matrix(g)if mode==3 else csr_matrix((g.ravel(),np.tile(np.arange(n),n),np.arange(n+1)*n),shape=(n,n))
  if mode==4:
   data_values=[];columns=[];ptr=[0]
   for row in g:
    for j in range(n-1,-1,-1):
     if row[j]==0:continue
     data_values += [.4*row[j],.6*row[j]];columns += [j,j]
    ptr.append(len(data_values))
   source_g=csr_matrix((data_values,columns,ptr),shape=(n,n))
  block=[f'{n} {p} {mode} {clusters} {int(distances)} {method+1} {metric+1}']
  block+=[' '.join(map(str,row))for row in x]
  if mode:block+=[' '.join(map(str,row))for row in g]
  key=hashlib.sha256(('\n'.join(block)+'\n').encode()).hexdigest()
  if key not in cache:
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    expected=linkage_tree(x,connectivity=source_g,n_clusters=clusters or None,
       return_distance=distances,linkage=methods[method],affinity=metrics[metric])
   source_calls+=1
   cache[key]=dict(children=np.asarray(expected[0]).reshape(-1,2).tolist(),components=int(expected[1]),
     leaves=int(expected[2]),parents=None if expected[3]is None else expected[3].tolist(),
     distances=expected[4].tolist()if distances else None,warnings=len(caught))
  else:cache_hits+=1
  stored=cache[key];warning_count+=stored['warnings']
  expected=(np.asarray(stored['children'],int).reshape(-1,2),stored['components'],stored['leaves'],
    None if stored['parents']is None else np.asarray(stored['parents'],int))
  if distances:expected+=(np.asarray(stored['distances'],float),)
  label=dict(label,n=n,p=p,mode=mode,clusters=clusters,distances=distances,linkage=methods[method],affinity=metrics[metric])
  cases.append((expected,label));lines.extend(block)
 for seed in [7,42,2026]:
  rng=np.random.default_rng(seed)
  for n,p in [(13,3),(23,5),(2,1)]:
   for data in ['normal','grid','duplicates','zeros']:
    x=rng.normal(size=(n,p))
    if data=='grid':x=rng.integers(-3,4,size=(n,p)).astype(float)
    if data=='duplicates':x[1]=x[0]
    if data=='zeros':x[:]=0
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
      for method in range(2):
       for metric in range(6):
        if mode!=0 and metric in [2,4]:continue
        if metric==3 and np.any(~np.any(x,axis=1)):continue
        values=x
        if metric==5:
         values=np.sqrt(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2))
         # Source consumes upper entries without connectivity and directed
         # row entries with connectivity. Symmetry/zero diagonal are not imposed.
         if data=='normal':values[np.tril_indices(n,-1)]+=.125
         np.fill_diagonal(values,np.arange(n)+1.)
        index=len(cases);clusters=[0,min(2,n-1),n-1][index%3];distances=(index//3)%2==0
        add(values,g,mode,clusters,distances,method,metric,dict(seed=seed,data=data,topology=topology))
 # Dense equal-weight heaps exceed CPython's 2,500-entry heapify threshold.
 for data in ['zeros','grid']:
  rng=np.random.default_rng(91);x=np.zeros((128,3))if data=='zeros'else rng.integers(-2,3,size=(128,3)).astype(float)
  for mode in [0,1,3,4]:
   for method in range(2):
    add(x,np.ones((128,128)),mode,7,True,method,0,dict(seed=91,data=data,topology='large_equal_heap'))
 # Named aliases, wider cosine rows, and signed precomputed distances.
 rng=np.random.default_rng(13)
 for p in [1,17,33]:
  x=rng.normal(size=(9,p));g=np.eye(9,k=1)
  for mode in [0,1,3]:
   for method in range(2):
    for metric in [3,6,7,8,9]:
     if mode!=0 and metric==9:continue
     add(x,g,mode,3,True,method,metric,dict(seed=13,data='normal',topology='aliases_wide'))
  d=-np.sqrt(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2))
  for mode in [0,1,3]:
   for method in range(2):
    add(d,g,mode,3,True,method,5,dict(seed=13,data='signed_precomputed',topology='directed'))
 if args.reference_cache:
  args.reference_cache.parent.mkdir(parents=True,exist_ok=True)
  args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 proc=subprocess.run([args.executable],input='\n'.join(lines)+'\n',capture_output=True,text=True)
 if proc.returncode:raise RuntimeError(proc.stderr+'\nlast output:\n'+proc.stdout[-1500:])
 rows=iter(proc.stdout.splitlines());worst=0
 for expected,label in cases:
  children,components,leaves,parents=expected[:4]
  children=np.asarray(children).reshape(-1,2)
  source_d=expected[4]if len(expected)==5 else np.array([])
  meta=next(rows).split();count=int(meta[2])
  actual_children=np.array([np.fromstring(next(rows),sep=' ',dtype=int)for _ in range(count)]).reshape(-1,2)
  parent_count=int(next(rows));actual_parent=np.fromstring(next(rows),sep=' ',dtype=int)
  distance_count=int(next(rows));actual_d=np.fromstring(next(rows),sep=' ')
  assert int(meta[0])==leaves and int(meta[1])==components,label
  assert (meta[3]=='T')==(label['mode']!=0)and(meta[4]=='T')==(components>1),label
  np.testing.assert_array_equal(actual_children,children+1,err_msg=str(label))
  np.testing.assert_array_equal(actual_parent,np.array([],int)if parents is None else parents+1,err_msg=str(label))
  assert len(actual_parent)==parent_count and len(actual_d)==distance_count,label
  np.testing.assert_allclose(actual_d,source_d,rtol=3e-8,atol=3e-8,err_msg=str(label))
  worst=max(worst,float(np.max(np.abs(actual_d-source_d)/(1+np.abs(source_d)),initial=0)))
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),groups=4*len(cases),max_scaled_error=worst,source_warnings=warning_count,
   sklearn=sklearn.__version__,scipy=scipy.__version__,numpy=np.__version__,
   executable_sha256=hashlib.sha256(Path(args.executable).read_bytes()).hexdigest(),
   source_sha256=hashlib.sha256((root/'src/sklearn_hierarchy.f90').read_bytes()).hexdigest(),
   validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   input_sha256=hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest(),
   configuration_inventory=[c[1]for c in cases],reference_provenance=provenance,
   source_calls=source_calls,reference_cache_hits=cache_hits,
   reference_cache_sha256=hashlib.sha256(args.reference_cache.read_bytes()).hexdigest()if args.reference_cache else None)
 print(json.dumps({k:v for k,v in report.items()if k!='configuration_inventory'},indent=2))
 if args.report:Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
