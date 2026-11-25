"""Ward trees: chain, structured moments, topology completion and partial forests."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,sklearn,scipy
from scipy.sparse import csr_matrix
from sklearn.cluster import ward_tree
import sklearn.cluster._agglomerative as source_ward
import sklearn.cluster._hierarchical_fast as source_fast
import scipy.cluster._hierarchy as source_hierarchy


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True);parser.add_argument('--report');parser.add_argument('--reference-cache',type=Path)
 args=parser.parse_args();cases=[];lines=[];warning_count=0;source_calls=0;cache_hits=0
 provenance=dict(sklearn=sklearn.__version__,scipy=scipy.__version__,numpy=np.__version__,
   files={m.__file__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()for m in [source_ward,source_fast,source_hierarchy]})
 cache={}
 if args.reference_cache and args.reference_cache.exists():
  saved=json.loads(args.reference_cache.read_text());assert saved['provenance']==provenance,'reference cache provenance changed'
  cache=saved['cases']
 for seed in [7,42,2026]:
  rng=np.random.default_rng(seed)
  for n,p in [(13,3),(23,5),(2,1),(1,2)]:
   for data in ['normal','grid','duplicates','zeros']:
    x=rng.normal(size=(n,p))
    if data=='grid':x=rng.integers(-3,4,size=(n,p)).astype(float)
    if data=='duplicates'and n>1:x[1]=x[0]
    if data=='zeros':x[:]=0
    for topology in ['none','chain','directed','random','split','empty','cancellation','diagonal']:
     g=np.zeros((n,n))
     if topology in ['chain','directed','split','cancellation']:
      for i in range(n-1):
       if topology=='split'and i==n//2-1:continue
       g[i,i+1]=1
       if topology!='directed':g[i+1,i]= -1 if topology=='cancellation'else 1
     if topology=='random':g=(rng.random((n,n))<.2).astype(float)
     if topology=='diagonal':np.fill_diagonal(g,1)
     if topology=='none'and n==1:continue
     for mode in ([0]if topology=='none'else [1,2,3,4,5]):
      source_g=None if mode==0 else (g!=0 if mode==2 else g)
      if mode in [3,5]:
       if mode==3:source_g=csr_matrix(g)
       else:
        source_g=csr_matrix((g.ravel(),np.tile(np.arange(n),n),np.arange(n+1)*n),shape=(n,n))
      if mode==4:
       data_values=[];columns=[];ptr=[0]
       for row in g:
        for j in range(n-1,-1,-1):
         if row[j]==0:continue
         data_values += [.4*row[j],.6*row[j]];columns += [j,j]
        ptr.append(len(data_values))
       source_g=csr_matrix((data_values,columns,ptr),shape=(n,n))
      for clusters in sorted(set([0,min(2,n),n])):
       for distances in [False,True]:
        key_data=x.tobytes()+repr((mode,clusters,distances)).encode()
        if source_g is not None:
         if isinstance(source_g,csr_matrix):
          key_data+=source_g.data.tobytes()+source_g.indices.tobytes()+source_g.indptr.tobytes()
         else:key_data+=source_g.tobytes()
        key=hashlib.sha256(key_data).hexdigest()
        if key not in cache:
         with warnings.catch_warnings(record=True)as caught:
          warnings.simplefilter('always')
          expected=ward_tree(x,connectivity=source_g,n_clusters=clusters or None,return_distance=distances)
         source_calls+=1
         cache[key]=dict(children=np.asarray(expected[0]).reshape(-1,2).tolist(),components=int(expected[1]),
           leaves=int(expected[2]),parents=None if expected[3]is None else expected[3].tolist(),
           distances=expected[4].tolist()if distances else None,warnings=len(caught))
        else:cache_hits+=1
        stored=cache[key];warning_count+=stored['warnings']
        expected=(np.asarray(stored['children'],int).reshape(-1,2),stored['components'],stored['leaves'],
          None if stored['parents']is None else np.asarray(stored['parents'],int))
        if distances:expected+= (np.asarray(stored['distances'],float),)
        label=dict(seed=seed,n=n,p=p,data=data,topology=topology,mode=mode,clusters=clusters,distances=distances)
        cases.append((expected,label))
        lines += [f'{n} {p} {mode} {clusters} {int(distances)}']
        lines += [' '.join(map(str,row))for row in x]
        if mode:lines += [' '.join(map(str,row))for row in g]
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
