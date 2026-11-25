"""Weighted Elkan KMeans: seeded dense/CSR fitting, restarts and queries."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,warnings
import numpy as np,scipy,sklearn
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_info
from sklearn.cluster import KMeans,k_means
import sklearn.cluster._kmeans as source

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True)
 ap.add_argument('--reference-cache',type=Path);ap.add_argument('--report',type=Path);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 provenance=dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,files={str(p.name):sha(p)for p in [Path(source.__file__),*(Path(source.__file__).parent/n for n in ['_k_means_elkan.pyx','_k_means_lloyd.pyx','_k_means_common.pyx'])]})
 cache={};cases=[];calls=hits=warns=0
 if args.reference_cache and args.reference_cache.exists():
  old=json.loads(args.reference_cache.read_text());assert old['provenance']==provenance;cache=old['cases']
 def add(x,k,method,w,seed,ninit=0,limit=25,tol=1e-4,kind='normal',initial=None,route=0):
  nonlocal calls,hits,warns
  n,p=x.shape;xd=x.toarray()if isinstance(x,csr_matrix)else x
  m=min(2,n)+2;q=np.vstack([xd[:min(2,n)],xd[-1:]+.17,np.zeros((1,p))]);qw=np.linspace(.5,1.5,m)
  # CSR callers supply query representation independently through outer setup.
  train_sparse=isinstance(x,csr_matrix);query_sparse=add.query_sparse
  if query_sparse:q=x[:m].copy()if kind=='noncanonical'else csr_matrix(q)
  init=initial if method=='array'else None
  native_method='k-means++'if method=='array'else method
  lines=[f'{n} {p} {k} {m} {int(train_sparse)} {int(query_sparse)} 1 {int(init is not None)} {ninit} {limit} {seed} {tol:.17g} {native_method} {route} elkan',
   ' '.join(map(str,w)),' '.join(map(str,qw))]
  if init is not None:lines+=[' '.join(map(str,row))for row in init]
  for data in [x,q]:
   if isinstance(data,csr_matrix):lines += [str(data.nnz),' '.join(map(str,data.indptr+1)),' '.join(map(str,data.indices+1)),' '.join(map(str,data.data))]
   else:lines+=[' '.join(map(str,row))for row in data]
  request='\n'.join(lines)+'\n';key=hashlib.sha256(request.encode()).hexdigest()
  label=dict(n=n,p=p,k=k,method=method,seed=seed,n_init=ninit,limit=limit,tol=tol,kind=kind,train_sparse=train_sparse,query_sparse=query_sparse,route=route)
  if key in cache:expected=cache[key];hits+=1
  else:
   kwargs=dict(n_clusters=k,init=init.copy()if init is not None else method,n_init='auto'if ninit==0 else ninit,max_iter=limit,tol=tol,random_state=seed,algorithm='elkan')
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always');model=KMeans(**kwargs).fit(x,sample_weight=w)
    if route:
     centers,labels,inertia,niter=k_means(x,sample_weight=w,return_n_iter=True,**kwargs)
     np.testing.assert_array_equal(labels,model.labels_);np.testing.assert_array_equal(centers,model.cluster_centers_)
     assert inertia==model.inertia_ and niter==model.n_iter_
    expected=dict(meta=[model.n_iter_,model.n_features_in_,model._n_init],centers=model.cluster_centers_.ravel(order='F').tolist(),labels=model.labels_.tolist(),inertia=[model.inertia_],predict=model.predict(q).tolist(),distance=model.transform(q).ravel(order='F').tolist(),score=[model.score(q,sample_weight=qw)])
   expected['warnings']=len(caught);cache[key]=expected;calls+=1
  warns+=expected['warnings'];cases.append((label,request,expected))
 for seed in ([7]if args.smoke else [0,7,42]):
  rng=np.random.default_rng(seed)
  for n in ([7]if args.smoke else [1,7,33,257]):
   for p in ([3]if args.smoke else [1,3,7]):
    base=rng.normal(size=(n,p));base+=np.arange(n)[:,None]%3*3
    for k in ([2]if args.smoke else sorted(set([1,min(3,n)]))):
     for kind in (['normal']if args.smoke else ['normal','duplicate','offset','zero']):
      x=base.copy()
      if kind=='duplicate':x=np.round(x)
      elif kind=='offset':x+=np.arange(p)*1e3+1e3
      elif kind=='zero':x[:]=0
      for method in ['k-means++','random','array']:
       for weighting in ['ones','fraction','zeros']:
        w=np.ones(n)
        if weighting=='fraction':w=np.arange(1,n+1,dtype=float)/n
        elif weighting=='zeros':w[::3]=0;w[-1]=1
        for train_sparse in [False,True]:
         for query_sparse in [False,True]:
          add.query_sparse=query_sparse;route=len(cases)%2
          add(csr_matrix(x)if train_sparse else x,k,method,w,seed,ninit=0 if weighting=='ones'else 3,
           limit=25 if weighting=='ones'else 1 if weighting=='fraction'else 2,kind=kind,initial=x[:k],route=route)
 if not args.smoke:
  rng=np.random.default_rng(123)
  for p in [1,3,7]:
   x=np.round(rng.normal(size=(9,p))*2);initial=x[[0,4,8]]
   for mode in ['all_zero','signed','noncanonical']:
    w=np.zeros(9)if mode=='all_zero'else np.array([1.,-.2,2.,.5,1.,-.1,1.,.5,2.])if mode=='signed'else np.ones(9)
    if mode=='noncanonical':
     data=[];indices=[];indptr=[0]
     for row in x:
      for j in range(p-1,-1,-1):
       data.extend([row[j]*.25,row[j]*.75,0.]);indices.extend([j,j,j])
      indptr.append(len(data))
     data_x=csr_matrix((data,indices,indptr),shape=x.shape)
    else:data_x=x
    for query_sparse in [False,True]:
     add.query_sparse=query_sparse
     for tolerance in [0.,1e-4]:add(data_x,3,'array',w,123,initial=initial,kind=mode,tol=tolerance,limit=5)
 if not args.smoke:
  for seed in [7,42]:
   rng=np.random.default_rng(seed)
   for p in [9,33,129]:
    x=rng.normal(size=(33,p))
    for k in [1,3,9]:
     for method in ['k-means++','random','array']:
      for train_sparse in [False,True]:
       for query_sparse in [False,True]:
        add.query_sparse=query_sparse
        add(csr_matrix(x)if train_sparse else x,k,method,np.linspace(.1,2,33),seed,ninit=3,
         limit=10,initial=x[:k],kind='wide',route=len(cases)%2)
 if args.reference_cache:
  args.reference_cache.parent.mkdir(parents=True,exist_ok=True);args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 request=''.join(c[1]for c in cases);run=subprocess.run([args.executable],input=request,text=True,capture_output=True)
 if run.returncode:raise RuntimeError(run.stderr+'\n'+run.stdout[-1000:])
 rows=iter(run.stdout.splitlines());groups=0;worst=0
 for label,_,expected in cases:
  for field in ['meta','centers','labels','inertia','predict','distance','score']:
   actual=np.fromstring(next(rows),sep=' ');want=np.asarray(expected[field]);groups+=1
   if field in ['meta','labels','predict']:np.testing.assert_array_equal(actual,want,err_msg=str((label,field)))
   else:
    np.testing.assert_allclose(actual,want,rtol=3e-8,atol=3e-8,equal_nan=True,err_msg=str((label,field)))
    finite=np.isfinite(actual)&np.isfinite(want)
    if finite.any():worst=max(worst,float(np.max(abs(actual[finite]-want[finite])/(1+abs(want[finite])))))
 assert next(rows,None)is None
 root=Path(__file__).resolve().parents[1]
 report=dict(runtime_blas=[{k:v for k,v in info.items()if k in ['internal_api','version','architecture','num_threads','prefix']}for info in threadpool_info()],configurations=len(cases),groups=groups,max_scaled_error=worst,source_warnings=warns,source_calls=calls,reference_cache_hits=hits,reference_provenance=provenance,reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,
  source_sha256={name:sha(root/'src'/name)for name in ['sklearn_kmeans_core.f90','sklearn_cluster.f90','sklearn_kmeans_seeding.f90']},validator_sha256=sha(__file__),executable_sha256=sha(args.executable),input_sha256=hashlib.sha256(request.encode()).hexdigest(),configuration_inventory=[c[0]for c in cases])
 print(json.dumps({k:v for k,v in report.items()if k!='configuration_inventory'},indent=2))
 if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
