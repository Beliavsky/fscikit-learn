"""Actual sklearn functional graphs and stable CSR sorting; retain raw ordering differences."""
from pathlib import Path
import argparse,hashlib,json,os,sys,subprocess,tempfile,warnings,concurrent.futures
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np,scipy,sklearn
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.neighbors import (NearestNeighbors,KNeighborsClassifier,KNeighborsRegressor,
 RadiusNeighborsClassifier,RadiusNeighborsRegressor,kneighbors_graph,radius_neighbors_graph,sort_graph_by_row_values)
from sklearn.neighbors._base import _is_sorted_by_data
import sklearn.neighbors._graph as graph_source,sklearn.neighbors._base as base_source
FIELDS=['meta','indptr','indices','data','dense','input_indptr','input_indices','input_data','initial_sorted']
EXACT=set(FIELDS)-{'data','dense','input_data'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def graph_cases(smoke):
 for seed in ([7]if smoke else[7,42]):
  rng=np.random.RandomState(seed)
  for n,p in ([(9,3)]if smoke else[(1,1),(9,3),(33,5),(65,5)]):
   original=rng.normal(size=(n,p));original[rng.random_sample((n,p))<.3]=0
   for kind in (['regular']if smoke else['regular','duplicate']):
    x=original.copy()
    if kind=='duplicate':x[:min(3,n)]=x[0]
    for metric,power in ([('euclidean',2.)]if smoke else[('euclidean',2.),('sqeuclidean',2.),('manhattan',2.),('cosine',2.),('minkowski',.5),('minkowski',np.inf),('precomputed',2.)]):
     dense=cdist(x,x)if metric=='precomputed'else x
     for sparse in [False,True]:
      if sparse and metric=='precomputed':
       mask=rng.random_sample((n,n))<.65;np.fill_diagonal(mask,True);ptr=[0];cols=[];data=[]
       for i in range(n):
        ids=np.flatnonzero(mask[i]);ids=ids[rng.permutation(len(ids))];cols.extend(ids);data.extend(dense[i,ids]);ptr.append(len(cols))
       a=csr_matrix((data,cols,ptr),shape=(n,n))
      else:a=csr_matrix(dense)if sparse else dense.copy()
      for family in [0,1]:
       for selector in (sorted(set([1,min(3,n),n]))if family==0 else[0.,.8,np.inf]):
        for include in [0,1,2]:
         yield dict(operation=0,seed=seed,kind=kind,x=a.copy(),storage=int(sparse)+(2 if metric=='precomputed'else 0),family=family,k=int(selector)if family==0 else 1,radius=float(selector)if family else 0.,include=include,metric=metric,power=power)
 if not smoke:
  z=np.array([[0.,0.],[2.,1.],[1.,2.],[4.,4.]])
  for family in [0,1]:
   for fitted in [1,2,3]:
    yield dict(operation=0,seed=0,kind='mutated_parameters',x=z,storage=0,family=family,k=1,radius=3.,include=0,metric='minkowski',power=1.,fit_metric='minkowski',fit_power=2.,fit_kind=fitted,mutation=1)
  large=np.random.RandomState(2026).normal(size=(65,5))
  for metric in ['euclidean','manhattan']:
   for include in [0,1,2]:
    yield dict(operation=0,seed=2026,kind='auto_tree_order',x=large,storage=0,family=1,k=1,radius=np.inf,include=include,metric=metric,power=2.,fit_kind=0,mode='distance')
  raw=csr_matrix(([1.,1.,2.,1.,2.,1.,-1.,2.],[2,0,2,1,1,2,0,0],[0,3,5,8]),shape=(3,3))
  missing=csr_matrix(([2.,1.,1.,1.,1.,1.,2.,1.],[2,1,2,2,0,1,0,1],[0,3,5,8]),shape=(3,3))
  for metric in ['euclidean','l2','manhattan','cityblock','l1','cosine','precomputed']:
   for family in [0,1]:
    for include in [0,1,2]:
     for mode in ['connectivity','distance']:
      yield dict(operation=0,seed=0,kind='noncanonical_csr',x=missing if metric=='precomputed'else raw,storage=3 if metric=='precomputed'else 1,family=family,k=1,radius=2.,include=include,metric=metric,power=2.,fit_kind=0,mode=mode)

def sort_cases(smoke):
 for seed in ([7]if smoke else[7,42]):
  rng=np.random.RandomState(seed)
  for n,p in ([(4,5)]if smoke else[(0,0),(1,4),(2,4),(9,11),(33,17)]):
   for pattern in ['equal','variable','leading_empty','empty']:
    ptr=[0];cols=[];data=[]
    for row in range(n):
     count=0 if pattern=='empty'or(pattern=='leading_empty'and row==0)else(4 if pattern=='equal'else int(rng.randint(7)))
     if p==0:count=0
     cols.extend(rng.randint(p,size=count));data.extend(rng.randint(-2,4,size=count).astype(float));ptr.append(len(cols))
    for copy in [False,True]:
     for warn in [False,True]:yield dict(operation=1,seed=seed,kind=pattern,x=csr_matrix((data,cols,ptr),shape=(n,p)),copy=copy,warn=warn)
  specials=[([0,4],[0,1,2,3],[np.nan,2.,1.,np.nan],(1,4)),([0,3],[0,1,2],[2.,np.nan,1.],(1,3)),([0,0,2],[0,1],[2.,1.],(2,2)),([0,0,0],[],[],(2,2)),([0,6],[0,0,1,1,2,2],[-0.,0.,np.inf,-np.inf,np.nan,np.nan],(1,3))]
  for ptr,cols,data,shape in specials:
   for copy in [False,True]:yield dict(operation=1,seed=seed,kind='nonfinite_or_boundary',x=csr_matrix((data,cols,ptr),shape=shape),copy=copy,warn=True)

def csr_text(a):
 return [str(a.nnz),' '.join(map(str,a.indptr+1)),' '.join(map(str,a.indices+1)),' '.join(map(str,a.data))]

def frame(a,original=None,sorted_before=False):
 return dict(meta=[*a.shape,a.nnz],indptr=(a.indptr+1).tolist(),indices=(a.indices+1).tolist(),data=a.data.tolist(),dense=a.toarray().ravel(order='F').tolist(),input_indptr=[]if original is None else(original.indptr+1).tolist(),input_indices=[]if original is None else(original.indices+1).tolist(),input_data=[]if original is None else original.data.tolist(),initial_sorted=[int(sorted_before)])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--reference-cache',type=Path);ap.add_argument('--smoke',action='store_true');ap.add_argument('--sort-only',action='store_true');ap.add_argument('--record-differences',action='store_true');a=ap.parse_args()
 provenance=dict(schema=2,python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,graph_sha256=sha(graph_source.__file__),base_sha256=sha(base_source.__file__),adapters='one-based CSR; owned Fortran returns; Python warning and identity objects not represented; raw auto/fitted brute Source dispatch')
 cache={}
 if a.reference_cache and a.reference_cache.exists():
  saved=json.loads(a.reference_cache.read_text());assert saved['provenance']==provenance;cache=saved['cases']
 cases=[];calls=hits=0
 fixtures=list(graph_cases(a.smoke))+list(sort_cases(a.smoke))
 for index,c in enumerate(fixtures):
  if a.sort_only and c['operation']!=1:continue
  x=c['x'].copy();n,p=x.shape
  if c['operation']==0:
   fitted=c.get('fit_kind',0 if c['storage']%2 else(index//3+index%3)%4);mutation=c.get('mutation',0);fit_metric=c.get('fit_metric',c['metric']);fit_power=c.get('fit_power',c['power'])
   text=['0',f"{n} {p} {c['storage']} {c['family']} {c['k']} {c['radius']} {c['include']} {fitted} {mutation}",f"connectivity {c['metric']} {c['power']} {fit_metric} {fit_power}"]
   mode=c.get('mode','distance'if(index//3)%2 else'connectivity');text[-1]=text[-1].replace('connectivity',mode)
   text+=csr_text(x)if c['storage']%2 else[' '.join(map(str,row))for row in x]
   label={k:v for k,v in c.items()if k!='x'};label.update(index=index,n=n,p=p,fit_kind=fitted,mutation=mutation,mode=mode)
  else:
   text=['1',f"{n} {p} {int(c['copy'])} {int(c['warn'])}",*csr_text(x)];label={k:v for k,v in c.items()if k!='x'};label.update(index=index,n=n,p=p)
  request='\n'.join(text)+'\n';key=hashlib.sha256(request.encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    try:
     if c['operation']==1:
      sorted_before=_is_sorted_by_data(x);out=sort_graph_by_row_values(x,copy=c['copy'],warn_when_not_sorted=c['warn']);expected=frame(out,x,sorted_before);expected['source_aliases_input']=out is x
     else:
      obj=x
      if fitted:
       classes=[NearestNeighbors,KNeighborsClassifier if c['family']==0 else RadiusNeighborsClassifier,KNeighborsRegressor if c['family']==0 else RadiusNeighborsRegressor]
       obj=classes[fitted-1](metric=fit_metric,p=fit_power,algorithm='brute',n_jobs=1)
       if fitted==1:obj.fit(x)
       elif fitted==2:obj.fit(x,np.arange(1,n+1)%2)
       else:obj.fit(x,np.arange(1,n+1,dtype=float))
       if mutation:obj.set_params(metric=c['metric'],p=c['power'])
      inc=[False,True,'auto'][c['include']]
      if c['family']==0:out=kneighbors_graph(obj,c['k'],mode=mode,metric=c['metric'],p=c['power'],include_self=inc)
      else:out=radius_neighbors_graph(obj,c['radius'],mode=mode,metric=c['metric'],p=c['power'],include_self=inc)
      expected=frame(out)
    except (ValueError,TypeError,IndexError,AttributeError,ArithmeticError)as error:expected=dict(error=type(error).__name__+': '+str(error))
   expected['warnings']=len(caught);cache[key]=expected;calls+=1
  cases.append((request,expected,label))
 if a.reference_cache:a.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 good=[c for c in cases if 'error'not in c[1]];bad=[c for c in cases if 'error'in c[1]];by_request={};differences=[];offset=0;width=len(FIELDS)+1
 while offset<len(good):
  chunk=good[offset:offset+32]
  with tempfile.TemporaryFile(mode='w+',encoding='utf-8')as output:
   proc=subprocess.run([a.executable],input=''.join(c[0]for c in chunk),text=True,stdout=output,stderr=subprocess.PIPE);output.seek(0);proc.stdout=output.read()
  lines=proc.stdout.splitlines();completed=len(lines)//width
  if proc.returncode or len(lines)!=len(chunk)*width:
   artifact=a.report.with_name(a.report.stem+f'_batch_{offset}')
   Path(str(artifact)+'.in').write_text(''.join(c[0]for c in chunk))
   Path(str(artifact)+'.stdout').write_text(proc.stdout);Path(str(artifact)+'.stderr').write_text(proc.stderr)
   Path(str(artifact)+'.json').write_text(json.dumps(dict(offset=offset,requested=len(chunk),returncode=proc.returncode,complete_frames=completed,lines=len(lines),case_headers=sum(row.startswith('CASE ')for row in lines),executable_sha256=sha(a.executable)),indent=2)+'\n')
  if not proc.returncode:assert len(lines)==len(chunk)*width,(offset,len(chunk),len(lines))
  else:assert completed<len(chunk),(offset,'native failure after full output',proc.returncode,proc.stderr)
  for j in range(completed):
   assert lines[j*width]==f'CASE {j+1}';by_request[good[offset+j][0]]=subprocess.CompletedProcess([],0,'\n'.join(lines[j*width:j*width+width])+'\n','')
  offset+=completed
  if proc.returncode:
   if not a.record_differences:raise RuntimeError(str(good[offset][2])+proc.stderr+proc.stdout[-600:])
   by_request[good[offset][0]]=proc;offset+=1
  else:assert completed==len(chunk)
 def native(c):return subprocess.run([a.executable],input=c[0],text=True,capture_output=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
  for c,proc in zip(bad,pool.map(native,bad)):by_request[c[0]]=proc
 groups=success=0;worst=0;field_max={}
 for request,e,label in cases:
  proc=by_request[request];text=proc.stdout+proc.stderr
  if proc.returncode:assert 'fscikit-learn:'in text and not any(s in text for s in['SIGFPE','SIGSEGV','Fortran runtime error']),(label,text)
  if 'error'in e:
   if not proc.returncode:differences.append(dict(case=label,field='Source rejection accepted',source=e['error']))
   continue
  if proc.returncode:differences.append(dict(case=label,field='native rejected Source result',diagnostic=text));continue
  success+=1;rows=iter(proc.stdout.splitlines());next(rows)
  for field in FIELDS:
   actual=np.fromstring(next(rows),sep=' ');want=np.asarray(e[field]);groups+=1
   try:
    if field in EXACT:np.testing.assert_array_equal(actual,want,err_msg=str((label,field)))
    else:np.testing.assert_allclose(actual,want,rtol=3.e-8,atol=3.e-8,equal_nan=True,err_msg=str((label,field)))
   except AssertionError:
    if not a.record_differences:raise
    differences.append(dict(case=label,field=field,actual=actual.tolist(),expected=want.tolist()))
   if actual.shape==want.shape:
    finite=np.isfinite(actual)&np.isfinite(want)
    if finite.any():
     error=float(np.max(abs(actual[finite]-want[finite])/(1+abs(want[finite]))));worst=max(worst,error);field_max[field]=max(field_max.get(field,0),error)
 if differences and not a.record_differences:raise AssertionError(differences)
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),source_successful_calls=len(good),native_successful_calls=success,source_rejections=len(bad),groups=groups,failed_comparison_groups=len(differences),strict_parity=not differences,max_scaled_error=worst,field_max_scaled_error=field_max,source_warnings=sum(c[1]['warnings']for c in cases),source_calls=calls,reference_cache_hits=hits,provenance=provenance,source_sha256=sha(root/'src/sklearn_neighbor_functions.f90'),neighbors_sha256=sha(root/'src/sklearn_neighbors.f90'),validator_sha256=sha(__file__),executable_sha256=sha(a.executable),reference_cache_sha256=sha(a.reference_cache)if a.reference_cache else None,input_sha256=hashlib.sha256(''.join(c[0]for c in cases).encode()).hexdigest(),inventory=[c[2]for c in cases],differences=differences)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in['inventory','differences']},indent=2))
if __name__=='__main__':main()
