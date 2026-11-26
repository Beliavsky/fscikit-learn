"""Joint K/radius-neighbor regression and self-query behavior against sklearn."""
from pathlib import Path
import argparse,hashlib,json,os,sys,subprocess,tempfile,warnings,concurrent.futures
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np,scipy,sklearn
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
import sklearn.neighbors._regression as source,sklearn.neighbors._base as base
FIELDS=['meta','targets','prediction','matrix_prediction','score'];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fixtures(smoke):
 for seed in ([7]if smoke else[0,7,42]):
  rng=np.random.RandomState(seed)
  for n,p in ([(17,3)]if smoke else[(1,1),(7,2),(17,3),(129,5)]):
   original=rng.normal(size=(n,p));target=rng.normal(size=(n,3))+2
   for kind in (['regular']if smoke else['regular','duplicate']):
    data=original.copy()
    if kind=='duplicate':data[:min(3,n)]=data[0]
    query=np.vstack([data[:min(2,n)],rng.normal(size=(3,p))+.1])
    for metric,power in ([('euclidean',2.)]if smoke else[('euclidean',2.),('manhattan',2.),('minkowski',.5),('cosine',2.),('precomputed',2.)]):
     x=data;q=query
     if metric=='precomputed':x=cdist(data,data);q=cdist(query,data)
     for weights in ['uniform','distance']:
      for t in [1,3]:
       for self_query in [False,True]:
        for family in [1,2]:
         for selector in (sorted(set([1,min(3,n),n]))if family==1 else[0.,.8,np.inf]):
          yield dict(seed=seed,x=x,q=x.copy()if self_query else q,y=target[:,:t],kind=kind,metric=metric,power=power,weights=weights,t=t,self_query=self_query,family=family,selector=selector)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--reference-cache',type=Path)
 ap.add_argument('--smoke',action='store_true');ap.add_argument('--regular',action='store_true');ap.add_argument('--record-differences',action='store_true');a=ap.parse_args()
 provenance=dict(schema=1,python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,source_sha256=sha(source.__file__),base_sha256=sha(base.__file__));cache={}
 if a.reference_cache and a.reference_cache.exists():
  saved=json.loads(a.reference_cache.read_text());assert saved['provenance']==provenance;cache=saved['cases']
 cases=[];calls=hits=0
 for index,c in enumerate(fixtures(a.smoke)):
  if a.regular and c['kind']!='regular':continue
  x,y,q=c['x'],c['y'],c['q'];n,p=x.shape;t=c['t'];m=len(q);matrix=t>1 or index%3==0;score_rank=int(t>1 or index%2==0);score_mode=1+index%2
  truth=.2*np.arange(1,m+1)[:,None]-.1*np.arange(1,t+1);sw=np.linspace(.2,1.5,m)
  if m>2:sw[0]=0
  k=int(c['selector'])if c['family']==1 else 1;radius=0. if c['family']==1 else c['selector']
  header=[str(v)for v in [n,p,t,m,c['family'],int(matrix),int(c['self_query']),score_mode,score_rank,k,radius,c['power'],c['metric'],c['weights']]]
  rows=[' '.join(map(str,row))+' '+' '.join(map(str,target))for row,target in zip(x,y)]+[' '.join(map(str,row))+' '+' '.join(map(str,target))for row,target in zip(q,truth)]+[' '.join(map(str,sw))]
  request='\n'.join([' '.join(header),*rows])+'\n';key=hashlib.sha256(request.encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   cls=KNeighborsRegressor if c['family']==1 else RadiusNeighborsRegressor
   options=dict(n_neighbors=k)if c['family']==1 else dict(radius=radius)
   model=cls(algorithm='brute',weights=c['weights'],metric=c['metric'],p=c['power'],n_jobs=1,**options)
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    try:
     model.fit(x,y if matrix else y[:,0]);expected=dict(meta=[model.n_features_in_,model.n_samples_fit_,t,int(matrix)],targets=y.ravel(order='F').tolist())
    except (ValueError,TypeError,ArithmeticError)as error:expected=dict(error=str(error))
    if 'error'not in expected:
     try:
      prediction=model.predict(None if c['self_query']else q);expected.update(prediction=prediction.ravel(order='F').tolist(),matrix_prediction=prediction.ravel(order='F').tolist())
     except (ValueError,TypeError,ArithmeticError)as error:expected.update(predict_error=str(error),prediction=[],matrix_prediction=[],score=[0.])
     if 'predict_error'not in expected:
      try:expected['score']=[model.score(None if c['self_query']else q,truth if score_rank else truth[:,0],sample_weight=sw if score_mode==2 else None)]
      except (ValueError,TypeError,ArithmeticError)as error:expected.update(score_error=str(error),score=[0.])
   expected['warnings']=len(caught);cache[key]=expected;calls+=1
  label={key:c[key]for key in ['seed','kind','metric','power','weights','t','self_query','family','selector']};label.update(index=index,n=n,p=p,m=m,matrix=matrix,score_mode=score_mode,score_rank=score_rank)
  changed=header.copy()
  if 'predict_error'in expected:changed[6]='2';changed[7]='0'
  elif 'score_error'in expected:changed[7]='0'
  eval_request='\n'.join([' '.join(changed),*rows])+'\n';cases.append((request,expected,label,eval_request))
 if a.reference_cache:a.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 good=[c for c in cases if 'error'not in c[1]];bad=[c for c in cases if 'error'in c[1]];by_request={};differences=[];offset=0;width=len(FIELDS)+1
 while offset<len(good):
  chunk=good[offset:offset+64]
  with tempfile.TemporaryFile(mode='w+',encoding='utf-8')as output:
   proc=subprocess.run([a.executable],input=''.join(c[3]for c in chunk),text=True,stdout=output,stderr=subprocess.PIPE);output.seek(0);proc.stdout=output.read()
  lines=proc.stdout.splitlines();completed=sum(row.startswith('CASE ')for row in lines)
  if not proc.returncode:assert len(lines)==len(chunk)*width,(offset,len(chunk),len(lines))
  for j in range(completed):
   assert lines[j*width]==f'CASE {j+1}'
   by_request[good[offset+j][0]]=subprocess.CompletedProcess([],0,'\n'.join(lines[j*width:j*width+width])+'\n','')
  offset+=completed
  if proc.returncode:
   if not a.record_differences:raise RuntimeError(str(good[offset][2])+proc.stderr+proc.stdout[-600:])
   by_request[good[offset][0]]=proc;offset+=1
  else:assert completed==len(chunk)
 def native(c):return subprocess.run([a.executable],input=c[0],text=True,capture_output=True)
 rejects=[c for c in good if 'predict_error'in c[1]or'score_error'in c[1]];probes={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
  for c,proc in zip(bad,pool.map(native,bad)):by_request[c[0]]=proc
  for c,proc in zip(rejects,pool.map(native,rejects)):probes[c[0]]=proc
 groups=fits=0;worst=0;field_max={}
 for request,e,label,_ in cases:
  proc=by_request[request];text=proc.stdout+proc.stderr
  if proc.returncode:assert 'fscikit-learn:'in text and not any(s in text for s in ['SIGFPE','SIGSEGV','Fortran runtime error']),(label,text)
  if 'error'in e:
   if not proc.returncode:differences.append(dict(case=label,field='Source fit rejection accepted',source=e['error']))
   continue
  if proc.returncode:differences.append(dict(case=label,field='native rejected Source evaluation',diagnostic=text));continue
  fits+=1;lines=proc.stdout.splitlines();assert len(lines)==width,(label,lines);rows=iter(lines);next(rows)
  for field in FIELDS:
   actual=np.fromstring(next(rows),sep=' ');want=np.asarray(e[field])
   if 'predict_error'in e and field in ['prediction','matrix_prediction','score']:continue
   if 'score_error'in e and field=='score':continue
   groups+=1
   try:
    if field=='meta':np.testing.assert_array_equal(actual,want,err_msg=str(label))
    else:np.testing.assert_allclose(actual,want,atol=3.e-8,rtol=3.e-8,equal_nan=True,err_msg=str((label,field)))
   except AssertionError:
    if not a.record_differences:raise
    differences.append(dict(case=label,field=field,actual=actual.tolist(),expected=want.tolist()))
   if actual.shape==want.shape:
    finite=np.isfinite(actual)&np.isfinite(want)
    if finite.any():
     err=float(np.max(abs(actual[finite]-want[finite])/(1+abs(want[finite]))));worst=max(worst,err);field_max[field]=max(field_max.get(field,0),err)
  assert next(rows,None)is None
  if 'predict_error'in e or'score_error'in e:
   probe=probes[request];text=probe.stdout+probe.stderr
   if probe.returncode:assert 'fscikit-learn:'in text and not any(s in text for s in ['SIGFPE','SIGSEGV','Fortran runtime error']),(label,text)
   else:differences.append(dict(case=label,field='Source evaluation rejection accepted',source=e.get('predict_error',e.get('score_error'))))
 if differences and not a.record_differences:raise AssertionError(differences)
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),successful_fits=fits,source_fit_rejections=len(bad),source_prediction_rejections=sum('predict_error'in c[1]for c in cases),source_score_rejections=sum('score_error'in c[1]for c in cases),groups=groups,failed_comparison_groups=len(differences),strict_parity=not differences,max_scaled_error=worst,field_max_scaled_error=field_max,source_warnings=sum(c[1]['warnings']for c in cases),source_calls=calls,reference_cache_hits=hits,provenance=provenance,source_sha256=sha(root/'src/sklearn_neighbors.f90'),validator_sha256=sha(__file__),executable_sha256=sha(a.executable),reference_cache_sha256=sha(a.reference_cache)if a.reference_cache else None,input_sha256=hashlib.sha256(''.join(c[0]for c in cases).encode()).hexdigest(),inventory=[c[2]for c in cases],differences=differences)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({key:value for key,value in report.items()if key not in ['inventory','differences']},indent=2))
if __name__=='__main__':main()
