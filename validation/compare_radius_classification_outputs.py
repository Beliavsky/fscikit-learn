"""Joint radius classification, outlier policies and IEEE boundaries vs sklearn."""
from pathlib import Path
import argparse,hashlib,json,os,sys,subprocess,tempfile,warnings,concurrent.futures
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np,scipy,sklearn
from scipy.spatial.distance import cdist
from sklearn.neighbors import RadiusNeighborsClassifier
import sklearn.neighbors._classification as source,sklearn.neighbors._base as base
import sklearn.metrics._classification as metrics
FIELDS=['meta','class_sizes','classes','outlier_labels','targets','prediction','vector_prediction','probability','vector_probability','score']
EXACT=set(FIELDS)-{'probability','vector_probability','score'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fixtures(smoke):
 for seed in ([7]if smoke else[7,42]):
  rng=np.random.RandomState(seed)
  for n,p in ([(9,3)]if smoke else[(1,1),(9,3),(65,5)]):
   original=rng.normal(size=(n,p));binary=rng.randint(2,size=(n,3));ragged=np.c_[rng.randint(4,size=n)*3-4,rng.randint(3,size=n)*5+2,np.full(n,42)]
   for kind in (['regular']if smoke else['regular','duplicate']):
    data=original.copy()
    if kind=='duplicate':data[:min(3,n)]=data[0]
    query=np.vstack([data[:min(2,n)],rng.normal(size=(3,p))+.1])
    for metric,power in ([('euclidean',2.)]if smoke else[('euclidean',2.),('manhattan',2.),('minkowski',.5),('cosine',2.),('precomputed',2.)]):
     x=data;q=query
     if metric=='precomputed':x=cdist(data,data);q=cdist(query,data)
     for weights in ['uniform','distance']:
      for label_kind,labels in [('binary',binary),('ragged',ragged)]:
       for t in [1,3]:
        for self_query in [False,True]:
         for radius in [0.,.8,np.inf]:
          yield dict(seed=seed,x=x,q=x.copy()if self_query else q,y=labels[:,:t],kind=kind,metric=metric,power=power,weights=weights,label_kind=label_kind,t=t,self_query=self_query,radius=radius)
 if not smoke:
  for t in [1,2]:
   yield dict(seed=0,x=np.array([[0.,1.],[1.,0.]]),q=np.array([[1.e-320,1.],[1.,1.e-320],[1.e-320,1.e-320],[0.,1.e-320]]),y=np.array([[-3,2],[7,9]])[:,:t],kind='extreme',metric='precomputed',power=2.,weights='distance',label_kind='reciprocal_overflow',t=t,self_query=False,radius=1.)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--reference-cache',type=Path)
 ap.add_argument('--smoke',action='store_true');ap.add_argument('--regular',action='store_true');ap.add_argument('--record-differences',action='store_true');a=ap.parse_args()
 provenance=dict(schema=3,outlier_dtype_adapter='Source labels are int64 with Python int/list outlier parameters; native default integers preserve values but do not represent Python dtype promotion rules',python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,source_sha256=sha(source.__file__),base_sha256=sha(base.__file__),metrics_sha256=sha(metrics.__file__));cache={}
 if a.reference_cache and a.reference_cache.exists():
  saved=json.loads(a.reference_cache.read_text());assert saved['provenance']==provenance;cache=saved['cases']
 cases=[];calls=hits=0
 for index,c in enumerate(fixtures(a.smoke)):
  if a.regular and c['kind']!='regular':continue
  x,y,q=c['x'],np.asarray(c['y'],dtype=np.int64),c['q'];n,p=x.shape;t=c['t'];m=len(q);matrix=t>1 or index%3==0;score_rank=int(t>1 or index%2==0);score_mode=1+index%2
  truth=(np.arange(m)[:,None]+np.arange(t))%2;sw=np.linspace(.2,1.5,m)
  if m>2:sw[0]=0
  outmode=(index//3+index%3)%4;outlabels=np.min(y,axis=0).copy()
  if t>1:outlabels[-1]=-99
  header=[str(v)for v in[n,p,t,m,int(matrix),int(c['self_query']),score_mode,score_rank,outmode,c['radius'],c['power'],c['metric'],c['weights']]]
  rows=[' '.join(map(str,row))+' '+' '.join(map(str,target))for row,target in zip(x,y)]+[' '.join(map(str,row))+' '+' '.join(map(str,target))for row,target in zip(q,truth)]+[' '.join(map(str,sw)),' '.join(map(str,outlabels))]
  request='\n'.join([' '.join(header),*rows])+'\n';key=hashlib.sha256(request.encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   model=RadiusNeighborsClassifier(radius=c['radius'],outlier_label=[None,-99,outlabels.tolist(),'most_frequent'][outmode],weights=c['weights'],metric=c['metric'],p=c['power'],algorithm='brute',n_jobs=1)
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always');model.fit(x,y if matrix else y[:,0]);classes=model.classes_ if model.outputs_2d_ else[model.classes_]
    expected=dict(meta=[model.n_features_in_,len(model._fit_X),t,int(model.outputs_2d_)],class_sizes=[len(v)for v in classes],classes=np.concatenate(classes).tolist(),outlier_labels=[]if model.outlier_label_ is None else np.asarray(model.outlier_label_).tolist(),targets=y.ravel(order='F').tolist())
    try:
     pred=model.predict(None if c['self_query']else q);prob=model.predict_proba(None if c['self_query']else q)
     expected.update(prediction=pred.ravel(order='F').tolist(),vector_prediction=pred.tolist()if t==1 else[],probability=np.concatenate([v.ravel(order='F')for v in(prob if model.outputs_2d_ else[prob])]).tolist(),vector_probability=prob.ravel(order='F').tolist()if t==1 else[])
    except (ValueError,TypeError,ArithmeticError)as error:expected.update(predict_error=str(error),prediction=[],vector_prediction=[],probability=[],vector_probability=[],score=[0.])
    if 'predict_error'not in expected:
     try:expected['score']=[model.score(None if c['self_query']else q,truth if score_rank else truth[:,0],sample_weight=sw if score_mode==2 else None)]
     except (ValueError,TypeError,ArithmeticError)as error:expected.update(score_error=str(error),score=[0.])
   expected['warnings']=len(caught);cache[key]=expected;calls+=1
  label={key:c[key]for key in['seed','kind','metric','power','weights','label_kind','t','self_query','radius']};label.update(index=index,n=n,p=p,m=m,matrix=matrix,score_mode=score_mode,score_rank=score_rank,outmode=outmode)
  changed=header.copy()
  if 'predict_error'in expected:changed[5]='2';changed[6]='0'
  elif 'score_error'in expected:changed[6]='0'
  cases.append((request,expected,label,'\n'.join([' '.join(changed),*rows])+'\n'))
 if a.reference_cache:a.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 by_request={};differences=[];offset=0;width=len(FIELDS)+1
 while offset<len(cases):
  chunk=cases[offset:offset+64]
  with tempfile.TemporaryFile(mode='w+',encoding='utf-8')as output:
   proc=subprocess.run([a.executable],input=''.join(c[3]for c in chunk),text=True,stdout=output,stderr=subprocess.PIPE);output.seek(0);proc.stdout=output.read()
  lines=proc.stdout.splitlines();completed=len(lines)//width
  if proc.returncode or len(lines)!=len(chunk)*width:
   artifact=a.report.with_name(a.report.stem+f'_batch_{offset}')
   Path(str(artifact)+'.in').write_text(''.join(c[3]for c in chunk))
   Path(str(artifact)+'.stdout').write_text(proc.stdout);Path(str(artifact)+'.stderr').write_text(proc.stderr)
   Path(str(artifact)+'.json').write_text(json.dumps(dict(offset=offset,requested=len(chunk),returncode=proc.returncode,complete_frames=completed,lines=len(lines),case_headers=sum(row.startswith('CASE ')for row in lines),executable_sha256=sha(a.executable)),indent=2)+'\n')
  if not proc.returncode:assert len(lines)==len(chunk)*width,(offset,len(chunk),len(lines))
  else:assert completed<len(chunk),(offset,'native failure after full output',proc.returncode,proc.stderr)
  for j in range(completed):
   assert lines[j*width]==f'CASE {j+1}'
   by_request[cases[offset+j][0]]=subprocess.CompletedProcess([],0,'\n'.join(lines[j*width:j*width+width])+'\n','')
  offset+=completed
  if proc.returncode:
   if not a.record_differences:raise RuntimeError(str(cases[offset][2])+proc.stderr+proc.stdout[-600:])
   by_request[cases[offset][0]]=proc;offset+=1
  else:assert completed==len(chunk)
 rejects=[c for c in cases if 'predict_error'in c[1]or'score_error'in c[1]]
 def native(c):return subprocess.run([a.executable],input=c[0],text=True,capture_output=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:probes={c[0]:proc for c,proc in zip(rejects,pool.map(native,rejects))}
 groups=fits=0;worst=0;field_max={}
 for request,e,label,_ in cases:
  proc=by_request[request];text=proc.stdout+proc.stderr
  if proc.returncode:
   assert 'fscikit-learn:'in text and not any(s in text for s in['SIGFPE','SIGSEGV','Fortran runtime error']),(label,text)
   differences.append(dict(case=label,field='native rejected Source evaluation',diagnostic=text));continue
  fits+=1;lines=proc.stdout.splitlines();assert len(lines)==width,(label,lines);rows=iter(lines);next(rows)
  for field in FIELDS:
   actual=np.fromstring(next(rows),sep=' ');want=np.asarray(e[field])
   if 'predict_error'in e and field not in['meta','class_sizes','classes','outlier_labels','targets']:continue
   if 'score_error'in e and field=='score':continue
   groups+=1
   try:
    if field in EXACT:np.testing.assert_array_equal(actual,want,err_msg=str((label,field)))
    else:np.testing.assert_allclose(actual,want,atol=3.e-8,rtol=3.e-8,equal_nan=True,err_msg=str((label,field)))
   except AssertionError:
    if not a.record_differences:raise
    differences.append(dict(case=label,field=field,actual=actual.tolist(),expected=want.tolist()))
   if actual.shape==want.shape:
    finite=np.isfinite(actual)&np.isfinite(want)
    if finite.any():
     err=float(np.max(abs(actual[finite]-want[finite])/(1+abs(want[finite]))));worst=max(worst,err);field_max[field]=max(field_max.get(field,0),err)
  if request in probes:
   probe=probes[request];text=probe.stdout+probe.stderr
   if probe.returncode:assert 'fscikit-learn:'in text and not any(s in text for s in['SIGFPE','SIGSEGV','Fortran runtime error']),(label,text)
   else:differences.append(dict(case=label,field='Source evaluation rejection accepted',source=e.get('predict_error',e.get('score_error'))))
 if differences and not a.record_differences:raise AssertionError(differences)
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),successful_fits=fits,source_prediction_rejections=sum('predict_error'in c[1]for c in cases),source_score_rejections=sum('score_error'in c[1]for c in cases),groups=groups,failed_comparison_groups=len(differences),strict_parity=not differences,max_scaled_error=worst,field_max_scaled_error=field_max,source_warnings=sum(c[1]['warnings']for c in cases),source_calls=calls,reference_cache_hits=hits,provenance=provenance,source_sha256=sha(root/'src/sklearn_neighbors.f90'),validator_sha256=sha(__file__),executable_sha256=sha(a.executable),reference_cache_sha256=sha(a.reference_cache)if a.reference_cache else None,input_sha256=hashlib.sha256(''.join(c[0]for c in cases).encode()).hexdigest(),inventory=[c[2]for c in cases],differences=differences)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({key:value for key,value in report.items()if key not in['inventory','differences']},indent=2))
if __name__=='__main__':main()
