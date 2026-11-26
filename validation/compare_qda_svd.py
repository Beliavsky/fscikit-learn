"""Direct QDA SVD factors, singular fits, priors and posterior behavior vs sklearn."""
from pathlib import Path
import argparse,hashlib,json,os,sys,subprocess,tempfile,warnings,concurrent.futures
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np,scipy,sklearn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sklearn.discriminant_analysis as source
FIELDS=['meta','classes','means','priors','covariance','scalings','projectors','directions','ranks','logdet','decision','log_proba','proba','predict','score']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fixtures(smoke):
 for seed in ([7]if smoke else[0,7,42]):
  rng=np.random.RandomState(seed)
  for counts,p in ([( [6,8],2)]if smoke else[([6,8],2),([2,3],7),([5,7,6],3),([33,17],1)]):
   y=np.repeat(np.arange(len(counts))*5-3,counts);x=rng.normal(size=(len(y),p))+.1*y[:,None];q=rng.normal(size=(5,p));truth=np.resize(np.unique(y),len(q));k=len(counts)
   for kind in (['regular']if smoke else['regular','duplicate','constant']):
    data=x.copy()
    if kind=='duplicate'and p>1:data[:,-1]=data[:,0]
    if kind=='constant':data[:]=0
    for reg in ([.2]if smoke else[0.,.2,1.]):
     for tol in ([1.e-4]if smoke else[0.,1.e-4,1.e3]):
      for mode in (['empirical','broadcast']if smoke else['empirical','uniform','scaled','broadcast','zero','negative','nan','inf','mismatch','empty']):
       priors=None
       if mode=='uniform':priors=np.ones(k)/k
       if mode=='scaled':priors=np.arange(1,k+1)*3.7
       if mode=='broadcast':priors=np.array([.25])
       if mode=='zero':priors=np.zeros(k)
       if mode=='negative':priors=np.ones(k);priors[0]=-1
       if mode=='nan':priors=np.ones(k);priors[-1]=np.nan
       if mode=='inf':priors=np.ones(k);priors[0]=np.inf
       if mode=='mismatch':priors=np.ones(k+1)
       if mode=='empty':priors=np.empty(0)
       yield dict(seed=seed,x=data,y=y,q=q,truth=truth,counts=counts,kind=kind,reg=reg,tol=tol,mode=mode,priors=priors)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--report',required=True,type=Path);ap.add_argument('--reference-cache',type=Path)
 ap.add_argument('--smoke',action='store_true');ap.add_argument('--full-rank',action='store_true');ap.add_argument('--record-differences',action='store_true');a=ap.parse_args()
 provenance=dict(schema=1,python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,source_sha256=sha(source.__file__));cache={}
 if a.reference_cache and a.reference_cache.exists():
  old=json.loads(a.reference_cache.read_text());assert old['provenance']==provenance;cache=old['cases']
 cases=[];hits=calls=0
 for index,c in enumerate(fixtures(a.smoke)):
  x,y,q,truth,priors=c['x'],c['y'],c['q'],c['truth'],c['priors'];n,p=x.shape;k=len(c['counts'])
  if a.full_rank and(c['kind']!='regular'or min(c['counts'])<=p):continue
  header=f"{n} {p} {len(q)} {-1 if priors is None else len(priors)} {c['reg']} {c['tol']}"
  train=[' '.join(map(str,row))+' '+str(label)for row,label in zip(x,y)];query=[' '.join(map(str,row))+' '+str(label)for row,label in zip(q,truth)]
  tail=[]if priors is None or len(priors)==0 else[' '.join(map(str,priors))];request='\n'.join([header,*train,*query,*tail])+'\n'
  meta_request='\n'.join([' '.join(header.split()[:2]+['0']+header.split()[3:]),*train,*tail])+'\n';key=hashlib.sha256(request.encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   model=QuadraticDiscriminantAnalysis(priors=priors,reg_param=c['reg'],tol=c['tol'],store_covariance=True)
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    try:
     model.fit(x,y)
     expected=dict(meta=[model.n_features_in_,len(model.classes_)],classes=model.classes_.tolist(),means=model.means_.ravel(order='F').tolist(),priors=model.priors_.tolist(),covariance=np.stack(model.covariance_,axis=2).ravel(order='F').tolist(),scalings=np.concatenate(model.scalings_).tolist(),projectors=np.stack([r@r.T for r in model.rotations_],axis=2).ravel(order='F').tolist(),directions=[len(s)for s in model.scalings_],ranks=[int(sum(s>c['tol']))for s in model.scalings_],logdet=[np.log(s).sum()for s in model.scalings_])
    except (ValueError,TypeError,np.linalg.LinAlgError)as error:expected=dict(error=str(error))
    if 'error'not in expected:
     try:expected.update(decision=model._decision_function(q).ravel(order='F').tolist(),log_proba=model.predict_log_proba(q).ravel(order='F').tolist(),proba=model.predict_proba(q).ravel(order='F').tolist(),predict=model.predict(q).tolist(),score=[model.score(q,truth)])
     except (ValueError,TypeError,ArithmeticError)as error:expected.update(query_error=str(error),decision=[],log_proba=[],proba=[],predict=[],score=[0.])
   expected['warnings']=len(caught);cache[key]=expected;calls+=1
  label={key:c[key]for key in ['seed','counts','kind','reg','tol','mode']};label.update(index=index,n=n,p=p)
  cases.append((request,expected,label,meta_request))
 if a.reference_cache:a.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 good=[c for c in cases if 'error'not in c[1]];bad=[c for c in cases if 'error'in c[1]];by_request={};differences=[];width=len(FIELDS)+1;offset=0
 while offset<len(good):
  chunk=good[offset:offset+64]
  with tempfile.TemporaryFile(mode='w+',encoding='utf-8')as out:
   proc=subprocess.run([a.executable],input=''.join(c[3]if 'query_error'in c[1]else c[0]for c in chunk),text=True,stdout=out,stderr=subprocess.PIPE);out.seek(0);proc.stdout=out.read()
  rows=proc.stdout.splitlines();completed=sum(row.startswith('CASE ')for row in rows)
  if not proc.returncode:assert len(rows)==len(chunk)*width,(offset,len(rows),len(chunk))
  for j in range(completed):
   assert rows[j*width]==f'CASE {j+1}'
   by_request[good[offset+j][0]]=subprocess.CompletedProcess([],0,'\n'.join(rows[j*width:j*width+width])+'\n','')
  offset+=completed
  if proc.returncode:
   if not a.record_differences:raise RuntimeError(str(good[offset][2])+proc.stderr+proc.stdout[-500:])
   by_request[good[offset][0]]=proc;offset+=1
  else:assert completed==len(chunk)
 def native(c):return subprocess.run([a.executable],input=c[0],text=True,capture_output=True)
 query_bad=[c for c in good if 'query_error'in c[1]];query_probes={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
  for c,proc in zip(bad,pool.map(native,bad)):by_request[c[0]]=proc
  for c,proc in zip(query_bad,pool.map(native,query_bad)):query_probes[c[0]]=proc
 groups=0;worst=0;fits=0;field_max={}
 for request,e,label,_ in cases:
  proc=by_request[request];text=proc.stdout+proc.stderr
  if proc.returncode:assert 'fscikit-learn:'in text and not any(s in text for s in ['SIGFPE','SIGSEGV','Fortran runtime error']),(label,text)
  if 'error'in e:
   if not proc.returncode:differences.append(dict(case=label,field='Source fit rejection accepted',source=e['error']))
   continue
  if proc.returncode:differences.append(dict(case=label,field='native rejected Source fit',diagnostic=text));continue
  fits+=1;rows=iter(proc.stdout.splitlines());assert len(proc.stdout.splitlines())==width,(label,proc.stdout);next(rows)
  for field in FIELDS:
   actual=np.fromstring(next(rows),sep=' ');want=np.asarray(e[field])
   if 'query_error'in e and field in ['decision','log_proba','proba','predict','score']:continue
   groups+=1
   try:
    if field in ['meta','classes','directions','ranks','predict']:np.testing.assert_array_equal(actual,want,err_msg=str((label,field)))
    else:np.testing.assert_allclose(actual,want,atol=3.e-8,rtol=3.e-8,equal_nan=True,err_msg=str((label,field)))
   except AssertionError:
    if not a.record_differences:raise
    differences.append(dict(case=label,field=field,actual=actual.tolist(),expected=want.tolist()))
   if actual.shape==want.shape:
    finite=np.isfinite(actual)&np.isfinite(want)
    if finite.any():
     err=float(np.max(abs(actual[finite]-want[finite])/(1+abs(want[finite]))));worst=max(worst,err);field_max[field]=max(field_max.get(field,0),err)
  assert next(rows,None)is None
  if 'query_error'in e:
   proc=query_probes[request];text=proc.stdout+proc.stderr
   if proc.returncode:assert 'fscikit-learn:'in text and not any(s in text for s in ['SIGFPE','SIGSEGV','Fortran runtime error']),(label,text)
   else:differences.append(dict(case=label,field='Source query rejection accepted',source=e['query_error']))
 if differences and not a.record_differences:raise AssertionError(differences)
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),source_successful_fits=len(good),successful_fits=fits,source_fit_rejections=len(bad),source_query_rejections=len(query_bad),groups=groups,failed_comparison_groups=len(differences),strict_parity=not differences,max_scaled_error=worst,field_max_scaled_error=field_max,source_warnings=sum(c[1]['warnings']for c in cases),source_calls=calls,reference_cache_hits=hits,provenance=provenance,source_sha256=sha(root/'src/sklearn_discriminant_analysis.f90'),executable_sha256=sha(a.executable),validator_sha256=sha(__file__),reference_cache_sha256=sha(a.reference_cache)if a.reference_cache else None,input_sha256=hashlib.sha256(''.join(c[0]for c in cases).encode()).hexdigest(),inventory=[c[2]for c in cases],differences=differences)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['inventory','differences']},indent=2))
if __name__=='__main__':main()
