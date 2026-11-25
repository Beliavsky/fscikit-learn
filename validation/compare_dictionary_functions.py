"""Functional batch/online dictionary learning and mutable callback traces."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,warnings
import numpy as np,scipy,sklearn
from sklearn.decomposition import dict_learning,dict_learning_online,MiniBatchDictionaryLearning
import sklearn.decomposition._dict_learning as source

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True)
 ap.add_argument('--reference-cache',type=Path);ap.add_argument('--report',type=Path);ap.add_argument('--smoke',action='store_true');ap.add_argument('--allow-roundoff-certificates',action='store_true');args=ap.parse_args()
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 provenance=dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,source_sha256=sha(source.__file__))
 cache={};cases=[];calls=hits=warns=0
 if args.reference_cache and args.reference_cache.exists():
  old=json.loads(args.reference_cache.read_text());assert old['provenance']==provenance;cache=old['cases']
 def add(x,k,mode,method,alpha,limit,init,action,seed,return_code=True,pd=False,pc=False,tol=0.,patience=-1,batch=3,shuffle=False,inner=1000):
  nonlocal calls,hits,warns
  n,p=x.shape;rng=np.random.default_rng(seed+100);d=rng.normal(size=(max(1,k),p));u=rng.normal(size=(n,max(1,k)))
  lines=[f'{mode} {n} {p} {k} {limit} {inner} {seed} {int(pd)} {int(pc)} {int(shuffle)} {batch} {int(return_code)} {init} {action} {method} {tol:.17g} {patience} {alpha:.17g}']
  lines+=[' '.join(map(str,row))for row in x]
  if init&1:lines+=[' '.join(map(str,row))for row in d]
  if init&2:lines+=[' '.join(map(str,row))for row in u]
  payload='\n'.join(lines)+'\n';key=hashlib.sha256(payload.encode()).hexdigest()
  label=dict(mode=mode,n=n,p=p,k=k,method=method,alpha=alpha,limit=limit,init=init,action=action,seed=seed,return_code=return_code,positive_dict=pd,positive_code=pc,tol=tol,patience=patience,batch=batch,shuffle=shuffle,inner=inner)
  if key in cache:expected=cache[key];hits+=1
  else:
   events=[]
   def callback(frame):
    if mode==0:
     i=frame['ii'];code=frame['code'];dictionary=frame['dictionary'];errors=frame['errors']
     if action==1 and i==0:dictionary*=.99;errors[0]*=1.001
     events.append([i,errors[-1],np.sum(code),np.sum(dictionary),np.sum(errors)])
    else:
     i=frame['i'];dictionary=frame['dictionary'];model=frame['self']
     if action==1 and i==0:dictionary*=.99
     events.append([i,frame['batch_cost'],np.sum(dictionary),np.sum(model._A),np.sum(model._B)])
   kw=dict(n_components=k or None,alpha=alpha,max_iter=limit,random_state=seed,positive_dict=pd,positive_code=pc)
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    if mode==0:
     code,dictionary,errors,niter=dict_learning(x,method=method,tol=tol,dict_init=d if init&1 else None,code_init=u if init&2 else None,callback=callback if action>=0 else None,return_n_iter=True,method_max_iter=inner,**kw);steps=0
    else:
     online_kw=dict(dict_init=d.copy() if init&1 else None,batch_size=batch,shuffle=shuffle,tol=tol,max_no_improvement=None if patience==-1 else patience)
     result=dict_learning_online(x,method=method,callback=callback if action>=0 else None,return_code=return_code,method_max_iter=inner,**online_kw,**kw)
     if return_code:code,dictionary=result
     else:dictionary=result;code=np.empty((0,0))
     saved=[row[:]for row in events];events.clear()
     online_kw['dict_init']=d.copy()if init&1 else None
     model=MiniBatchDictionaryLearning(fit_algorithm=method,transform_algorithm='lasso_'+method,transform_alpha=alpha,transform_max_iter=inner,callback=callback if action>=0 else None,**online_kw,**kw).fit(x)
     np.testing.assert_allclose(model.components_,dictionary,rtol=0,atol=0)
     assert np.array_equal(np.asarray(saved),np.asarray(events))
     events=saved;niter=int(model.n_iter_);steps=int(model.n_steps_);errors=[]
   expected=dict(meta=[n,p,len(dictionary),int(niter),steps,int(code.size>0),len(events),len(errors)],code=code.ravel(order='F').tolist(),dictionary=dictionary.ravel(order='F').tolist(),errors=np.asarray(errors).tolist(),events=np.asarray(events).ravel().tolist(),warnings=len(caught));cache[key]=expected;calls+=1
  warns+=expected['warnings'];cases.append((label,payload,expected))
 for seed in ([7]if args.smoke else [7,42]):
  rng=np.random.default_rng(seed)
  for n,p in ([(7,3)]if args.smoke else [(7,3),(4,5)]):
   x=rng.normal(size=(n,p))
   for k in ([2]if args.smoke else [1,4]):
    for method in ['cd','lars']:
     for alpha in ([.1]if args.smoke else [.1,1.]):
      for limit in [0,1,6]:
       for warm in [False,True]:
        for action in [-1,0,1]:
         add(x,k,0,method,alpha,limit,3 if warm else 0,action,seed)
         if not args.smoke:add(x,k,0,method,alpha,limit,3 if warm else 0,action,seed,tol=1e-8)
         for codes in ([True]if args.smoke else [False,True]):
          add(x,k,1,method,alpha,limit,1 if warm else 0,action,seed,codes,shuffle=(limit%2==1))
          if not args.smoke:add(x,k,1,method,alpha,limit,1 if warm else 0,action,seed,codes,shuffle=(limit%2==1),tol=.001,patience=10)
 if not args.smoke:
  for method in ['cd','lars']:
   for pd,pc in [(True,False),(False,True),(True,True)]:
    if method=='lars'and pc:continue
    for k in [0,2,5]:
     for mode in [0,1]:
      add(np.abs(np.random.default_rng(2026).normal(size=(6,3))),k,mode,method,.2,11,0,0,2026,pd=pd,pc=pc)
  for mode in [0,1]:
   for limit in [0,1,11]:
    for kind in ['zero','constant','one_sample']:
     x=np.zeros((5,3))if kind=='zero'else np.ones((5,3))
     if kind=='one_sample':x=np.array([[1.,2.,3.]])
     add(x,2,mode,'cd',.3,limit,0,0,42,tol=.001,patience=2,batch=2)
 if not args.smoke:
  x=np.random.default_rng(123).normal(size=(8,3))
  for mode in [0,1]:
   for method in ['cd','lars']:
    for inner in [0,1]:add(x,2,mode,method,.2,1,3 if mode==0 else 1,0,123,inner=inner)
    add(x,2,mode,method,0.,1,3 if mode==0 else 1,0,123)
  for init in [1,2]:add(x,2,0,'cd',.2,3,init,0,123,tol=1e-8)
 if args.reference_cache:
  args.reference_cache.parent.mkdir(parents=True,exist_ok=True);args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 payload=''.join(c[1]for c in cases);run=subprocess.run([args.executable],input=payload,text=True,capture_output=True)
 if run.returncode:raise RuntimeError(run.stderr+'\n'+run.stdout[-1000:])
 rows=iter(run.stdout.splitlines());groups=0;worst=0;certificates=[];direct=0
 for label,request,expected in cases:
  meta=np.fromstring(next(rows),dtype=np.int64,sep=' ')
  fields={field:np.fromstring(next(rows),sep=' ')for field in ['code','dictionary','errors','events']}
  want_meta=np.asarray(expected['meta']);failures=[];groups+=5
  if not np.array_equal(meta,want_meta):failures.append('meta')
  else:direct+=1
  for field,actual in fields.items():
   want=np.asarray(expected[field])
   if actual.shape!=want.shape or not np.allclose(actual,want,rtol=3e-8,atol=3e-8,equal_nan=True):failures.append(field)
   else:
    direct+=1;finite=np.isfinite(actual)&np.isfinite(want)
    if finite.any():worst=max(worst,float(np.max(abs(actual[finite]-want[finite])/(1+abs(want[finite])))))
  if not failures:continue
  assert args.allow_roundoff_certificates,(label,failures,meta,want_meta)
  n,p,k=map(int,meta[:3]);x=np.array([list(map(float,row.split()))for row in request.splitlines()[1:n+1]])
  u=fields['code'].reshape(n,k,order='F');v=fields['dictionary'].reshape(k,p,order='F')
  su=np.asarray(expected['code']).reshape(n,k,order='F');sv=np.asarray(expected['dictionary']).reshape(k,p,order='F')
  scale=1+np.linalg.norm(x);eps=np.finfo(float).eps
  if label['limit']==0:
   assert label['mode']==0 and failures==['code']
   # The native Jacobi SVD uses zero left columns for null singular values.
   # Prove exact reconstruction equivalence of the exposed low-rank factors.
   np.testing.assert_allclose(u@v,su@sv,atol=64*eps*scale,rtol=64*eps)
   changed=np.max(abs(u-su),axis=0)>3e-8
   assert np.all(np.linalg.norm(v[changed],axis=1)<=64*eps*scale)
   assert np.all(np.linalg.norm(sv[changed],axis=1)<=64*eps*scale)
   kind='zero_iteration_null_code'
  else:
   assert label['mode']==0 and label['tol']==0
   assert set(failures)<=set(['meta','dictionary','errors','events'])
   assert np.array_equal(meta[[0,1,2,4,5]],want_meta[[0,1,2,4,5]])
   a=fields['errors'];b=np.asarray(expected['errors']);common=min(len(a),len(b));assert common>=3
   np.testing.assert_allclose(a[:common],b[:common],rtol=3e-8,atol=3e-8)
   budget=64*eps*max(n,p,k)*(1+max(abs(a[-1]),abs(b[-1])))
   assert abs(a[common-1]-a[common-2])<=budget and abs(b[common-1]-b[common-2])<=budget
   assert np.all(abs(a[common-1:]-a[common-1])<=budget)and np.all(abs(b[common-1:]-b[common-1])<=budget)
   # Independently recompute both objectives and prove identical fitted signals.
   objective=lambda c,d:.5*np.sum((x-c@d)**2)+label['alpha']*np.sum(abs(c))
   assert abs(objective(u,v)-a[-1])<=budget and abs(objective(su,sv)-b[-1])<=budget
   np.testing.assert_allclose(u@v,su@sv,rtol=3e-8,atol=3e-8)
   changed=np.max(abs(v-sv),axis=1)>3e-8*(1+np.max(abs(sv),axis=1))
   assert np.all(np.max(abs(u[:,changed]),axis=0)<=64*eps*scale)
   assert np.all(np.max(abs(su[:,changed]),axis=0)<=64*eps*scale)
   for dictionary in [v,sv]:assert np.all(np.linalg.norm(dictionary,axis=1)<=1+64*eps*scale)
   ev=fields['events'].reshape(-1,5);se=np.asarray(expected['events']).reshape(-1,5);count=min(len(ev),len(se))
   np.testing.assert_allclose(ev[:count],se[:count],rtol=3e-8,atol=3e-8)
   for events in [ev,se]:
    assert np.all(events[:,0]%5==0)and np.all(events[:,0]<max(meta[3],want_meta[3]))
    for event in events[count:]:assert event[0]>=common-1 and abs(event[1]-a[-1])<=budget
   kind='roundoff_stopping_and_unused_atoms'
  certificates.append(dict(configuration=label,kind=kind,fields=failures,native_meta=meta.tolist(),source_meta=want_meta.tolist()))
 assert next(rows,None)is None
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),groups=groups,direct_groups=direct,certificate_groups=groups-direct,certificates=certificates,max_scaled_error=worst,source_warnings=warns,source_calls=calls,reference_cache_hits=hits,reference_provenance=provenance,
  reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,source_sha256=sha(root/'src/sklearn_dictionary.f90'),validator_sha256=sha(__file__),executable_sha256=sha(args.executable),input_sha256=hashlib.sha256(payload.encode()).hexdigest(),configuration_inventory=[c[0]for c in cases])
 print(json.dumps({k:v for k,v in report.items()if k not in ['configuration_inventory','certificates']},indent=2))
 if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
