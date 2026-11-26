"""Actual sklearn graphical_lasso functional covariance, precision and full costs."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings,sys,concurrent.futures
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np,sklearn,scipy
from sklearn.covariance import graphical_lasso
import sklearn.covariance._graph_lasso as source

def fixtures():
 for seed in [0,7,42]:
  rng=np.random.default_rng(seed)
  for p in [2,3,5,9]:
   for kind in ['spd','rank_deficient','diagonal','asymmetric','indefinite']:
    x=rng.normal(size=(p+5 if kind!='rank_deficient'else 2,p));c=x.T@x/len(x)
    if kind=='diagonal':c=np.diag(np.diag(c))
    if kind=='asymmetric':c[0,1]+=.2
    if kind=='indefinite':c[0,0]=-.5
    for mode in ['cd','lars']:
     for alpha in [0.,.01,.3,10.]:
      for limit in [0,1,3,100]:
       yield dict(cov=c,mode=mode,alpha=alpha,limit=limit,tol=1.e-4,enet_tol=1.e-6,eps=np.finfo(float).eps,seed=seed,kind=kind)
 c=np.array([[2.,.8],[.8,1.]])
 for mode in ['cd','lars']:
  for field in ['alpha','tol','enet_tol','eps']:
   for value in [0.,np.inf]:
    if field in ['tol','enet_tol']and value==0:continue
    for limit in [0,1,3]:
     controls=dict(cov=c,mode=mode,alpha=.3,limit=limit,tol=1.e-4,enet_tol=1.e-4,eps=np.finfo(float).eps,kind=field)
     controls[field]=value;yield controls

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--report',required=True,type=Path)
 ap.add_argument('--reference-cache',type=Path);ap.add_argument('--record-differences',action='store_true')
 ap.add_argument('--regularized',action='store_true',help='Separate strict grid with finite positive penalties and finite eps; the default retains every case.')
 args=ap.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 provenance=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,source_sha256=sha(source.__file__))
 cache={}
 if args.reference_cache and args.reference_cache.exists():
  saved=json.loads(args.reference_cache.read_text());assert saved['provenance']==provenance;cache=saved['cases']
 cases=[];calls=hits=0
 for index,c in enumerate(fixtures()):
  if args.regularized and not (np.isfinite(c['alpha'])and c['alpha']>0 and np.isfinite(c['eps'])):continue
  a=c['cov'];lines=[f'{len(a)} {c["alpha"]} {c["limit"]} {c["tol"]} {c["enet_tol"]} {c["eps"]} {c["mode"]}']
  lines+=[' '.join(map(str,row))for row in a];request='\n'.join(lines)+'\n';key=hashlib.sha256(request.encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    try:
     cov,precision,costs,it=graphical_lasso(a,c['alpha'],mode=c['mode'],tol=c['tol'],enet_tol=c['enet_tol'],max_iter=c['limit'],eps=c['eps'],return_costs=True,return_n_iter=True)
     costs=np.asarray(costs).reshape(-1,2)
     expected=dict(meta=[it,len(costs)],covariance=cov.ravel(order='F').tolist(),precision=precision.ravel(order='F').tolist(),costs=costs.ravel(order='F').tolist())
    except (ValueError,ArithmeticError,np.linalg.LinAlgError)as error:expected=dict(error=str(error))
   expected['warnings']=len(caught);cache[key]=expected;calls+=1
  cases.append((request,expected,dict(index=index,**{k:v for k,v in c.items()if k!='cov'},p=len(a))))
 if args.reference_cache:args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 good=[c for c in cases if 'error'not in c[1]];bad=[c for c in cases if 'error'in c[1]];differences=[]
 def native(c):return subprocess.run([args.executable],input=c[0],text=True,capture_output=True)
 if args.record_differences:
  by_request={};offset=0
  while offset<len(good):
   result=subprocess.run([args.executable],input=''.join(c[0]for c in good[offset:]),text=True,capture_output=True)
   lines=result.stdout.splitlines();completed=sum(row.startswith('CASE ')for row in lines)
   for j in range(completed):
    assert lines[j*5]==f'CASE {j+1}'
    by_request[good[offset+j][0]]=subprocess.CompletedProcess([],0,'\n'.join(lines[j*5:j*5+5])+'\n','')
   offset+=completed
   if result.returncode:
    by_request[good[offset][0]]=result;offset+=1
   else:assert offset==len(good)
  with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
   for c,result in zip(bad,pool.map(native,bad)):by_request[c[0]]=result
  results=[by_request[c[0]]for c in cases]
  outputs=[];successful=[]
  for c,result in zip(cases,results):
   request,expected,label=c;out=result.stdout+result.stderr
   assert 'SIGFPE'not in out and 'Fortran runtime error'not in out,(label,out)
   if 'error'in expected:
    if not result.returncode:differences.append(dict(case=label,field='Source rejection accepted by native',source=expected['error']))
   elif result.returncode:
    assert 'fscikit-learn:'in out,(label,out)
    differences.append(dict(case=label,field='native rejected Source fit',diagnostic=next(row.strip()for row in out.splitlines()if 'fscikit-learn:'in row)))
   else:
    successful.append(c);outputs+=result.stdout.splitlines()[1:]
  good=successful;rows=iter(outputs)
 else:
  proc=subprocess.run([args.executable],input=''.join(c[0]for c in good),text=True,capture_output=True)
  if proc.returncode:
   completed=sum(row.startswith('CASE ')for row in proc.stdout.splitlines())
   raise RuntimeError(str(good[completed][2])+'\n'+proc.stdout[-200:]+'\n'+proc.stderr)
  rows=iter(proc.stdout.splitlines())
 groups=0;worst=0
 for index,(_,expected,label)in enumerate(good,1):
  if not args.record_differences:assert next(rows)==f'CASE {index}'
  for field in ['meta','covariance','precision','costs']:
   actual=np.fromstring(next(rows),sep=' ');want=np.asarray(expected[field]);groups+=1
   try:
    if field=='meta':np.testing.assert_array_equal(actual,want,err_msg=str(label))
    else:np.testing.assert_allclose(actual,want,atol=3.e-8,rtol=3.e-8,equal_nan=True,err_msg=str((label,field)))
   except AssertionError:
    if not args.record_differences:raise
    differences.append(dict(case=label,field=field,actual=actual.tolist(),expected=want.tolist()))
   if field!='meta'and actual.shape==want.shape:
    finite=np.isfinite(actual)&np.isfinite(want)
    if finite.any():worst=max(worst,float(np.max(abs(actual[finite]-want[finite])/(1+abs(want[finite])))))
 assert next(rows,None)is None
 for request,expected,label in (bad if not args.record_differences else []):
  proc=subprocess.run([args.executable],input=request,text=True,capture_output=True);out=proc.stdout+proc.stderr
  assert proc.returncode and 'fscikit-learn:'in out,(label,expected,out)
  assert 'SIGFPE'not in out and 'Fortran runtime error'not in out,(label,out)
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),successful_fits=len(good),source_rejections=len(bad),groups=groups,max_scaled_error=worst,source_calls=calls,reference_cache_hits=hits,source_warnings=sum(c[1]['warnings']for c in cases),provenance=provenance,executable_sha256=sha(args.executable),validator_sha256=sha(__file__),source_sha256=sha(root/'src/sklearn_covariance.f90'),input_sha256=hashlib.sha256(''.join(c[0]for c in cases).encode()).hexdigest(),inventory=[c[2]for c in cases])
 report.update(selection='regularized'if args.regularized else'all',source_successful_fits=len(cases)-len(bad),reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,
  strict_parity=not differences,failed_comparison_groups=len(differences),numeric_failed_groups=sum(d['field']in ['meta','covariance','precision','costs']for d in differences),differences=differences)
 args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['inventory','differences']},indent=2))
if __name__=='__main__':main()
