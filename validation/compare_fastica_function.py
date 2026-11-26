"""Actual sklearn fastica whitening, unmixing, sources, means and iteration counts."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,warnings,concurrent.futures
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np,sklearn,scipy
from sklearn.decomposition import fastica
import sklearn.decomposition._fastica as source

def fixtures(smoke=False):
 for seed in ([7]if smoke else[0,7,42]):
  rng=np.random.RandomState(seed)
  for n,p in ([(51,3)]if smoke else[(17,2),(51,3),(73,5)]):
   signal=rng.laplace(size=(n,p));signal[:,1]=rng.uniform(-2,2,n)
   x=signal@rng.normal(size=(p,p))+np.arange(p)
   xc=x-x.mean(axis=0);values,vectors=np.linalg.eigh(xc.T@xc/n)
   prewhite=xc@((vectors/np.sqrt(values))@vectors.T)
   for solver in ['svd','eigh']:
    for algorithm in ['parallel','deflation']:
     for fun in ['logcosh','exp','cube']:
      for whiten in ['unit-variance','arbitrary-variance','none']:
       for k in [1,p]:
        for initial in [False,True]:
         count=p if whiten=='none'else k
         w=rng.normal(size=(count,count))if initial else None
         limit=1 if k==1 else 25
         c=dict(x=x,k=k,w=w,seed=seed,solver=solver,algorithm=algorithm,fun=fun,whiten=whiten,
          limit=limit,tol=1.e-4,alpha=1.5 if fun=='logcosh'else 1.,compute=not initial,kind='raw_none'if whiten=='none'else'whitening')
         yield c
         if fun=='cube':yield dict(c,fun='custom_cube')
         if whiten=='none':yield dict(c,x=prewhite,kind='prewhite')
         if whiten=='none'and fun=='cube':yield dict(c,x=prewhite,kind='prewhite',fun='custom_cube')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
 ap.add_argument('--reference-cache',type=Path);ap.add_argument('--smoke',action='store_true')
 ap.add_argument('--standard-inputs',action='store_true',help='Internal whitening plus explicitly prewhitened inputs for whiten=none; the full grid retains raw none inputs.')
 ap.add_argument('--record-differences',action='store_true');ap.add_argument('--single-step',action='store_true');args=ap.parse_args()
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 provenance=dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,source_sha256=sha(source.__file__))
 cache={}
 if args.reference_cache and args.reference_cache.exists():
  saved=json.loads(args.reference_cache.read_text());assert saved['provenance']==provenance;cache=saved['cases']
 cases=[];calls=hits=0
 for index,c in enumerate(fixtures(args.smoke)):
  if args.standard_inputs and c['kind']=='raw_none':continue
  if args.single_step:c=dict(c,limit=1)
  x,w=c['x'],c['w'];n,p=x.shape;initial=0 if w is None else len(w)
  lines=[f'{n} {p} {c["k"]} {c["seed"]} {c["limit"]} {initial} {int(c["compute"])} {c["tol"]} {c["alpha"]} {c["algorithm"]} {c["whiten"]} {c["fun"]} {c["solver"]}']
  lines+=[' '.join(map(str,row))for row in x]
  if w is not None:lines+=[' '.join(map(str,row))for row in w]
  request='\n'.join(lines)+'\n';key=hashlib.sha256(request.encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    try:
     def cube_callback(z,alpha=1.):return z**3,np.mean(3*z**2,axis=-1)
     K,W,S,mean,it=fastica(x,n_components=c['k'],algorithm=c['algorithm'],whiten=False if c['whiten']=='none'else c['whiten'],fun=cube_callback if c['fun']=='custom_cube'else c['fun'],fun_args={'alpha':c['alpha']},max_iter=c['limit'],tol=c['tol'],w_init=w,whiten_solver=c['solver'],random_state=c['seed'],compute_sources=c['compute'],return_X_mean=True,return_n_iter=True)
     expected=dict(meta=[it,int(K is not None),int(mean is not None),int(S is not None)],whitening=[]if K is None else K.ravel(order='F').tolist(),unmixing=W.ravel(order='F').tolist(),sources=[]if S is None else S.ravel(order='F').tolist(),mean=[]if mean is None else mean.tolist())
    except (ValueError,ArithmeticError,np.linalg.LinAlgError)as error:expected=dict(error=str(error))
   expected['warnings']=len(caught);cache[key]=expected;calls+=1
  cases.append((request,expected,dict(index=index,n=n,p=p,**{k:v for k,v in c.items()if k not in ['x','w']},initial=initial)))
 if args.reference_cache:args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 good=[c for c in cases if 'error'not in c[1]];bad=[c for c in cases if 'error'in c[1]];differences=[]
 if args.record_differences:
  by_request={};offset=0
  while offset<len(good):
   proc=subprocess.run([args.executable],input=''.join(c[0]for c in good[offset:]),text=True,capture_output=True)
   rows_=proc.stdout.splitlines();completed=sum(row.startswith('CASE ')for row in rows_)
   for j in range(completed):
    assert rows_[j*6]==f'CASE {j+1}'
    by_request[good[offset+j][0]]=subprocess.CompletedProcess([],0,'\n'.join(rows_[j*6:j*6+6])+'\n','')
   offset+=completed
   if proc.returncode:by_request[good[offset][0]]=proc;offset+=1
   else:assert offset==len(good)
  def native(c):return subprocess.run([args.executable],input=c[0],text=True,capture_output=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
   for c,proc in zip(bad,pool.map(native,bad)):by_request[c[0]]=proc
  matched=[];output=[]
  for c in cases:
   proc=by_request[c[0]];out=proc.stdout+proc.stderr
   if proc.returncode:assert 'fscikit-learn:'in out and 'SIGFPE'not in out and 'Fortran runtime error'not in out,(c[2],out)
   if 'error'in c[1]:
    if not proc.returncode:differences.append(dict(case=c[2],field='Source rejection accepted by native',source=c[1]['error']))
   elif proc.returncode:differences.append(dict(case=c[2],field='native rejected Source fit',diagnostic=next(line.strip()for line in out.splitlines()if 'fscikit-learn:'in line)))
   else:matched.append(c);output+=proc.stdout.splitlines()[1:]
  good=matched;rows=iter(output)
 else:
  proc=subprocess.run([args.executable],input=''.join(c[0]for c in good),text=True,capture_output=True)
  if proc.returncode:
   completed=sum(row.startswith('CASE ')for row in proc.stdout.splitlines());raise RuntimeError(str(good[completed][2])+'\n'+proc.stdout[-200:]+'\n'+proc.stderr)
  rows=iter(proc.stdout.splitlines())
 groups=0;worst=0
 for index,(_,expected,label)in enumerate(good,1):
  if not args.record_differences:assert next(rows)==f'CASE {index}'
  for field in ['meta','whitening','unmixing','sources','mean']:
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
 for request,expected,label in (bad if not args.record_differences else[]):
  proc=subprocess.run([args.executable],input=request,text=True,capture_output=True);out=proc.stdout+proc.stderr
  assert proc.returncode and 'fscikit-learn:'in out and 'SIGFPE'not in out and 'Fortran runtime error'not in out,(label,out)
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),successful_fits=len(good),source_rejections=len(bad),groups=groups,max_scaled_error=worst,source_calls=calls,reference_cache_hits=hits,source_warnings=sum(c[1]['warnings']for c in cases),provenance=provenance,input_sha256=hashlib.sha256(''.join(c[0]for c in cases).encode()).hexdigest(),source_sha256=sha(root/'src/sklearn_fast_ica.f90'),executable_sha256=sha(args.executable),validator_sha256=sha(__file__),reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,inventory=[c[2]for c in cases])
 report.update(selection=('single_step_'if args.single_step else'')+('standard_inputs'if args.standard_inputs else'all'),strict_parity=not differences,source_successful_fits=len(cases)-len(bad),failed_comparison_groups=len(differences),differences=differences)
 args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['inventory','differences']},indent=2))
if __name__=='__main__':main()
