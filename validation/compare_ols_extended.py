"""Actual sklearn LinearRegression weighted dense/CSR and multiple-target fits."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,warnings,concurrent.futures
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np,sklearn,scipy
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
import sklearn.linear_model._base as source

def fixtures(smoke=False):
 for seed in ([7]if smoke else[0,7,42]):
  rng=np.random.default_rng(seed)
  for n,p in ([(17,3)]if smoke else[(1,1),(7,3),(7,11),(33,5),(67,9)]):
   for kind in (['regular']if smoke else['regular','duplicate','zero']):
    x=rng.normal(size=(n,p))+.5;y=rng.normal(size=(n,3))+2
    if kind=='duplicate'and p>1:x[:,-1]=x[:,0]
    if kind=='zero':x[:]=0
    for t in [1,3]:
     for sparse in [False,True]:
      for positive in [False,True]:
       for intercept in [False,True]:
        for weights in ['none','fraction','scalar','zero']:
         w=None
         if weights=='fraction':w=np.linspace(.1,1.2,n);w[::4]=0
         if weights=='scalar':w=2.
         if weights=='zero':w=np.zeros(n)
         yield dict(x=x,y=y[:,:t],sparse=sparse,positive=positive,intercept=intercept,w=w,kind=kind,seed=seed,weights=weights)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
 ap.add_argument('--reference-cache',type=Path);ap.add_argument('--smoke',action='store_true')
 ap.add_argument('--unconstrained',action='store_true');ap.add_argument('--record-differences',action='store_true');args=ap.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 provenance=dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,source_sha256=sha(source.__file__))
 cache={}
 if args.reference_cache and args.reference_cache.exists():
  saved=json.loads(args.reference_cache.read_text());assert saved['provenance']==provenance;cache=saved['cases']
 cases=[];calls=hits=0
 baseline=list(fixtures(args.smoke))
 # Preserve the original grid and add queries outside duplicate-column subspaces.
 extra=[dict(c,query_unseen=True)for c in baseline if c['kind']=='duplicate'and c['positive']and not c['sparse']]
 for index,c in enumerate(baseline+extra):
  if args.unconstrained and c['positive']:continue
  x,y,w=c['x'],c['y'],c['w'];n,p=x.shape;t=y.shape[1];m=4
  matrix=t>1 or index%3==0;qsparse=index%2==1;q=x[:1]+np.arange(m)[:,None]*.1
  if c.get('query_unseen',False):q=q+np.linspace(-.2,.3,p)[None,:]
  mode=0 if w is None else 2 if np.ndim(w)==0 else 1
  lines=[f'{n} {p} {t} {m} {int(c["sparse"])} {int(qsparse)} {int(matrix)} {int(c["intercept"])} {int(c["positive"])} {mode}']
  for a,b in zip(x,y):lines+=[' '.join(map(str,a)),' '.join(map(str,b))]
  lines+=[' '.join(map(str,row))for row in q]
  if mode:lines+=[' '.join(map(str,w))if mode==1 else str(w)]
  request='\n'.join(lines)+'\n';key=hashlib.sha256(request.encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   a=csr_matrix(x)if c['sparse']else x;b=csr_matrix(q)if qsparse else q
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    try:
     model=LinearRegression(fit_intercept=c['intercept'],positive=c['positive']).fit(a,y if matrix else y[:,0],sample_weight=w)
     singular=getattr(model,'singular_',None)
     qy=np.arange(1,m+1)[:,None]*.2-np.arange(1,t+1)*.1
     expected=dict(meta=[model.n_features_in_,t,getattr(model,'rank_',0),int(singular is not None)],coef=np.asarray(model.coef_).reshape(t,p).ravel(order='F').tolist(),intercept=np.asarray(model.intercept_).reshape(-1).tolist()if c['intercept']else[0.]*t,prediction=np.asarray(model.predict(b)).reshape(m,t).ravel(order='F').tolist(),singular=[]if singular is None else singular.tolist(),score=[model.score(b,qy if matrix else qy[:,0],sample_weight=np.linspace(.5,1.5,m))])
    except (ValueError,TypeError,ArithmeticError,np.linalg.LinAlgError,RuntimeError)as error:expected=dict(error=str(error))
   expected['warnings']=len(caught);cache[key]=expected;calls+=1
  label={k:v for k,v in c.items()if k not in ['x','y','w']};label.update(index=index,n=n,p=p,t=t,matrix=matrix,query_sparse=qsparse)
  cases.append((request,expected,label))
 if args.reference_cache:args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 good=[c for c in cases if 'error'not in c[1]];bad=[c for c in cases if 'error'in c[1]];differences=[]
 if args.record_differences:
  by_request={};offset=0
  while offset<len(good):
   proc=subprocess.run([args.executable],input=''.join(c[0]for c in good[offset:]),text=True,capture_output=True)
   lines=proc.stdout.splitlines();completed=sum(row.startswith('CASE ')for row in lines)
   for j in range(completed):
    assert lines[j*7]==f'CASE {j+1}'
    by_request[good[offset+j][0]]=subprocess.CompletedProcess([],0,'\n'.join(lines[j*7:j*7+7])+'\n','')
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
   elif proc.returncode:differences.append(dict(case=c[2],field='native rejected Source fit',diagnostic=next(row.strip()for row in out.splitlines()if 'fscikit-learn:'in row)))
   else:matched.append(c);output+=proc.stdout.splitlines()[1:]
  good=matched;rows=iter(output)
 else:
  proc=subprocess.run([args.executable],input=''.join(c[0]for c in good),text=True,capture_output=True)
  if proc.returncode:
   completed=sum(row.startswith('CASE ')for row in proc.stdout.splitlines());raise RuntimeError(str(good[completed][2])+'\n'+proc.stdout[-500:]+'\n'+proc.stderr)
  rows=iter(proc.stdout.splitlines())
 groups=0;worst=0
 for index,(_,expected,label)in enumerate(good,1):
  if not args.record_differences:assert next(rows)==f'CASE {index}'
  for field in ['meta','coef','intercept','prediction','singular','score']:
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
  assert proc.returncode and 'fscikit-learn:'in out and 'SIGFPE'not in out and 'Fortran runtime error'not in out,(label,expected,out)
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),successful_fits=len(good),source_rejections=len(bad),groups=groups,max_scaled_error=worst,source_calls=calls,reference_cache_hits=hits,source_warnings=sum(c[1]['warnings']for c in cases),provenance=provenance,source_sha256={n:sha(root/'src'/n)for n in ['sklearn_ols_core.f90','sklearn_linear_model.f90','sklearn_ridge_solver.f90']},executable_sha256=sha(args.executable),validator_sha256=sha(__file__),reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,input_sha256=hashlib.sha256(''.join(c[0]for c in cases).encode()).hexdigest(),inventory=[c[2]for c in cases])
 report.update(selection='unconstrained'if args.unconstrained else'all',strict_parity=not differences,source_successful_fits=len(cases)-len(bad),failed_comparison_groups=len(differences),differences=differences)
 args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['inventory','differences']},indent=2))
if __name__=='__main__':main()
