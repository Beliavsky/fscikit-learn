"""Actual sklearn Ridge class solvers, weighted centering, CSR and multi-target fits."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys,warnings,concurrent.futures
import numpy as np,scipy,sklearn
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
import sklearn.linear_model._ridge as source
import sklearn.linear_model._base as base

def fixtures(smoke):
 for seed in ([7]if smoke else [0,7,42]):
  rng=np.random.default_rng(seed)
  for n in ([7]if smoke else [1,7,33]):
   for p in ([3]if smoke else [1,3,9]):
    for t in [1,3]:
     x=rng.normal(size=(n,p))*.3+2;y=rng.normal(size=(n,t))+np.arange(t)+3
     for solver in ['auto','svd','cholesky','sparse_cg','lsqr','sag','saga','lbfgs']:
      for sparse in [False,True]:
       if sparse and solver=='svd':continue
       for intercept in [False,True]:
        if sparse and intercept and solver not in ['auto','sparse_cg','lsqr','sag','lbfgs']:continue
        for weights in range(3):
         w=None if weights==0 else np.linspace(.2,2,n)if weights==1 else 2.
         yield dict(x=x,y=y,solver=solver,sparse=sparse,intercept=intercept,w=w,seed=seed)
 # Source's special sparse-SAG intercept route and zero/signed weight behavior.
 rng=np.random.default_rng(17);x=rng.normal(size=(17,3))*.1;y=rng.normal(size=(17,2))
 for solver in ['auto','lsqr','sparse_cg','sag','saga','cholesky','lbfgs']:
  for sparse in [False,True]:
   for intercept in [False,True]:
    if sparse and intercept and solver in ['cholesky','saga']:continue
    for mode in ['zeros','all_zero','signed','zero_target']:
     if mode=='all_zero'and intercept:continue
     if mode=='signed'and(solver not in ['sag','saga']or sparse and intercept):continue
     w=np.ones(17);w[::3]=0
     if mode=='all_zero':w[:]=0
     if mode=='signed':w[::3]=-.1
     yield dict(x=x,y=y if mode!='zero_target'else np.zeros_like(y),solver=solver,sparse=sparse,intercept=intercept,w=w,seed=17,limit=-1 if sparse and solver=='sag'and intercept else 10,tol=.01 if sparse and solver=='sag'and intercept else 1e-4)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--report',required=True,type=Path);ap.add_argument('--reference-cache',type=Path);ap.add_argument('--smoke',action='store_true');ap.add_argument('--record-differences',action='store_true',help='Retain failed numeric comparisons in a diagnostic report; default remains strict.');args=ap.parse_args()
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 provenance=dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,ridge_sha256=sha(source.__file__),base_sha256=sha(base.__file__))
 cache={}
 if args.reference_cache and args.reference_cache.exists():
  saved=json.loads(args.reference_cache.read_text());assert saved['provenance']==provenance;cache=saved['cases']
 cases=[];calls=hits=0
 for index,c in enumerate(fixtures(args.smoke)):
  x,y=c['x'],c['y'];n,p=x.shape;t=y.shape[1];m=4
  matrix=t>1 or index%3==0;qsparse=index%2==1
  q=x[:1]+np.arange(m)[:,None]*.1;qw=np.linspace(.5,1.5,m);qy=np.arange(1,m+1)[:,None]*.1-np.arange(1,t+1)*.2
  w=c['w'];wmode=0 if w is None else 2 if np.ndim(w)==0 else 1
  positive=c['solver']=='lbfgs';limit=c.get('limit',[1,5,25][index%3]);tol=c.get('tol',[1e-4,1e-6,0.][index%3]);alpha=.7
  penalties=t>1 and index%2==0;penalty=np.linspace(.3,1.1,t)if penalties else alpha
  lines=[f'{n} {p} {t} {m} {int(c["sparse"])} {int(qsparse)} {int(matrix)} {int(c["intercept"])} {int(positive)} {limit} {c["seed"]} {wmode} {int(penalties)} {tol} {alpha} {c["solver"]}']
  for a,b in zip(x,y):lines+=[' '.join(map(str,a)),' '.join(map(str,b))]
  lines+=[' '.join(map(str,row))for row in q];lines+=[' '.join(map(str,qw))]
  if wmode:lines+=[' '.join(map(str,w))if wmode==1 else str(w)]
  if penalties:lines+=[' '.join(map(str,penalty))]
  request='\n'.join(lines)+'\n';key=hashlib.sha256(request.encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   a=csr_matrix(x)if c['sparse']else x;b=csr_matrix(q)if qsparse else q;target=y if matrix else y[:,0]
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always');fit_error=None
    try:model=Ridge(alpha=penalty,solver=c['solver'],fit_intercept=c['intercept'],positive=positive,max_iter=None if limit==-1 else limit,tol=tol,random_state=c['seed']).fit(a,target,sample_weight=w)
    except (ValueError,ArithmeticError)as error:fit_error=str(error)
    if fit_error is not None:
     expected=dict(fit_error=fit_error,score_error=None,warnings=len(caught))
    else:
     score_error=None
     try:query_score=model.score(b,qy if matrix else qy[:,0],sample_weight=qw)
     except ValueError as error:
      query_score=None;score_error=str(error)
     expected=dict(meta=[model.n_features_in_,t,int(matrix)],solver=model.solver_,iterations=[-1]*t if model.n_iter_ is None else model.n_iter_.tolist(),coef=np.asarray(model.coef_).reshape(t,p).ravel(order='F').tolist(),intercept=np.asarray(model.intercept_).reshape(-1).tolist()if c['intercept']else [0.]*t,prediction=np.asarray(model.predict(b)).reshape(m,t).ravel(order='F').tolist(),score=[query_score]if score_error is None else None,score_error=score_error,warnings=len(caught))
   cache[key]=expected;calls+=1
  request=request.replace('\n',f' {int(expected.get("score_error")is None)}\n',1)
  cases.append((request,expected,dict(index=index,n=n,p=p,t=t,solver=c['solver'],sparse=c['sparse'],matrix=matrix,intercept=c['intercept'],limit=limit,tol=tol,weights=wmode)))
 if args.reference_cache:args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 fit_rejected=[(request,label,expected['fit_error'])for request,expected,label in cases if expected.get('fit_error')is not None]
 successful=[c for c in cases if c[1].get('fit_error')is None]
 request=''.join(c[0]for c in successful);r=subprocess.run([args.executable],input=request,text=True,capture_output=True)
 if r.returncode:
  markers=[line for line in r.stdout.splitlines()if line.startswith('CASE ')]
  number=int(markers[-1].split()[1])-1 if markers else 0
  raise RuntimeError(str(successful[number][2])+'\n'+r.stderr+'\n'+r.stdout[-1000:])
 rows=iter(r.stdout.splitlines());groups=0;worst=0;differences=[]
 for case_id,(_,expected,label)in enumerate(successful,1):
  assert next(rows)==f'CASE {case_id}'
  for field in ['meta','solver','iterations','coef','intercept','prediction','score']:
   row=next(rows);groups+=1
   if field=='solver':assert row.strip()==expected[field],(label,row,expected[field]);continue
   if field=='score'and expected.get('score_error')is not None:
    groups-=1;continue
   actual=np.fromstring(row,sep=' ');want=np.asarray(expected[field])
   if field in ['meta','iterations']:
    try:np.testing.assert_array_equal(actual,want,err_msg=str((label,field)))
    except AssertionError:
     if not args.record_differences or field=='meta':raise
     differences.append(dict(case=label,field=field,actual=actual.tolist(),expected=want.tolist()))
   else:
    try:np.testing.assert_allclose(actual,want,rtol=3e-8,atol=3e-8,equal_nan=True,err_msg=str((label,field)))
    except AssertionError:
     if not args.record_differences:raise
     differences.append(dict(case=label,field=field,actual=actual.tolist(),expected=want.tolist()))
    finite=np.isfinite(actual)&np.isfinite(want)
    if finite.any():worst=max(worst,float(np.max(abs(actual[finite]-want[finite])/(1+abs(want[finite])))))
 assert next(rows,None)is None
 rejected=[(request,label)for request,expected,label in successful if expected.get('score_error')is not None]
 def probe(item):
  request,label=item
  r=subprocess.run([args.executable,'force-score'],input=request,text=True,capture_output=True)
  output=r.stdout+r.stderr
  assert r.returncode and 'variance score targets must be finite'in output,(label,output)
  assert 'SIGFPE'not in output and 'Fortran runtime error'not in output,(label,output)
 def fit_probe(item):
  request,label,error=item
  r=subprocess.run([args.executable],input=request,text=True,capture_output=True);output=r.stdout+r.stderr
  assert r.returncode and 'fscikit-learn:'in output,(label,error,output)
  if 'step_size * alpha_scaled'in error:assert 'step times penalty'in output,(label,error,output)
  assert 'SIGFPE'not in output and 'Fortran runtime error'not in output,(label,output)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
  list(pool.map(probe,rejected));list(pool.map(fit_probe,fit_rejected))
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),successful_fits=len(successful),source_fit_rejections=len(fit_rejected),groups=groups,source_score_rejections=len(rejected),max_scaled_error=worst,source_calls=calls,reference_cache_hits=hits,source_warnings=sum(c[1]['warnings']for c in cases),provenance=provenance,input_sha256=hashlib.sha256(request.encode()).hexdigest(),executable_sha256=sha(args.executable),validator_sha256=sha(__file__),reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,source_sha256={n:sha(root/'src'/n)for n in ['sklearn_ridge_estimator_core.f90','sklearn_ridge_solver.f90','sklearn_linear_model.f90']},inventory=[c[2]for c in cases])
 report.update(strict_parity=not differences,failed_numeric_groups=len(differences),differences=differences)
 args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['inventory','differences']},indent=2))
if __name__=='__main__':main()
