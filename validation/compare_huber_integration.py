"""Fitted warm-start Huber prototypes in weighted native compositions."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,VotingRegressor,StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import HuberRegressor
import sklearn.linear_model._huber as huber_source
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
a=ap.parse_args();records=[];payload=[]
for epsilon,seed,n in itertools.product([1.35,1.8,2.],[0,42],[24,31]):
 p=3 if n==24 else 8;rng=np.random.RandomState(2917+n);x=rng.normal(size=(n,p));y=2*x[:,0]+x[:,1]*x[:,2]
 w=np.resize([1.,0.,.5,2.,3.],n);weights=w if seed==42 else None;query=x+.137
 source_error=None;expected=None;optimizer_runs=[];phase='prototype'
 original_minimize=huber_source.optimize.minimize
 def observe(fun,*args,**kwargs):
  result=original_minimize(fun,*args,**kwargs)
  if fun is huber_source._huber_loss_and_gradient:
   design,target,eps,alpha,weight=kwargs['args']
   optimizer_runs.append(dict(phase=phase,n=design.shape[0],p=design.shape[1],
    nonzero_weights=int(np.count_nonzero(weight)),epsilon=eps,alpha=alpha,
    status=int(result.status),iterations=int(result.nit),message=str(result.message),
    initial_point=np.asarray(args[0]).tolist(),point=result.x.tolist(),objective=float(result.fun)))
  return result
 huber_source.optimize.minimize=observe
 try:
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   base=HuberRegressor(epsilon=epsilon,alpha=.01,tol=1e-8,max_iter=200,warm_start=True).fit(x,0*y,sample_weight=weights)
   grid={'alpha':[.01,.1],'epsilon':[1.35,1.8],'fit_intercept':[True,False]}
   phase='search'
   search=GridSearchCV(base,grid,cv=3,error_score='raise',n_jobs=1).fit(x,y,sample_weight=weights)
   phase='selector'
   selector=SelectFromModel(base).fit(x,y,sample_weight=weights);prefit=SelectFromModel(base,prefit=True).fit(x)
   phase='pipeline'
   pipe=Pipeline([('scale',StandardScaler()),('boost',clone(base))]).fit(x,y,boost__sample_weight=weights)
   other=clone(base).set_params(epsilon=2.,alpha=.1)
   phase='voting'
   vote=VotingRegressor([('first',base),('second',other)],weights=[1.,2.],n_jobs=1).fit(x,y,sample_weight=weights)
   phase='bagging'
   bag=BaggingRegressor(base,n_estimators=4,max_samples=.75,max_features=.8,bootstrap=True,oob_score=True,random_state=seed,n_jobs=1).fit(x,y,sample_weight=weights)
   phase='stacking'
   stack=StackingRegressor([('first',base),('second',other)],cv=3,n_jobs=1).fit(x,y,sample_weight=weights)
   cv=search.cv_results_
   expected=dict(mean=cv['mean_test_score'].tolist(),std=cv['std_test_score'].tolist(),
    splits=np.column_stack([cv[f'split{k}_test_score'] for k in range(3)]).ravel().tolist(),ranks=cv['rank_test_score'].tolist(),
    best_index=[int(search.best_index_)+1],best_score=[search.best_score_],best_prediction=search.predict(query).tolist(),
    selector_importance=np.abs(selector.estimator_.coef_).tolist(),selector_mask=selector.get_support().astype(int).tolist(),
    prefit_importance=np.abs(prefit.estimator_.coef_).tolist(),prefit_mask=prefit.get_support().astype(int).tolist(),
    original_prediction=base.predict(query).tolist(),pipeline=pipe.predict(query).tolist(),voting=vote.predict(query).tolist(),
    voting_components=vote.transform(query).ravel().tolist(),bagging=bag.predict(query).tolist(),
    bagging_samples=(np.stack(bag.estimators_samples_,axis=1)+1).ravel().tolist(),
    bagging_features=(np.stack(bag.estimators_features_,axis=1)+1).ravel().tolist(),bagging_seeds=bag._seeds.tolist(),
    oob=bag.oob_prediction_.tolist(),oob_score=[bag.oob_score_],
    pipeline_score=[pipe.score(x,y,sample_weight=weights)],stacking=stack.predict(query).tolist(),stacking_components=stack.transform(query).ravel().tolist())
 except (ValueError,ZeroDivisionError) as error:
  source_error=dict(type=type(error).__name__,message=str(error))
 huber_source.optimize.minimize=original_minimize
 records.append(dict(optimizer_runs=optimizer_runs,case=dict(epsilon=epsilon,seed=seed,n=n,p=p,weighted=weights is not None),source=expected,source_error=source_error,differences=[]))
 text=f'{n} {p} {seed} {epsilon} {int(weights is not None)}\n'+'\n'.join(' '.join(map(str,row)) for row in x)+'\n'
 text+=' '.join(map(str,y))+'\n'+' '.join(map(str,w))+'\n';
 if source_error:records[-1]['input']=text
 else:payload.append(text)
text=''.join(payload);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.with_suffix('.in').write_text(text)
p=subprocess.run([str(a.executable)],input=text,text=True,capture_output=True,timeout=180)
a.report.with_suffix('.stdout').write_text(p.stdout);a.report.with_suffix('.stderr').write_text(p.stderr)
lines=p.stdout.splitlines();valid=[r for r in records if r['source'] is not None];fields=list(valid[0]['source']);width=len(fields);assert width==24
integer={'ranks','best_index','selector_mask','prefit_mask','bagging_samples','bagging_features','bagging_seeds','component_seeds'}
for i,r in enumerate(valid):
 chunk=lines[width*i:width*(i+1)];r['native']={}
 if len(chunk)!=width:r['differences'].append('missing native output');continue
 for key,line in zip(fields,chunk):
  got=np.fromstring(line,sep=' ');want=np.asarray(r['source'][key]);r['native'][key]=got.tolist()
  ok=got.shape==want.shape and (np.array_equal(got,want) if key in integer else np.allclose(got,want,rtol=3e-8,atol=3e-8,equal_nan=True))
  if not ok:r['differences'].append(key)
for r in records:
 if r['source'] is not None:continue
 rejected=subprocess.run([str(a.executable)],input=r['input'],text=True,capture_output=True,timeout=60)
 output=rejected.stdout+rejected.stderr
 r['native_error']=dict(returncode=rejected.returncode,output=output)
 if not(rejected.returncode!=0 and 'fscikit-learn:' in output and not any(v in output for v in ['SIGFPE','SIGSEGV','Fortran runtime error'])):r['differences'].append('expected controlled Source fit rejection')
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
report=dict(configurations=len(records),records=records,failed_configurations=sum(bool(r['differences']) for r in records),native_returncode=p.returncode,
 input_sha256=hashlib.sha256(text.encode()).hexdigest(),sklearn=sklearn.__version__,sha256={str(f):sha(f) for f in [Path(__file__),a.executable,Path('src/sklearn_huber_solver.f90'),Path('src/sklearn_linear_model.f90'),Path('src/sklearn_sgd_clone.f90'),Path('src/sklearn_base_dispatch.f90'),Path('src/sklearn_search_adapters.f90'),Path('src/sklearn_model_selection_features.f90')]})
a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['records','sha256']});assert p.returncode==0 and report['failed_configurations']==0
