"""Weighted classic classification boosting in search, selection and compositions."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier,BaggingClassifier,VotingClassifier,StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
ap.add_argument('--subsample',action='store_true')
a=ap.parse_args();records=[];payload=[]
for (loss,classes),seed,n in itertools.product([('log_loss',2),('log_loss',3),('exponential',2)],[0,42],[24,31]):
 p=3 if n==24 else 8;rng=np.random.RandomState(2917+n);x=rng.normal(size=(n,p));y=(np.arange(n)%classes)*17-9;rng.shuffle(y)
 w=np.resize([1.,0.,.5,2.,3.],n);weights=w if seed==42 else None;query=x+.137
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  base=GradientBoostingClassifier(loss=loss,n_estimators=3,max_depth=2,random_state=seed,max_features=.7,subsample=(.65 if seed==42 else .8) if a.subsample else 1.).fit(x,y,sample_weight=weights)
  grid=dict(ccp_alpha=[0.,.01],criterion=['friedman_mse','squared_error'],max_features=[.5,None],max_leaf_nodes=[None,3])
  if a.subsample:grid['subsample']=[.5,.8]
  search=GridSearchCV(base,grid,cv=3,error_score='raise',n_jobs=1).fit(x,y,sample_weight=weights)
  selector=SelectFromModel(base).fit(x,y,sample_weight=weights);prefit=SelectFromModel(base,prefit=True).fit(x)
  pipe=Pipeline([('scale',StandardScaler()),('boost',clone(base))]).fit(x,y,boost__sample_weight=weights)
  other=clone(base).set_params(loss='log_loss',n_estimators=4,random_state=seed+1)
  vote=VotingClassifier([('first',base),('second',other)],weights=[1.,2.],voting='soft',n_jobs=1).fit(x,y,sample_weight=weights)
  bag=BaggingClassifier(base,n_estimators=4,max_samples=.75,max_features=.8,bootstrap=True,oob_score=True,random_state=seed,n_jobs=1).fit(x,y,sample_weight=weights)
  stack=StackingClassifier([('first',base),('second',other)],final_estimator=GradientBoostingClassifier(n_estimators=3,max_depth=1,random_state=19),stack_method='decision_function' if seed==42 else 'auto',cv=3,n_jobs=1).fit(x,y,sample_weight=weights)
  cv=search.cv_results_
  expected=dict(mean=cv['mean_test_score'].tolist(),std=cv['std_test_score'].tolist(),
   splits=np.column_stack([cv[f'split{k}_test_score'] for k in range(3)]).ravel().tolist(),ranks=cv['rank_test_score'].tolist(),
   best_index=[int(search.best_index_)+1],best_score=[search.best_score_],best_prediction=search.predict(query).tolist(),
   selector_importance=selector.estimator_.feature_importances_.tolist(),selector_mask=selector.get_support().astype(int).tolist(),
   prefit_importance=prefit.estimator_.feature_importances_.tolist(),prefit_mask=prefit.get_support().astype(int).tolist(),
   original_prediction=base.predict(query).tolist(),pipeline=pipe.predict(query).tolist(),voting=vote.predict(query).tolist(),
   voting_components=vote.transform(query).ravel().tolist(),bagging=bag.predict(query).tolist(),
   bagging_samples=(np.stack(bag.estimators_samples_,axis=1)+1).ravel().tolist(),
   bagging_features=(np.stack(bag.estimators_features_,axis=1)+1).ravel().tolist(),bagging_seeds=bag._seeds.tolist(),
   component_seeds=[e.random_state for e in bag.estimators_],oob=bag.oob_decision_function_.ravel().tolist(),oob_score=[bag.oob_score_],
   pipeline_score=[pipe.score(x,y,sample_weight=weights)],stacking=stack.predict(query).tolist(),stacking_components=stack.transform(query).ravel().tolist(),bagging_probability=bag.predict_proba(query).ravel().tolist(),voting_probability=vote.predict_proba(query).ravel().tolist(),stacking_probability=stack.predict_proba(query).ravel().tolist())
 records.append(dict(case=dict(loss=loss,seed=seed,n=n,p=p,weighted=weights is not None),source=expected,differences=[]))
 text=f'{n} {p} {seed} {loss} {int(weights is not None)}\n'+'\n'.join(' '.join(map(str,row)) for row in x)+'\n'
 text+=' '.join(map(str,y))+'\n'+' '.join(map(str,w))+'\n';payload.append(text)
text=''.join(payload);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.with_suffix('.in').write_text(text)
p=subprocess.run([str(a.executable)]+(['--subsample'] if a.subsample else []),input=text,text=True,capture_output=True,timeout=180)
a.report.with_suffix('.stdout').write_text(p.stdout);a.report.with_suffix('.stderr').write_text(p.stderr)
lines=p.stdout.splitlines();fields=list(records[0]['source']);width=len(fields);assert width==28
integer={'best_prediction','original_prediction','pipeline','voting','bagging','stacking','ranks','best_index','selector_mask','prefit_mask','bagging_samples','bagging_features','bagging_seeds','component_seeds'}
for i,r in enumerate(records):
 chunk=lines[width*i:width*(i+1)];r['native']={}
 if len(chunk)!=width:r['differences'].append('missing native output');continue
 for key,line in zip(fields,chunk):
  got=np.fromstring(line,sep=' ');want=np.asarray(r['source'][key]);r['native'][key]=got.tolist()
  ok=got.shape==want.shape and (np.array_equal(got,want) if key in integer else np.allclose(got,want,rtol=2e-10,atol=2e-12,equal_nan=True))
  if not ok:r['differences'].append(key)
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
report=dict(configurations=len(records),records=records,failed_configurations=sum(bool(r['differences']) for r in records),native_returncode=p.returncode,
 input_sha256=hashlib.sha256(text.encode()).hexdigest(),sklearn=sklearn.__version__,sha256={str(f):sha(f) for f in [Path(__file__),Path('app/reference_gbc_integration_driver.f90'),a.executable,Path('src/sklearn_classic_gradient_boosting.f90'),Path('src/sklearn_boosting_loss.f90'),Path('src/sklearn_base_dispatch.f90'),Path('src/sklearn_search_adapters.f90'),Path('src/sklearn_model_selection_features.f90')]})
a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['records','sha256']});assert p.returncode==0 and len(lines)==len(records)*width and report['failed_configurations']==0
