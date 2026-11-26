"""Fitted warm forest templates must be cloned without learned trees/OOB state."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor,DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.feature_selection import SelectFromModel
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
a=ap.parse_args();records=[];payloads=[]
for family,mode,seed,n in itertools.product([0,1,2,3],[0,1],[0,7,42],[24,31]):
 rng=np.random.RandomState(1913+n);x=rng.normal(size=(n,3 if n==24 else 8));yr=2*x[:,0]+x[:,1]*x[:,2];yi=(x[:,0]+.2*x[:,2]>0).astype(int);y=yr if family%2==0 else yi
 cls=[[RandomForestRegressor,ExtraTreesRegressor],[RandomForestClassifier,ExtraTreesClassifier],[DecisionTreeRegressor,ExtraTreeRegressor],[DecisionTreeClassifier,ExtraTreeClassifier]][family][mode]
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  params=dict(random_state=seed,max_depth=3,max_features=.7)
  if family<2:params.update(warm_start=True,n_estimators=5,bootstrap=True,oob_score=True,n_jobs=1)
  base=cls(**params).fit(x,np.zeros_like(y))
  grid=dict(max_features=[1,1.,.5,'sqrt','log2',None,x.shape[1]+3])
  search=GridSearchCV(base,grid,cv=3,error_score='raise',n_jobs=1).fit(x,y)
  selector=SelectFromModel(base).fit(x,y);prefit=SelectFromModel(base,prefit=True).fit(x)
 best=search.best_estimator_;cv=search.cv_results_
 expected=dict(mean=cv['mean_test_score'].tolist(),std=cv['std_test_score'].tolist(),
               splits=np.column_stack([cv[f'split{k}_test_score'] for k in range(3)]).ravel().tolist(),ranks=cv['rank_test_score'].tolist(),
               metadata=[int(search.best_index_)+1,len(best.estimators_) if family<2 else 1,int(getattr(best,'bootstrap',False)),int(getattr(best,'warm_start',False)),int(hasattr(best,'oob_score_'))],
               prediction=search.predict(x).tolist(),probabilities=search.predict_proba(x).ravel().tolist() if family%2==1 else [],
               selector_importance=selector.estimator_.feature_importances_.tolist(),selector_mask=selector.get_support().astype(int).tolist(),
               prefit_importance=prefit.estimator_.feature_importances_.tolist(),prefit_mask=prefit.get_support().astype(int).tolist(),
               original_prediction=base.predict(x).tolist(),original_metadata=[int(hasattr(base,'oob_score_')),len(base.estimators_) if family<2 else 1],best_score=[search.best_score_])
 records.append(dict(case=dict(family=family,mode=mode,seed=seed,n=n,p=x.shape[1]),source=expected,differences=[]))
 text=f'{n} {x.shape[1]} {family} {mode} {seed}\n'+'\n'.join(' '.join(map(str,row)) for row in x)+'\n'
 text+=' '.join(map(str,yr))+'\n'+' '.join(map(str,yi))+'\n';payloads.append(text)
payload=''.join(payloads);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.with_suffix('.in').write_text(payload)
p=subprocess.run([str(a.executable)],input=payload,text=True,capture_output=True)
a.report.with_suffix('.stdout').write_text(p.stdout);a.report.with_suffix('.stderr').write_text(p.stderr);lines=p.stdout.splitlines()
for i,r in enumerate(records):
 chunk=lines[14*i:14*(i+1)];r['native']={}
 if len(chunk)!=14:r['differences'].append('missing native output');continue
 for (key,expected),line in zip(r['source'].items(),chunk):
  got=np.fromstring(line,sep=' ');r['native'][key]=got.tolist();want=np.asarray(expected)
  integer=key in ['ranks','metadata','selector_mask','prefit_mask','original_metadata'] or (r['case']['family']%2==1 and key in ['prediction','original_prediction'])
  ok=got.shape==want.shape and (np.array_equal(got,want) if integer else np.allclose(got,want,rtol=2e-10,atol=2e-12,equal_nan=True))
  if not ok:r['differences'].append(key)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
report=dict(features=True,configurations=len(records),records=records,failed_configurations=sum(bool(r['differences']) for r in records),native_returncode=p.returncode,
            input_sha256=hashlib.sha256(payload.encode()).hexdigest(),sklearn=sklearn.__version__,sha256={str(path):sha(path) for path in [Path(__file__),a.executable,Path('src/sklearn_ensemble.f90'),Path('src/sklearn_search_adapters.f90'),Path('src/sklearn_model_selection_features.f90'),Path('src/sklearn_tree.f90'),Path('src/sklearn_tree_features.f90')]})
a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['records','sha256']});assert not p.returncode and not report['failed_configurations']
