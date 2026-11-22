"""Reference engine for native cross_validate and cross_val_predict."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import warnings
import subprocess
os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import sklearn
from sklearn.model_selection import cross_validate,cross_val_predict,cross_val_score,KFold,StratifiedKFold
from sklearn.linear_model import Ridge,ElasticNet,LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_classif,f_regression
from compare_function_transformer import ROOT,resolve

fit_targets=[]
predict_calls=0
score_calls=0


class FitSpy:
    def fit(self,X,y=None,**params):
        fit_targets.append(None if y is None else np.asarray(y).copy())
        return super().fit(X,y,**params)


class SpyRidge(FitSpy,Ridge):
    def predict(self,X):
        global predict_calls
        predict_calls+=1
        return super().predict(X)


class SpyLogistic(FitSpy,LogisticRegression):pass
class SpyBayes(FitSpy,GaussianNB):pass
class SpyNeighbors(FitSpy,KNeighborsClassifier):pass
class SpyPipeline(FitSpy,Pipeline):pass


def mean_error(estimator,x,y):
    global score_calls
    score_calls+=1
    return float(np.mean(estimator.predict(x)-y))


def scaled_moment(estimator,x,y=None):
    global score_calls
    score_calls+=1
    z=estimator.transform(x)
    return float(np.mean(z*z))


def nonfinite(estimator,x,y):
    global score_calls
    score_calls+=1
    return np.inf if len(x)%2==0 else np.nan


def model_for(c):
    family=c['family'];spy=c['mode']==1
    if family in [0,11]:return Ridge(alpha=.3,solver='svd')
    if family==10:return SpyRidge(alpha=.3,solver='svd')
    if family==1:return ElasticNet(alpha=.08,l1_ratio=.4,tol=1e-11)
    if family==2:return (SpyLogistic if spy else LogisticRegression)(C=.8,tol=1e-10,max_iter=200)
    if family==3:return (SpyBayes if spy else GaussianNB)()
    if family==4:return KNeighborsRegressor(n_neighbors=3,algorithm='brute')
    if family==5:return (SpyNeighbors if spy else KNeighborsClassifier)(n_neighbors=3,algorithm='brute')
    if family==9:return StandardScaler()
    steps=[('scale',StandardScaler())]
    if family==8:steps.append(('select',SelectKBest(f_classif if c['integer_target'] else f_regression,k=2)))
    steps.append(('model',LogisticRegression(C=.8,tol=1e-10,max_iter=200) if family==7 else Ridge(alpha=.3,solver='svd')))
    return (SpyPipeline if family==7 and spy else Pipeline)(steps)


def fixtures(mode):
    for seed in [0,42,2026]:
        rng=np.random.RandomState(seed)
        for family in range(12):
            if mode==1 and family in [9,11]:continue
            for config in range(8):
                classifier=family in [2,3,5,7]
                if mode==1 and family==3 and config in [2,7]:continue
                if mode==1 and family==5 and config in [2,3,6,7]:continue
                n,p,m,cv=36+config%2,4,6,3+config%2
                x=rng.normal(size=(n,p));q=rng.normal(size=(m,p))
                yr=2*x[:,0]-1.2*x[:,2]+.3+rng.normal(scale=.3,size=n)
                if classifier:
                    classes=np.array([-4,9,15])[:3 if mode==0 and config%2 else 2]
                    yi=classes[np.argmax(np.c_[x[:,0],-x[:,0]+x[:,1],x[:,2]][:,:len(classes)],axis=1)]
                    if mode==1 and config in [5,6]:
                        yi=np.resize([-4,9],n);yi[5]=15
                    if mode==1 and config==7:
                        yi=np.resize([-4,9,15],n);yi[5]=23
                else:yi=np.rint(yr).astype(int)
                method=0
                if mode==1 and classifier:method={0:0,1:1,2:2,3:3,4:0,5:1,6:3,7:2}[config]
                weights=rng.uniform(.2,2,size=n);weights[:2]=0
                c=dict(seed=seed,mode=mode,family=family,config=config,n=n,p=p,m=m,cv=cv,
                       weighted=int(family in [0,2,3,6,7,8,10] and config%2==1),
                       integer_target=int(config%3==2),method=method,classification=int(classifier))
                yield c,x,q,yr,yi,weights


def read_matrix(lines):
    rows,cols=map(int,next(lines).split())
    return np.array([np.fromstring(next(lines),sep=' ') for _ in range(rows)]).reshape(rows,cols)


def run(mode):
    global fit_targets,predict_calls,score_calls
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
    parser.add_argument('--executable',type=Path);args=parser.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_validation_driver')
    checks=[];errors={};warning_counts={};count=0
    for c,x,q,yr,yi,weights in fixtures(mode):
        family,config=c['family'],c['config'];classifier=bool(c['classification']);no_target=family==9
        y=None if no_target else yi if classifier or c['integer_target'] else yr
        if classifier and not c['integer_target']:y=y.astype(float)
        model=model_for(c);method=['predict','predict_proba','decision_function','predict_log_proba'][c['method']]
        if mode==0 and config in [4,6]:
            tests=[np.array([i]) if config==6 else np.r_[i,np.arange(i,len(x),5)] for i in range(4 if config==6 else 3)]
            folds=[(np.flatnonzero(~np.isin(np.arange(len(x)),test)),test) for test in tests]
        elif mode==1 and config==4:
            tests=[np.arange(len(x)-i-1,-1,-3) for i in range(3)]
            folds=[(np.flatnonzero(~np.isin(np.arange(len(x)),test)),test) for test in tests]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                folds=list((StratifiedKFold(c['cv']) if classifier else KFold(c['cv'])).split(x,y))
        params={}
        if c['weighted']:params['model__sample_weight' if family in [6,7,8] else 'sample_weight']=weights
        fit_targets=[];predict_calls=0;score_calls=0
        with warnings.catch_warnings(record=True) as caught,np.errstate(all='ignore'):
            warnings.simplefilter('always')
            if mode==0:
                scoring='accuracy' if classifier else None
                if config in [1,4] and not classifier:scoring='neg_mean_squared_error'
                if config==4 and classifier:scoring='balanced_accuracy'
                if config==5:scoring='f1_macro' if classifier else 'neg_root_mean_squared_error'
                if config==2:scoring=dict(primary='accuracy' if classifier else 'neg_mean_squared_error',
                                          secondary='neg_log_loss' if classifier else 'r2')
                if config==3:scoring=mean_error
                if family==9:scoring=scaled_moment
                if family==11:scoring=nonfinite
                train=config%2==1 or config in [2,6];estimators=config not in [0,3];indices=config not in [0,2]
                result=cross_validate(model,x,y,cv=folds,scoring=scoring,params=params,return_train_score=train,
                                      return_estimator=estimators,return_indices=indices,error_score='raise')
                scalar=cross_val_score(model,x,y,cv=folds,scoring=scoring,params=params,error_score='raise') if config==3 else np.array([])
                names=list(scoring) if isinstance(scoring,dict) else ['score']
                want_test=np.column_stack([result['test_'+name] for name in names])
                want_train=np.column_stack([result['train_'+name] for name in names]) if train else np.empty((0,0))
                expected_fit_calls=len(fit_targets) if family==10 else 0
                expected_score_calls=score_calls
                stored=[]
                if estimators:
                    for child in result['estimator']:
                        prediction=child.transform(q) if no_target else child.predict(q)[:,None]
                        probability=child.predict_proba(q) if classifier else np.empty((0,0))
                        classes=child.classes_ if classifier else np.array([],dtype=int)
                        stored.append((prediction,probability,classes))
            else:
                predicted=cross_val_predict(model,x,y,cv=folds,method=method,params=params)
                if predicted.ndim==1:predicted=predicted[:,None]
                expected_fit_calls=len(fit_targets) if classifier or family==10 else 0
                expected_response_calls=predict_calls if family==10 else 0
                expected_range=[int(min(np.min(t) for t in fit_targets)),int(max(np.max(t) for t in fit_targets))] if classifier else [2147483647,-2147483647]
        for warning in caught:
            key=warning.category.__name__+': '+str(warning.message)
            warning_counts[key]=warning_counts.get(key,0)+1
        header=[c[k] for k in ['mode','family','config','n','p','m','cv','weighted','integer_target','method']]
        payload=' '.join(map(str,header))+'\n'+''.join(' '.join(f'{v:.17g}' for v in np.asarray(a).ravel(order='F'))+'\n'
                                                       for a in [x,q,yr,yi,weights])
        native=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
        assert native.returncode==0,(c,native.stdout,native.stderr)
        lines=iter(native.stdout.splitlines());metadata=np.fromstring(next(lines),sep=' ',dtype=int)
        actuals={};wanted={}
        if mode==0:
            np.testing.assert_array_equal(metadata,[len(folds),c['p'],len(names),int(estimators),int(indices),
                                                   expected_fit_calls,expected_score_calls],err_msg=str(c))
            actual_names=[next(lines) for _ in names];assert actual_names==names,(c,actual_names,names)
            actuals.update(test_score=read_matrix(lines),train_score=read_matrix(lines))
            wanted.update(test_score=want_test,train_score=want_train)
            width=int(next(lines));actuals['scalar_score']=np.fromstring(next(lines),sep=' ')
            assert len(actuals['scalar_score'])==width,c;wanted['scalar_score']=scalar
            fit_time=np.fromstring(next(lines),sep=' ');score_time=np.fromstring(next(lines),sep=' ')
            assert fit_time.shape==score_time.shape==(len(folds),) and np.all(fit_time>=0) and np.all(score_time>=0),c
            if indices:
                for train_ids,test_ids in folds:
                    tr,te=map(int,next(lines).split());actual_train=np.fromstring(next(lines),sep=' ',dtype=int)-1
                    actual_test=np.fromstring(next(lines),sep=' ',dtype=int)-1
                    np.testing.assert_array_equal(actual_train,train_ids,err_msg=str(c))
                    np.testing.assert_array_equal(actual_test,test_ids,err_msg=str(c))
                    assert (tr,te)==(len(train_ids),len(test_ids)),c
            for i,(prediction,probability,classes) in enumerate(stored):
                actuals[f'estimator_prediction_{i}']=read_matrix(lines);wanted[f'estimator_prediction_{i}']=prediction
                actuals[f'estimator_probability_{i}']=read_matrix(lines);wanted[f'estimator_probability_{i}']=probability
                width=int(next(lines));actual_classes=np.fromstring(next(lines),sep=' ',dtype=int)
                assert len(actual_classes)==width,c
                np.testing.assert_array_equal(actual_classes,classes,err_msg=str(c))
            extra=['metadata','scorer_names','indices','timing_shapes','estimator_classes']
        else:
            np.testing.assert_array_equal(metadata,[expected_fit_calls,expected_response_calls,*expected_range],err_msg=str(c))
            actuals['prediction']=read_matrix(lines);wanted['prediction']=predicted
            extra=['fit_response_counts','encoded_target_range']
        assert next(lines,None) is None
        tolerance=3e-6 if family in [2,7] else 3e-8
        for name,want in wanted.items():
            actual=actuals[name]
            if classifier and (name=='prediction' and method=='predict' or name.startswith('estimator_prediction')):
                np.testing.assert_array_equal(actual,want,err_msg=f'{c} {name}')
            else:
                np.testing.assert_allclose(actual,want,rtol=tolerance,atol=tolerance,equal_nan=True,err_msg=f'{c} {name}')
                finite=np.isfinite(actual)&np.isfinite(want)
                errors[name]=max(errors.get(name,0.),float(np.max(abs(actual[finite]-want[finite]),initial=0)))
            checks.append(dict(**c,name=name,passed=True))
        checks.extend(dict(**c,name=name,passed=True) for name in extra)
        count+=1
        if count%50==0:print(f'{count} CV mode {mode} fixtures passed',flush=True)
    source=Path(sklearn.__file__).parent;stem='cross_validate' if mode==0 else 'cross_val_predict'
    report=dict(profile=args.profile,fixtures=count,checks=checks,sklearn=sklearn.__version__,numpy=np.__version__,
                max_absolute_errors=errors,source_warnings=warning_counts,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                entry_sha256=hashlib.sha256(Path(sys.argv[0]).read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in
                               ['model_selection/_validation.py','model_selection/_split.py']})
    (ROOT/'validation/results').mkdir(exist_ok=True)
    (ROOT/'validation/results'/f'{stem}_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} {stem} groups / {count} fixtures passed')
