"""Native GridSearchCV/RandomizedSearchCV complete candidate/fold result tables."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import warnings
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import sklearn
import scipy
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.base import clone
from search_rank_certificate import certify
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from compare_function_transformer import ROOT, resolve


class AlphaDistribution:
    def rvs(self, random_state):
        return .01 + 10*random_state.uniform()


class CountedRidge(Ridge):
    fit_calls = 0
    set_calls = 0
    def fit(self, X, y, sample_weight=None):
        type(self).fit_calls += 1
        return super().fit(X, y, sample_weight=sample_weight)
    def set_params(self, **params):
        type(self).set_calls += 1
        return super().set_params(**params)


def special(estimator, x, y):
    return [np.nan, np.inf, -np.inf, 1.][round(estimator.alpha)]


def configure(family, config, random):
    models = [lambda: Ridge(solver='svd'), lambda: ElasticNet(tol=1e-11),
              lambda: LogisticRegression(tol=1e-10, max_iter=200), GaussianNB,
              lambda: KNeighborsRegressor(algorithm='brute'), lambda: KNeighborsClassifier(algorithm='brute'),
              lambda: Pipeline([('scale',StandardScaler()),('model',Ridge(solver='svd'))]),
              lambda: Pipeline([('scale',StandardScaler()),('select',SelectFromModel(Ridge(solver='svd'))),
                                ('model',Ridge(solver='svd'))]), lambda: Ridge(solver='svd'),
              lambda: CountedRidge(solver='svd')]
    grids = [dict(fit_intercept=[True,False],alpha=[.01,.3,3.]),
             dict(l1_ratio=[.25,.8],alpha=[.02,.15],fit_intercept=[True,False]),
             dict(fit_intercept=[True,False],C=[.1,1.,10.]),dict(var_smoothing=[1e-9,.05,1.]),
             dict(weights=['uniform','distance'],n_neighbors=[1,3,5]),
             dict(weights=['uniform','distance'],n_neighbors=[1,3,5]),
             dict(scale__with_mean=[True,False],model__alpha=[.03,1.,15.]),
             dict(select__threshold=['mean','median'],model__alpha=[.2,2.],select__estimator__alpha=[.1,1.]),
             dict(alpha=[0.,1.,2.,3.]),dict(fit_intercept=[True,False],alpha=[.01,.3,3.])]
    grid = grids[family]
    if family == 0 and config in [4,6]:
        grid = [grid, dict(alpha=[30.])]
    if family == 9 and random and config % 2 == 0:
        grid['alpha'] = AlphaDistribution()
    return models[family](), grid


def fixtures():
    for seed in [0,42,2026]:
        rng = np.random.RandomState(seed)
        for family in range(10):
            for config in range(9):
                n,p,m,cv = 36+config%3,4,7,3+config%2
                x = rng.normal(size=(n,p)); q = rng.normal(size=(m,p))
                yr = 2*x[:,0]-1.2*x[:,2]+.3+rng.normal(scale=.4,size=n)
                qr = 2*q[:,0]-1.2*q[:,2]+.3+rng.normal(scale=.4,size=m)
                classification = family in [2,3,5]
                if classification:
                    count = 3 if config%2 else 2
                    classes = np.array([-4,9,15])[:count]
                    def labels(z):
                        return classes[np.argmax(np.c_[z[:,0],-z[:,0]+z[:,1],z[:,2]][:,:count],axis=1)]
                    yi,qi = labels(x),labels(q)
                    qi[:count]=classes
                else:
                    yi,qi = np.rint(yr).astype(int),np.rint(qr).astype(int)
                weights = rng.uniform(.2,2,size=n);weights[:2]=0
                c = dict(seed=seed,family=family,config=config,random=config%2,n_iter=3,n=n,p=p,m=m,cv=cv,
                         weighted=int(family in [0,2,3,6,7,9] and config%2==0),
                         classification=int(classification),integer_target=int(config%3==2))
                yield c,x,q,yr,qr,yi,qi,weights


def matrix(lines):
    rows,cols = map(int,next(lines).split())
    return np.array([np.fromstring(next(lines),sep=' ') for _ in range(rows)]).reshape(rows,cols)


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
    parser.add_argument('--executable',type=Path);args=parser.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_search_driver')
    checks=[];errors={};warning_counts={};count=0;certificates=[]
    for c,x,q,yr,qr,yi,qi,weights in fixtures():
        family,config=c['family'],c['config'];classification=bool(c['classification'])
        base,grid=configure(family,config,c['random'])
        scoring = 'accuracy' if classification else 'r2'
        if config==1:scoring='neg_log_loss' if classification else 'neg_mean_squared_error'
        if config==2:scoring='roc_auc' if classification else 'neg_mean_absolute_error'
        if config==7:scoring='balanced_accuracy' if classification else 'neg_mean_absolute_error'
        if config==0:scoring=None
        refit=config not in [3,6]
        if config in [4,6]:
            scoring=dict(primary='accuracy' if classification else 'neg_mean_squared_error',
                         secondary='neg_log_loss' if classification else 'r2')
            refit='secondary' if config==4 else False
        if config==5:refit=lambda results:len(results['params'])-1
        if family==8:scoring=special
        y=yi if classification or c['integer_target'] else yr
        query_y=qi if classification or c['integer_target'] else qr
        if classification and not c['integer_target']:y=y.astype(float)
        if config in [7,8]:
            tests=[np.array([i]) if config==8 else np.arange(i,len(x),4+i+1) for i in range(4 if config==8 else 3)]
            folds=[(np.flatnonzero(~np.isin(np.arange(len(x)),test)),test) for test in tests]
        else:
            folds=list((StratifiedKFold(c['cv']) if classification else KFold(c['cv'])).split(x,y))
        options=dict(cv=folds,scoring=scoring,refit=refit,return_train_score=config%2==0,error_score='raise')
        search=(RandomizedSearchCV(base,grid,n_iter=c['n_iter'],random_state=c['seed'],**options) if c['random'] else
                GridSearchCV(base,grid,**options))
        params={}
        if c['weighted']:params['model__sample_weight' if family in [6,7] else 'sample_weight']=weights
        CountedRidge.fit_calls=0;CountedRidge.set_calls=0
        with warnings.catch_warnings(record=True) as caught,np.errstate(all='ignore'):
            warnings.simplefilter('always');search.fit(x,y,**params)
            prediction=np.empty((0,0));prob=np.empty((0,0));decision=np.empty((0,0));score=np.nan
            if refit:
                prediction=search.predict(q)[:,None]
                if classification:
                    prob=search.predict_proba(q)
                    if family==2:
                        decision=search.decision_function(q)
                        if decision.ndim==1:decision=decision[:,None]
                # Family 8 uses a single callable scorer, even for configs 4/6.
                score=search.score(q,query_y)
        expected_fit_calls=CountedRidge.fit_calls;expected_set_calls=CountedRidge.set_calls
        for warning in caught:
            key=warning.category.__name__+': '+str(warning.message)
            warning_counts[key]=warning_counts.get(key,0)+1
        header=[c[k] for k in ['family','config','seed','random','n_iter','n','p','m','cv','weighted',
                             'classification','integer_target']]
        payload=' '.join(map(str,header))+'\n'+''.join(' '.join(f'{v:.17g}' for v in np.asarray(a).ravel(order='F'))+'\n'
                                                       for a in [x,q,yr,qr,yi,qi,weights])
        run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
        assert run.returncode==0,(c,run.stdout,run.stderr)
        lines=iter(run.stdout.splitlines());metadata=np.fromstring(next(lines),sep=' ',dtype=int)
        best_and_score=np.fromstring(next(lines),sep=' ');nc,nm=metadata[4:6]
        names=[next(lines) for _ in range(nm)]
        actual=dict(mean_test=matrix(lines),std_test=matrix(lines),rank_test=matrix(lines))
        for k in range(nm):actual[f'split_test_{k}']=matrix(lines)
        if config%2==0:
            actual.update(mean_train=matrix(lines),std_train=matrix(lines))
            for k in range(nm):actual[f'split_train_{k}']=matrix(lines)
        actual.update(prediction=matrix(lines),probability=matrix(lines),decision=matrix(lines))
        fit_time=matrix(lines);score_time=matrix(lines)
        got_folds=[]
        for _ in folds:
            tr,te=map(int,next(lines).split());train=np.fromstring(next(lines),sep=' ',dtype=int)-1
            test=np.fromstring(next(lines),sep=' ',dtype=int)-1;assert len(train)==tr and len(test)==te
            got_folds.append((train,test))
        got_params=[]
        for _ in range(nc):
            setting={}
            for _ in range(int(next(lines))):
                name=next(lines);kind=int(next(lines));text=next(lines)
                setting[name]=float(text) if kind==1 else int(text) if kind==2 else bool(int(text)) if kind==3 else text
            got_params.append(setting)
        assert next(lines,None) is None
        expected_names=list(scoring) if isinstance(scoring,dict) else ['score']
        np.testing.assert_array_equal(names,expected_names,err_msg=str(c))
        wanted=dict(prediction=prediction,probability=prob,decision=decision)
        for name in ['mean_test','std_test','rank_test']+(['mean_train','std_train'] if config%2==0 else []):
            wanted[name]=np.column_stack([search.cv_results_[name+'_'+metric] for metric in expected_names])
        for k,metric in enumerate(expected_names):
            wanted[f'split_test_{k}']=np.column_stack([search.cv_results_[f'split{j}_test_{metric}'] for j in range(len(folds))])
            if config%2==0:
                wanted[f'split_train_{k}']=np.column_stack([search.cv_results_[f'split{j}_train_{metric}'] for j in range(len(folds))])
        source_rank=wanted['rank_test'].copy()
        certified_rank,witnesses=certify(actual['mean_test'],wanted['mean_test'],actual['rank_test'],source_rank,len(folds))
        wanted['rank_test']=certified_rank
        for witness in witnesses:certificates.append(dict(**c,**witness))
        selected=getattr(search,'best_index_',-1)
        if hasattr(search,'best_index_') and not callable(refit):
            metric=expected_names.index(refit) if isinstance(scoring,dict) else 0
            selected=int(np.argmin(certified_rank[:,metric]))
            if selected != search.best_index_:
                # Only the ordering changes certified above may alter selection.
                assert witnesses
                if refit:
                    chosen=clone(base).set_params(**got_params[selected]);chosen.fit(x,y,**params)
                    search.best_estimator_=chosen
                    wanted['prediction']=chosen.predict(q)[:,None]
                    if classification:
                        wanted['probability']=chosen.predict_proba(q)
                        if family==2:
                            response=chosen.decision_function(q)
                            wanted['decision']=response[:,None] if response.ndim==1 else response
                    score=search.score(q,query_y)
            search.best_score_=wanted['mean_test'][selected,metric]
        expected_metadata=[selected+1,int(hasattr(search,'best_score_')),len(folds),c['p'],
                           len(search.cv_results_['params']),len(expected_names),int(search.multimetric_),
                           expected_fit_calls if family==9 else 0,expected_set_calls if family==9 else 0]
        np.testing.assert_array_equal(metadata,expected_metadata,err_msg=str(c))
        tolerance=3e-6 if family==2 else 3e-8
        for name,want in wanted.items():
            got=actual[name]
            if name in ['rank_test'] or name=='prediction' and classification:
                np.testing.assert_array_equal(got,want,err_msg=f'{c} {name}')
            else:
                np.testing.assert_allclose(got,want,rtol=tolerance,atol=tolerance,equal_nan=True,err_msg=f'{c} {name}')
                finite=np.isfinite(got)&np.isfinite(want)
                errors[name]=max(errors.get(name,0.),float(np.max(abs(got[finite]-want[finite]),initial=0)))
            checks.append(dict(**c,name=name,passed=True))
        np.testing.assert_allclose(best_and_score,[getattr(search,'best_score_',np.nan),score],rtol=tolerance,
                                   atol=tolerance,equal_nan=True,err_msg=str(c))
        for (got_train,got_test),(train,test) in zip(got_folds,folds):
            np.testing.assert_array_equal(got_train,train,err_msg=str(c));np.testing.assert_array_equal(got_test,test,err_msg=str(c))
        assert len(got_params)==len(search.cv_results_['params']),c
        for got,want in zip(got_params,search.cv_results_['params']):
            assert got.keys()==want.keys(),(c,got,want)
            for name,value in want.items():
                if isinstance(value,(float,np.floating)):
                    np.testing.assert_allclose(got[name],value,atol=1e-14,rtol=1e-14,err_msg=str(c))
                else:assert got[name]==value,(c,got,want)
        assert fit_time.shape==score_time.shape==(nc,len(folds)) and np.all(fit_time>=0) and np.all(score_time>=0),c
        for name in ['metadata','best_score_and_score','folds','parameters','feature_metric_names','timing_shapes']:
            checks.append(dict(**c,name=name,passed=True))
        count+=1
        if count%50==0:print(f'{count} search fixtures passed',flush=True)
    source=Path(sklearn.__file__).parent
    report=dict(profile=args.profile,fixtures=count,checks=checks,sklearn=sklearn.__version__,numpy=np.__version__,
                scipy=scipy.__version__,rank_source_sha256=hashlib.sha256((Path(scipy.__file__).parent/'stats/_stats_py.py').read_bytes()).hexdigest(),
                max_absolute_errors=errors,source_warnings=warning_counts,rank_certificates=certificates,
                certificate_sha256=hashlib.sha256((ROOT/'validation/search_rank_certificate.py').read_bytes()).hexdigest(),native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in
                               ['model_selection/_search.py','model_selection/_validation.py','model_selection/_split.py']})
    (ROOT/'validation/results').mkdir(exist_ok=True)
    (ROOT/'validation/results'/f'search_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} search groups / {count} fixtures passed')


if __name__=='__main__':main()
