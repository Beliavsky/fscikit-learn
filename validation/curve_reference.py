"""Direct sklearn reference engine for learning/validation curves and permutation tests."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import subprocess
import warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import sklearn
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.model_selection import learning_curve,validation_curve,permutation_test_score,KFold,StratifiedKFold
from sklearn.linear_model import Ridge,ElasticNet,LogisticRegression,SGDRegressor,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from compare_function_transformer import ROOT,resolve

trace=[]
fit_calls=partial_calls=set_calls=score_calls=0
trace_partial=False


def record(x,y,weights):
    y=np.asarray(y) if y is not None else None
    trace.append([len(x),0. if y is None else float(np.sum(y)),
                  0. if y is None else float(np.dot(x[:,0],y)),0 if weights is None else len(weights)])


class Spy:
    def fit(self,X,y=None,**params):
        global fit_calls
        fit_calls+=1;record(X,y,params.get('sample_weight',params.get('model__sample_weight')))
        return super().fit(X,y,**params)
    def partial_fit(self,X,y=None,*args,**params):
        global partial_calls
        if trace_partial:
            partial_calls+=1;record(X,y,args[0] if args else params.get('sample_weight'))
        return super().partial_fit(X,y,*args,**params)
    def set_params(self,**params):
        global set_calls
        set_calls+=1
        return super().set_params(**params)


class SpyRidge(Spy,Ridge):pass
class SpyElastic(Spy,ElasticNet):pass
class SpyLogistic(Spy,LogisticRegression):pass
class SpyBayes(Spy,GaussianNB):pass
class SpyKNR(Spy,KNeighborsRegressor):pass
class SpyKNC(Spy,KNeighborsClassifier):pass
class SpySGDR(Spy,SGDRegressor):pass
class SpySGDC(Spy,SGDClassifier):pass
class SpyScale(Spy,StandardScaler):pass
class SpyPipeline(Spy,Pipeline):pass


class Counter(RegressorMixin,BaseEstimator):
    def __init__(self,offset=0.):self.offset=offset
    def fit(self,X,y=None,sample_weight=None):
        self.fits_=getattr(self,'fits_',0)+1;return self
    def partial_fit(self,X,y=None,sample_weight=None,classes=None):
        self.fits_=getattr(self,'fits_',0)+1;return self


class SpyCounter(Spy,Counter):pass


def mean_error(model,x,y):
    global score_calls
    score_calls+=1;return float(np.mean(model.predict(x)-y))


def scaled_moment(model,x,y=None):
    global score_calls
    score_calls+=1;z=model.transform(x);return float(np.mean(z*z))


def counter_score(model,x,y):
    global score_calls
    score_calls+=1;return model.fits_+.001*model.offset


def nonfinite_score(model,x,y):
    global score_calls
    score_calls+=1;return np.inf if len(x)%2==0 else np.nan


def constructor(family):
    constructors=[lambda:SpyRidge(alpha=.4,solver='svd'),lambda:SpyElastic(alpha=.08,l1_ratio=.4,tol=1e-11),
                  lambda:SpyLogistic(C=.8,tol=1e-10,max_iter=200),lambda:SpyBayes(var_smoothing=.05),
                  lambda:SpyKNR(n_neighbors=1,algorithm='brute'),lambda:SpyKNC(n_neighbors=1,algorithm='brute'),
                  lambda:SpyPipeline([('scale',StandardScaler()),('model',Ridge(alpha=.4,solver='svd'))]),
                  lambda:SpySGDR(learning_rate='constant',eta0=.02,max_iter=6,tol=None,shuffle=False,random_state=0),
                  lambda:SpySGDC(loss='log_loss',learning_rate='constant',eta0=.02,max_iter=6,tol=None,
                                shuffle=False,random_state=0),SpyScale,SpyCounter]
    return constructors[family]()


def fixtures(mode):
    for seed in [0,42,2026]:
        rng=np.random.RandomState(seed)
        for family in range(11):
            if mode==2 and family==9:continue
            increments=[0,1] if mode==0 and family in [3,7,8,10] else [0]
            for incremental in increments:
                for config in range(8):
                    n,p,ncv=36+config%2,4,3+config%2
                    x=rng.normal(size=(n,p));yr=2*x[:,0]-x[:,2]+.3+rng.normal(scale=.2,size=n)
                    classifier=family in [2,3,5,8]
                    yi=np.resize([-4,9,15] if config==6 else [-4,9],n) if classifier else np.rint(yr).astype(int)
                    weights=rng.uniform(.2,2,size=n);groups=np.arange(n)//3
                    weighted=int(family in [0,2,3,6,7,8,10] and config%2==1 and (not incremental or family==10))
                    c=dict(seed=seed,mode=mode,family=family,config=config,n=n,p=p,cv=ncv,
                           weighted=weighted,integer_target=int(config%3==2),incremental=incremental,
                           nperm=[1,3,5,9][config%4],grouped=int(config%2==1))
                    yield c,x,yr,yi,weights,groups


def matrix(lines):
    rows,cols=map(int,next(lines).split())
    return np.array([np.fromstring(next(lines),sep=' ') for _ in range(rows)]).reshape(rows,cols)


def run(mode):
    global trace,fit_calls,partial_calls,set_calls,score_calls,trace_partial
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
    parser.add_argument('--executable',type=Path);args=parser.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_curve_driver')
    checks=[];errors={};warning_counts={};count=0
    for c,x,yr,yi,weights,groups in fixtures(mode):
        family,config=c['family'],c['config'];incremental=bool(c['incremental'])
        classifier=family in [2,3,5,8];no_target=family==9
        y=None if no_target else yi if classifier or c['integer_target'] else yr
        if classifier and not c['integer_target']:y=y.astype(float)
        model=constructor(family)
        if config==6 and incremental and family in [3,7,8]:model.fit(x,y)
        if config==4:
            folds=[(np.arange(len(x)-6),np.arange(len(x)-6,len(x))),(np.arange(12),np.arange(12,len(x)))]
        elif config==5:
            folds=[(np.flatnonzero(np.arange(len(x))%3!=i),np.arange(i,len(x),3)) for i in range(3)]
        else:
            folds=list((StratifiedKFold(c['cv']) if classifier else KFold(c['cv'])).split(x,y))
        scoring='accuracy' if classifier else 'r2' if config%3==0 else 'neg_mean_squared_error'
        if config==3:scoring=mean_error
        if no_target:scoring=scaled_moment
        if family==10:scoring=counter_score
        if config==7 and family==0:scoring=nonfinite_score
        params={}
        if c['weighted']:params['model__sample_weight' if family==6 else 'sample_weight']=weights
        sizes={0:np.linspace(.1,1,5),1:np.array([1.,.5,.5,.05]),2:np.array([12,3,12,18]),
               3:np.array([.25,.6,1.]),4:np.array([5,15,len(x)-6]),
               5:np.array([.3,.7,1.]),6:np.array([4,12,18]),7:np.array([1e-8,.5,1.])}[config]
        parameter_name='alpha';param_range=[.4,.01,3.,.4]
        if family==2:parameter_name='C';param_range=[.1,1.,10.,.1]
        if family==3:parameter_name='var_smoothing';param_range=[.01,.05,.3,.01]
        if family in [4,5]:parameter_name='n_neighbors';param_range=[1,3,5,1]
        if family==6:parameter_name='model__alpha'
        if family==9:parameter_name='with_mean';param_range=[True,False,True]
        if family==10:parameter_name='offset'
        trace=[];fit_calls=partial_calls=set_calls=score_calls=0
        trace_partial=incremental and (config%2==1 or family==10)
        source_failure=None
        with warnings.catch_warnings(record=True) as caught,np.errstate(all='ignore'):
            warnings.simplefilter('always')
            if mode==0:
                try:
                    want=learning_curve(model,x,y,train_sizes=sizes,cv=folds,scoring=scoring,
                                        exploit_incremental_learning=incremental,shuffle=config%2==1,random_state=c['seed'],
                                        return_times=config%2==0,params=params,error_score='raise')
                except ValueError as error:
                    source_failure=str(error)
                    assert ('at least 2 classes' in source_failure or 'greater than one' in source_failure or
                            '0 sample' in source_failure or 'n_samples=0' in source_failure), (c,source_failure)
                if source_failure is None:
                    planned=[(train.copy(),test.copy()) for train,test in folds]
                    if config%2==1:
                        shuffler=np.random.RandomState(c['seed'])
                        planned=[(shuffler.permutation(train),test) for train,test in planned]
                    wanted=dict(train=want[1],test=want[2],actual_sizes=np.minimum(want[0][:,None],
                                np.array([len(train) for train,test in folds])[None,:]))
            elif mode==1:
                want=validation_curve(model,x,y,param_name=parameter_name,param_range=param_range,cv=folds,
                                      scoring=scoring,params=params,error_score='raise')
                wanted=dict(train=want[0],test=want[1]);planned=folds
            else:
                # Use the splitter itself for default stratification so folds regenerate for each permuted y.
                splitter=folds if config in [4,5] else StratifiedKFold(c['cv']) if classifier else KFold(c['cv'])
                want=permutation_test_score(model,x,y,groups=groups if c['grouped'] else None,cv=splitter,
                                            n_permutations=c['nperm'],random_state=c['seed'],scoring=scoring,params=params)
                wanted=dict(score_and_pvalue=np.array([want[0],want[2]]),permutation_scores=want[1])
        for warning in caught:
            key=warning.category.__name__+': '+str(warning.message)
            warning_counts[key]=warning_counts.get(key,0)+1
        header=[c[k] for k in ['mode','family','config','n','p','cv','weighted','integer_target','incremental','seed','nperm','grouped']]
        payload=' '.join(map(str,header))+'\n'+''.join(' '.join(f'{v:.17g}' for v in np.asarray(a).ravel(order='F'))+'\n'
                                                       for a in [x,yr,yi,weights,groups])
        native=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
        if source_failure is not None:
            assert native.returncode!=0 and 'fscikit-learn:' in native.stdout,(c,source_failure,native.stdout,native.stderr)
            expected='X must be nonempty' if '0 sample' in source_failure or 'n_samples=0' in source_failure else 'classes'
            assert expected in native.stdout,(c,source_failure,native.stdout)
            checks.append(dict(**c,name='source_rejection',source_error=source_failure,passed=True))
            count+=1
            continue
        assert native.returncode==0,(c,native.stdout,native.stderr)
        lines=iter(native.stdout.splitlines());metadata=np.fromstring(next(lines),sep=' ',dtype=int)
        actual={};extra=[];init_calls=len(folds) if incremental and family==10 else 0
        if mode==0:
            np.testing.assert_array_equal(metadata,[len(want[0]),len(folds),int(config%2==0),fit_calls,partial_calls,
                                                   score_calls,init_calls,set_calls],err_msg=str(c))
            np.testing.assert_array_equal(np.fromstring(next(lines),sep=' ',dtype=int),want[0],err_msg=str(c))
            actual=dict(train=matrix(lines),test=matrix(lines),actual_sizes=matrix(lines))
            if config%2==0:
                fit_time,score_time=matrix(lines),matrix(lines)
                assert fit_time.shape==score_time.shape==want[1].shape and np.all(fit_time>=0) and np.all(score_time>=0),c
            extra=['metadata','train_sizes','timing_presence']
        elif mode==1:
            np.testing.assert_array_equal(metadata,[len(param_range),len(folds),fit_calls,partial_calls,score_calls,
                                                   init_calls,set_calls],err_msg=str(c))
            actual=dict(train=matrix(lines),test=matrix(lines));extra=['metadata','parameter_order']
        else:
            np.testing.assert_array_equal(metadata,[c['nperm'],fit_calls,partial_calls,score_calls,init_calls,set_calls],err_msg=str(c))
            actual['score_and_pvalue']=np.fromstring(next(lines),sep=' ')
            actual['permutation_scores']=np.fromstring(next(lines),sep=' ');extra=['metadata']
        if mode in [0,1]:
            for train,test in planned:
                tr,te=map(int,next(lines).split());got_train=np.fromstring(next(lines),sep=' ',dtype=int)-1
                got_test=np.fromstring(next(lines),sep=' ',dtype=int)-1
                assert (tr,te)==(len(train),len(test)),c
                np.testing.assert_array_equal(got_train,train,err_msg=str(c));np.testing.assert_array_equal(got_test,test,err_msg=str(c))
            extra.append('folds')
        actual['fit_trace']=matrix(lines);wanted['fit_trace']=np.array(trace).reshape(-1,4)
        assert next(lines,None) is None
        tolerance=3e-6 if family==2 else 3e-8
        for name,expected in wanted.items():
            got=actual[name]
            np.testing.assert_allclose(got,expected,rtol=tolerance,atol=tolerance,equal_nan=True,err_msg=f'{c} {name}')
            if mode==2 and name=='score_and_pvalue':assert got[1]==expected[1],(c,got,expected)
            finite=np.isfinite(got)&np.isfinite(expected)
            errors[name]=max(errors.get(name,0.),float(np.max(abs(got[finite]-expected[finite]),initial=0)))
            checks.append(dict(**c,name=name,passed=True))
        checks.extend(dict(**c,name=name,passed=True) for name in extra)
        count+=1
        if count%50==0:print(f'{count} curve mode {mode} fixtures passed',flush=True)
    stem=['learning_curve','validation_curve','permutation_test_score'][mode];source=Path(sklearn.__file__).parent
    report=dict(profile=args.profile,fixtures=count,checks=checks,sklearn=sklearn.__version__,numpy=np.__version__,
                max_absolute_errors=errors,source_warnings=warning_counts,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                entry_sha256=hashlib.sha256(Path(sys.argv[0]).read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in
                               ['model_selection/_validation.py','model_selection/_split.py']})
    (ROOT/'validation/results').mkdir(exist_ok=True)
    (ROOT/'validation/results'/f'{stem}_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} {stem} groups / {count} fixtures passed')
