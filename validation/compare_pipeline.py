"""Sequential pipeline ordering, supervised transforms and bidirectional nesting."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn import preprocessing as pp,impute,decomposition,feature_selection as fs
from sklearn.linear_model import LinearRegression,Ridge,LogisticRegression,PoissonRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from compare_function_transformer import ROOT,resolve
from pipeline_knn_certificate import certify

COUNTS=[0,0]

def affine(x):COUNTS[0]+=1;return 2*np.asarray(x)+3

def undo(x):COUNTS[1]+=1;return (np.asarray(x)-3)/2

def expand(x):
    COUNTS[0]+=1;x=np.asarray(x);return np.c_[x,x*x]

def expand_names(model,input):return [str(x)+'_original' for x in input]+[str(x)+'_square' for x in input]


def configure(task,kind,config):
    standard=pp.StandardScaler();minmax=pp.MinMaxScaler(feature_range=(-1,2))
    principal=decomposition.PCA(n_components=2,whiten=config%2==0,svd_solver='full')
    score=fs.f_classif if task==2 and config<2 else fs.f_regression
    function=pp.FunctionTransformer(affine,undo,feature_names_out='one-to-one')
    variants=[lambda:[('scale',standard)],
              lambda:[('impute',impute.SimpleImputer(strategy='median')),('scale',standard)],
              lambda:[('pca',principal),('range',minmax)],
              lambda:[('select',fs.SelectKBest(score_func=score,k=2)),('scale',standard)],
              lambda:[('select',fs.SelectPercentile(score_func=score,percentile=50)),('scale',standard)],
              lambda:[('encode',pp.TargetEncoder(target_type='auto' if task==2 else 'continuous',
                                               smooth=1,cv=3,random_state=42)),('scale',standard)],
              lambda:[('columns',ColumnTransformer([('processed',Pipeline([
                  ('impute',impute.SimpleImputer(strategy='median')),('scale',pp.StandardScaler())]),[2,0])],
                  remainder='passthrough',sparse_threshold=0)),('scale',standard)],
              lambda:[('union',FeatureUnion([('latent',Pipeline([('scale',pp.StandardScaler()),
                  ('pca',decomposition.PCA(n_components=1,whiten=config%2==0,svd_solver='full'))])),
                  ('raw','passthrough')],transformer_weights={'raw':.5})),('scale',standard)],
              lambda:[('nested',Pipeline([('scale',pp.StandardScaler()),('range',minmax)])),('affine',function)],
              lambda:[('expand',pp.FunctionTransformer(expand,feature_names_out=expand_names)),
                      ('vary',fs.VarianceThreshold()),('scale',standard)],
              lambda:[('off','passthrough'),('robust',pp.RobustScaler()),
                      ('quantile',pp.QuantileTransformer(n_quantiles=7,subsample=None))],
              lambda:[('affine',function),('range',minmax)],
              lambda:[('ordinal',pp.OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)),
                      ('onehot',pp.OneHotEncoder(handle_unknown='ignore',sparse_output=False))]]
    stages=[('off','passthrough')] if kind>=13 else variants[kind]()
    if task==0:tail='passthrough'
    elif task==1:
        tail=[LinearRegression(),Ridge(alpha=.3,solver='svd'),KNeighborsRegressor(n_neighbors=3,weights='distance',algorithm='brute'),
              LinearRegression()][config]
    elif task==2:
        tail=[GaussianNB(),LogisticRegression(C=.7,tol=1e-10,max_iter=300),
              KNeighborsClassifier(n_neighbors=3,weights='distance',algorithm='brute'),GaussianNB()][config]
    elif task==3:tail=TransformedTargetRegressor(regressor=Ridge(alpha=.3,solver='svd'))
    else:tail=PoissonRegressor(alpha=.3,tol=1e-10,max_iter=300)
    if config==3 and task>0:
        first=('select',fs.SelectKBest(score_func=fs.f_classif,k=2)) if kind==14 else ('identity',pp.FunctionTransformer(feature_names_out='one-to-one'))
        tail=Pipeline([first,('estimator',tail)])
    return Pipeline(stages+[('final',tail)])


def fixtures():
    for seed in [0,42,2026]:
        rng=np.random.RandomState(seed)
        for task in range(5):
            for kind in range(13):
                if task==3 and kind in [3,4,5]:continue
                for config in range(4):
                    n=[51,80,91,63][config];p=4;m=7
                    x=rng.normal(size=(n,p));q=rng.normal(size=(m,p))*.7
                    if kind in [5,12]:x=np.round(x);q=np.round(q)
                    yr=np.exp(.4+.2*x[:,0]-.15*x[:,1])+rng.uniform(.01,.1,n)
                    order=np.argsort(x[:,0]+.4*x[:,1]);groups=np.empty(n,dtype=int)
                    groups[order]=np.arange(n)*(2+int(config%2==1 and seed!=42))//n;yi=groups*3-2
                    ym=np.c_[yr,2*yr+rng.normal(size=n)*.1]
                    weights=rng.uniform(.2,2,n);weights[:5]=0
                    if kind in [1,6]:x[1,0]=np.nan;x[3,2]=np.nan;q[2,0]=np.nan;q[4,2]=np.nan
                    if kind==9 and config==0:x[:,0]=1;q[:,0]=1
                    c=dict(seed=seed,task=task,kind=kind,config=config,n=n,p=p,m=m,named=config%2,
                           weighted=int(task in [1,2,3] and config in [1,3]),real_class=int(task==2 and config>=2))
                    yield c,x,q,yr,yi,ym,weights

    # Independent named-handoff fixtures leave the original random streams intact.
    for seed in [0,42,2026]:
        rng=np.random.RandomState(seed+10013)
        for task in [1,2,3]:
            n=63;p=4;m=7;x=rng.normal(size=(n,p));q=rng.normal(size=(m,p))*.7
            yr=np.exp(.4+.2*x[:,0]-.15*x[:,1])+rng.uniform(.01,.1,n)
            groups=np.empty(n,dtype=int);groups[np.argsort(x[:,0]+.4*x[:,1])]=np.arange(n)*3//n;yi=groups*3-2
            ym=np.c_[yr,2*yr+rng.normal(size=n)*.1];weights=rng.uniform(.2,2,n);weights[:5]=0
            c=dict(seed=seed,task=task,kind=13,config=3,n=n,p=p,m=m,named=1,weighted=1,real_class=int(task==2))
            yield c,x,q,yr,yi,ym,weights

    for seed in [0,42,2026]:
        rng=np.random.RandomState(seed+10014);n=63;p=4;m=7
        x=rng.normal(size=(n,p));q=rng.normal(size=(m,p))*.7
        groups=np.empty(n,dtype=int);groups[np.argsort(x[:,0]+.4*x[:,1])]=np.arange(n)*3//n;yi=groups*3-2
        yr=yi.astype(float);ym=np.c_[yr,2*yr+rng.normal(size=n)*.1]
        weights=rng.uniform(.2,2,n);weights[:5]=0
        c=dict(seed=seed,task=1,kind=14,config=3,n=n,p=p,m=m,named=1,weighted=1,real_class=0)
        yield c,x,q,yr,yi,ym,weights


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);args=ap.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_pipeline_driver')
    checks=[];count=0;errors={};warning_counts={};certificates=[]
    for c,x,q,yr,yi,ym,weights in fixtures():
        task=c['task'];config=c['config'];kind=c['kind'];n=len(x)
        features=['field_'+str(i+1) for i in range(c['p'])]
        data=pd.DataFrame(x,columns=features) if c['named'] else x
        queries=pd.DataFrame(q,columns=features) if c['named'] else q
        y=yi.astype(float) if c['real_class'] else yi
        if task!=2:y=ym if task==3 else yr
        if kind==14:y=yi
        graph=configure(task,kind,config);COUNTS[:]=[0,0]
        with warnings.catch_warnings(record=True) as caught,np.errstate(all='ignore'):
            warnings.simplefilter('always')
            if task==0:
                if config%2:fitted=np.asarray(graph.fit_transform(data,y))
                else:graph.fit(data,y);fitted=np.asarray(graph.transform(data))
                prefix=graph
            else:
                route='final__estimator__sample_weight' if config==3 else 'final__sample_weight'
                graph.fit(data,y,**({route:weights} if c['weighted'] else {}))
                prefix=graph[:-1];fitted=np.asarray(prefix.transform(data))
            train=np.asarray(prefix.transform(data));query=np.asarray(prefix.transform(queries))
            proba=np.empty((0,0));decision=np.empty((0,0))
            if task==0:
                predicted=np.asarray(graph.transform(queries));training_prediction=np.asarray(graph.transform(data));scores=[]
            else:
                predicted=np.asarray(graph.predict(queries)).reshape((len(q),-1))
                training_prediction=np.asarray(graph.predict(data)).reshape((n,-1))
                if task==2:
                    proba=graph.predict_proba(queries)
                    if config==1:decision=np.asarray(graph.decision_function(queries)).reshape((len(q),-1))
                    scores=[graph.score(data,yi),graph.score(data,yi,sample_weight=weights)]
                elif task==4:scores=[graph.score(data,yr)]
                else:scores=[graph.score(data,y),graph.score(data,y,sample_weight=weights)]
            names=list(prefix.get_feature_names_out(features)) if kind>=13 else list(prefix.get_feature_names_out())
            inverse=np.asarray(prefix.inverse_transform(train)) if kind in [0,2,3,4,8,9,10,11,13,14] else np.empty((0,0))
            classes=np.asarray(graph.classes_,dtype=int) if task==2 else np.empty(0,dtype=int)
            expected_counts=COUNTS.copy()
        for w in caught:warning_counts[str(w.message)]=warning_counts.get(str(w.message),0)+1
        header=[task,kind,config,n,c['p'],len(q),c['named'],c['weighted'],c['real_class']]
        payload='\n'.join([' '.join(map(str,header)),*[' '.join(format(v,'.17g') for v in a.ravel(order='F'))
                        for a in [x,q,yr,yi,ym,weights]]])+'\n'
        result=subprocess.run([str(exe)],input=payload,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert result.returncode==0,(c,result.stdout)
        lines=iter(result.stdout.splitlines());meta=np.fromstring(next(lines),sep=' ',dtype=int)
        np.testing.assert_array_equal(meta,[c['p'],0 if task==0 else (1 if task==2 else 2),2 if task==3 else 1,
                                          *expected_counts,len(classes)],err_msg=str(c))
        actual_classes=np.fromstring(next(lines),sep=' ',dtype=int)
        np.testing.assert_array_equal(actual_classes,classes,err_msg=str(c))
        checks.append(dict(**c,name='metadata_and_calls',passed=True))
        expected=[('fit_features',fitted),('training_features',train),('query_features',query),('prediction',predicted),
                  ('training_prediction',training_prediction),('probability',proba),('decision',decision),('inverse',inverse)]
        actuals={}
        for name,want in expected:
            shape=tuple(map(int,next(lines).split()));assert shape==want.shape,(c,name,shape,want.shape)
            actuals[name]=np.fromstring(next(lines),sep=' ').reshape(shape,order='F')
        size=int(next(lines));assert size==len(scores),(c,size,scores)
        actual_scores=np.fromstring(next(lines),sep=' ')
        actual_names=[next(lines) for _ in range(int(next(lines)))];assert actual_names==names,(c,actual_names,names)
        proof=int(next(lines));oracle='sklearn'
        tolerance=3e-6 if task==4 or task==2 and config==1 else 3e-8
        expected=dict(expected)
        if proof:
            assert config==2 and task in [1,2]
            def matrix():
                shape=tuple(map(int,next(lines).split()))
                return np.fromstring(next(lines),sep=' ').reshape(shape,order='F')
            native_fit=matrix();native_targets=matrix();proofs=[(matrix(),matrix()),(matrix(),matrix())]
            tail=graph.steps[-1][1]
            np.testing.assert_allclose(native_fit,tail._fit_X,rtol=3e-10,atol=3e-10,err_msg=str(c))
            np.testing.assert_array_equal(native_targets.ravel(),y,err_msg=str(c))
            checks.append(dict(**c,name='neighbor_training_data',passed=True))
            mismatch=any(not np.allclose(actuals[name],expected[name],rtol=tolerance,atol=tolerance,equal_nan=True)
                         for name in ['prediction','training_prediction','probability'])
            if mismatch:
                forced,outputs,witnesses=certify(tail,native_fit,native_targets,
                    [actuals['query_features'],actuals['training_features']],
                    [query,train],proofs,y,task==2)
                expected['prediction']=outputs[0][0];expected['training_prediction']=outputs[1][0]
                if task==2:expected['probability']=outputs[0][1]
                scores=[forced.score(outputs[1][2],y),forced.score(outputs[1][2],y,sample_weight=weights)]
                certificates.append(dict(**c,witnesses=witnesses))
                oracle='source_knn_geometry_and_aggregation_certificate'
        else:assert not(config==2 and task in [1,2])
        if config==3 and task>0:
            actual_nested_names=[next(lines) for _ in range(int(next(lines)))]
            expected_nested_names=list(getattr(graph.steps[-1][1],'feature_names_in_',[]))
            assert actual_nested_names==expected_nested_names,(c,actual_nested_names,expected_nested_names)
            checks.append(dict(**c,name='final_pipeline_input_names',passed=True))
        if kind==14:
            assert int(next(lines))==1
            forwarded_scores=np.fromstring(next(lines),sep=' ')
            np.testing.assert_allclose(forwarded_scores,graph.steps[-1][1].steps[0][1].scores_,rtol=3e-9,atol=3e-9)
            checks.append(dict(**c,name='integer_regression_target_forwarding',passed=True))
        for name,want in expected.items():
            actual=actuals[name]
            if task==2 and name in ['prediction','training_prediction']:
                np.testing.assert_array_equal(actual,want,err_msg=str(c))
            else:
                np.testing.assert_allclose(actual,want,rtol=tolerance,atol=tolerance,equal_nan=True,err_msg=f'{c} {name}')
            finite=np.isfinite(actual)&np.isfinite(want)
            errors[name]=max(errors.get(name,0.),float(np.max(abs(actual[finite]-want[finite]),initial=0)))
            checks.append(dict(**c,name=name,passed=True,
                               oracle=oracle if name in ['prediction','training_prediction','probability'] else 'sklearn'))
        np.testing.assert_allclose(actual_scores,scores,rtol=tolerance,atol=tolerance,equal_nan=True,err_msg=str(c))
        checks.append(dict(**c,name='scores',passed=True,oracle=oracle))
        checks.append(dict(**c,name='feature_names',passed=True));assert next(lines,None) is None
        count+=1
        if count%100==0:print(f'{count} pipeline fixtures passed',flush=True)
    root=Path(sklearn.__file__).parent
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,
                fixtures=count,checks=checks,max_absolute_errors=errors,source_warnings=warning_counts,
                knn_certificates=certificates,certificate_sha256=hashlib.sha256((ROOT/'validation/pipeline_knn_certificate.py').read_bytes()).hexdigest(),
                native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                               ['pipeline.py','preprocessing/_target_encoder.py','feature_selection/_univariate_selection.py',
                                'neighbors/_base.py','neighbors/_regression.py','neighbors/_classification.py']})
    (ROOT/'validation/results').mkdir(exist_ok=True)
    (ROOT/'validation/results'/f'pipeline_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {count} pipeline fixtures passed')

if __name__=='__main__':main()
