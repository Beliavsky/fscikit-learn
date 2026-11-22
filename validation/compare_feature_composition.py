"""Dense composition reference engine; wrappers select ColumnTransformer/FeatureUnion."""
from pathlib import Path
import argparse
import hashlib
import json
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import subprocess
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn import preprocessing as pp
from sklearn import impute,decomposition,feature_selection
from compare_function_transformer import ROOT,resolve


def affine(x):return 2*np.asarray(x)+3

def expand(x):
    x=np.asarray(x);return np.c_[x,x*x]

def expanded_names(model,input):return [str(n)+'_original' for n in input]+[str(n)+'_square' for n in input]


def make_model(kind,config):
    models=[lambda:pp.FunctionTransformer(affine,feature_names_out='one-to-one'),
            lambda:pp.StandardScaler(with_mean=config%2==0),
            lambda:pp.MinMaxScaler(feature_range=(-2,3)),lambda:pp.MaxAbsScaler(),
            lambda:pp.RobustScaler(with_centering=config%2==0),lambda:impute.SimpleImputer(strategy='median'),
            lambda:pp.OneHotEncoder(handle_unknown='ignore',sparse_output=False),
            lambda:pp.OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),
            lambda:impute.KNNImputer(n_neighbors=3,keep_empty_features=True,add_indicator=True),
            lambda:impute.MissingIndicator(error_on_new=False),
            lambda:pp.PowerTransformer(standardize=config%2==0),
            lambda:pp.QuantileTransformer(n_quantiles=7,subsample=None),
            lambda:pp.KBinsDiscretizer(n_bins=3,strategy='uniform',encode='ordinal' if config%2==0 else 'onehot-dense'),
            lambda:pp.SplineTransformer(n_knots=3,degree=2,include_bias=config%2==0),
            lambda:decomposition.PCA(n_components=1,whiten=config%2==0,svd_solver='full'),
            lambda:decomposition.TruncatedSVD(n_components=1,random_state=0,n_iter=20),
            lambda:feature_selection.VarianceThreshold(),
            lambda:FeatureUnion([('center',pp.StandardScaler()),('raw','passthrough')]),
            lambda:ColumnTransformer([('first',pp.StandardScaler(),[0])],remainder='passthrough'),
            lambda:pp.FunctionTransformer(expand,feature_names_out=expanded_names)]
    return models[kind]()


def fixtures(family):
    for seed in [0,42,2026]:
        rng=np.random.RandomState(seed)
        for kind in range(20):
            for config in range(6 if family==0 else 4):
                n=[7,19,31][config%3];p=5;m=6
                x=rng.uniform(.25,4,size=(n,p));q=rng.uniform(.25,4,size=(m,p))
                if kind in [6,7]:
                    if kind==7 or config in [0,1]:x=np.round(x,1);q=np.round(q,1)
                    # Include small/large exponents and negative zero in numeric names.
                    if config==0:x[:,0]=np.resize([0.,-.0,1e-7,1e16,np.finfo(float).max,np.nextafter(0.,1.),np.pi],n)
                if kind in [5,8,9]:
                    x[1,0]=np.nan;x[3,2]=np.nan;q[0,0]=np.nan;q[2,2]=np.nan
                    if kind==8 and config==3:x[:,2]=np.nan
                if kind==16:x[:,0]=2;q[:,0]=2
                c=dict(seed=seed,kind=kind,config=config,n=n,p=p,m=m,operation=config%3,named=config%2)
                yield c,x,q


def run(family):
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);args=ap.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_composition_driver')
    checks=[];count=0;errors={};warning_counts={}
    for c,x,q in fixtures(family):
        p=c['p'];kind=c['kind'];config=c['config'];named=c['named'];operation=c['operation']
        features=['field_'+str(i+1) for i in range(p)]
        data=pd.DataFrame(x,columns=features) if named else x
        query_data=pd.DataFrame(q,columns=features) if named else q
        model=make_model(kind,config)
        with warnings.catch_warnings(record=True) as caught,np.errstate(all='ignore'):
            warnings.simplefilter('always')
            if operation==2:
                selected=data.iloc[:,[2,0]] if named else x[:,[2,0]]
                model.fit(selected if family==0 else data)
            calls=[0]
            if family==0:
                def selector(X):calls[0]+=1;return [2,0]
                columns=[[2,0],[True,False,True,False,False],features[2::-2] if named else [2,0],selector][config%4]
                remainder=['drop','passthrough',pp.MinMaxScaler(feature_range=(-1,2))][config%3]
                graph=ColumnTransformer([('main',model,columns),('raw','passthrough',[2,0]),
                                        ('ignored','drop',[4]),('empty',pp.StandardScaler(),[])],
                                       remainder=remainder,sparse_threshold=0,
                                       transformer_weights=dict(main=-.75,raw=2,remainder=.5))
                if config==4:
                    graph.verbose_feature_names_out=False
                    graph.transformers[1]=('raw','passthrough',[3,1])
                if config==5:graph.verbose_feature_names_out=lambda branch,feature:branch+'::'+feature
            else:
                graph=FeatureUnion([('main',model),('raw','passthrough'),('ignored','drop')],
                                   transformer_weights=dict(main=-.75,raw=2))
                if config==3 and kind in [9,14,15,17,18,19]:graph.verbose_feature_names_out=False
            if operation==1:graph.fit(data);fitted=np.asarray(graph.transform(data))
            else:fitted=np.asarray(graph.fit_transform(data))
            train=np.asarray(graph.transform(data))
            if family==0 and named and config%2:
                query_data=pd.DataFrame(np.c_[np.full(len(q),99),q[:,[2,0,3,1]]],
                                        columns=['new',features[2],features[0],features[3],features[1]])
            query=np.asarray(graph.transform(query_data));names=list(graph.get_feature_names_out())
        for w in caught:warning_counts[str(w.message)]=warning_counts.get(str(w.message),0)+1
        header=[family,c['n'],p,len(q),kind,config,operation,named]
        payload='\n'.join([' '.join(map(str,header)),*[' '.join(format(v,'.17g') for v in a.ravel(order='F')) for a in [x,q]]])+'\n'
        result=subprocess.run([str(exe)],input=payload,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert result.returncode==0,(c,result.stdout)
        lines=iter(result.stdout.splitlines());meta=np.fromstring(next(lines),sep=' ',dtype=int)
        nout=-1 if family==1 and operation==1 else fitted.shape[1]
        np.testing.assert_array_equal(meta,[p,nout,len(graph.transformers_) if family==0 else 3,calls[0]],err_msg=str(c))
        checks.append(dict(**c,name='metadata',passed=True))
        for name,want in [('fit_transform',fitted),('training_transform',train),('query_transform',query)]:
            shape=tuple(map(int,next(lines).split()));assert shape==want.shape,(c,name,shape,want.shape)
            actual=np.fromstring(next(lines),sep=' ').reshape(shape,order='F')
            tolerance=3e-6 if kind==10 else 3e-9
            np.testing.assert_allclose(actual,want,rtol=tolerance,atol=tolerance,equal_nan=True,err_msg=f'{c} {name}')
            finite=np.isfinite(actual)&np.isfinite(want)
            errors[name]=max(errors.get(name,0.),float(np.max(abs(actual[finite]-want[finite]),initial=0)))
            checks.append(dict(**c,name=name,passed=True))
        actual_names=[next(lines) for _ in range(int(next(lines)))]
        assert actual_names==names,(c,actual_names,names)
        checks.append(dict(**c,name='feature_names',passed=True))
        number=int(next(lines));ranges=[list(map(int,next(lines).split())) for _ in range(number)]
        if family==0:
            expected=[[graph.output_indices_[key].start,graph.output_indices_[key].stop]
                      for key in ['main','raw','ignored','empty','remainder']]
            assert ranges==expected,(c,ranges,expected)
            checks.append(dict(**c,name='output_indices',passed=True))
        else:assert number==0
        assert next(lines,None) is None
        count+=1
        if count%100==0:print(f'{count} composition fixtures passed',flush=True)
    root=Path(sklearn.__file__).parent;stem='column_transformer' if family==0 else 'feature_union'
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,
                fixtures=count,checks=checks,max_absolute_errors=errors,source_warnings=warning_counts,
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                               ['compose/_column_transformer.py','pipeline.py','preprocessing/_function_transformer.py']})
    (ROOT/'validation/results').mkdir(exist_ok=True)
    (ROOT/'validation/results'/f'{stem}_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {count} {stem} fixtures passed')
