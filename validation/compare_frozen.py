"""FrozenEstimator contracts against sklearn; base endpoints are separate diagnostics."""
from pathlib import Path
import argparse,copy,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np
import scipy,sklearn
from sklearn.base import clone,is_classifier,is_regressor
from sklearn.utils.validation import check_is_fitted
from sklearn.frozen import FrozenEstimator
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,PolynomialFeatures,Normalizer,Binarizer,FunctionTransformer
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression,Ridge,LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
FIELDS=['prediction','probability','log_probability','decision','transform','inverse','score','score_samples']
METHODS=['transform','inverse_transform','predict','predict_proba','predict_log_proba','decision_function',
         'score','get_feature_names_out','fit_transform','fit_predict','score_samples']

def shifted(y,code,offset):
    result=y.copy()
    if code==21:result=1-result
    else:result[:,0]+=offset
    return result

def fitted(model):
    try:check_is_fitted(model);return True
    except Exception:return False

def fit_model(model,x,yr,yc,code,skip=False):
    if code in [6,7,8] and skip:return
    if code in [0,1,2,3,4,5,6,7,8,9]:model.fit(x)
    elif code in [12,13,15,17,19,23,25,26]:model.fit(x,yc[:,0])
    elif code==20:model.fit(x,yr)
    elif code==21:model.fit(x,yc)
    else:model.fit(x,yr[:,0])

def make_model(code,x,yr,yc,seed,variant):
    mlp=dict(hidden_layer_sizes=(4,),activation='tanh',max_iter=3,batch_size=5,random_state=seed)
    factories={0:StandardScaler,1:MinMaxScaler,2:RobustScaler,3:PolynomialFeatures,
       4:lambda:PCA(n_components=2,svd_solver='full'),5:lambda:TruncatedSVD(n_components=2,algorithm='arpack',random_state=seed),
       6:Normalizer,7:Binarizer,8:lambda:FunctionTransformer(feature_names_out='one-to-one'),9:VarianceThreshold,
       10:LinearRegression,11:Ridge,12:GaussianNB,13:lambda:KNeighborsClassifier(n_neighbors=3),
       14:lambda:KNeighborsRegressor(n_neighbors=3),15:DummyClassifier,16:DummyRegressor,
       17:lambda:LogisticRegression(C=.8,tol=1e-12,max_iter=500),18:lambda:MLPRegressor(**mlp),
       19:lambda:MLPClassifier(**mlp),20:lambda:MLPRegressor(**mlp),21:lambda:MLPClassifier(**mlp),
       22:lambda:Pipeline([('scale',StandardScaler()),('model',Ridge())]),
       23:lambda:Pipeline([('scale',StandardScaler()),('model',GaussianNB())])}
    if code==24:model=Pipeline([('scale',FrozenEstimator(StandardScaler().fit(x))),('model',Ridge())])
    elif code==25:model=Pipeline([('frozen',FrozenEstimator(GaussianNB().fit(x,yc[:,0])))])
    elif code==26:model=FrozenEstimator(GaussianNB().fit(x,yc[:,0]))
    else:model=factories[code]()
    fit_model(model,x,yr,yc,code,skip=variant%2==0)
    return model

def values(model,q,qr,qc,code):
    result={name:np.array([]) for name in FIELDS}
    if hasattr(model,'predict'):result['prediction']=np.asarray(model.predict(q)).reshape(len(q),-1).ravel(order='F')
    if hasattr(model,'predict_proba'):result['probability']=model.predict_proba(q).ravel(order='F')
    if hasattr(model,'predict_log_proba'):
        with np.errstate(divide='ignore'):result['log_probability']=model.predict_log_proba(q*1000).ravel(order='F')
    if hasattr(model,'decision_function'):result['decision']=np.asarray(model.decision_function(q)).ravel(order='F')
    if hasattr(model,'transform'):
        z=model.transform(q);result['transform']=z.ravel(order='F')
        if hasattr(model,'inverse_transform'):result['inverse']=model.inverse_transform(z).ravel(order='F')
    if hasattr(model,'score'):
        if code==4:score=model.score(q)
        elif code in [12,13,15,17,19,23,25,26]:score=model.score(q,qc[:,0])
        elif code==20:score=model.score(q,qr)
        elif code==21:score=model.score(q,qc)
        else:score=model.score(q,qr[:,0])
        result['score']=np.array([score])
    if hasattr(model,'score_samples'):result['score_samples']=model.score_samples(q)
    return {key:np.asarray(value,dtype=float).copy() for key,value in result.items()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True,type=Path)
    parser.add_argument('--profile',default='checked');parser.add_argument('--report',type=Path)
    parser.add_argument('--case',type=int);args=parser.parse_args()
    inputs=[];records=[];configurations=[]
    for case,(code,seed,p,variant) in enumerate(itertools.product(range(27),[0,7,42],[3,5],range(3))):
        if args.case is not None and case!=args.case:continue
        n=17;nq=7;rng=np.random.RandomState(seed+131);x=rng.normal(0,.4,(n,p));q=rng.normal(0,.5,(nq,p))
        if variant==1 and code==9:x[:,-1]=0;q[:,-1]=0
        elif variant==2:x[:,1]=x[:,0]*.7+.03*rng.normal(size=n)
        yr=np.column_stack([1+x[:,0]-.4*x[:,1],np.sin(x[:,0])+x[:,2],x[:,0]*x[:,1]])
        qr=np.column_stack([1+q[:,0]-.4*q[:,1],np.sin(q[:,0])+q[:,2],q[:,0]*q[:,1]])
        classes=2 if code in [17,19] else 3
        yc=np.column_stack([3+4*(np.arange(n)%classes),np.arange(n)%2]);qc=np.column_stack([3+4*(np.arange(nq)%classes),np.arange(nq)%2])
        if code==21:yc=np.column_stack([np.arange(n)%2,(np.arange(n)//2)%2]);qc=np.column_stack([np.arange(nq)%2,(np.arange(nq)//2)%2])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            base=make_model(code,x,yr,yc,seed,variant)
            other=make_model(code,x*.7+2,yr*3+2,shifted(yc,code,202),seed+1,variant)
            frozen=FrozenEstimator(base);alias=clone(frozen)
            for phase in range(4):
                if phase==1:
                    assert frozen.fit(np.nan,'ignored',sample_weight=np.nan) is frozen
                    assert frozen.fit(np.empty((0,1)),np.array([])) is frozen
                elif phase==2:fit_model(base,x+1,yr*2+1,shifted(yc,code,101),code)
                elif phase==3:alias.set_params(estimator=other)
                current=other if phase==3 else base
                caps=np.array([float(fitted(frozen))]+[float(hasattr(frozen,name)) for name in METHODS]+
                              [1. if is_classifier(frozen) else (2. if is_regressor(frozen) else 0.)])
                expected=values(frozen,q,qr,qc,code);baseline=values(current,q,qr,qc,code)
                for name in FIELDS:np.testing.assert_equal(expected[name],baseline[name],err_msg=(case,phase,name))
                names=[]
                if (hasattr(frozen,'transform') and hasattr(frozen,'get_feature_names_out') and
                    not (code in [6,7] and getattr(frozen,'n_features_in_',0)==0)):
                    names=list(frozen.get_feature_names_out([f'x{j}' for j in range(p)]))
                records.append(dict(case=case,code=code,phase=phase,fields=expected,caps=caps,names=names,
                                    classes=np.asarray(getattr(frozen,'classes_',[]),dtype=float),
                                    features=getattr(frozen,'n_features_in_',0),
                                    identity=np.array([float(alias is frozen),float(frozen.get_params(True)['estimator'] is frozen.get_params(False)['estimator'])]),
                                    target_identity=np.array([float(frozen.get_params()['estimator'] is current)])))
        configurations.append(dict(case=case,code=code,seed=seed,features=p,variant=variant))
        inputs.append(' '.join(map(str,[code,n,p,nq,seed,variant])))
        for array in [x,yr,yc,q,qr,qc]:inputs.extend(' '.join(format(v,'.17g') for v in row) for row in array)
    exe=args.executable.resolve();run=subprocess.run([str(exe)],input=str(len(configurations))+'\n'+'\n'.join(inputs)+'\n',text=True,capture_output=True,cwd=ROOT)
    assert run.returncode==0,(run.returncode,run.stdout[-2000:],run.stderr[-2000:])
    checks=[];endpoints=[];metadata=[];actual={};index=-1;names_remaining=0;actual_names=[]
    def check(field,got,want,rtol=0.,atol=0.):
        got=np.asarray(got);want=np.asarray(want);np.testing.assert_allclose(got,want,rtol=rtol,atol=atol,err_msg=str((index,field,records[index]['case'],records[index]['phase'])))
        checks.append(dict(fixture=index,field=field,passed=True,max_abs_error=float(np.max(np.abs(got[np.isfinite(got)]-want[np.isfinite(got)]),initial=0))))
    for line in run.stdout.splitlines():
        if names_remaining:
            actual_names.append(line);names_remaining-=1;continue
        parts=line.split()
        if not parts:continue
        if parts[0]=='BEGIN':index+=1;actual={};actual_names=[]
        elif parts[0]=='NAMES':names_remaining=int(parts[1])
        elif parts[0]=='END':
            r=records[index];assert set(actual)=={'identity','target_identity','capabilities','features','classes'}|set(FIELDS)|{'base_'+field for field in FIELDS}
            check('identity',actual['identity'],r['identity']);check('target_identity',actual['target_identity'],r['target_identity'])
            check('capabilities',actual['capabilities'],r['caps']);check('classes',actual['classes'],r['classes'])
            assert actual_names==r['names'],(index,actual_names,r['names'])
            checks.append(dict(fixture=index,field='feature_names',passed=True,max_abs_error=0.))
            native_features=int(actual['features'][0]);source_features=r['features']
            if source_features is None:source_features=0
            configured_features=configurations[index//4]['features']
            assert native_features in [0,configured_features],(index,native_features,configured_features)
            metadata.append(dict(fixture=index,native_features=native_features,source_features=int(source_features)))
            for field in FIELDS:
                check('delegated_'+field,actual[field],actual['base_'+field])
                want=r['fields'][field];got=actual[field];assert got.shape==want.shape,(index,field,got.shape,want.shape)
                passed=np.allclose(got,want,rtol=2e-7,atol=2e-8,equal_nan=True)
                finite=np.isfinite(got)&np.isfinite(want)
                endpoints.append(dict(fixture=index,field=field,passed=bool(passed),max_abs_error=float(np.max(np.abs(got[finite]-want[finite]),initial=0)),
                                      nonfinite_mismatch=int(np.count_nonzero(np.isfinite(got)!=np.isfinite(want)))))
        else:
            assert parts[0] not in actual;actual[parts[0]]=np.array(list(map(float,parts[1:])))
    assert index+1==len(records)
    invalid=dict(unbound='not fitted',unfitted='not fitted',missing_predict='has no predict',missing_transform='has no transform',
       wrong_regression_kind='use predict_class',wrong_classifier_kind='use predict',query_width='feature',query_nan='NaN or infinity',
       names_width='feature',score_count='label sizes',wrong_classes='not a classifier',unsupported='not fitted')
    rejections=[]
    for mode,message in invalid.items():
        result=subprocess.run([str(exe),mode],text=True,capture_output=True,cwd=ROOT);output=result.stdout+result.stderr
        assert result.returncode!=0 and message.lower() in output.lower(),(mode,output)
        assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(mode,output)
        rejections.append(dict(case=mode,passed=True))
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,cases=len(configurations),stages=len(records),
       checks=checks,rejections=rejections,configurations=configurations,endpoint_diagnostics=endpoints,feature_metadata=metadata,
       nonfinite_source_values=sum(int(np.count_nonzero(~np.isfinite(a))) for r in records for a in r['fields'].values()),
       validator_sha256=sha(__file__),executable_sha256=sha(exe),native_source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
         [ROOT/'src'/name for name in ['sklearn_frozen.f90','sklearn_transformer_protocol.f90','sklearn_pipeline.f90','sklearn_feature_composition.f90']]},
       source_sha256={name:sha(Path(sklearn.__file__).parent/name) for name in ['frozen/_frozen.py','pipeline.py','base.py','utils/validation.py']})
    output=args.report or ROOT/f'validation/results/frozen_{args.profile}.json';output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(len(checks),'required contract groups;',len(configurations),'cases;',len(records),'stages;',len(rejections),'diagnostics passed')
    print(sum(not c['passed'] for c in endpoints),'of',len(endpoints),'base endpoint diagnostics differ; they are not contract passes')
if __name__=='__main__':main()
