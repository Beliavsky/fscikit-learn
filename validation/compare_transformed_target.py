"""TransformedTargetRegressor with cloned transforms, scalar and joint targets."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import warnings
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT","2")
os.environ.setdefault("OMP_NUM_THREADS","2")
import numpy as np
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer,StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler,PowerTransformer,QuantileTransformer
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score
from compare_weighted_trees import certify_cart
from transform_callbacks import callbacks
from compare_function_transformer import ROOT,resolve


def fixtures():
    for seed in [0,42,2026]:
        rng=np.random.RandomState(seed)
        for base in range(5):
            for trans in range(8):
                for case in range(6):
                    n,p,k=[(1,1,1),(7,3,1),(23,3,2),(107,4,3),(251,2,1),(31,5,1)][case]
                    if base==3 and k>1:continue
                    x=rng.normal(size=(n,p));q=rng.normal(size=(5,p))*.7
                    y=np.exp(.7+.2*x[:,0,None]+rng.normal(size=(n,k))*.1)+np.arange(k)[None,:]
                    if case==5:y[:]=4.
                    mode=0
                    if trans==0:mode=[0,1,4,5,10,6][case]
                    if trans==7:mode=[7,2,1,5,10,0][case]
                    if mode==7:y[0,0]=0
                    if mode==10 and n>2:y[2,0]=7.7
                    w=rng.uniform(.1,2,size=n)
                    if n>7:w[:3]=0
                    weighted=int(base in [0,1,3] and case in [1,2,3])
                    if case==4:w[0]=-.25
                    c=dict(seed=seed,base=base,transformer=trans,case=case,n=n,p=p,k=k,mode=mode,
                           rank=2 if case in [1,2,3] else 1,weighted=weighted,check=int(case!=4),alpha=.2)
                    yield c,x,y,q,w


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);args=ap.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_transformed_target_driver')
    checks=[];count=0;errors={};source_warnings={};score_roundoff=[];tree_certificates=[];error_context={};scaled_errors={}
    for c,x,y,q,w in fixtures():
        f,inv=callbacks(c['mode']);n,p,k=c['n'],c['p'],c['k']
        bases=[None,Ridge(alpha=c['alpha'],solver='svd'),KNeighborsRegressor(n_neighbors=min(3,n),weights='distance'),
               DecisionTreeRegressor(max_depth=3,random_state=0),DummyRegressor()]
        transformer=None
        if c['transformer']==1:transformer=StandardScaler()
        if c['transformer']==2:transformer=MinMaxScaler(feature_range=(-2,3))
        if c['transformer']==3:transformer=MaxAbsScaler()
        if c['transformer']==4:transformer=RobustScaler()
        if c['transformer']==5:transformer=PowerTransformer()
        if c['transformer']==6:transformer=QuantileTransformer(n_quantiles=min(17,n),subsample=None)
        if c['transformer']==7:
            transformer=FunctionTransformer(f,inv,validate=True).fit(np.ones((2,k+1)))
            transformer.validate=False
        kw=dict(regressor=bases[c['base']],transformer=transformer,check_inverse=bool(c['check']))
        if c['transformer']==0 and c['mode']!=0:kw.update(func=f,inverse_func=inv)
        model=TransformedTargetRegressor(**kw)
        original=y[:,0] if c['rank']==1 else y
        with warnings.catch_warnings(record=True) as caught,np.errstate(invalid='ignore',over='ignore'):
            warnings.simplefilter('always')
            model.fit(x,original,**(dict(sample_weight=w) if c['weighted'] else {}))
            raw=np.asarray(model.regressor_.predict(q)).reshape((len(q),-1))
            predicted=np.asarray(model.predict(q)).reshape((len(q),-1))
            train=np.asarray(model.predict(x)).reshape((n,-1))
            scores=np.array([r2_score(y,train),r2_score(y,train,sample_weight=w)])
            fitted=model.transformer_;is_function=isinstance(fitted,FunctionTransformer)
            consistent=True
            if c['check']:
                selected=y[::max(1,n//10)]
                consistent=bool(np.allclose(selected,fitted.inverse_transform(fitted.transform(selected))))
            ft_consistent=-1;ft_width=-1
            if is_function:
                ft_consistent=1;ft_width=getattr(fitted,'n_features_in_',0)
                if fitted.check_inverse and fitted.func is not None and fitted.inverse_func is not None:
                    selected=y[::max(1,n//100)]
                    ft_consistent=int(np.allclose(selected,fitted.inverse_transform(fitted.transform(selected)),rtol=1e-7,atol=1e-9))
            stats=[]
            if c['transformer']==1:stats=np.r_[fitted.mean_,fitted.var_,fitted.scale_]
            if c['transformer']==2:stats=np.r_[fitted.data_min_,fitted.data_max_,fitted.scale_,fitted.min_]
            if c['transformer']==3:stats=fitted.scale_
            if c['transformer']==4:stats=np.r_[fitted.center_,fitted.scale_]
            if c['transformer']==5:stats=np.r_[fitted.lambdas_,fitted._scaler.mean_,fitted._scaler.scale_]
            if c['transformer']==6:stats=np.r_[fitted.quantiles_.ravel(order='F'),fitted.references_]
        for message in caught:source_warnings[str(message.message)]=source_warnings.get(str(message.message),0)+1
        header=[n,p,k,len(q),c['base'],c['transformer'],c['mode'],c['rank'],c['weighted'],c['check'],c['alpha']]
        payload='\n'.join([' '.join(map(str,header)),*[' '.join(format(v,'.17g') for v in a.ravel(order='F'))
                        for a in [x,y,q,w]]])+'\n'
        run=subprocess.run([str(exe)],input=payload,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert run.returncode==0,(c,run.stdout)
        lines=iter(run.stdout.splitlines())
        meta=np.fromstring(next(lines),sep=' ',dtype=int)
        np.testing.assert_array_equal(meta,[c['rank'],k,raw.shape[1],p,c['check'],int(consistent),ft_consistent,ft_width],err_msg=str(c))
        checks.append(dict(**c,name='metadata',passed=True))
        tolerance=3e-6 if c['transformer']==5 else 3e-9
        latent_shape=tuple(map(int,next(lines).split()))
        latent=np.fromstring(next(lines),sep=' ').reshape(latent_shape,order='F')
        np.testing.assert_allclose(latent,model.transformer_.transform(y),rtol=tolerance,atol=tolerance)
        checks.append(dict(**c,name='training_transform',passed=True))
        nodes=[];node_stats=[];node_values=[]
        for _ in range(int(next(lines))):
            nodes.append(np.fromstring(next(lines),sep=' ',dtype=int))
            node_stats.append(np.fromstring(next(lines),sep=' '))
            node_values.append(np.fromstring(next(lines),sep=' '))
        expected=dict(raw_query=raw,prediction=predicted,training_prediction=train,score=scores,transformer_state=stats)
        actuals={}
        for name,want in expected.items():
            if name in ['raw_query','prediction','training_prediction']:
                shape=tuple(map(int,next(lines).split()));assert shape==want.shape,(c,name,shape,want.shape)
            else:shape=np.shape(want)
            actuals[name]=np.fromstring(next(lines),sep=' ').reshape(shape,order='F')
        if c['base']==3 and any(not np.allclose(actuals[name],expected[name],rtol=tolerance,atol=tolerance,equal_nan=True)
                                for name in ['raw_query','prediction','training_prediction']):
            both,witnesses=certify_cart(np.array(nodes),np.array(node_stats),np.array(node_values),x,latent[:,0],
                w if c['weighted'] else np.ones(n),np.vstack([q,x]),None,1,2,3,0,0.,1,0,0)
            both=np.asarray(both).reshape((-1,1))
            expected['raw_query']=both[:len(q)]
            expected['prediction']=model.transformer_.inverse_transform(both[:len(q)])
            expected['training_prediction']=model.transformer_.inverse_transform(both[len(q):])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                expected['score']=np.array([r2_score(y,expected['training_prediction']),
                    r2_score(y,expected['training_prediction'],sample_weight=w)])
            tree_certificates.append(dict(**c,witnesses=witnesses))
        for name,want in expected.items():
            actual=actuals[name]
            if name=='score' and c['case']==5:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    native_scores=np.array([r2_score(y,actuals['training_prediction']),
                                            r2_score(y,actuals['training_prediction'],sample_weight=w)])
                if not np.allclose(actual,want,rtol=tolerance,atol=tolerance,equal_nan=True):
                    score_roundoff.append(dict(**c,source_scores=want.tolist(),native_scores=actual.tolist()))
                want=native_scores
            np.testing.assert_allclose(actual,want,rtol=tolerance,atol=tolerance,equal_nan=True,err_msg=f'{c} {name}')
            finite=np.isfinite(actual)&np.isfinite(want)
            delta=abs(actual[finite]-np.asarray(want)[finite])
            difference=float(np.max(delta,initial=0))
            if difference>errors.get(name,-1):
                error_context[name]=dict(**c,max_absolute_error=difference)
            errors[name]=max(errors.get(name,0.),difference)
            scaled_errors[name]=max(scaled_errors.get(name,0.),float(np.max(
                delta/(tolerance+tolerance*abs(np.asarray(want)[finite])),initial=0)))
            checks.append(dict(**c,name=name,passed=True))
        assert next(lines,None) is None
        count+=1
        if count%100==0:print(f'{count} transformed-target fixtures passed',flush=True)
    root=Path(sklearn.__file__).parent
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,fixtures=count,checks=checks,
                max_absolute_errors=errors,max_tolerance_fractions=scaled_errors,error_context=error_context,
                tree_runtime=__import__("sklearn_tree_runtime").tree_runtime_provenance(),source_warnings=source_warnings,constant_target_score_roundoff=score_roundoff,tree_certificates=tree_certificates,
                native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                               ['compose/_target.py','preprocessing/_function_transformer.py','preprocessing/_data.py','utils/validation.py']})
    (ROOT/'validation/results').mkdir(exist_ok=True)
    (ROOT/'validation/results'/f'transformed_target_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {count} transformed-target fixtures passed')


if __name__=='__main__':main()
