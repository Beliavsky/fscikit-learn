"""Positive linear options through fitted native composition/cloning adapters."""
import argparse,hashlib,itertools,json,os,subprocess,warnings
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['LOKY_MAX_CPU_COUNT']='2'
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.multioutput import MultiOutputRegressor,RegressorChain
from scipy.optimize import nnls
from compare_positive_linear import parse,sha,preprocess,objective

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True,type=Path)
    parser.add_argument('--profile',choices=['development','checked','release'],default='development')
    parser.add_argument('--report',type=Path,default=ROOT/'validation/results/positive_composition.json')
    args=parser.parse_args();exe=args.executable.resolve();checks=[];fixture=0;certificates=[]
    for seed,shape,kind,base_kind,intercept in itertools.product([0,42,2026],[(21,4),(31,5)],range(5),[0,1],[False,True]):
        for weighted in ([False,True] if kind<3 else [False]):
            rng=np.random.RandomState(seed);n,p=shape;x=rng.normal(size=shape);q=rng.normal(size=(7,p))
            y=x@rng.normal(size=(p,2))+.3*rng.normal(size=(n,2))+np.array([1.7,-.8])
            w=.1+2*rng.random_sample(n) if weighted else np.ones(n)
            if weighted:w[::5]=0
            base=(LinearRegression(positive=True,fit_intercept=intercept) if base_kind==0 else
                  Ridge(alpha=2.,positive=True,fit_intercept=intercept,solver='lbfgs',tol=1.e-13,max_iter=100000))
            if kind==0:source=Pipeline([('scale',StandardScaler()),('model',base)])
            if kind==1:source=TransformedTargetRegressor(regressor=base,transformer=StandardScaler())
            if kind==2:source=BaggingRegressor(estimator=base,n_estimators=5,max_samples=.75,max_features=.75,random_state=seed)
            if kind==3:source=MultiOutputRegressor(base)
            if kind==4:source=RegressorChain(base)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                kw=({'model__sample_weight':w} if kind==0 else {'sample_weight':w}) if weighted else {}
                source.fit(x,y[:,0] if kind<3 else y,**kw)
            public_prediction=source.predict(q).copy()
            def exact_ridge(model,design,target,weights,label):
                if not isinstance(model,Ridge):return
                a,b=preprocess(design,target,weights,intercept)
                exact,_=nnls(np.vstack([a,np.sqrt(model.alpha)*np.eye(a.shape[1])]),np.r_[b,np.zeros(a.shape[1])])
                old=model.coef_.copy();gs=a.T@(a@old-b)+model.alpha*old;ge=a.T@(a@exact-b)+model.alpha*exact
                ps=np.where(old>0,gs,np.minimum(gs,0));pe=np.where(exact>0,ge,np.minimum(ge,0))
                bound=float((np.linalg.norm(ps)+np.linalg.norm(pe))/model.alpha)
                distance=float(np.linalg.norm(old-exact));roundoff=128*np.finfo(float).eps*max(1.,np.linalg.norm(old))
                assert distance<=bound*(1+1.e-6)+roundoff,(fixture,label,distance,bound,roundoff)
                old_cost=objective(a,b,old,0.,model.alpha);exact_cost=objective(a,b,exact,0.,model.alpha)
                assert exact_cost<=old_cost+1.e-10*max(1.,old_cost)
                certificates.append(dict(fixture=fixture,model=label,coefficient_distance=distance,
                    coefficient_distance_bound=bound,roundoff_bound=float(roundoff),alpha=model.alpha,
                    source_projected_norm=float(np.linalg.norm(ps)),exact_projected_norm=float(np.linalg.norm(pe)),
                    source_objective=old_cost,exact_objective=exact_cost))
                xm=np.average(design,axis=0,weights=weights) if intercept else np.zeros(design.shape[1])
                ym=np.average(target,weights=weights) if intercept else 0.
                model.coef_=exact;model.intercept_=float(ym-xm@exact)
            if kind==0:exact_ridge(source[-1],source[0].transform(x),y[:,0],w,'pipeline')
            if kind==1:exact_ridge(source.regressor_,x,source.transformer_.transform(y[:,[0]]).ravel(),w,'target')
            if kind==2:
                for j,model in enumerate(source.estimators_):
                    effective=w*np.bincount(source.estimators_samples_[j],minlength=n)
                    exact_ridge(model,x[:,source.estimators_features_[j]],y[:,0],effective,f'bag{j}')
            if kind==3:
                for j,model in enumerate(source.estimators_):exact_ridge(model,x,y[:,j],np.ones(n),f'multi{j}')
            if kind==4:
                for j,model in enumerate(source.estimators_):
                    design=np.column_stack([x,y[:,:j]])
                    exact_ridge(model,design,y[:,j],np.ones(n),f'chain{j}')
            expected=[source.predict(q)[:,None] if kind<3 else source.predict(q)]
            if base_kind==1:
                certificates.append(dict(fixture=fixture,model='composition_prediction',
                    public_prediction=public_prediction.ravel(order='F').tolist(),
                    exact_prediction=source.predict(q).ravel(order='F').tolist(),
                    difference=float(np.max(abs(public_prediction-source.predict(q))))))
            names=['predict']
            def estimator(model,prefix):
                assert model.positive
                expected.extend([model.coef_[:,None],np.array([model.intercept_,1.])[:,None]])
                names.extend([prefix+'coef',prefix+'intercept_positive'])
            if kind==0:estimator(source[-1],'pipeline_')
            if kind==1:estimator(source.regressor_,'target_')
            if kind==2:
                expected.append(np.asarray(source._seeds)[:,None]);names.append('seeds')
                for j,model in enumerate(source.estimators_):
                    expected.extend([(source.estimators_features_[j]+1)[:,None],(source.estimators_samples_[j]+1)[:,None]])
                    names.extend([f'bag{j}_features',f'bag{j}_samples']);estimator(model,f'bag{j}_')
            if kind==3:
                for j,model in enumerate(source.estimators_):estimator(model,f'multi{j}_')
            if kind==4:
                expected.append((source.order_+1)[:,None]);names.append('order')
                for j,model in enumerate(source.estimators_):estimator(model,f'chain{j}_')
            header=[n,p,len(q),kind,base_kind,int(intercept),seed,int(weighted),2.]
            payload=' '.join(map(str,header))+'\n'+'\n'.join(' '.join(format(v,'.17g') for v in a.ravel(order='F'))
                                                          for a in [x,y,q,w])+'\n'
            proc=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
            assert proc.returncode==0,(fixture,header,proc.stdout,proc.stderr)
            actual=parse(proc.stdout);assert len(actual)==len(expected),(fixture,len(actual),len(expected))
            for name,got,want in zip(names,actual,expected):
                np.testing.assert_allclose(got,want,atol=3.e-8,rtol=3.e-7,err_msg=str((fixture,header,name)))
                if name.endswith('coef'):assert np.min(got)>=0
                checks.append(dict(fixture=fixture,field=name,oracle='scipy_augmented_nnls' if base_kind else 'sklearn',
                                   max_abs_error=float(np.max(abs(got-want)))))
            fixture+=1
    report=dict(profile=args.profile,fixtures=fixture,checks=checks,certificates=certificates,sklearn_version=sklearn.__version__,
                executable_sha256=sha(exe),validator_sha256=sha(__file__),
                core_validator_sha256=sha(ROOT/'validation/compare_positive_linear.py'))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'PASS: {fixture} positive composition fixtures, {len(checks)} groups')


if __name__=='__main__':main()
