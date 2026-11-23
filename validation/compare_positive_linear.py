"""Positive OLS/Ridge/coordinate regression and CV against sklearn 1.6.1."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import warnings
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['LOKY_MAX_CPU_COUNT']='2'
import numpy as np
import scipy
import sklearn
from scipy.optimize import nnls,lsq_linear
from sklearn.linear_model import LinearRegression,Ridge,ElasticNet,Lasso,ElasticNetCV,LassoCV
from sklearn.metrics import r2_score

ROOT=Path(__file__).resolve().parents[1]
REJECTIONS={
    **{name:'positive coefficients are unsupported for this regressor' for name in
       ['bayesian','ard','omp','huber','ransac','theil_sen','pa','tweedie','poisson','gamma']},
    'ridge_cv':'RidgeCV does not support positive coefficients','empty':'nonempty','width':'sample counts differ',
    'unfitted':'not fitted','x_nan':'NaN or infinity','y_inf':'NaN or infinity','length':'sample counts differ',
    'data_range':'positive linear data exceeds working range','coefficient_range':'NNLS coefficient exceeds working range',
    'ridge_alpha_nan':'positive linear alpha must be finite','ridge_alpha_negative':'alpha must be nonnegative',
    'elastic_alpha_nan':'positive elastic controls must be finite','elastic_ratio_nan':'positive elastic controls must be finite',
    'elastic_tol_nan':'positive elastic controls must be finite','weights_length':'bagging linear weight length mismatch',
    'weights_nan':'bagging linear weights must be finite','weights_negative':'bagging linear weights must be nonnegative',
    'weights_zero':'bagging linear weights must have a positive sum','nnls_iterations':'NNLS iteration limit must be positive',
    'prediction_range':'positive linear product exceeds working range'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(output):
    tokens=iter(output.split());fields=[]
    while True:
        try:n=int(next(tokens))
        except StopIteration:return fields
        p=int(next(tokens));fields.append(np.array([float(next(tokens)) for _ in range(n*p)]).reshape((n,p),order='F'))


def preprocess(x,y,w,intercept):
    xm=np.average(x,axis=0,weights=w) if intercept else np.zeros(x.shape[1])
    ym=np.average(y,weights=w) if intercept else 0.
    return (x-xm)*np.sqrt(w)[:,None],(y-ym)*np.sqrt(w)


def objective(a,b,coef,l1,l2):
    residual=a@coef-b
    return float(.5*np.sum(residual*residual)+l1*np.sum(abs(coef))+.5*l2*np.sum(coef*coef))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True,type=Path)
    parser.add_argument('--profile',choices=['development','checked','release'],default='development')
    parser.add_argument('--report',type=Path,default=ROOT/'validation/results/positive_linear.json')
    parser.add_argument('--start-fixture',type=int,default=0)
    args=parser.parse_args();exe=args.executable.resolve();fixtures=[];checks=[];certificates=[]
    for seed,shape,scenario,kind,intercept in itertools.product(
            [0,42,2026],[(13,5),(5,9),(20,4)],range(6),range(4),[False,True]):
        for positive in ([False,True] if scenario==0 else [True]):
            for weighted in ([False,True] if kind<2 else [False]):
                fixtures.append((seed,shape,scenario,kind,intercept,positive,weighted,1))
    for seed,shape,scenario,kind,intercept,positive,grid in itertools.product(
            [0,42,2026],[(17,5),(9,12)],range(3),[4,5],[False,True],[False,True],[0,1]):
        fixtures.append((seed,shape,scenario,kind,intercept,positive,False,grid))
    for index,(seed,shape,scenario,kind,intercept,positive,weighted,grid) in enumerate(fixtures):
        if index<args.start_fixture:continue
        rng=np.random.RandomState(seed);n,p=shape;x=rng.normal(size=shape);q=rng.normal(size=(4,p))
        truth=rng.normal(size=p);y=x@truth+.2*rng.normal(size=n)+1.7
        if scenario==1:x[:,-1]=x[:,0];q[:,-1]=q[:,0]
        if scenario==2:x[:,-1]=0;q[:,-1]=0
        if scenario==3:x[:,-1]=1;q[:,-1]=1
        if scenario==4:
            scale=10.**np.linspace(-3,3,p);x*=scale;q*=scale
        if scenario==5:x=abs(x);q=abs(q);y=-abs(y)
        w=.1+2*rng.random_sample(n) if weighted else np.ones(n)
        if weighted:w[::4]=0
        alpha=[.01,.3,2.][scenario%3];ratio=[.2,.7,1.][scenario%3]
        steps=100000;tol=1.e-11;cv=3;eps=.01;alphas=np.array([.7,.03,.2,.01]);ratios=np.array([.3,.8,1.])
        if kind==0:source=LinearRegression(positive=positive,fit_intercept=intercept)
        if kind==1:source=Ridge(alpha=alpha,positive=positive,fit_intercept=intercept,
                                solver='lbfgs' if positive else 'svd',tol=1.e-13,max_iter=100000)
        if kind==2:source=ElasticNet(alpha=alpha,l1_ratio=ratio,positive=positive,fit_intercept=intercept,
                                    max_iter=steps,tol=tol)
        if kind==3:source=Lasso(alpha=alpha,positive=positive,fit_intercept=intercept,max_iter=steps,tol=tol)
        if kind==4:source=ElasticNetCV(alphas=alphas if grid else None,n_alphas=len(alphas),l1_ratio=ratios,
            eps=eps,cv=cv,positive=positive,fit_intercept=intercept,max_iter=steps,tol=tol,n_jobs=1)
        if kind==5:source=LassoCV(alphas=alphas if grid else None,n_alphas=len(alphas),eps=eps,cv=cv,
            positive=positive,fit_intercept=intercept,max_iter=steps,tol=tol,n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');source.fit(x,y,**({'sample_weight':w} if weighted else {}))
        header=[n,p,len(q),kind,int(intercept),int(positive),int(weighted),steps,tol,alpha,ratio,
                cv,grid,len(alphas),len(ratios),eps]
        arrays=[x,y,q,w,alphas,ratios]
        payload=' '.join(map(str,header))+'\n'+'\n'.join(' '.join(format(v,'.17g') for v in a.ravel(order='F'))
                                                       for a in arrays)+'\n'
        proc=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
        assert proc.returncode==0,(index,header,proc.stdout,proc.stderr)
        actual=parse(proc.stdout);assert len(actual)==(9 if kind>=4 else 6),(index,proc.stdout)
        a,b=preprocess(x,y,w,intercept)
        reference_coef=source.coef_.copy();reference_intercept=float(source.intercept_);oracle='sklearn'
        kkt_bound=3.e-7*max(1.,np.linalg.norm(a)*np.linalg.norm(b))
        if kind==0 and positive:
            public_gradient=a.T@(a@reference_coef-b)
            public_pg=np.where(reference_coef>0,public_gradient,np.minimum(public_gradient,0))
            if np.max(abs(public_pg))>kkt_bound:
                corrected=lsq_linear(a,b,bounds=(0,np.inf),method='bvls',tol=1.e-13,max_iter=10000)
                assert corrected.success
                corrected_pg=a.T@(a@corrected.x-b)
                corrected_pg=np.where(corrected.x>1.e-10,corrected_pg,np.minimum(corrected_pg,0))
                assert np.max(abs(corrected_pg))<=kkt_bound
                old_cost=objective(a,b,reference_coef,0.,0.);new_cost=objective(a,b,corrected.x,0.,0.)
                assert new_cost<=old_cost
                certificates.append(dict(fixture=index,kind='scipy_1_15_nnls_qr_tail',
                    public_kkt_violation=float(np.max(abs(public_pg))),corrected_kkt_violation=float(np.max(abs(corrected_pg))),
                    kkt_bound=float(kkt_bound),public_objective=old_cost,corrected_objective=new_cost))
                reference_coef=corrected.x
                xm=np.average(x,axis=0,weights=w) if intercept else np.zeros(p)
                ym=np.average(y,weights=w) if intercept else 0.
                reference_intercept=float(ym-xm@reference_coef);oracle='scipy_bvls_corrected_nnls'
        if kind==1 and positive:
            exact,_=nnls(np.vstack([a,np.sqrt(alpha)*np.eye(p)]),np.r_[b,np.zeros(p)])
            np.testing.assert_allclose(actual[0].ravel(),exact,atol=3.e-8,rtol=3.e-7,
                                       err_msg=str((index,'augmented Ridge NNLS')))
            checks.append(dict(fixture=index,field='ridge_augmented_solution',
                max_abs_error=float(np.max(abs(actual[0].ravel()-exact)))))
            xm=np.average(x,axis=0,weights=w) if intercept else np.zeros(p)
            ym=np.average(y,weights=w) if intercept else 0.
            exact_intercept=float(ym-xm@exact)
            public_close=(np.allclose(reference_coef,exact,atol=3.e-8,rtol=3.e-7)
                          and np.allclose(source.predict(x),x@exact+exact_intercept,atol=3.e-8,rtol=3.e-7)
                          and np.allclose(source.predict(q),q@exact+exact_intercept,atol=3.e-8,rtol=3.e-7))
            if not public_close:
                gs=a.T@(a@reference_coef-b)+alpha*reference_coef
                ge=a.T@(a@exact-b)+alpha*exact
                ps=np.where(reference_coef>0,gs,np.minimum(gs,0));pe=np.where(exact>0,ge,np.minimum(ge,0))
                distance=float(np.linalg.norm(reference_coef-exact))
                distance_bound=float((np.linalg.norm(ps)+np.linalg.norm(pe))/alpha)
                # Strong convexity and the normal-cone residual certify the
                # discrepancy caused by public Ridge's finite L-BFGS stopping.
                assert distance<=distance_bound*(1+1.e-6)+1.e-10,(index,distance,distance_bound)
                source_cost=objective(a,b,reference_coef,0.,alpha);exact_cost=objective(a,b,exact,0.,alpha)
                assert exact_cost<=source_cost+1.e-10*max(1.,source_cost)
                certificates.append(dict(fixture=index,kind='ridge_public_optimizer',coefficient_distance=distance,
                    coefficient_distance_bound=distance_bound,source_projected_norm=float(np.linalg.norm(ps)),
                    exact_projected_norm=float(np.linalg.norm(pe)),alpha=alpha,
                    source_objective=source_cost,exact_objective=exact_cost))
                reference_coef=exact
                reference_intercept=exact_intercept;oracle='scipy_augmented_nnls'
        nonunique=False
        if kind==0 and positive and not np.allclose(actual[0].ravel(),reference_coef,atol=3.e-8,rtol=3.e-7):
            native_coef=actual[0].ravel();difference=native_coef-reference_coef
            image=float(np.linalg.norm(a@difference))
            image_bound=512*np.finfo(float).eps*max(a.shape)*max(1.,np.linalg.norm(a)*(
                np.linalg.norm(native_coef)+np.linalg.norm(reference_coef)),np.linalg.norm(b))
            ng=a.T@(a@native_coef-b);ng=np.where(native_coef>0,ng,np.minimum(ng,0))
            rg=a.T@(a@reference_coef-b);rg=np.where(reference_coef>1.e-10,rg,np.minimum(rg,0))
            assert image<=image_bound and np.max(abs(ng))<=kkt_bound and np.max(abs(rg))<=kkt_bound,(
                index,'nonunique',image,image_bound,ng,rg)
            certificates.append(dict(fixture=index,kind='nonunique_nnls',image_norm=image,image_bound=float(image_bound),
                native_kkt_violation=float(np.max(abs(ng))),reference_kkt_violation=float(np.max(abs(rg))),
                kkt_bound=float(kkt_bound),coefficient_distance=float(np.linalg.norm(difference)),
                query_prediction_difference=float(np.linalg.norm(q@difference))))
            nonunique=True
        rank=0 if positive or kind>=2 else int(np.linalg.matrix_rank(a,
            tol=np.finfo(float).eps*max(a.shape)*np.linalg.svd(a,compute_uv=False)[0]))
        expected=[reference_coef[:,None],[[reference_intercept]],(x@reference_coef+reference_intercept)[:,None],
                  (q@reference_coef+reference_intercept)[:,None],
                  np.array([r2_score(y,x@reference_coef+reference_intercept),rank])[:,None]]
        names=['coef','intercept','train_predict','query_predict','score_rank']
        for name,got,want in zip(names,actual,expected):
            comparison_oracle=oracle
            if nonunique:
                comparison_oracle='nonunique_nnls_optimality_certificate'
                native_coef=actual[0].ravel()
                xm=np.average(x,axis=0,weights=w) if intercept else np.zeros(p)
                ym=np.average(y,weights=w) if intercept else 0.
                native_intercept=float(ym-xm@native_coef)
                if name=='coef':
                    checks.append(dict(fixture=index,field=name,oracle=comparison_oracle,
                                       max_abs_error=float(np.max(abs(got-want)))))
                    continue
                if name=='intercept':want=[[native_intercept]]
                if name=='train_predict':want=(x@native_coef+native_intercept)[:,None]
                if name=='query_predict':want=(q@native_coef+native_intercept)[:,None]
                if name=='score_rank':want=np.array([r2_score(y,x@native_coef+native_intercept),rank])[:,None]
            try:np.testing.assert_allclose(got,want,atol=3.e-8,rtol=3.e-7)
            except AssertionError as error:raise AssertionError((index,header,scenario,name,str(error))) from error
            checks.append(dict(fixture=index,field=name,oracle=comparison_oracle,max_abs_error=float(np.max(abs(got-want)))))
        iterations,converged=actual[5].ravel()
        assert iterations>=0 and iterations<=steps and converged in [0,1]
        checks.append(dict(fixture=index,field='iteration_contract',max_abs_error=0.))
        if kind>=4:
            effective_ratio=source.l1_ratio_ if kind==4 else 1.
            path=np.asarray(source.mse_path_)
            if path.ndim==2:path=path[None,:,:]
            expected_grid=np.asarray(source.alphas_)
            if expected_grid.ndim==1:expected_grid=np.tile(expected_grid[:,None],(1,path.shape[0]))
            else:expected_grid=expected_grid.T
            values=[np.array([source.alpha_,effective_ratio,np.mean(path,axis=2).min()])[:,None],
                    expected_grid,np.transpose(path,(1,2,0)).reshape((path.shape[1]*cv,path.shape[0]),order='F')]
            for name,got,want in zip(['selected','alpha_grid','mse_path'],actual[6:],values):
                np.testing.assert_allclose(got,want,atol=3.e-8,rtol=3.e-7,err_msg=str((index,name)))
                checks.append(dict(fixture=index,field=name,max_abs_error=float(np.max(abs(got-want)))))
            alpha=float(source.alpha_);ratio=float(effective_ratio)
        if kind==0:l1=l2=0.
        elif kind==1:l1=0.;l2=alpha
        else:
            l1=n*alpha*(1. if kind in [3,5] else ratio)
            l2=n*alpha*(0. if kind in [3,5] else 1-ratio)
        coef=actual[0].ravel()
        if positive:assert np.min(coef)>=0
        native_objective=objective(a,b,coef,l1,l2);source_objective=objective(a,b,reference_coef,l1,l2)
        np.testing.assert_allclose(native_objective,source_objective,atol=3.e-8,rtol=3.e-7)
        gradient=a.T@(a@coef-b)+l2*coef
        if positive:
            gradient+=l1
            pg=np.where(coef>0,gradient,np.minimum(gradient,0))
            bound=3.e-7*max(1.,np.linalg.norm(a)*np.linalg.norm(b),l2*np.linalg.norm(coef),l1)
            assert np.max(abs(pg))<=bound,(index,'KKT',pg,bound)
        checks.append(dict(fixture=index,field='objective_nonnegativity_kkt',max_abs_error=abs(native_objective-source_objective)))
        if index%100==0:print(f'{index}/{len(fixtures)} fixtures passed',flush=True)
    rejected=[]
    for name,message in REJECTIONS.items():
        proc=subprocess.run([str(exe),name],text=True,capture_output=True)
        assert proc.returncode!=0 and message.lower() in (proc.stdout+proc.stderr).lower(),(name,proc.stdout,proc.stderr)
        assert 'SIGFPE' not in proc.stderr and 'Fortran runtime error' not in proc.stderr,(name,proc.stderr)
        rejected.append(name)
    from sklearn.linear_model import _base,_coordinate_descent
    sources={str(Path(path).relative_to(Path(sklearn.__file__).parent)):sha(path)
             for path in [_base.__file__,_coordinate_descent.__file__]}
    report=dict(profile=args.profile,fixtures=len(fixtures),start_fixture=args.start_fixture,checks=checks,rejections=rejected,
        certificates=certificates,sklearn_version=sklearn.__version__,scipy_version=scipy.__version__,
        executable_sha256=sha(exe),validator_sha256=sha(__file__),source_sha256=sources,
        nnls_source_sha256=sha(ROOT/'validation/results/scipy_1_15_3_cython_nnls.pyx'))
    args.report.parent.mkdir(exist_ok=True,parents=True)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'PASS: {len(fixtures)} fixtures from {args.start_fixture}, {len(checks)} reference groups, {len(rejected)} rejections')


if __name__=='__main__':main()
