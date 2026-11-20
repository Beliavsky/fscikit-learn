"""BayesianGaussianMixture posterior, restart, warm-start and sampling contracts."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import warnings
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import sklearn
from sklearn.mixture import BayesianGaussianMixture
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


class Source(BayesianGaussianMixture):
    def _initialize(self,x,resp):
        super()._initialize(x,resp)
        self.traces.append([])
        self.current_trace=self.traces[-1]

    def _compute_lower_bound(self,log_resp,log_prob_norm):
        value=super()._compute_lower_bound(log_resp,log_prob_norm)
        if self.current_trace is None:
            self.traces.append([]); self.current_trace=self.traces[-1]
        self.current_trace.append(value)
        return value


def fit_source(model,x,operation=1):
    model.traces=[]; model.current_trace=None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',ConvergenceWarning)
        if operation==2: return model.fit_predict(x)
        model.fit(x)


def expand(array,kind,k,p):
    if kind=='full': return np.moveaxis(array,0,2)
    if kind=='tied': return np.repeat(np.asarray(array)[:,:,None],k,axis=2)
    if kind=='diag': return np.stack([np.diag(v) for v in array],axis=2)
    return np.stack([np.eye(p)*v for v in array],axis=2)


def snapshot(model,x,query,prefix=''):
    k,p=model.means_.shape; kind=model.covariance_type
    concentration=model.weight_concentration_
    if isinstance(concentration,tuple): concentration=np.column_stack(concentration)
    else: concentration=concentration[:,None]
    prior=np.asarray(model.covariance_prior_)
    if kind=='diag': prior=np.diag(prior)
    if kind=='spherical': prior=np.eye(p)*prior
    histories=[h for h in model.traces if h]
    history=max(histories,key=lambda h:h[-1]) if histories else []
    values=dict(weights=model.weights_,means=model.means_,concentration=concentration,
        mean_precision=model.mean_precision_,degrees=np.broadcast_to(model.degrees_of_freedom_,(k,)),
        covariances=expand(model.covariances_,kind,k,p),precisions=expand(model.precisions_,kind,k,p),
        precision_cholesky=expand(model.precisions_cholesky_,kind,k,p),lower_bound=[model.lower_bound_],history=history,
        mean_prior=model.mean_prior_,covariance_prior=prior,
        scalar_priors=[model.weight_concentration_prior_,model.mean_precision_prior_,model.degrees_of_freedom_prior_],
        predict=model.predict(query),proba=model.predict_proba(query),score_samples=model.score_samples(query),
        score=[model.score(x)],meta=[p,k,model.n_iter_,int(model.converged_)])
    return {prefix+name:np.asarray(value).copy() for name,value in values.items()}


def sampling_checks(actual,model,n_draw):
    k,p=model.means_.shape; weights=model.weights_; means=model.means_
    cov=expand(model.covariances_,model.covariance_type,k,p)
    mean=weights@means; population=np.zeros((p,p)); fourth=np.zeros((p,p))
    for c in range(k):
        delta=means[c]-mean; cc=cov[:,:,c]
        population+=weights[c]*(cc+np.outer(delta,delta))
        fourth+=weights[c]*(np.outer(delta**2,delta**2)+np.outer(delta**2,np.diag(cc))+
            np.outer(np.diag(cc),delta**2)+4*np.outer(delta,delta)*cc+np.outer(np.diag(cc),np.diag(cc))+2*cc**2)
    mean_error=7*np.sqrt(np.diag(population)/n_draw)+1e-3
    covariance_error=7*np.sqrt(np.maximum(0,fourth-population**2)/n_draw)+1e-3
    assert np.all(abs(actual['sample_mean']-mean)<=mean_error),(actual['sample_mean'],mean,mean_error)
    observed=actual['sample_covariance'].reshape((p,p),order='F')
    assert np.all(abs(observed-population)<=covariance_error),(observed,population,covariance_error)
    assert np.all(abs(actual['sample_counts']-n_draw*weights)<=7*np.sqrt(n_draw*weights*(1-weights))+2)
    assert actual['sample_counts'].sum()==n_draw
    np.testing.assert_array_equal(actual['sample_meta'],[1,1])
    return ['sample_mean','sample_covariance','sample_counts','sample_meta']


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--executable',type=Path); args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable: executable=args.executable.resolve()
    else:
        proc=subprocess.run(['fpm','run','--flag',FLAGS,'--target','reference_bayesian_mixture_driver','--runner'],
                            cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if proc.returncode: raise RuntimeError(proc.stdout)
        paths=[ROOT/s.strip().strip(chr(34)) for s in proc.stdout.splitlines()
               if s.strip().strip(chr(34)).endswith(('reference_bayesian_mixture_driver','reference_bayesian_mixture_driver.exe'))]
        paths=[p for p in paths if p.is_file()]; assert len(paths)==1,proc.stdout
        executable=paths[0].resolve()
    checks=[]
    for seed,n,p0 in ((7,60,3),(42,72,4),(0,48,2),(2026,89,5)):
        rng=np.random.default_rng(seed); transform=rng.normal(size=(p0,p0))/np.sqrt(p0)+np.eye(p0)
        original=rng.normal(size=(n,p0))@transform
        original+=np.array([-4.,0.,5.])[np.arange(n)%3,None]*np.linspace(.5,1.2,p0)
        cases=[]
        for prior_type in ('dirichlet_process','dirichlet_distribution'):
            for covariance in ('full','tied','diag','spherical'):
                for init in ('kmeans','random','random_from_data','k-means++'):
                    cases.append((f'{prior_type}_{covariance}_{init}',dict(weight_concentration_prior_type=prior_type,
                        covariance_type=covariance,init_params=init),{}))
                cases.append((f'{prior_type}_{covariance}_restarts',dict(weight_concentration_prior_type=prior_type,
                    covariance_type=covariance,init_params='random',n_init=3),{}))
                cases.append((f'{prior_type}_{covariance}_priors',dict(weight_concentration_prior_type=prior_type,
                    covariance_type=covariance,mean_precision_prior=2.3,weight_concentration_prior=.2,
                    degrees_of_freedom_prior=p0+1.7),dict(priors=True,sample=20000)))
        for init in ('kmeans','random','random_from_data','k-means++'):
            cases.append((f'zero_iter_{init}',dict(max_iter=0,n_init=3,init_params=init,covariance_type='diag'),{}))
        for covariance in ('full','tied','diag','spherical'):
            cases.append((f'one_iter_{covariance}',dict(max_iter=1,covariance_type=covariance),{}))
            cases.append((f'one_component_{covariance}',dict(n_components=1,covariance_type=covariance),dict(sample=20000)))
        cases += [('zero_tolerance',dict(tol=0,max_iter=5,init_params='random'),{}),
            ('warm_same',dict(warm_start=True,n_init=3),dict(operation=3)),
            ('warm_shifted',dict(warm_start=True,covariance_type='tied',weight_concentration_prior_type='dirichlet_distribution'),dict(operation=4)),
            ('warm_zero',dict(warm_start=True,max_iter=0),dict(operation=3)),
            ('refit_dimensions',{},dict(operation=5)),('fit_predict',{},dict(operation=2)),
            ('constant',dict(n_components=2,reg_covar=.01),dict(constant=True)),
            ('univariate',dict(covariance_type='spherical',init_params='k-means++'),dict(univariate=True)),
            ('translated',dict(covariance_type='diag'),dict(translate=True)),
            ('wide_budget',dict(max_iter=1000000000,tol=1e9),{})]
        for name,overrides,extra in cases:
            x=original.copy()
            if extra.get('constant'): x[:]=2
            if extra.get('univariate'): x=x[:,:1]
            if extra.get('translate'): x+=100
            p=x.shape[1]; query=np.vstack([x+.13,rng.normal(size=(11,p))])
            options=dict(n_components=4,max_iter=200,tol=1e-4,random_state=seed)
            options.update(overrides)
            pm=pc=None
            if extra.get('priors'):
                pm=rng.normal(size=p)
                if options['covariance_type'] in ('full','tied'):
                    a=rng.normal(size=(p,p)); pc=a.T@a+.5*np.eye(p); native_cov=pc
                elif options['covariance_type']=='diag': pc=rng.uniform(.3,2,p); native_cov=np.diag(pc)
                else: pc=.7; native_cov=np.eye(p)*pc
                options.update(mean_prior=pm,covariance_prior=pc)
            model=Source(**options); operation=extra.get('operation',1); n_draw=extra.get('sample',0)
            data,output=RESULTS/'bayesian_mixture_input.txt',RESULTS/'bayesian_mixture_actual.txt'
            with data.open('w') as stream:
                stream.write(f'{n} {p} {len(query)} {operation} {n_draw}\n')
                stream.write(f'{model.n_components} {model.max_iter} {model.n_init} {seed} {model.tol} {model.reg_covar}\n')
                stream.write(f'{model.weight_concentration_prior or 0} {model.mean_precision_prior or 1} '
                             f'{model.degrees_of_freedom_prior or 0} {model.warm_start}\n')
                stream.write(f'{model.covariance_type} {model.weight_concentration_prior_type} {model.init_params}\n')
                stream.write(f'{pm is not None} {pc is not None}\n')
                np.savetxt(stream,x,fmt='%.17g'); np.savetxt(stream,query,fmt='%.17g')
                if pm is not None: np.savetxt(stream,pm[None,:],fmt='%.17g')
                if pc is not None: np.savetxt(stream,native_cov,fmt='%.17g')
            subprocess.run([str(executable),str(data),str(output)],cwd=ROOT,check=True)
            actual={}
            for line in output.read_text().splitlines():
                key,*values=line.split(); actual[key]=np.array(values,dtype=float)
            expected={}; labels=fit_source(model,x,operation)
            if operation==2: expected['fit_predict']=labels
            if operation>=3:
                expected.update(snapshot(model,x,query,'first_'))
                if operation==3: fit_source(model,x)
                elif operation==4: fit_source(model,x+.2)
                else:
                    model.set_params(warm_start=False,n_components=2,covariance_type='diag',mean_prior=None,covariance_prior=None)
                    fit_source(model,x[:,:1])
            p=model.n_features_in_; expected.update(snapshot(model,x[:,:p],query[:,:p]))
            sample_names=[]
            if n_draw: sample_names=sampling_checks(actual,model,n_draw)
            assert actual.keys()==expected.keys()|set(sample_names),(seed,name,actual.keys()^expected.keys())
            for key,value in expected.items():
                tol=0 if key.endswith(('meta','predict')) else 3e-8
                np.testing.assert_allclose(actual[key],np.asarray(value).ravel(order='F'),atol=tol,rtol=tol,
                                           err_msg=f'seed={seed} case={name} {key}')
                checks.append(dict(seed=seed,case=name,name=key,passed=True))
            for key in sample_names: checks.append(dict(seed=seed,case=name,name=key,passed=True))
        print(f'seed {seed}: {len(cases)} configurations passed',flush=True)
    (RESULTS/'bayesian_mixture.json').write_text(json.dumps(dict(sklearn=sklearn.__version__,checks=checks),indent=2)+'\n')
    print(f'{len(checks)} BayesianGaussianMixture result groups passed')


if __name__=='__main__': main()
