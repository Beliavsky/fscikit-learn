"""Installed sklearn IterativeImputer: outputs, sequence, RNG replay and names."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import warnings
from collections import Counter
import numpy as np
import scipy
import sklearn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

ROOT=Path(__file__).resolve().parents[1]
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'

def resolve(profile):
    flags=['--profile','release'] if profile=='release' else ['--flag',FLAGS]
    r=subprocess.run(['fpm','run',*flags,'--target','reference_iterative_impute_driver','--runner'],
                     cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/line.strip().strip('"') for line in r.stdout.splitlines()
           if line.strip().strip('"').endswith(('reference_iterative_impute_driver.exe','reference_iterative_impute_driver'))]
    paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,r.stdout
    return paths[0]

def fixtures():
    for seed in [0,7,42,2026,314159]:
        rng=np.random.RandomState(seed)
        for case in range(60):
            n,p,m=24,5+case%3,9
            if case%11==0:p=1
            x=rng.normal(size=(n,p));q=rng.normal(size=(m,p))
            if p>1:
                x[:,1]+=.8*x[:,0];q[:,1]+=.8*q[:,0]
            x[rng.uniform(size=x.shape)<.25]=np.nan
            q[rng.uniform(size=q.shape)<.4]=np.nan
            if p>1:x[:,-1]=rng.normal(size=n) # complete feature can be missing at transform
            if case%7==0:x[:,0]=np.nan
            if case%13==0:x[:]=np.nan
            if case%17==0:q[:]=np.nan
            if case%19==0:x[:]=0;q[:]=0;x[0,0]=np.nan;q[1,0]=np.nan
            if case%23==0:x=rng.normal(size=x.shape)
            sentinel=-99 if case%9==0 else np.nan
            kw=dict(max_iter=0 if case%6==0 else 4,tol=1.e-3 if case%2 else 0.,
                    n_nearest_features=1 if case%4 in [0,1] else None,
                    random_state=seed,skip_complete=bool(case%2),keep_empty_features=bool(case//2%2),
                    add_indicator=bool(case//3%2),sample_posterior=bool(case%3==0),
                    initial_strategy=['mean','median','most_frequent','constant'][case%4],fill_value=.25,
                    imputation_order=['ascending','descending','roman','arabic','random'][case//4%5],
                    missing_values=sentinel)
            bounds=case%5==0
            if bounds:
                kw.update(min_value=np.arange(p)*.1-1,max_value=np.arange(p)*.2+1)
            else:kw.update(min_value=-np.inf if case%10==1 else -3.,max_value=np.inf if case%10==1 else 3.)
            if np.isfinite(sentinel):x[np.isnan(x)]=sentinel;q[np.isnan(q)]=sentinel
            yield seed,case,kw,bounds,x,q
        if seed!=0:continue
        for n,p in [(1,7),(2,9),(3,12)]:
            for posterior in [False,True]:
                for keep in [False,True]:
                    for nearest in [None,1]:
                        x=rng.normal(size=(n,p));q=rng.normal(size=(4,p))
                        x[0,::2]=np.nan;q[0,::2]=np.nan;q[1,1::2]=np.nan
                        kw=dict(max_iter=3,tol=1.e-3,n_nearest_features=nearest,random_state=seed,
                                skip_complete=False,keep_empty_features=keep,add_indicator=True,
                                sample_posterior=posterior,initial_strategy='mean',fill_value=0.,
                                imputation_order='random',missing_values=np.nan,min_value=-3.,max_value=3.)
                        case=1000+n*100+int(posterior)*10+int(keep)*2+int(nearest is not None)
                        yield seed,case,kw,False,x,q

def encode(kw,bounds,x,q):
    def line(a):return ' '.join(str(v) for v in a)
    n,p=x.shape
    lines=[f"{n} {p} {len(q)} {kw['max_iter']} {kw['n_nearest_features'] or 0} {kw['random_state']}",
           line(int(kw[key]) for key in ['sample_posterior','skip_complete','keep_empty_features','add_indicator'])+
           f" {int(np.isfinite(kw['missing_values']))} {int(bounds)}",
           f"{kw['initial_strategy']} {kw['imputation_order']}",
           f"{kw['tol']} {kw['fill_value']} {kw['min_value'] if not bounds else -3} {kw['max_value'] if not bounds else 3}"]
    if bounds:lines.extend([line(kw['min_value']),line(kw['max_value'])])
    lines.extend(line(row) for a in [x,q] for row in a)
    return '\n'.join(lines)+'\n'

def expected_results(model,x,q):
    z=model.fit_transform(x.copy());a=model.transform(q.copy());b=model.transform(q.copy())
    return dict(output=z,query=a,query_repeat=b,
                metadata=np.array([model.n_iter_,getattr(model,'n_features_with_missing_',0),
                                   len(model.imputation_sequence_),z.shape[1]]),
                statistics=np.nan_to_num(model.initial_imputer_.statistics_,nan=0),
                names=model.get_feature_names_out())

def covariance_replay(kw,x,q,records):
    """Replay sklearn with a certified alternative thin-SVD null-space basis.

    No installed module is patched. Each sklearn BayesianRidge fit is still
    checked against the recorded native fit on the replay's CURRENT inputs.
    Covariance substitution is allowed only when wide centered data have a
    non-unique retained null-space completion. This is a conditional oracle,
    explicitly distinct from direct sklearn-default output parity.
    """
    certificates=[]
    class AuditedBayesianRidge(BayesianRidge):
        fitted=0
        def fit(self,X,y):
            super().fit(X,y)
            position=type(self).fitted;type(self).fitted+=1
            record=records[position];alpha,lam,intercept,*coef=record['parameters']
            np.testing.assert_allclose(np.r_[self.alpha_,self.lambda_,self.intercept_,self.coef_],
                                       record['parameters'],rtol=3e-9,atol=3e-9)
            sigma=record['sigma'];center=X-X.mean(axis=0)
            _,s,vh=np.linalg.svd(center,full_matrices=False)
            k=min(X.shape);p=X.shape[1];rank=np.linalg.matrix_rank(center)
            if k<p and rank<k:
                # Nonzero row-space action is unique. The remaining k-rank
                # directions may be ANY orthonormal completion in the null
                # space, but their covariance eigenvalues are fixed.
                np.testing.assert_allclose(sigma,sigma.T,rtol=3e-11,atol=3e-11)
                spectrum=np.r_[1/(alpha*s*s+lam),np.zeros(p-k)]
                eigenvalues=np.linalg.eigvalsh(sigma)
                np.testing.assert_allclose(eigenvalues,np.sort(spectrum),rtol=3e-9,atol=3e-9)
                target=((center@vh.T)/(alpha*s*s+lam))@vh
                action=center@sigma
                np.testing.assert_allclose(action,target,rtol=3e-9,atol=3e-9)
                certificates.append(dict(step=position,n_rows=X.shape[0],n_predictors=p,rank=int(rank),
                    eigenvalue_error=float(np.max(abs(eigenvalues-np.sort(spectrum)),initial=0)),
                    row_space_error=float(np.max(abs(action-target),initial=0)),
                    sklearn_covariance_difference=float(np.max(abs(self.sigma_-sigma),initial=0))))
                self.sigma_=sigma.copy()
            else:
                np.testing.assert_allclose(sigma,self.sigma_,rtol=3e-9,atol=3e-9)
            return self
    replay=IterativeImputer(**kw,estimator=AuditedBayesianRidge())
    result=expected_results(replay,x,q)
    assert AuditedBayesianRidge.fitted==len(records) and certificates
    return replay,result,certificates

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);args=ap.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile)
    checks=[];source_warnings=Counter();cases=0;max_errors={};conditional=[]
    for seed,case,kw,bounds,x,q in fixtures():
        original=x.copy();query_original=q.copy();model=IterativeImputer(**kw)
        with warnings.catch_warnings(record=True) as caught:
            expected=expected_results(model,x,q)
        source_warnings.update(str(w.message) for w in caught)
        z=expected['output'];a=expected['query'];b=expected['query_repeat']
        run=subprocess.run([str(exe)],input=encode(kw,bounds,x,q),cwd=ROOT,text=True,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert run.returncode==0,(seed,case,run.stdout)
        lines=run.stdout.splitlines();actual=dict(metadata=np.fromstring(lines[0],sep=' ',dtype=int),
            output=np.fromstring(lines[1],sep=' ').reshape(z.shape,order='F'),
            query=np.fromstring(lines[2],sep=' ').reshape(a.shape,order='F'),
            query_repeat=np.fromstring(lines[3],sep=' ').reshape(b.shape,order='F'),
            statistics=np.fromstring(lines[4],sep=' '),names=np.array(lines[7].split()))
        records=[]
        for j in range(int(actual['metadata'][2])):
            header=np.fromstring(lines[10+4*j],sep=' ',dtype=int)
            neighbors=np.fromstring(lines[11+4*j],sep=' ',dtype=int)-1
            parameters=np.fromstring(lines[12+4*j],sep=' ')
            covariance=np.fromstring(lines[13+4*j],sep=' ').reshape((header[1],header[1]),order='F')
            records.append(dict(feature=header[0]-1,neighbors=neighbors,parameters=parameters,sigma=covariance))
        original_errors={key:float(np.max(abs(actual[key]-expected[key]),initial=0))
                         for key in ['output','query','query_repeat']}
        needs_replay=any(not np.allclose(actual[key],expected[key],rtol=3e-11,atol=3e-11)
                         for key in ['output','query','query_repeat'])
        if needs_replay:
            assert kw['sample_posterior'],(seed,case,original_errors)
            with warnings.catch_warnings(record=True):model,expected,certificates=covariance_replay(kw,x,q,records)
            conditional.append(dict(seed=seed,case=case,reason='nonunique wide thin-SVD null-space completion',
                                    direct_default_max_errors=original_errors,certificates=certificates))
        for key,want in expected.items():
            if key in ['metadata','names']:np.testing.assert_array_equal(actual[key],want,err_msg=f'{seed} {case} {key}')
            else:
                np.testing.assert_allclose(actual[key],want,rtol=3e-11,atol=3e-11,err_msg=f'{seed} {case} {key}')
                error=float(np.max(np.abs(actual[key]-want),initial=0))
                max_errors[key]=max(max_errors.get(key,0.),error)
            checks.append(dict(seed=seed,case=case,name=key,
                               oracle='sklearn_with_certified_covariance' if needs_replay else 'sklearn',passed=True))
        for j,triplet in enumerate(model.imputation_sequence_):
            np.testing.assert_array_equal(records[j]['feature'],triplet.feat_idx,err_msg=f'{seed} {case} sequence')
            np.testing.assert_array_equal(records[j]['neighbors'],triplet.neighbor_feat_idx,err_msg=f'{seed} {case} sequence')
        checks.append(dict(seed=seed,case=case,name='sequence',
                           oracle='sklearn_with_certified_covariance' if needs_replay else 'sklearn',passed=True))
        np.testing.assert_array_equal(x,original);np.testing.assert_array_equal(q,query_original)
        np.testing.assert_array_equal(np.fromstring(lines[8],sep=' ').reshape(x.shape,order='F'),original)
        np.testing.assert_array_equal(np.fromstring(lines[9],sep=' ').reshape(q.shape,order='F'),query_original)
        # Imputation must preserve every observed value and its indicator mask.
        mask=np.isnan(x) if np.isnan(kw['missing_values']) else x==kw['missing_values']
        valid=np.flatnonzero(~mask.all(axis=0) | kw['keep_empty_features'])
        np.testing.assert_array_equal(np.fromstring(lines[5],sep=' ',dtype=int)-1,valid)
        np.testing.assert_array_equal(np.fromstring(lines[6],sep=' ',dtype=int)-1,np.flatnonzero(mask.any(axis=0)))
        observed=~mask[:,valid];np.testing.assert_array_equal(actual['output'][:,:len(valid)][observed],x[:,valid][observed])
        query_mask=np.isnan(q) if np.isnan(kw['missing_values']) else q==kw['missing_values']
        observed=~query_mask[:,valid]
        np.testing.assert_array_equal(actual['query'][:,:len(valid)][observed],q[:,valid][observed])
        checks.append(dict(seed=seed,case=case,name='input_observed_identity',oracle='identity',passed=True))
        cases+=1
        if cases%60==0:print(f'{cases} iterative imputation fixtures passed',flush=True)
    root=Path(sklearn.__file__).parent
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,
                fixtures=cases,checks=checks,source_warnings=dict(source_warnings),max_absolute_errors=max_errors,
                conditional_covariance_replays=conditional,
                native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest()
                               for p in ['impute/_iterative.py','impute/_base.py','linear_model/_bayes.py']})
    (ROOT/'validation/results'/f'iterative_impute_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {cases} fixtures passed')

if __name__=='__main__':main()
