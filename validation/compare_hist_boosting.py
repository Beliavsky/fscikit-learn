"""Numeric histogram boosting vs installed sklearn, all stages and bin maps."""
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
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier

ROOT=Path(__file__).resolve().parents[1]
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'

def fixtures():
    for seed in [0,7,42,2026,314159]:
        rng=np.random.RandomState(seed)
        for loss in ['squared_error','absolute_error','poisson','gamma','quantile','binary','multiclass','one_class']:
            for case in range(12):
                n=[1,47,91,300][case%4];p=1+case%6;m=13
                x=rng.normal(size=(n,p));q=rng.normal(size=(m,p));signal=x[:,0]+.2*rng.normal(size=n)
                if loss=='poisson':y=rng.poisson(np.exp(.3*signal)).astype(float);y[0]+=1
                elif loss=='gamma':y=np.exp(.3*signal)+.1
                elif loss=='one_class':y=np.full(n,7)
                elif loss=='binary':y=np.where(signal>0,7,-3)
                elif loss=='multiclass':y=np.array([-3,7,11])[rng.randint(3,size=n)]
                else:y=signal
                if case%3==0:x[rng.uniform(size=x.shape)<.2]=np.nan;q[:2]=np.nan
                if case%7==0:x[:,0]=np.nan
                if case==8:x[:]=1;q[0]=np.nan
                if case==9:x[0]=-np.inf;x[-1]=np.inf;q[0]=-np.inf;q[-1]=np.inf
                w=rng.uniform(.1,3,size=n);w[::7]=0;w[-1]=1
                mono=rng.randint(-1,2,size=p) if case==7 and loss!='multiclass' else np.zeros(p,int)
                yield dict(seed=seed,case=case,loss=loss,kind=int(loss in ['binary','multiclass','one_class']),
                           max_iter=8,min_leaf=[1,5,20][case%3],max_leaves=[2,3,7,None][case%4],
                           max_depth=[None,1,3][case%3],max_bins=[2,7,31,255][case%4],
                           rate=.1 if case%2 else .3,l2=.5 if case%2 else 0.,tol=1.e-7,
                           early=bool(case%4==0 or case in [7,10]),patience=3,weighted=bool(case%2),balanced=bool(case%3==1),
                           warm=3 if case%5==0 else 0,quantile=.3),x,q,y,w,mono
    rng=np.random.RandomState(2718);n=200017
    x=rng.normal(size=(n,1));q=np.array([[-1.],[0.],[1.],[np.nan]])
    y=np.sin(x[:,0])+.1*rng.normal(size=n)
    yield dict(seed=0,case=5000,loss='squared_error',kind=0,max_iter=2,min_leaf=20,max_leaves=3,
               max_depth=None,max_bins=7,rate=.1,l2=.3,tol=1.e-7,early=False,patience=3,
               weighted=False,balanced=False,warm=0,quantile=.3),x,q,y,np.ones(n),np.zeros(1,int)
    yield dict(seed=0,case=5001,loss='binary',kind=1,max_iter=2,min_leaf=20,max_leaves=3,
               max_depth=None,max_bins=7,rate=.1,l2=.3,tol=1.e-7,early=False,patience=3,
               weighted=False,balanced=False,warm=0,quantile=.3),x,q,np.where(x[:,0]>0,7,-3),np.ones(n),np.zeros(1,int)
    for loss,n,weighted in [('absolute_error',5,False),('absolute_error',6,True),('quantile',6,False)]:
        x=np.arange(n,dtype=float)[:,None];q=np.array([[0.],[1.],[np.nan]])
        yield dict(seed=0,case=6000+n+int(weighted),loss=loss,kind=0,max_iter=3,min_leaf=1,max_leaves=3,
                   max_depth=None,max_bins=7,rate=.1,l2=0.,tol=1.e-7,early=False,patience=3,
                   weighted=weighted,balanced=False,warm=0,quantile=.3),x,q,np.full(n,1e308),np.ones(n),np.zeros(1,int)
    x=np.arange(6,dtype=float)[:,None];q=np.array([[-1.],[0.],[1.]])
    yield dict(seed=0,case=7000,loss='squared_error',kind=0,max_iter=3,min_leaf=2147483647,max_leaves=3,
               max_depth=None,max_bins=7,rate=.1,l2=0.,tol=1.e-7,early=False,patience=3,
               weighted=False,balanced=False,warm=0,quantile=.3),x,q,np.arange(6,dtype=float),np.ones(6),np.zeros(1,int)
    n=63;x=np.arange(n,dtype=float)[:,None];q=np.array([[0.],[30.],[np.nan]])
    yield dict(seed=0,case=8000,loss='one_class',kind=1,max_iter=150,min_leaf=1,max_leaves=3,
               max_depth=None,max_bins=7,rate=.1,l2=0.,tol=0.,early=True,patience=3,
               weighted=True,balanced=False,warm=0,quantile=.3),x,q,np.full(n,7),np.full(n,1e20),np.zeros(1,int)

def encode(c,x,q,y,w,monotonic):
    n,p=x.shape
    header=[c['kind'],n,p,len(q),c['max_iter'],c['max_leaves'] or 0,c['max_depth'] or 0,c['min_leaf'],
            c['max_bins'],c['seed'],int(c['early']),c['patience'],int(c['weighted']),int(c['balanced']),c['warm']]
    loss='log_loss' if c['kind'] else c['loss']
    def line(a):return ' '.join(str(v) for v in a)
    lines=[line(header),f"{loss} {c['rate']} {c['l2']} {c['tol']} {c['quantile']}"]
    lines.extend(line(row) for a in [x,q] for row in a)
    lines.extend([line(y),line(w),line(monotonic)])
    return '\n'.join(lines)+'\n'

def resolve(profile):
    flags=['--profile','release'] if profile=='release' else ['--flag',FLAGS]
    r=subprocess.run(['fpm','run',*flags,'--target','reference_hist_boosting_driver','--runner'],
                     cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/s.strip().strip('"') for s in r.stdout.splitlines()
           if s.strip().strip('"').endswith(('reference_hist_boosting_driver.exe','reference_hist_boosting_driver'))]
    paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,r.stdout
    return paths[0]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path)
    ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--cases',type=int,nargs='+');args=ap.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile)
    checks=[];source_warnings=Counter();cases=0;errors={}
    for c,x,q,y,w,mono in fixtures():
        if args.cases is not None and c['case'] not in args.cases:continue
        kw=dict(max_iter=c['max_iter'],max_leaf_nodes=c['max_leaves'],max_depth=c['max_depth'],
                min_samples_leaf=c['min_leaf'],max_bins=c['max_bins'],learning_rate=c['rate'],
                l2_regularization=c['l2'],random_state=c['seed'],early_stopping=c['early'],
                validation_fraction=None,n_iter_no_change=c['patience'],tol=c['tol'],
                monotonic_cst=mono if np.any(mono) else None)
        if c['kind']:
            model=HistGradientBoostingClassifier(**kw,class_weight='balanced' if c['balanced'] else None)
        else:
            model=HistGradientBoostingRegressor(**kw,loss=c['loss'],quantile=c['quantile'] if c['loss']=='quantile' else None)
        fit_kw={'sample_weight':w} if c['weighted'] else {}
        with warnings.catch_warnings(record=True) as caught:
            if c['warm']:
                model.set_params(max_iter=c['warm']);model.fit(x,y,**fit_kw)
                model.set_params(warm_start=True,max_iter=c['max_iter'])
            model.fit(x,y,**fit_kw)
            expected=dict(metadata=np.array([model.n_iter_,model.n_trees_per_iteration_,x.shape[1]]),
                          baseline=model._baseline_prediction.ravel(),raw_train=model._raw_predict(x),
                          raw_query=model._raw_predict(q),staged_raw=np.stack(list(model._staged_raw_predict(q)),axis=2),
                          predictions=model.predict(q),train_score=model.train_score_,
                          bins=model._bin_mapper.n_bins_non_missing_)
            if c['kind']:expected.update(probabilities=model.predict_proba(q),classes=model.classes_)
        source_warnings.update(str(v.message) for v in caught)
        run=subprocess.run([str(exe)],input=encode(c,x,q,y,w,mono),cwd=ROOT,text=True,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert run.returncode==0,(c,run.stdout)
        lines=run.stdout.splitlines();niter,nt,p=map(int,lines[0].split())
        def array(i):return np.fromstring(lines[i],sep=' ')
        actual=dict(metadata=array(0),baseline=array(1),raw_train=array(2).reshape((len(x),nt),order='F'),
                    raw_query=array(3).reshape((len(q),nt),order='F'),
                    staged_raw=array(4).reshape((len(q),nt,niter),order='F'),predictions=array(5),
                    train_score=array(7),bins=array(9))
        if c['kind']:actual.update(probabilities=array(6).reshape((len(q),max(2,nt)),order='F'),classes=array(8))
        for key,want in expected.items():
            if key in ['metadata','classes','bins'] or key=='predictions' and c['kind']:
                np.testing.assert_array_equal(actual[key],want,err_msg=f'{c} {key}')
            else:
                absolute_tolerance=0. if c['case']==8000 and key in ['train_score','probabilities'] else 3e-10
                np.testing.assert_allclose(actual[key],want,rtol=3e-10,atol=absolute_tolerance,err_msg=f'{c} {key}')
                errors[key]=max(errors.get(key,0.),float(np.max(abs(actual[key]-want),initial=0)))
            checks.append(dict(**c,name=key,passed=True))
        for j in range(p):np.testing.assert_array_equal(array(10+j),model._bin_mapper.bin_thresholds_[j],err_msg=str(c))
        checks.append(dict(**c,name='bin_thresholds',passed=True))
        for k in range(nt):
            want=np.array([len(trees[k].nodes) for trees in model._predictors])
            np.testing.assert_array_equal(array(10+p+k),want,err_msg=f'{c} node counts')
        checks.append(dict(**c,name='node_counts',passed=True))
        cases+=1
        if cases%96==0:print(f'{cases} histogram boosting fixtures passed',flush=True)
    root=Path(sklearn.__file__).parent
    source=['ensemble/_hist_gradient_boosting/gradient_boosting.py','_loss/loss.py','_loss/_loss.pyx.tp']
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,fixtures=cases,cases_filter=args.cases,
                checks=checks,max_absolute_errors=errors,source_warnings=dict(source_warnings),
                native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source})
    (ROOT/'validation/results'/f'hist_boosting_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {cases} histogram boosting fixtures passed')

if __name__=='__main__':main()
