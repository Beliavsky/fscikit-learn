"""Multi-task CV paths plus seeded random/base zero-penalty coordinate descent."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import warnings
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import sklearn
from sklearn.linear_model import MultiTaskElasticNetCV,MultiTaskLassoCV,MultiTaskElasticNet,MultiTaskLasso
from sklearn.model_selection import KFold,PredefinedSplit
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def rows(x,y):return [' '.join(f'{v:.17g}' for v in row) for row in np.column_stack([x,y])]
def logical(value):return 'T' if value else 'F'


def run(executable,lines):
    inp=RESULTS/'multitask_cv_input.txt';out=RESULTS/'multitask_cv_output.txt';inp.write_text('\n'.join(lines)+'\n')
    proc=subprocess.run([str(executable),str(inp),str(out)],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert proc.returncode==0,proc.stdout
    actual={}
    for line in out.read_text().splitlines():
        name,*values=line.split();actual[name]=np.array(values,dtype=float)
    return actual


def compare(actual,expected,checks,kind,seed,case,equivalences,allowed_ratios=None):
    assert actual.keys()==expected.keys(),(kind,seed,case)
    for name,value in expected.items():
        tol=0 if name.endswith(('meta','fold_iterations','fold_labels','fold_sizes')) else 3.e-8
        want=np.asarray(value).ravel(order='F');oracle='sklearn'
        if name.endswith('selected') and actual[name][0]==0 and want[0]==0 and actual[name][1]!=want[1]:
            assert allowed_ratios is not None and actual[name][1] in allowed_ratios
            # Both L1/L2 strengths are identically zero; only the inactive ratio differs.
            assert actual[name][0]*actual[name][1]==want[0]*want[1]==0
            assert actual[name][0]*(1-actual[name][1])==want[0]*(1-want[1])==0
            np.testing.assert_allclose(actual[name][[0,2]],want[[0,2]],atol=tol,rtol=tol)
            equivalences.append(dict(kind=kind,seed=seed,case=case,name=name,native_ratio=float(actual[name][1]),source_ratio=float(want[1])))
            oracle='inactive ratio at zero alpha'
        else:
            np.testing.assert_allclose(actual[name],want,atol=tol,rtol=tol,
                                       err_msg=f'kind={kind} seed={seed} case={case} {name}')
        checks.append(dict(kind=kind,seed=seed,case=case,name=name,passed=True,oracle=oracle))


def cv_cases(executable,checks,equivalences):
    for seed,n,p,t in [(0,32,6,3),(7,29,5,2),(42,37,9,4),(2026,23,3,1)]:
        rng=np.random.RandomState(seed);x0=rng.normal(size=(n,p));beta=rng.normal(size=(p,t));beta[3:]=0
        y0=x0@beta+np.arange(t)+.05*rng.normal(size=(n,t));qx=rng.normal(size=(8,p));qy=qx@beta+np.arange(t)
        for kind in [1,2]:
            for case in range(24):
                x=x0.copy();y=y0.copy();q=qx.copy();target=qy.copy();ops=[]
                c=dict(cv=3,n_alphas=6,max_iter=2000,random_state=seed,eps=.01,tol=1.e-7,l1_ratio=.5,
                       fit_intercept=True,selection='cyclic',alphas=None,test_fold=None,warm_start=False)
                def add(op,a,b,**changes):
                    c.update(changes);ops.append((op,np.array(a,copy=True),np.array(b,copy=True),c.copy()))
                if case==1:c.update(fit_intercept=False)
                if case==2:c.update(alphas=np.array([.05,.5,.15,.5,.01]))
                if case==3:c.update(l1_ratio=np.array([.2,.8,1.])) if kind==1 else c.update(n_alphas=3,cv=7)
                if case==4:c.update(l1_ratio=np.array([0.,.5]),alphas=np.array([1.,.1,0.]),max_iter=200,tol=1.e-6)
                if case==5:c.update(selection='random')
                if case==6:c.update(selection='random',max_iter=1,tol=0.)
                if case==7:c.update(selection='random',l1_ratio=np.array([.1,.9]),alphas=np.array([.5,.02,.1,.1]))
                if case==8:c.update(n_alphas=1)
                if case==9:y[:]=2.;target[:]=2.;c.update(l1_ratio=np.array([.9,.1]))
                if case==10:x[:]=2.;q[:]=2.
                if case==11:y*=1.e-20;target*=1.e-20
                if case==12:y=y[:,:1];target=target[:,:1]
                if case==13:c.update(test_fold=np.where(np.arange(n)%4==0,-1,np.where(np.arange(n)%3==0,8,-2)))
                if case==14:
                    fold=np.full(n,-1);fold[-4:]=99;c.update(test_fold=fold)
                if case==15:c.update(cv=n,n_alphas=3)
                if case==16:c.update(eps=2.,n_alphas=4)
                if case==17:x[:,-1]=3.;q[:,-1]=3.;x[:,-2]=x[:,0];q[:,-2]=q[:,0]
                if case==18:c.update(alphas=np.array([100.,1000.]))
                if case==19:c.update(alphas=np.array([0.]),max_iter=200,tol=1.e-6)
                if case==20:
                    x=rng.normal(size=(15,23));b=rng.normal(size=(23,t));b[3:]=0
                    y=x@b+.05*rng.normal(size=(15,t));q=rng.normal(size=(8,23));target=q@b
                    c.update(n_alphas=4,max_iter=5000,tol=1.e-6,eps=.05)
                if case==21:c.update(cv=5,n_alphas=100,max_iter=1000,eps=.001,tol=1.e-4)
                if case==23:c.update(l1_ratio=np.array([.01,.99]),n_alphas=5,eps=.1 if kind==1 else 1.)
                add(1,x,y)
                if case==22:
                    x=np.column_stack([x,np.ones(len(x))]);y=np.column_stack([y,np.zeros(len(y))])
                    q=np.column_stack([q,np.ones(len(q))]);target=np.column_stack([target,np.zeros(len(target))])
                    add(1,x,y,warm_start=True,n_alphas=4)
                add(2,q,target);add(3,q,target)
                source=(MultiTaskElasticNetCV if kind==1 else MultiTaskLassoCV)(n_jobs=1)
                lines=[f'{kind} {len(ops)}'];expected={};last=None
                for i,(op,a,b,config) in enumerate(ops,1):
                    kw={k:v for k,v in config.items() if k not in ('test_fold','warm_start')}
                    if kind==2:kw.pop('l1_ratio')
                    splitter=PredefinedSplit(config['test_fold']) if config['test_fold'] is not None else KFold(config['cv'])
                    kw['cv']=splitter;source.set_params(**kw);prefix=f'{i}_'
                    ratios=np.atleast_1d(config['l1_ratio']) if kind==1 else np.array([1.])
                    vector=np.atleast_1d(config['l1_ratio']) if np.ndim(config['l1_ratio']) else np.array([])
                    na=0 if config['alphas'] is None else len(config['alphas'])
                    scalar=float(config['l1_ratio']) if not len(vector) else .5
                    lines.extend([f'{op} {a.shape[0]} {a.shape[1]} {b.shape[1]}',
                        f"{config['cv']} {config['n_alphas']} {config['max_iter']} {config['random_state']} {na} {len(vector)} {int(config['test_fold'] is not None)}",
                        f"{config['eps']:.17g} {config['tol']:.17g} {scalar:.17g} {logical(config['fit_intercept'])} {logical(config['warm_start'])}",config['selection']])
                    if na:lines.append(' '.join(f'{v:.17g}' for v in config['alphas']))
                    if len(vector):lines.append(' '.join(f'{v:.17g}' for v in vector))
                    if config['test_fold'] is not None:lines.append(' '.join(map(str,config['test_fold'])))
                    lines.extend(rows(a,b));result=None
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        if op==1:
                            recorded=[];original=type(source).path
                            def observe(*args,**kwargs):
                                kwargs.pop('return_n_iter',None)
                                values=original(*args,return_n_iter=True,**kwargs)
                                recorded.append((np.asarray(values[3]).copy(),np.asarray(values[2]).copy()))
                                return values[:3]
                            source.path=observe;source.fit(a,b)
                            folds=list(splitter.split(a,b));nf=len(folds);nr=len(ratios)
                            nalphas=len(np.atleast_1d(source.alphas_).ravel())//nr if config['alphas'] is None else na
                            alpha_matrix=np.asarray(source.alphas_)
                            if alpha_matrix.ndim==1:alpha_matrix=np.tile(alpha_matrix,(nr,1))
                            fold_iterations=np.array([v[0] for v in recorded]).reshape((nr,nf,nalphas)).transpose(2,1,0)
                            fold_gaps=np.array([v[1] for v in recorded]).reshape((nr,nf,nalphas)).transpose(2,1,0)
                            mse=np.asarray(source.mse_path_).reshape((nr,nalphas,nf)).transpose(1,2,0)
                            labels=np.unique(config['test_fold']) if config['test_fold'] is not None else np.arange(nf)
                            labels=labels[labels!=-1]
                            last=dict(alphas=alpha_matrix.T,mse=mse,fold_iterations=fold_iterations,fold_gaps=fold_gaps,
                                      fold_labels=labels,fold_sizes=[len(test) for _,test in folds],
                                      selected=[source.alpha_,getattr(source,'l1_ratio_',1.),float(np.min(mse.mean(axis=1)))],nfolds=nf)
                        elif op==2:result=source.predict(a)
                        elif op==3:result=[source.score(a,b)]
                    if result is not None:expected[prefix+'result']=result
                    for key,value in last.items():
                        if key!='nfolds':expected[prefix+key]=np.asarray(value).copy()
                    expected[prefix+'coef']=source.coef_.copy();expected[prefix+'intercept']=np.broadcast_to(source.intercept_,(source.coef_.shape[0],)).copy()
                    # The full-data refit uses the data from the latest fit operation.
                    fit_y=next(v[2] for v in reversed(ops[:i]) if v[0]==1)
                    centered=fit_y-fit_y.mean(axis=0) if config['fit_intercept'] else fit_y
                    expected[prefix+'fit']=[source.dual_gap_,source.tol*np.sum(centered**2)]
                    expected[prefix+'meta']=[source.n_features_in_,source.n_iter_,last['nfolds']]
                compare(run(executable,lines),expected,checks,kind,seed,case,equivalences,ratios)
            print(f'CV kind {kind} seed {seed}: 24 configurations passed',flush=True)


def base_cases(executable,checks,equivalences):
    for seed,n,p,t in [(0,32,6,3),(7,29,5,2),(42,37,9,4),(2026,23,3,1)]:
        rng=np.random.RandomState(seed);x0=rng.normal(size=(n,p));coef=rng.normal(size=(p,t));coef[3:]=0
        y0=x0@coef+np.arange(t)+.05*rng.normal(size=(n,t))
        for kind in [3,4]:
            for case in range(10):
                x=x0.copy();y=y0.copy();ops=[]
                c=dict(alpha=.08,l1_ratio=.5,tol=1.e-8,max_iter=2000,random_state=seed,fit_intercept=True,warm_start=False,selection='random')
                def add(op,**changes):c.update(changes);ops.append((op,c.copy()))
                if case==2:c.update(max_iter=1,tol=0.)
                if case==3:c.update(alpha=0.,selection='cyclic',max_iter=300,tol=1.e-6)
                if case==4:c.update(alpha=.5,l1_ratio=0.,max_iter=300,tol=1.e-6)
                if case==5:y[:]=0
                if case==6:x[:]=2.
                if case==7:c.update(fit_intercept=False)
                if case==8:c.update(alpha=0.,max_iter=300,tol=1.e-6)
                if case==9:c.update(alpha=100.,selection='cyclic')
                if case==1:add(1,alpha=.5,warm_start=True);add(1,alpha=.08)
                else:add(1)
                add(2);add(3);source=(MultiTaskElasticNet if kind==3 else MultiTaskLasso)();lines=[f'{kind} {len(ops)}'];expected={}
                for i,(op,config) in enumerate(ops,1):
                    kw=config.copy()
                    if kind==4:kw.pop('l1_ratio')
                    source.set_params(**kw);prefix=f'{i}_'
                    lines.extend([f'{op} {len(x)} {x.shape[1]} {y.shape[1]}',
                        f"{config['alpha']:.17g} {config['l1_ratio']:.17g} {config['tol']:.17g} {config['max_iter']} {config['random_state']} {logical(config['fit_intercept'])} {logical(config['warm_start'])}",config['selection'],*rows(x,y)])
                    result=None
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        if op==1:source.fit(x,y)
                        elif op==2:result=source.predict(x)
                        elif op==3:result=[source.score(x,y)]
                    if result is not None:expected[prefix+'result']=result
                    centered=y-y.mean(axis=0) if config['fit_intercept'] else y
                    expected[prefix+'coef']=source.coef_.copy();expected[prefix+'intercept']=np.broadcast_to(source.intercept_,(source.coef_.shape[0],)).copy()
                    expected[prefix+'fit']=[source.dual_gap_,source.tol*np.sum(centered**2)]
                    expected[prefix+'meta']=[source.n_features_in_,source.n_iter_]
                compare(run(executable,lines),expected,checks,kind,seed,case,equivalences)
            print(f'base kind {kind} seed {seed}: 10 configurations passed',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path);args=parser.parse_args();RESULTS.mkdir(exist_ok=True)
    if args.executable:executable=args.executable.resolve()
    else:
        proc=subprocess.run(['fpm','run','--flag',FLAGS,'--target','reference_multitask_cv_driver','--runner'],cwd=ROOT,
                            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/line.strip().strip('"') for line in proc.stdout.splitlines()
               if line.strip().strip('"').endswith(('reference_multitask_cv_driver','reference_multitask_cv_driver.exe'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,proc.stdout
        executable=paths[0]
    checks=[];equivalences=[];cv_cases(executable,checks,equivalences);base_cases(executable,checks,equivalences)
    (RESULTS/'multitask_cv.json').write_text(json.dumps(dict(sklearn=sklearn.__version__,checks=checks,zero_alpha_equivalences=equivalences),indent=2)+'\n')
    print(f'{len(checks)} multi-task CV/random-coordinate result groups passed')


if __name__=='__main__':main()
