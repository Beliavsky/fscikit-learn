"""Dense L2 LogisticRegressionCV paths, scoring and refit semantics versus sklearn."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import os
import subprocess
import warnings
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import sklearn
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile):
    opts=['--profile',profile] if profile=='release' else ['--flag',FLAGS]
    p=subprocess.run(['fpm','run',*opts,'--target','reference_logistic_cv_driver','--runner'],cwd=ROOT,
                     text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/s.strip().strip('"') for s in p.stdout.splitlines()
           if s.strip().strip('"').endswith(('reference_logistic_cv_driver','reference_logistic_cv_driver.exe'))]
    paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,p.stdout
    return paths[0]


def gradient(beta,x,y,classes,w,c,intercept):
    p=x.shape[1]
    logits=x@beta[:,:p].T+(beta[:,p] if intercept else 0.)
    target=np.searchsorted(classes,y)
    if len(classes)==2:
        difference=expit(logits)-target[:,None]
    else:
        difference=softmax(logits,axis=1)-np.eye(len(classes))[target]
    difference*=w[:,None]
    g=difference.T@x+beta[:,:p]/c
    return max(np.max(abs(g)),np.max(abs(difference.sum(axis=0))) if intercept else 0.)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--executable',type=Path)
    parser.add_argument('--profile',choices=['checked','release'],default='checked')
    args=parser.parse_args();RESULTS.mkdir(exist_ok=True)
    exe=args.executable.resolve() if args.executable else resolve(args.profile)
    checks=[];source_warnings=[];source_gauges=[];fixtures=0

    def run(seed,case,x,y,query,truth,*,cs=4,refit=True,intercept=True,scoring='accuracy',
            sample_weight=None,class_weight=None,cv=3,max_iter=1000):
        nonlocal fixtures
        classes=np.unique(y);k=len(classes);m=1 if k==2 else k;p=x.shape[1]
        q=p+int(intercept)
        folds=list(StratifiedKFold(cv).split(x,y)) if isinstance(cv,int) else cv
        options=dict(Cs=cs,refit=refit,fit_intercept=intercept,scoring=scoring,cv=cv,
                     solver='newton-cg',tol=1.e-12,max_iter=0 if max_iter==0 else 2000,class_weight=class_weight)
        test_weight=1+np.arange(len(query))%4*.25
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model=LogisticRegressionCV(**options).fit(x,y,sample_weight=sample_weight)
            path_classes=classes[1:] if k==2 else classes
            paths=np.stack([model.coefs_paths_[c] for c in path_classes])
            bias=model.intercept_.copy();decision=model.decision_function(query)
            if k>2 and intercept:
                # The multinomial intercept has a free common-shift mode.
                shifts=paths[:,:,:,-1].mean(axis=0)
                paths[:,:,:,-1]-=shifts[None,:,:]
                final_shift=float(bias.mean());bias-=final_shift;decision=decision-final_shift
                source_gauges.append(dict(seed=seed,case=case,path_shifts=shifts.tolist(),final_shift=final_shift))
            scores=model.scores_[path_classes[0]]
            for c in path_classes:np.testing.assert_array_equal(model.scores_[c],scores)
            indices=(np.repeat(np.argmax(scores.sum(axis=0)),len(folds)) if refit else np.argmax(scores,axis=1))
            initialization=np.mean([paths[:,f,j,:] for f,j in enumerate(indices)],axis=0)
            if class_weight is None:
                cw=np.ones(k)
            elif isinstance(class_weight,str):
                cw=np.array([len(y)/(k*np.count_nonzero(y==c)) for c in classes])
            else:
                cw=np.array([class_weight.get(c,1.) for c in classes])
            expected=dict(classes=classes,path_classes=path_classes,grid=model.Cs_,selected=model.C_,cw=cw,
                          scores=scores,paths=paths,init=initialization,coef=model.coef_,bias=bias,
                          proba=model.predict_proba(query),decision=decision,predict=model.predict(query),
                          score=[model.score(query,truth)],weighted_score=[model.score(query,truth,sample_weight=test_weight)],
                          base_weighted_accuracy=[accuracy_score(truth,model.predict(query),sample_weight=test_weight)],
                          features=[p],refit_classes=[4,9],refit_features=[1],refit_grid=[1.e-4,1.e4])
            signed=test_weight.copy();signed[0]=-.5
            expected['base_signed_accuracy']=[accuracy_score(truth,model.predict(query),sample_weight=signed)]
        source_warnings.extend(dict(seed=seed,case=case,category=w.category.__name__,message=str(w.message)) for w in caught)
        explicit=not isinstance(cs,int);classmode=0 if class_weight is None else 1 if isinstance(class_weight,str) else 2
        lines=[f'{len(x)} {p} {len(query)} {len(cs) if explicit else 0} '
               f'{0 if isinstance(cv,int) else len(folds)} {int(sample_weight is not None)} {classmode}',
               f'{4 if explicit else cs} {cv if isinstance(cv,int) else 3} {max_iter} 1.e-10 '
               f'{int(intercept)} {int(refit)} {scoring}']
        if explicit:lines+=[' '.join(map(str,cs))]
        lines+=[' '.join(map(str,row)) for row in x]+[' '.join(map(str,y))]
        if sample_weight is not None:lines+=[' '.join(map(str,sample_weight))]
        lines+=[' '.join(map(str,row)) for row in query]+[' '.join(map(str,truth)),' '.join(map(str,test_weight))]
        if classmode==2:lines+=[str(k),' '.join(map(str,cw))]
        if not isinstance(cv,int):
            for train,test in folds:lines+=[f'{len(train)} {len(test)}',' '.join(map(str,train+1)),' '.join(map(str,test+1))]
        inp=RESULTS/f'logistic_cv_{args.profile}_input.txt';out=RESULTS/f'logistic_cv_{args.profile}_output.txt'
        inp.write_text('\n'.join(lines)+'\n')
        proc=subprocess.run([str(exe),str(inp),str(out)],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert proc.returncode==0,(seed,case,proc.stdout)
        actual={s.split()[0]:np.asarray(s.split()[1:],float) for s in out.read_text().splitlines()}
        counts=actual.pop('native_iterations');converged=actual.pop('native_converged')
        assert len(counts)==len(folds)*len(model.Cs_) and np.all((counts>=int(max_iter>0))&(counts<=max_iter))
        assert np.all(converged==int(max_iter>0)),(seed,case,'native convergence state')
        checks.append(dict(seed=seed,case=case,name='native_iterations',passed=True,oracle='native_iteration_bounds'))
        native_scores=actual['scores'].reshape(scores.shape,order='F')
        expected['best']=1+(np.repeat(np.argmax(native_scores.sum(axis=0)),len(folds)) if refit else np.argmax(native_scores,axis=1))
        native_paths=actual['paths'].reshape((m,len(folds),len(model.Cs_),q),order='F')
        weights=np.ones(len(y)) if sample_weight is None else sample_weight
        effective=weights*cw[np.searchsorted(classes,y)]
        maximum=0.
        for f,(train,test) in enumerate(folds):
            expected[f'train{f+1}']=train+1;expected[f'test{f+1}']=test+1
            if max_iter>0:
                for j,c in enumerate(model.Cs_):
                    maximum=max(maximum,gradient(native_paths[:,f,j,:],x[train],y[train],classes,effective[train],c,intercept))
        if refit and max_iter>0:
            beta=actual['coef'].reshape(m,p,order='F')
            if intercept:beta=np.column_stack([beta,actual['bias']])
            maximum=max(maximum,gradient(beta,x,y,classes,effective,actual['selected'][0],intercept))
        if max_iter>0:
            assert maximum<1.e-7,(seed,case,'raw L2 gradient',maximum)
        else:
            assert np.all(native_paths==0) and np.all(actual['coef']==0) and np.all(actual['bias']==0)
        checks.append(dict(seed=seed,case=case,name='gradient',passed=True,
                           oracle='convex_stationarity' if max_iter>0 else 'zero_iteration_identity'))
        assert actual.keys()==expected.keys(),(actual.keys(),expected.keys())
        for name,want in expected.items():
            want=np.asarray(want).ravel(order='F')
            if name in ['classes','path_classes','predict','features','best','refit_classes','refit_features'] or name.startswith(('train','test')):
                np.testing.assert_array_equal(actual[name],want,err_msg=f'{seed} {case} {name}')
            else:
                np.testing.assert_allclose(actual[name],want,rtol=3.e-7,atol=3.e-8,equal_nan=True,err_msg=f'{seed} {case} {name}')
            checks.append(dict(seed=seed,case=case,name=name,passed=True,
                               oracle='score_selection_identity' if name=='best' else 'sklearn'))
        fixtures+=1

    for seed in [0,7,42,2026,314159]:
        rng=np.random.RandomState(seed);case=0
        for k in [2,3,5]:
            n=107;p=3;classes=np.arange(k)*17-31
            x=rng.normal(size=(n,p))+[.3,-.5,.2]
            latent=x@rng.normal(size=(p,k))+rng.normal(size=(n,k))*2
            target=np.argmax(latent,axis=1);target[:6*k]=np.tile(rng.permutation(k),6)
            y=classes[target];query=rng.normal(size=(23,p));truth=np.resize(classes,23)
            for scoring in ['accuracy','balanced_accuracy','neg_log_loss']+(['roc_auc'] if k==2 else []):
                for intercept in [False,True]:
                    for refit in [False,True]:
                        case+=1
                        cs=[1.,.03,.3,.3] if case%3==0 else 1 if case%3==1 else 4
                        sw=None if case%2==0 else rng.lognormal(0,.3,n)
                        if sw is not None:sw[::13]=0
                        cw=None if case%3==0 else 'balanced' if case%3==1 else dict(zip(classes,np.linspace(.5,2.,k)))
                        run(seed,case,x,y,query,truth,cs=cs,refit=refit,intercept=intercept,
                            scoring=scoring,sample_weight=sw,class_weight=cw)
        x=rng.normal(size=(83,3));classes=np.array([-8,4,27]);y=rng.permutation(np.resize(classes,83))
        query=rng.normal(size=(17,3));truth=np.resize(classes,17)
        for refit in [False,True]:
            case+=1
            run(seed,case,x,y,query,truth,cs=[.03,.3,3.],class_weight='balanced',
                sample_weight=rng.uniform(.2,2.,83),refit=refit,scoring='neg_log_loss')
        folds=[(np.r_[np.arange(60)[::-1],0,0],np.arange(30,83)),
               (np.arange(20,83),np.r_[np.arange(20),2])]
        case+=1
        run(seed,case,x,y,query,truth,cv=folds,cs=[1.,.2,.2],scoring='balanced_accuracy')
        case+=1
        run(seed,case,x,y,query,truth,cv=[folds[0]],cs=10,refit=False)
        case+=1
        run(seed,case,x,y,query,truth,cs=[.4],intercept=False,class_weight={-8:0.,4:1.,27:2.},scoring='neg_log_loss')
        classes=np.array([-8,4]);y=np.resize(classes,83);truth=np.resize(classes,17)
        one_class=[(np.flatnonzero(y==-8),np.arange(83))]
        for refit in [False,True]:
            case+=1
            run(seed,case,x,y,query,truth,cv=one_class,cs=[.1,1.],intercept=False,refit=refit)
        case+=1
        run(seed,case,x,y,query,truth,cv=[(np.arange(83),np.flatnonzero(y==-8))],
            cs=[.1,1.],intercept=False,scoring='roc_auc')
        case+=1
        run(seed,case,x,y,query,truth,cs=[.1,1.],intercept=False,scoring='balanced_accuracy',
            sample_weight=(y==4).astype(float))
        for refit in [False,True]:
            case+=1
            run(seed,case,x,y,query,truth,cs=[1.,.1],max_iter=0,refit=refit)
        print(f'seed {seed}: logistic CV passed',flush=True)
    root=Path(inspect.getfile(LogisticRegressionCV)).parent.parent
    paths=[root/'linear_model/_logistic.py',root/'model_selection/_split.py',root/'utils/class_weight.py',
           root/'metrics/_classification.py',root/'metrics/_scorer.py']
    report=dict(sklearn=sklearn.__version__,profile=args.profile,fixtures=fixtures,checks=checks,source_warnings=source_warnings,
                source_gauges=source_gauges,
                source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (RESULTS/f'logistic_cv_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} logistic-CV groups across {fixtures} fixtures passed')


if __name__=='__main__':main()
