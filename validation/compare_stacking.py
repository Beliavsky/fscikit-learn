"""Dense stacking stages and RidgeCV LOO against sklearn 1.6.1."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import subprocess
import warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, RidgeCV, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.svm import LinearSVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score
from compare_weighted_trees import certify_cart
from scipy.special import expit, softmax, logsumexp

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def logistic_certificate(native,model,x,y,w,query):
    """Convex stationarity and a segment Hessian bound for Newton solutions."""
    weight=np.ones(len(y)) if w is None else w
    design=np.column_stack([x,np.ones(len(x))])
    beta=np.column_stack([native['coef'],native['bias']])
    reference=np.column_stack([model.coef_,model.intercept_])
    binary=len(model.classes_)==2
    target=np.searchsorted(model.classes_,y)
    def state(b):
        logits=design@b.T
        if binary:
            probability=expit(logits);difference=probability-target[:,None]
            loss=np.logaddexp(0,logits[:,0])-target*logits[:,0]
        else:
            probability=softmax(logits,axis=1);difference=probability-np.eye(len(model.classes_))[target]
            loss=logsumexp(logits,axis=1)-logits[np.arange(len(y)),target]
        gradient=(difference*weight[:,None]).T@design
        gradient[:,:-1]+=b[:,:-1]/model.C
        objective=weight@loss+.5*np.sum(b[:,:-1]**2)/model.C
        return probability,gradient,objective
    _,original_gradient,original_objective=state(beta)
    _,_,reference_objective=state(reference)
    assert np.max(abs(original_gradient))<=1.e-9
    assert original_objective<=reference_objective+256*np.finfo(float).eps*max(1,abs(reference_objective))
    gauge_gap=0.
    if not binary:
        original_difference=beta-reference
        beta=beta-beta.mean(axis=0,keepdims=True)
        reference=reference-reference.mean(axis=0,keepdims=True)
        gauge_gap=float(np.linalg.norm(original_difference-(beta-reference)))
    pn,gn,on=state(beta);ps,gs,os_=state(reference)
    assert np.max(abs(gn))<=1.e-9,(np.max(abs(gn)),on,os_)
    assert on<=os_+256*np.finfo(float).eps*max(1,abs(os_))
    curvature=np.minimum((pn*(1-pn))[:,0],(ps*(1-ps))[:,0]) if binary else np.minimum(pn.min(axis=1),ps.min(axis=1))
    lower=design.T@((weight*curvature)[:,None]*design)
    lower[:-1,:-1]+=np.eye(x.shape[1])/model.C
    mu=float(np.linalg.eigvalsh(lower)[0]);assert mu>0
    bound=(np.linalg.norm(gn)+np.linalg.norm(gs))/mu
    gap=float(np.linalg.norm(beta-reference))
    assert gap<=bound+256*np.finfo(float).eps*max(1,np.linalg.norm(beta))
    query_design=np.column_stack([query,np.ones(len(query))])
    return dict(native_gradient=float(np.linalg.norm(gn)),source_gradient=float(np.linalg.norm(gs)),
                curvature=mu,parameter_gap=gap,parameter_bound=float(bound),
                gauge_gap=gauge_gap,score_bound=float(np.linalg.norm(query_design,axis=1).max()*(bound+gauge_gap)))


def resolve(profile):
    opts = ['--profile', profile] if profile else ['--flag', FLAGS]
    run = subprocess.run(['fpm', 'run', *opts, '--target', 'reference_stacking_driver', '--runner'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    paths = [ROOT/line.strip().strip('"') for line in run.stdout.splitlines()
        if line.strip().strip('"').endswith(('reference_stacking_driver', 'reference_stacking_driver.exe'))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, run.stdout
    return paths[0]


def base(mode, code, seed):
    if mode == 0:
        return [LogisticRegression(C=.7, tol=1.e-10, max_iter=500, solver='newton-cg'), GaussianNB(),
            RidgeClassifier(alpha=.7), DecisionTreeClassifier(max_depth=2, random_state=seed),
            ExtraTreeClassifier(max_depth=2, random_state=seed), KNeighborsClassifier(n_neighbors=3, algorithm='brute'),
            DummyClassifier(), LinearSVC(C=.7, tol=1.e-10, max_iter=10000, dual=False)][code]
    return [LinearRegression(), Ridge(alpha=.7), KNeighborsRegressor(n_neighbors=3, algorithm='brute'),
        DecisionTreeRegressor(max_depth=2, random_state=seed), ExtraTreeRegressor(max_depth=2, random_state=seed),
        DummyRegressor(), KernelRidge(kernel='rbf', gamma=.4, alpha=.7)][code]


def final_model(mode, code):
    if code < 0:
        # The native classifier uses its Newton solver at a strict tolerance;
        # source uses the matching optimizer, rather than default LBFGS stopping.
        return (LogisticRegression(solver='newton-cg', tol=1.e-11, max_iter=500) if mode == 0 else
                RidgeCV(alphas=[.1, 1, 10], store_cv_results=True))
    return base(mode, [0, 1, 2, 3][code], 0)


def fixtures():
    witness=json.loads((ROOT/'validation/fixtures/stacking_logistic_roundoff.json').read_text())
    yield witness['cfg'],*(np.array(witness[key]) for key in ['x','y','w','q']),[]
    for seed in [0, 7, 42, 2026]:
        for mode in range(2):
            for case in range(16):
                rng = np.random.default_rng(9182+seed*100+mode*20+case)
                n,p,m=36,3,9
                k=2 if case in [1, 5, 10, 12] else 3
                ids=np.arange(n)%k; rng.shuffle(ids)
                x=rng.normal(size=(n,p))*.6; x[:,0]+=ids*1.8
                q=rng.normal(size=(m,p)); y=np.array([-13,7,29])[:k][ids] if mode==0 else x@np.array([.7,-1.2,.4])+.1*rng.normal(size=n)
                w=np.ones(n)
                cfg=dict(mode=mode,case=case,seed=seed,choices=[0,1,2],drop=[0,0,0],final=-1,cv=3,
                         passthrough=0,prefit=0,weighted=0,method=0,fi=1)
                plans=[]
                if case==1:cfg['passthrough']=1
                elif case==2:cfg['drop']=[0,1,0]
                elif case==3:cfg['choices']=[3,4,0]
                elif case==4:
                    cfg.update(choices=[0,1,3] if mode==0 else [0,1,3],weighted=1,passthrough=1)
                    w=rng.uniform(.2,2,n);w[::8]=0
                elif case==5:cfg.update(prefit=1)
                elif case==6:cfg.update(prefit=1,method=3 if mode==0 else 0)
                elif case==7:
                    perm=rng.permutation(n)
                    plans=[(perm[12:],perm[:12]),(np.r_[perm[:12],perm[24:]],perm[12:24]),(perm[:24],perm[24:])]
                elif case==8:cfg.update(method=3 if mode==0 else 0,final=0)
                elif case==9:cfg.update(choices=[0,2,7] if mode==0 else [0,1,6],method=2 if mode==0 else 0,final=2 if mode==0 else 1)
                elif case==10:cfg.update(choices=[1,3,5] if mode==0 else [0,1,2],method=1 if mode==0 else 0)
                elif case==11:cfg.update(final=1)
                elif case==12:cfg.update(final=3,choices=[3,4,0])
                elif case==13:cfg.update(choices=[6,0,1] if mode==0 else [5,0,1])
                elif case==14:
                    cfg.update(prefit=1,weighted=1,choices=[5,0,1] if mode==0 else [2,0,1])
                    w=rng.uniform(.5,2,n)
                elif case==15:
                    cfg.update(choices=[3,4,6] if mode==0 else [3,4,0],cv=2,final=3)
                    if mode==0:
                        y=np.ones(n,dtype=int)*7;y[0]=-13
                    plans=[(np.arange(18,36),np.arange(18)),(np.arange(18),np.arange(18,36))]
                yield cfg,x,y,w,q,plans
        for n,p in [(20,3),(6,10),(1,2),(8,1)]:
            for intercept in [0,1]:
                for weighted in [0,1]:
                    rng=np.random.default_rng(seed+762+n*4+p+intercept+weighted*7)
                    x=rng.normal(size=(n,p));q=rng.normal(size=(3,p));y=rng.normal(size=n);w=rng.uniform(.5,2,n)
                    if n>2:w[::4]=0
                    cfg=dict(mode=2,case=100+n*100+p*10+intercept*2+weighted,seed=seed,choices=[],drop=[],
                        final=-1,cv=0,passthrough=0,prefit=0,weighted=weighted,method=0,fi=intercept)
                    yield cfg,x,y,w,q,[]


def run(executable,profile):
    checks=[];certificates=[];optimizer_certificates=[];source_warnings=[];count=0
    for cfg,x,y,w,q,plans in fixtures():
        n,p=x.shape;m=len(q);mode=cfg['mode'];k=len(np.unique(y)) if mode==0 else 0
        inp,out=RESULTS/f'stacking_{profile or "checked"}_input.txt',RESULTS/f'stacking_{profile or "checked"}_output.txt'
        with inp.open('w') as f:
            f.write(' '.join(map(str,[mode,n,p,m,len(cfg['choices']),cfg['final'],cfg['cv'],cfg['passthrough'],
                cfg['prefit'],cfg['weighted'],cfg['method'],len(plans),cfg['seed'],cfg['fi']]))+'\n')
            for a in [np.array(cfg['choices'])[None],np.array(cfg['drop'])[None],x,y[None],w[None],q]:
                np.savetxt(f,a,fmt='%.17g')
            for train,test in plans:
                f.write(f'{len(train)} {len(test)}\n');np.savetxt(f,(train+1)[None],fmt='%d');np.savetxt(f,(test+1)[None],fmt='%d')
        proc=subprocess.run([str(executable),str(inp),str(out)],capture_output=True,text=True)
        assert proc.returncode==0,(cfg,proc.stdout,proc.stderr)
        rows=iter(out.read_text().splitlines())
        def vector(dtype=float):return np.fromstring(next(rows),sep=' ',dtype=dtype)
        def matrix(nrow):return np.array([vector() for _ in range(nrow)])
        def equal(a,b,name,exact=False,oracle='sklearn'):
            np.testing.assert_allclose(a,b,rtol=0 if exact else 3.e-8,atol=0 if exact else 3.e-9,
                                       equal_nan=True,err_msg=str((cfg,name)))
            checks.append(dict(seed=cfg['seed'],mode=mode,case=cfg['case'],name=name,oracle=oracle,passed=True))
        if mode==3:
            heads=int(vector(int)[0]);coef=matrix(heads);bias=vector();prob=matrix(m);pred=vector()
            model=LogisticRegression(C=1,solver='newton-cg',tol=1.e-12,max_iter=500).fit(x,y,sample_weight=w)
            equal(coef,model.coef_,'roundoff regression coefficients');equal(bias,model.intercept_,'roundoff regression intercept')
            equal(prob,model.predict_proba(q),'roundoff regression probabilities');equal(pred,model.predict(q),'roundoff regression labels',True)
            logits=x@coef.T+bias;logits-=logits.max(axis=1,keepdims=True)
            probability=np.exp(logits);probability/=probability.sum(axis=1,keepdims=True)
            difference=(probability-np.eye(heads)[np.searchsorted(np.unique(y),y)])*w[:,None]
            gradient=max(abs(difference.T@x+coef).max(),abs(difference.sum(axis=0)).max())
            assert gradient<1.e-9,gradient
            checks.append(dict(seed=cfg['seed'],mode=mode,case=cfg['case'],name='weighted logistic stationarity',oracle='convex_gradient',passed=True))
            assert list(rows)==[];count+=1;continue
        if mode==2:
            meta=vector();scores=vector();coef=vector();bias=vector()[0];pred=vector()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model=RidgeCV(alphas=[.1,1,10],fit_intercept=bool(cfg['fi']),store_cv_results=True).fit(x,y,sample_weight=w if cfg['weighted'] else None)
            equal(meta,[model.alpha_,model.best_score_],'LOO alpha/score')
            equal(scores,-np.mean(model.cv_results_,axis=0),'LOO candidate scores')
            equal(coef,model.coef_,'LOO coefficients');equal(bias,model.intercept_,'LOO intercept');equal(pred,model.predict(q),'LOO predictions')
            assert list(rows)==[];count+=1;continue
        features,width,bases,nclasses,flags=vector(int)
        classes=vector(int) if k else None
        active=vector(int)-1;outs=vector(int);methods=[next(rows).strip() for _ in active]
        predicted=vector();trained_pred=vector()
        proba=matrix(m) if flags&1 else None
        decision=matrix(m) if flags&2 and int(vector(int)[0]) else None
        score=vector()[0];training=matrix(n);transformed=matrix(m);train_transformed=matrix(n)
        def state():
            tag,a,b=vector(int)
            if tag==1:
                labels=vector(int) if k else None;importance=vector();nodes=[];stats=[];values=[]
                for _ in range(a):nodes.append(vector(int));stats.append(vector());values.append(vector())
                return dict(tag=tag,classes=labels,importance=importance,nodes=np.array(nodes),stats=np.array(stats),values=np.array(values),seed=b)
            if tag==2:return dict(tag=tag,coef=matrix(a),bias=vector())
            if tag==5:return dict(tag=tag,coef=matrix(a),bias=vector(),gradient=vector())
            if tag==3:return dict(tag=tag,coef=vector(),bias=vector()[0],meta=vector(),scores=vector())
            if tag==4:return dict(tag=tag,coef=vector(),bias=vector()[0])
            return dict(tag=0)
        final_state=state();full_states=[state() for _ in active]
        folds=[];fold_states=[]
        for _ in range(int(vector(int)[0])):
            nt,nv=vector(int);train=vector(int)-1;test=vector(int)-1
            assert len(train)==nt and len(test)==nv
            folds.append((train,test));fold_states.append([state() for _ in active])
        originals=[base(mode,c,cfg['seed']) for c in cfg['choices']]
        if cfg['prefit']:
            for model,drop in zip(originals,cfg['drop']):
                if not drop:model.fit(x,y)
        names=[(f'model{i+1}','drop' if drop else model) for i,(model,drop) in enumerate(zip(originals,cfg['drop']))]
        cv='prefit' if cfg['prefit'] else plans or cfg['cv']
        final=final_model(mode,cfg['final'])
        if cfg['final']==3:final.set_params(random_state=cfg['seed'])
        ctor=StackingClassifier if k else StackingRegressor
        opts=dict(estimators=names,final_estimator=final,cv=cv,passthrough=bool(cfg['passthrough']))
        if k:opts['stack_method']=['auto','predict_proba','decision_function','predict'][cfg['method']]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');expected=ctor(**opts).fit(x,y,sample_weight=w if cfg['weighted'] else None)
        equal([features,width,bases,nclasses],[p,transformed.shape[1],len(expected.estimators_),k],'metadata',True)
        equal(active,np.flatnonzero(np.array(cfg['drop'])==0),'active indices',True)
        assert methods==(expected.stack_method_ if k else ['predict']*len(active))
        equal(outs,np.array([1 if method=='predict' or k<=2 else k for method in methods]),'response widths',True)
        if k:equal(classes,expected.classes_,'global classes',True)
        target=np.searchsorted(classes,y) if k else y
        expected_plans=[] if cfg['prefit'] else (plans or list((StratifiedKFold(cfg['cv']) if k else KFold(cfg['cv'])).split(x,target)))
        equal(len(folds),len(expected_plans),'fold count',True)
        for i,(actual,reference) in enumerate(zip(folds,expected_plans)):
            equal(actual[0],reference[0],f'fold {i} train',True);equal(actual[1],reference[1],f'fold {i} test',True)
        any_conditioned=False
        def evaluate(native,model,fitx,fity,fitw,query,name):
            nonlocal any_conditioned
            method=name[1]
            raw=getattr(model,method)(query)
            if raw.ndim==1:raw=raw[:,None]
            if native['tag']==2:
                score_native=query@native['coef'].T+native['bias']
                probability_native=(np.column_stack([1-expit(score_native[:,0]),expit(score_native[:,0])])
                                    if len(model.classes_)==2 else softmax(score_native,axis=1))
                predicted_native=model.classes_[np.argmax(probability_native,axis=1)][:,None]
                response_native=(probability_native if method=='predict_proba' else score_native
                                 if method=='decision_function' else predicted_native)
                if not np.allclose(response_native,raw,rtol=3.e-8,atol=3.e-9):
                    witness=logistic_certificate(native,model,fitx,fity,fitw,query)
                    optimizer_certificates.append(dict(cfg=cfg,name=name[0],kind='logistic',**witness))
                    any_conditioned=True;raw=response_native
            if native['tag']==5:
                native_raw=query@native['coef'].T+native['bias']
                if not np.allclose(native_raw,raw,rtol=3.e-8,atol=3.e-9):
                    assert model.loss=='squared_hinge' and model.penalty=='l2' and not model.dual
                    augmented=np.column_stack([fitx,np.full(len(fitx),model.intercept_scaling)])
                    z=np.column_stack([native['coef'],native['bias']/model.intercept_scaling])
                    zs=np.column_stack([model.coef_,model.intercept_/model.intercept_scaling])
                    target_binary=np.column_stack([np.where(fity==c,1.,-1.) for c in model.classes_])
                    if len(model.classes_)==2:target_binary=target_binary[:,1:]
                    cost=model.C*(np.ones(len(fity)) if fitw is None else fitw)
                    margin=np.maximum(0,1-target_binary*(augmented@z.T))
                    source_margin=np.maximum(0,1-target_binary*(augmented@zs.T))
                    grad=z-2*((target_binary*margin*cost[:,None]).T@augmented)
                    source_grad=zs-2*((target_binary*source_margin*cost[:,None]).T@augmented)
                    norm=np.linalg.norm(grad,axis=1);source_norm=np.linalg.norm(source_grad,axis=1)
                    initial=np.maximum(1,np.linalg.norm(2*((target_binary*cost[:,None]).T@augmented),axis=1))
                    assert np.all(norm<=model.tol*initial+128*np.finfo(float).eps*initial),(cfg,name,norm,initial)
                    objective=.5*np.sum(z*z,axis=1)+np.sum(cost[:,None]*margin**2,axis=0)
                    source_objective=.5*np.sum(zs*zs,axis=1)+np.sum(cost[:,None]*source_margin**2,axis=0)
                    assert np.all(objective<=source_objective+128*np.finfo(float).eps*np.maximum(1,abs(source_objective)))
                    query_aug=np.column_stack([query,np.full(len(query),model.intercept_scaling)])
                    # L2 regularization of all augmented coefficients gives
                    # strong convexity >=1, bounding parameter and score error.
                    bound=np.linalg.norm(query_aug,axis=1)[:,None]*(norm+source_norm)[None,:]
                    gap=abs(native_raw-raw)
                    rounding=128*np.finfo(float).eps*np.maximum(1,np.maximum(abs(native_raw),abs(raw)))
                    assert np.all(gap<=bound+rounding),(cfg,name,gap.max(),bound.max())
                    optimizer_certificates.append(dict(cfg=cfg,name=name[0],kind='linear_svc',native_gradient=norm.tolist(),
                        source_gradient=source_norm.tolist(),score_gap=float(gap.max()),score_bound=float((bound+rounding).max())))
                    any_conditioned=True;raw=native_raw
            if native['tag']==1:
                # Verify source node/leaf responses and the full random split
                # protocol before using an alternative optimal CART solution.
                if k and method=='predict':
                    source_prob=model.predict_proba(query)
                elif k:source_prob=raw
                else:source_prob=raw
                if k:
                    native_classes=native['classes'];equal(native_classes,model.classes_,name[0]+' classes',True)
                else:native_classes=None
                response,cert=certify_cart(native['nodes'],native['stats'],native['values'],fitx,fity,
                    np.ones(len(fity)) if fitw is None else fitw,query,native_classes,1,2,2,0,0.,
                    (2 if isinstance(model,(ExtraTreeClassifier,ExtraTreeRegressor)) else 0)+(1 if not k else 0),0,int(native['seed']))
                # A matching tree stays a direct source comparison. A changed
                # response must use a recorded source-conditioned certificate.
                if not np.allclose(response,source_prob,rtol=3.e-8,atol=3.e-9,equal_nan=True):
                    any_conditioned=True;certificates.append(dict(cfg=cfg,name=name[0],splits=cert))
                if k and method=='predict':raw=native_classes[np.argmax(response,axis=1)][:,None]
                else:raw=response
            return raw
        query_parts=[];train_parts=[];oof_parts=[]
        for b,(index,method) in enumerate(zip(active,methods)):
            model=expected.estimators_[b]
            fy=y if cfg['prefit'] else target
            fw=None if cfg['prefit'] or not cfg['weighted'] else w
            both=evaluate(full_states[b],model,x,fy,fw,np.vstack([q,x]),(f'base {b}',method))
            rawq,rawtrain=both[:m],both[m:]
            if cfg['prefit']:oof=rawtrain.copy()
            else:
                oof=np.zeros((n,k if method=='predict_proba' else rawtrain.shape[1]))
                for f,(train,test) in enumerate(folds):
                    fm=clone(originals[index]).fit(x[train],target[train],**(dict(sample_weight=w[train]) if cfg['weighted'] else {}))
                    raw=evaluate(fold_states[f][b],fm,x[train],target[train],w[train] if cfg['weighted'] else None,
                                 x[test],(f'base {b} fold {f}',method))
                    if k and method=='predict_proba':
                        aligned=np.zeros((len(test),k));aligned[:,fm.classes_]=raw;raw=aligned
                    oof[test]=raw
            if method=='predict_proba' and k==2:rawq=rawq[:,1:];rawtrain=rawtrain[:,1:];oof=oof[:,1:]
            query_parts.append(rawq);train_parts.append(rawtrain);oof_parts.append(oof)
        if cfg['passthrough']:query_parts.append(q);train_parts.append(x);oof_parts.append(x)
        qmeta=np.hstack(query_parts);tmeta=np.hstack(train_parts);oof=np.hstack(oof_parts)
        oracle='source_conditioned_stacking' if any_conditioned else 'sklearn'
        equal(training,oof,'OOF training design',oracle=oracle);equal(transformed,qmeta,'query transform',oracle=oracle);equal(train_transformed,tmeta,'training transform',oracle=oracle)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            conditioned_final=clone(final).fit(training,target,**(dict(sample_weight=w) if cfg['weighted'] else {}))
        source_warnings.extend(dict(cfg=cfg,stage='final',category=warning.category.__name__,message=str(warning.message))
                               for warning in caught)
        logistic_refined=False
        if final_state['tag'] in [2,3,4]:
            if final_state['tag']==2 and not (np.allclose(final_state['coef'],conditioned_final.coef_,rtol=3.e-8,atol=3.e-9)
                and np.allclose(final_state['bias'],conditioned_final.intercept_,rtol=3.e-8,atol=3.e-9)):
                witness=logistic_certificate(final_state,conditioned_final,training,target,
                    w if cfg['weighted'] else None,np.vstack([transformed,train_transformed]))
                optimizer_certificates.append(dict(cfg=cfg,name='final',kind='logistic',**witness))
                any_conditioned=True;logistic_refined=True;oracle='source_conditioned_stacking'
                for name in ['final coefficients','final intercept']:
                    checks.append(dict(seed=cfg['seed'],mode=mode,case=cfg['case'],name=name,
                                       oracle='source_logistic_stationarity',passed=True))
            else:
                equal(final_state['coef'],conditioned_final.coef_,'final coefficients',oracle='source_on_verified_meta')
                equal(final_state['bias'],conditioned_final.intercept_,'final intercept',oracle='source_on_verified_meta')
            if final_state['tag']==3:
                equal(final_state['meta'],[conditioned_final.alpha_,conditioned_final.best_score_],'final LOO selection',oracle='source_on_verified_meta')
                equal(final_state['scores'],-conditioned_final.cv_results_.mean(axis=0),'final LOO scores',oracle='source_on_verified_meta')
        if final_state['tag']==1:
            raw=evaluate(final_state,conditioned_final,training,target,w if cfg['weighted'] else None,np.vstack([transformed,train_transformed]),
                         ('final','predict_proba' if k else 'predict'))
            if k:
                final_probability=raw[:m]
                final_pred=final_state['classes'][np.argmax(raw[:m],axis=1)]
                final_train=final_state['classes'][np.argmax(raw[m:],axis=1)]
            else:
                final_pred,final_train=raw[:m,0],raw[m:,0]
            oracle='source_conditioned_stacking' if any_conditioned else 'sklearn'
        else:
            final_pred=conditioned_final.predict(transformed);final_train=conditioned_final.predict(train_transformed)
        if logistic_refined:
            query_score=transformed@final_state['coef'].T+final_state['bias']
            train_score=train_transformed@final_state['coef'].T+final_state['bias']
            final_probability=(np.column_stack([1-expit(query_score[:,0]),expit(query_score[:,0])]) if k==2 else softmax(query_score,axis=1))
            train_probability=(np.column_stack([1-expit(train_score[:,0]),expit(train_score[:,0])]) if k==2 else softmax(train_score,axis=1))
            final_pred=conditioned_final.classes_[np.argmax(final_probability,axis=1)]
            final_train=conditioned_final.classes_[np.argmax(train_probability,axis=1)]
        equal(predicted,classes[final_pred.astype(int)] if k else final_pred,'prediction',bool(k),oracle)
        equal(trained_pred,classes[final_train.astype(int)] if k else final_train,'training prediction',bool(k),oracle)
        equal(score,accuracy_score(y,classes[final_train.astype(int)],sample_weight=w) if k else r2_score(y,final_train,sample_weight=w),'score',oracle=oracle)
        if flags&1:equal(proba,final_probability if final_state['tag']==1 or logistic_refined else conditioned_final.predict_proba(transformed),'probabilities',oracle=oracle)
        if flags&2:
            ref=query_score if logistic_refined else conditioned_final.decision_function(transformed)
            equal(decision,ref[:,None] if ref.ndim==1 else ref,'decision response',oracle=oracle)
        if not any_conditioned:
            equal(transformed,expected.transform(q),'ordinary source transform')
            equal(predicted,expected.predict(q),'ordinary source prediction',bool(k))
        equal(matrix(n),x,'training input preserved',True,'input_identity');equal(matrix(m),q,'query input preserved',True,'input_identity')
        equal(vector(),y,'target preserved',True,'input_identity');equal(vector(),w,'weights preserved',True,'input_identity')
        assert list(rows)==[];count+=1
    source=Path(sklearn.__file__).parent
    report=dict(sklearn=sklearn.__version__,profile=profile or 'checked',fixtures=count,checks=checks,cart_certificates=certificates,
        optimizer_certificates=optimizer_certificates,
        source_warnings=source_warnings,
        source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
        ['ensemble/_stacking.py','model_selection/_validation.py','model_selection/_split.py','linear_model/_ridge.py']})
    (RESULTS/f'stacking_{profile or "checked"}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} stacking/LOO groups across {count} fixtures passed; {len(certificates)} CART certificates')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['release','debug']);parser.add_argument('--executable',type=Path)
    args=parser.parse_args();RESULTS.mkdir(exist_ok=True)
    run(args.executable.resolve() if args.executable else resolve(args.profile),args.profile)
