"""Binary threshold wrappers and fold curves versus installed sklearn 1.6.1."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import os
import subprocess
import warnings
from decimal import Decimal,localcontext
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import sklearn
from sklearn_tree_runtime import tree_runtime_provenance
from sklearn import config_context
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import FixedThresholdClassifier,TunedThresholdClassifierCV,StratifiedKFold,StratifiedShuffleSplit
from sklearn.utils._response import _get_response_values_binary
from sklearn.metrics import make_scorer,accuracy_score,balanced_accuracy_score,f1_score,precision_score,recall_score,jaccard_score

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile):
    opts=['--profile','release'] if profile=='release' else ['--flag',FLAGS]
    result=subprocess.run(['fpm','run',*opts,'--target','reference_threshold_driver','--runner'],cwd=ROOT,
                          text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/s.strip().strip('"') for s in result.stdout.splitlines()
           if s.strip().strip('"').endswith(('reference_threshold_driver','reference_threshold_driver.exe'))]
    paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,result.stdout
    return paths[0]


def base_model(kind):
    return {1:lambda:LogisticRegression(C=.7,tol=1e-12,max_iter=2000,solver='newton-cg'),
            2:GaussianNB,3:lambda:RidgeClassifier(alpha=.7,solver='svd'),4:KNeighborsClassifier,
            5:lambda:DecisionTreeClassifier(max_depth=3,random_state=0),6:DummyClassifier,
            7:lambda:LogisticRegressionCV(Cs=[.1,1.],cv=2,tol=1e-12,max_iter=2000,solver='newton-cg')}[kind]()



def certify_tree_boundaries(model,xtrain,ytrain,weights,xtest,positive,native_leaves,source_response,
                            native_response,source_grid,native_grid):
    """Prove threshold crossings are confined to leaf-count summation roundoff."""
    train_leaves,test_leaves=native_leaves
    assert np.all(train_leaves==train_leaves.astype(int)) and np.all(test_leaves==test_leaves.astype(int))
    source_train=model.apply(xtrain);source_test=model.apply(xtest)
    eps=np.finfo(float).eps;gamma=len(xtrain)*eps/(1-len(xtrain)*eps)
    global_bound=(2*gamma+eps+gamma*eps)/(1-gamma)+eps
    grid_bound=global_bound+8*eps
    np.testing.assert_allclose(native_response,source_response,rtol=0,atol=2*global_bound)
    np.testing.assert_allclose(native_grid,source_grid,rtol=0,atol=grid_bound)
    rows,points=np.nonzero((native_response[:,None]>=native_grid[None,:]) !=
                           (source_response[:,None]>=source_grid[None,:]))
    assert len(rows)>0,'a curve mismatch without a response-boundary crossing is not certifiable'
    records=[];verified={}
    for row,point in zip(rows,points):
        key=(int(source_test[row]),int(test_leaves[row]))
        if key not in verified:
            source_ids=np.flatnonzero(source_train==key[0]);native_ids=np.flatnonzero(train_leaves==key[1])
            np.testing.assert_array_equal(native_ids,source_ids)
            with localcontext() as context:
                context.prec=100
                mass=sum((Decimal.from_float(float(weights[i])) for i in source_ids),Decimal(0))
                numerator=sum((Decimal.from_float(float(weights[i])) for i in source_ids if ytrain[i]==positive),Decimal(0))
                assert mass>0;ratio=numerator/mass
                g=len(source_ids)*eps/(1-len(source_ids)*eps)
                bound=abs(float(ratio))*(2*g+eps+g*eps)/(1-g)+eps
                source_error=abs(Decimal.from_float(float(source_response[row]))-ratio)
                native_error=abs(Decimal.from_float(float(native_response[row]))-ratio)
                assert source_error<=Decimal.from_float(bound) and native_error<=Decimal.from_float(bound)
            verified[key]=(bound,ratio)
            records.append(dict(source_leaf=key[0],native_leaf=key[1],training_rows=source_ids.tolist(),
                                exact_mass=str(mass),exact_positive_mass=str(numerator),exact_ratio=str(ratio),
                                forward_bound=bound,source_error=float(source_error),native_error=float(native_error)))
        bound,ratio=verified[key]
        with localcontext() as context:
            context.prec=100
            assert abs(Decimal.from_float(float(source_response[row]))-ratio)<=Decimal.from_float(bound)
            assert abs(Decimal.from_float(float(native_response[row]))-ratio)<=Decimal.from_float(bound)
        assert abs(native_response[row]-native_grid[point])<=2*bound+grid_bound
        assert abs(source_response[row]-source_grid[point])<=2*bound+grid_bound
    return dict(crossing_rows=rows.tolist(),crossing_points=points.tolist(),grid_bound=grid_bound,
                max_response_difference=float(np.max(abs(native_response-source_response))),leaf_certificates=records)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path)
    parser.add_argument('--profile',choices=['checked','release'],default='checked')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,7,42,2026,314159]);args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True);exe=args.executable.resolve() if args.executable else resolve(args.profile)
    runtime=tree_runtime_provenance()
    assert runtime['feature_threshold']==0., 'native tree baseline requires runtime FEATURE_THRESHOLD=0'
    checks=[];source_warnings=[];fixture_controls=[];certificates=[];fixtures=0
    def run(seed,case,x,y,query,truth,kind,method,scoring,mode,refit,positive_option,explicit,weighted,store=True):
        nonlocal fixtures
        classes=np.unique(y);positive=classes[0] if positive_option==1 else classes[-1]
        if positive_option==0 and scoring in ['f1','precision','recall','jaccard']:positive=1
        fit_weight=(.3+np.arange(len(y))%9*.19) if weighted in [1,3] else None
        score_weight=(.2+np.arange(len(y))%7*.23) if weighted in [2,3] else None
        if fit_weight is not None:fit_weight[::13]=0
        if score_weight is not None:score_weight[::11]=0
        val_weight=score_weight if score_weight is not None else fit_weight
        count=[2,11,31,100][case%4];cv_count=[2,3,5][case%3]
        thresholds=([0.,.2,.2,.7,1.] if method!='decision_function' and kind!=3 else [-3.,-.1,-.1,1.,3.]) if explicit else count
        fraction=.27
        if mode==0:
            cv=StratifiedKFold(cv_count);folds=list(cv.split(x,y))
        elif mode==1:
            cv=fraction;folds=list(StratifiedShuffleSplit(1,test_size=fraction,random_state=0).split(x,y))
        elif mode in [2,4]:
            folds=[(np.r_[np.arange(60)[::-1],0,0],np.r_[np.arange(30,len(y)),33])]
            if mode==2:folds.append((np.arange(20,len(y)),np.r_[np.arange(30),2]))
            cv=folds
        else:
            cv='prefit';folds=[(None,np.arange(len(y)))]
        metrics=dict(accuracy=accuracy_score,balanced_accuracy=balanced_accuracy_score,
                     f1=f1_score,precision=precision_score,recall=recall_score,jaccard=jaccard_score)
        def utility(truth,prediction,*,sample_weight=None,pos_label=None):
            costs=np.where((truth==positive)&(prediction!=positive),3.,
                           np.where((truth!=positive)&(prediction==positive),1.,0.))
            return -np.average(costs,weights=sample_weight)
        func=utility if scoring=='utility' else metrics[scoring]
        options={}
        if positive_option>0 or scoring in ['f1','precision','recall','jaccard']:
            options['pos_label']=int(positive)
        if scoring in ['accuracy','balanced_accuracy']:options={}
        # Accuracy/balanced accuracy have no positive-label parameter. A scorer
        # wrapper with a pos_label argument carries the response orientation.
        if positive_option>0 and scoring in ['accuracy','balanced_accuracy']:
            raw_metric=func
            def oriented(truth,prediction,*,sample_weight=None,pos_label=None):
                return raw_metric(truth,prediction,sample_weight=sample_weight)
            func=oriented;options={'pos_label':int(positive)}
        with config_context(enable_metadata_routing=True),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');base=base_model(kind)
            if kind not in [4,6]:base.set_fit_request(sample_weight='fit_weight')
            scorer=make_scorer(func,**options).set_score_request(sample_weight='score_weight')
            if mode==3:
                base.fit(x,y,**({'sample_weight':fit_weight} if fit_weight is not None else {}))
            cutoff_auto=case%3==0;cutoff=.5 if method!='decision_function' and kind!=3 else 0.
            if case%17==0:cutoff=float('inf')
            if case%19==0:cutoff=float('-inf')
            fixed=FixedThresholdClassifier(base,response_method=method,
                    threshold='auto' if cutoff_auto else cutoff,
                    pos_label=None if positive_option==0 else int(classes[0] if positive_option==1 else classes[-1]))
            if mode!=3:fixed.fit(x,y,**({'fit_weight':fit_weight} if fit_weight is not None else {}))
            tuned=TunedThresholdClassifierCV(base,response_method=method,scoring=scorer,thresholds=thresholds,
                        cv=cv,refit=refit,random_state=0,store_cv_results=store)
            kwargs={}
            if fit_weight is not None and mode!=3:kwargs['fit_weight']=fit_weight
            if val_weight is not None:kwargs['score_weight']=val_weight
            tuned.fit(x,y,**kwargs)
            expected=dict(classes=tuned.classes_,best=[tuned.best_threshold_,tuned.best_score_],
                          fixed_predict=fixed.predict(query),predict=tuned.predict(query),
                          score=[fixed.score(query,truth),tuned.score(query,truth)],
                          weighted_score=[fixed.score(query,truth,sample_weight=1+.1*np.arange(1,len(query)+1)),
                                          tuned.score(query,truth,sample_weight=1+.1*np.arange(1,len(query)+1))],
                          features=[x.shape[1]],positive=[positive],refit_classes=[4,9],refit_features=[1],refit_stored=[0])
            if kind!=3:expected.update(fixed_proba=fixed.predict_proba(query),proba=tuned.predict_proba(query))
            if kind in [1,3,7]:expected.update(fixed_decision=fixed.decision_function(query),decision=tuned.decision_function(query))
            curves=[];fold_models=[];fold_responses=[]
            for i,(train,test) in enumerate(folds,1):
                fitted=base if train is None else clone(base).fit(x[train],y[train],
                        **({'sample_weight':fit_weight[train]} if fit_weight is not None else {}))
                response,_=_get_response_values_binary(fitted,x[test],
                        ['predict_proba','decision_function'] if method=='auto' else method,
                        pos_label=tuned._curve_scorer._get_pos_label())
                expected[f'response{i}']=response;fold_models.append(fitted);fold_responses.append(response)
                if store:
                    curves.append(tuned._curve_scorer(fitted,x[test],y[test],
                                  **({'sample_weight':val_weight[test]} if val_weight is not None else {})))
            if store:
                expected.update(thresholds=tuned.cv_results_['thresholds'],mean_scores=tuned.cv_results_['scores'],
                                fold_thresholds=np.stack([c[1] for c in curves]),fold_scores=np.stack([c[0] for c in curves]))
        source_warnings.extend(dict(seed=seed,case=case,category=w.category.__name__,message=str(w.message)) for w in caught)
        lines=[f'{len(x)} {x.shape[1]} {len(query)} {kind} {int(fit_weight is not None)} {int(score_weight is not None)} '
               f'{mode} {int(refit)} {positive_option} {len(thresholds) if explicit else 0} {count} {cv_count} {int(store)} {int(cutoff_auto)}',
               f'{method} {scoring} {cutoff} {fraction}']
        lines+=[' '.join(map(str,row)) for row in x]+[' '.join(map(str,y))]
        if fit_weight is not None:lines+=[' '.join(map(str,fit_weight))]
        if score_weight is not None:lines+=[' '.join(map(str,score_weight))]
        lines+=[' '.join(map(str,row)) for row in query]+[' '.join(map(str,truth))]
        if explicit:lines+=[' '.join(map(str,thresholds))]
        if mode in [2,4]:
            for train,test in folds:lines+=[f'{len(train)} {len(test)}',' '.join(map(str,train+1)),' '.join(map(str,test+1))]
        inp=RESULTS/f'threshold_{args.profile}_input.txt';out=RESULTS/f'threshold_{args.profile}_output.txt'
        inp.write_text('\n'.join(lines)+'\n')
        proc=subprocess.run([str(exe),str(inp),str(out)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        assert proc.returncode==0,(seed,case,kind,method,scoring,mode,proc.stdout)
        actual={s.split()[0]:np.asarray(s.split()[1:],float) for s in out.read_text().splitlines()}
        leaf_data={}
        if kind==5:
            for i in range(1,len(folds)+1):leaf_data[i]=(actual.pop(f'leaf_train{i}'),actual.pop(f'leaf_test{i}'))
        conditioned=False
        if store:
            shape=(len(folds),len(expected['thresholds']))
            native_grids=actual['fold_thresholds'].reshape(shape,order='F')
            native_scores=actual['fold_scores'].reshape(shape,order='F')
            conditioned_scores=expected['fold_scores'].copy()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for f,(train,test) in enumerate(folds):
                    raw=actual[f'response{f+1}'];grid=native_grids[f]
                    negative=classes[0] if positive==classes[-1] else classes[-1]
                    replay=np.array([func(y[test],np.where(raw>=cut,positive,negative),
                                    sample_weight=None if val_weight is None else val_weight[test],**options) for cut in grid])
                    np.testing.assert_allclose(native_scores[f],replay,rtol=1e-12,atol=1e-12,
                                               err_msg=f'{seed} {case} fold {f} native curve/metric identity')
                    if not np.allclose(native_scores[f],expected['fold_scores'][f],rtol=3e-7,atol=3e-8):
                        assert kind==5,(seed,case,'only a proved tree probability boundary is conditional')
                        ids=np.arange(len(y)) if train is None else train
                        cert=certify_tree_boundaries(fold_models[f],x[ids],y[ids],
                                np.ones(len(ids)) if fit_weight is None else fit_weight[ids],x[test],positive,
                                leaf_data[f+1],fold_responses[f],raw,curves[f][1],grid)
                        crossing=np.unique(cert['crossing_points'])
                        conditioned_scores[f,crossing]=replay[crossing]
                        np.testing.assert_allclose(native_scores[f],conditioned_scores[f],rtol=3e-7,atol=3e-8)
                        certificates.append(dict(seed=seed,case=case,fold=f,**cert));conditioned=True
            checks.append(dict(seed=seed,case=case,name='curve_metric_identity',passed=True,oracle='metric_replay_identity'))
            if conditioned:
                original_means=expected['mean_scores'].copy()
                modified=np.mean([np.interp(expected['thresholds'],curve[1],row)
                                  for curve,row in zip(curves,conditioned_scores)],axis=0)
                native_means=actual['mean_scores'];best=int(np.argmax(original_means))
                assert best==int(np.argmax(native_means)),(seed,case,'boundary certificate cannot replace selected threshold')
                np.testing.assert_allclose(modified[best],original_means[best],rtol=1e-13,atol=1e-13,
                                           err_msg='boundary certificate cannot replace selected score')
                expected['mean_scores']=modified;expected['fold_scores']=conditioned_scores
        for i,(train,test) in enumerate(folds,1):
            expected[f'train{i}']=np.array([],int) if train is None else train+1;expected[f'test{i}']=test+1
        assert actual.keys()==expected.keys(),(actual.keys(),expected.keys())
        for name,want in expected.items():
            want=np.asarray(want).ravel(order='F')
            exact=(name in ['classes','fixed_predict','predict','features','positive','refit_classes','refit_features','refit_stored']
                   or name.startswith(('train','test')))
            if exact:np.testing.assert_array_equal(actual[name],want,err_msg=f'{seed} {case} {name}')
            else:np.testing.assert_allclose(actual[name],want,rtol=3e-7,atol=3e-8,err_msg=f'{seed} {case} {name}')
            oracle='input_identity' if name in ['features','positive','refit_classes','refit_features','refit_stored'] else 'sklearn'
            if conditioned and name in ['mean_scores','fold_scores']:oracle='source_conditioned_tree_boundary'
            checks.append(dict(seed=seed,case=case,name=name,passed=True,oracle=oracle))
        fixture_controls.append(dict(seed=seed,case=case,base=kind,response=method,scoring=scoring,mode=mode,refit=refit,
                                     positive_option=positive_option,thresholds=thresholds,weight_mode=weighted,stored=store))
        fixtures+=1

    for seed in args.seeds:
        rng=np.random.RandomState(seed);n=87;p=3;query=rng.normal(size=(19,p));x=rng.normal(size=(n,p))
        target=(x[:,0]-.4*x[:,1]+rng.normal(size=n)*1.2>.6).astype(int);target[:12]=np.resize([0,1],12)
        case=0
        for kind in range(1,6):
            for method in ['auto','predict_proba','decision_function']:
                if (kind==3 and method=='predict_proba') or (kind in [2,4,5] and method=='decision_function'):continue
                for scoring in ['balanced_accuracy','accuracy','f1','precision','recall','jaccard','utility']:
                    case+=1;positive_option=case%3;classes=np.array([0,1]) if positive_option==0 else np.array([-3,7])
                    y=classes[target];truth=np.resize(classes,len(query));mode=case%5
                    weighted=2 if kind==4 else case%3
                    run(seed,case,x,y,query,truth,kind,method,scoring,mode,mode not in [3,4],positive_option,
                        explicit=case%2==0,weighted=weighted,store=case%11!=0)
        for kind in [6,7]:
            for mode in [0,1,2,3,4]:
                case+=1;y=np.where(target==1,7,-3);truth=np.resize([-3,7],len(query))
                run(seed,case,x,y,query,truth,kind,'auto','balanced_accuracy',mode,mode not in [3,4],1,
                    explicit=kind==6 or mode%2==0,weighted=2 if kind==6 else 1)
        for positive_option in [0,1]:
            case+=1;balanced=np.resize([-3,7],80)
            run(seed,case,x[:80],balanced,query,np.resize([-3,7],len(query)),6,'auto','balanced_accuracy',3,False,
                positive_option,explicit=True,weighted=0)
        for kind,mode,refit,scoring,positive_option in [(1,0,True,'utility',1),(5,1,False,'balanced_accuracy',2)]:
            case+=1;y=np.where(target==1,7,-3);truth=np.resize([-3,7],len(query))
            run(seed,case,x,y,query,truth,kind,'auto',scoring,mode,refit,positive_option,explicit=False,weighted=3)
        print(f'seed {seed}: {case} threshold fixtures passed',flush=True)
    root=Path(inspect.getfile(TunedThresholdClassifierCV)).parent.parent
    paths=[root/'model_selection/_classification_threshold.py',root/'model_selection/_split.py',
           root/'metrics/_scorer.py',root/'metrics/_classification.py',root/'utils/_response.py']
    paths+=list({Path(inspect.getfile(type(base_model(k)))) for k in range(1,8)})
    report=dict(sklearn=sklearn.__version__,profile=args.profile,fixtures=fixtures,seeds=args.seeds,checks=checks,
                source_warnings=source_warnings,tree_runtime=runtime,tree_boundary_certificates=certificates,fixture_controls=fixture_controls,source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                                             for p in sorted(set(paths))})
    (RESULTS/f'threshold_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} threshold groups across {fixtures} fixtures passed')


if __name__=='__main__':main()
