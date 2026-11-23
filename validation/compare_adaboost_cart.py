"""Seeded SAMME/CART learners, staged outputs and source boundary states."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import warnings

os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from compare_image_graph import fields
from sklearn_tree_runtime import tree_runtime_provenance

REJECTIONS={
    'depth':'invalid AdaBoost CART size controls','leaf':'invalid AdaBoost CART size controls',
    'split':'invalid AdaBoost CART size controls','features_negative':'invalid AdaBoost CART max_features',
    'features_large':'invalid AdaBoost CART max_features','seed':'random_state must be nonnegative',
    'criterion':'invalid AdaBoost CART criterion','splitter':'invalid AdaBoost CART splitter',
    'impurity_negative':'negative AdaBoost CART impurity decrease','unfitted_log':'AdaBoost is not fitted',
    'unfitted_staged_score':'AdaBoost is not fitted','score_weights_zero':'score weights sum to zero',
    'score_weights_length':'score weight count mismatch','score_target_length':'score target count mismatch',
}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(actual,expected,index,field):
    expected=np.asarray(expected)
    assert actual.shape==expected.shape,(index,field,actual.shape,expected.shape)
    if np.issubdtype(expected.dtype,np.integer):
        np.testing.assert_array_equal(actual,expected,err_msg=f'{index} {field}');return 0.
    for classify in [np.isnan,np.isposinf,np.isneginf]:
        np.testing.assert_array_equal(classify(actual),classify(expected),err_msg=f'{index} {field}')
    finite=np.isfinite(expected)
    np.testing.assert_allclose(actual[finite],expected[finite],rtol=2.e-11,atol=2.e-12,
                               err_msg=f'{index} {field}')
    return float(np.max(np.abs(actual[finite]-expected[finite]),initial=0.))


def raw_tree(estimator):
    tree=estimator.tree_;n=tree.node_count;nodes=np.zeros((n,7),dtype=np.int64)
    statistics=np.column_stack([np.zeros(n),tree.weighted_n_node_samples,tree.impurity])
    nodes[:,4]=tree.n_node_samples
    for at in range(n):
        left,right=tree.children_left[at],tree.children_right[at]
        if left<0:nodes[at,3]=estimator.classes_[np.argmax(tree.value[at,0])]
        else:
            nodes[at,0]=tree.feature[at]+1;nodes[at,1:3]=[left+1,right+1]
            nodes[at,6]=tree.missing_go_to_left[at];statistics[at,0]=tree.threshold[at]
            nodes[left,5]=nodes[at,5]+1;nodes[right,5]=nodes[at,5]+1
    return nodes,statistics,tree.value[:,0,:]


def fixtures():
    controls=[(1,0,0,0,1,2,0.),(2,0,0,0,1,2,0.),(3,1,0,2,2,2,0.),(4,2,0,1,1,5,0.),
              (3,0,1,0,1,2,0.),(4,1,1,2,1,2,0.),(2,2,1,1,1,2,0.),(5,0,0,0,3,7,.01)]
    for seed,shape,control,count,rate,weighted in itertools.product([0,42,2026],
            [(37,3,2),(49,4,3),(61,7,5),(43,5,1),(97,6,10)],controls,[1,6,17],[.25,1.],[False,True]):
        yield dict(seed=seed,shape=shape,control=control,count=count,rate=rate,weighted=weighted,special='ordinary')
    for special,count in itertools.product(['exponential_overflow','large_finite_alpha','infinite_alpha',
            'tiny_rate','initial_sum_overflow','tiny_weights','one_active','equal_features','signed_scores','nan_scores'],[1,9]):
        yield dict(seed=42,shape=(8,2,2),control=controls[0],count=count,rate=1.,weighted=True,special=special)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True,type=Path)
    parser.add_argument('--profile',choices=['development','checked','release'],default='development')
    parser.add_argument('--report',required=True,type=Path);parser.add_argument('--start-fixture',type=int,default=0)
    args=parser.parse_args();exe=args.executable.resolve();cases=list(fixtures())
    checks=[];inventories=[];warnings_seen=Counter();stage_errors=[];normalization=[];nonfinite=[];fit_errors=[]
    for index,cfg in enumerate(cases):
        if index<args.start_fixture:continue
        rng=np.random.RandomState(cfg['seed']);n,p,k=cfg['shape'];depth,criterion,splitter,features,leaf,split,impurity=cfg['control']
        x=rng.normal(size=(n,p));query=rng.normal(size=(7,p));w=rng.uniform(.1,2.,size=n);w[0]=0
        if k==1:y=np.full(n,21,dtype=int)
        elif k==2:y=np.where(x[:,0]+.4*x[:,1]+.6*rng.normal(size=n)>0,7,-4)
        else:
            coefficients=rng.normal(size=(p,k));y=(np.argmax(x@coefficients+.6*rng.normal(size=(n,k)),axis=1)*7-11).astype(int)
        score_w=rng.uniform(.1,3.,size=n);rate=cfg['rate'];special=cfg['special'];mode=int(index%11==0)
        if special!='ordinary':
            x[:,0]=np.arange(n);x[:,1]=0;y=np.array([-4,-4,7,-4,7,7,-4,7]);w=np.ones(n);score_w=np.ones(n)
            if special=='exponential_overflow':rate=1000.
            elif special=='large_finite_alpha':rate=1.e308
            elif special=='infinite_alpha':rate=np.finfo(float).max
            elif special=='tiny_rate':rate=np.nextafter(0.,1.)
            elif special=='initial_sum_overflow':w[:]=np.finfo(float).max
            elif special=='tiny_weights':w*=1.e-320
            elif special=='one_active':w[:]=0;w[1]=1
            elif special=='equal_features':x[:,1]=x[:,0]
            elif special=='signed_scores':score_w=np.array([-1.,2.,-3.,4.,-1.,2.,-3.,4.])
            elif special=='nan_scores':score_w[0]=np.nan
        tree=DecisionTreeClassifier(max_depth=depth,criterion=['gini','entropy','log_loss'][criterion],
            splitter=['best','random'][splitter],max_features=features or None,min_samples_leaf=leaf,
            min_samples_split=split,min_impurity_decrease=impurity)
        options=dict(estimator=tree,n_estimators=cfg['count'],learning_rate=rate,random_state=cfg['seed'])
        expected=[]
        with warnings.catch_warnings(record=True) as caught,np.errstate(all='ignore'):
            warnings.simplefilter('always');model=AdaBoostClassifier(**options)
            weights=w if cfg['weighted'] else None
            if special=='initial_sum_overflow':
                try:model.fit(x,y,sample_weight=weights)
                except (ValueError,ZeroDivisionError) as error:
                    normalization.append(dict(fixture=index,error=type(error).__name__,message=str(error)))
                else:raise AssertionError('expected raw source weight normalization failure')
                weights=w/w.max();model=AdaBoostClassifier(**options)
            try:model.fit(x,y,sample_weight=weights)
            except ValueError as error:
                assert 'worse than random' in str(error),str(error)
                fit_errors.append(dict(fixture=index,message=str(error)));model=None
            if model is not None:
                retained=len(model.estimators_);classes=len(model.classes_);channels=1 if classes==2 else classes
                decision=lambda data:model.decision_function(data).reshape((len(data),channels))
                expected=[('metadata',np.array([p,classes,retained,retained],dtype=np.int64)[:,None]),
                    ('classes',model.classes_[:,None]),('weights',model.estimator_weights_[:,None]),
                    ('errors',model.estimator_errors_[:,None]),('importance',model.feature_importances_[:,None]),
                    ('train_decision',decision(x)),('query_decision',decision(query)),
                    ('proba',model.predict_proba(query)),('log_proba',model.predict_log_proba(query)),
                    ('predict',model.predict(query)[:,None])]
                if classes==1:
                    for name in ['staged_decision_function','staged_predict_proba','staged_predict','staged_score']:
                        try:list(getattr(model,name)(query) if name!='staged_score' else getattr(model,name)(x,y))
                        except ZeroDivisionError as error:stage_errors.append(dict(fixture=index,method=name,error=str(error)))
                        else:raise AssertionError('expected source single-class staged division failure')
                    stages=np.zeros((len(query),1,retained));prob=np.ones((len(query),1,retained))
                    predicted=np.full((len(query),retained),model.classes_[0],dtype=int)
                    staged_score=np.full((retained,1),model.score(x,y));weighted_score=np.full((retained,1),model.score(x,y,score_w))
                else:
                    stages=np.stack([v.reshape((len(query),channels)) for v in model.staged_decision_function(query)],axis=2)
                    prob=np.stack(list(model.staged_predict_proba(query)),axis=2)
                    predicted=np.stack(list(model.staged_predict(query)),axis=1)
                    staged_score=np.array(list(model.staged_score(x,y)))[:,None]
                    weighted_score=np.array(list(model.staged_score(x,y,sample_weight=score_w)))[:,None]
                expected.extend([('staged_decision',stages.reshape((len(query)*channels,retained),order='F')),
                    ('staged_proba',prob.reshape((len(query)*classes,retained),order='F')),('staged_predict',predicted),
                    ('scores',np.array([model.score(x,y),model.score(x,y,score_w)])[:,None]),
                    ('staged_score',staged_score),('staged_weighted_score',weighted_score),
                    ('tree_prediction',np.column_stack([t.predict(query) for t in model.estimators_])),
                    ('tree_importance',np.column_stack([t.feature_importances_ for t in model.estimators_])),
                    ('seeds',np.array([t.random_state for t in model.estimators_],dtype=np.int64)[:,None]),
                    ('depths',np.array([t.get_depth() for t in model.estimators_],dtype=np.int64)[:,None]),
                    ('leaves',np.array([t.get_n_leaves() for t in model.estimators_],dtype=np.int64)[:,None]),
                    ('prototype_unfitted',np.zeros((2,1),dtype=np.int64))])
                for t,estimator in enumerate(model.estimators_):
                    nodes,statistics,probability=raw_tree(estimator)
                    expected.extend([(f'tree_{t}_nodes',nodes),(f'tree_{t}_statistics',statistics),
                                     (f'tree_{t}_probability',probability)])
                if mode:
                    model.fit(x,y);expected.append(('unweighted_refit',decision(query)))
            warnings_seen.update(str(warning.message) for warning in caught)
        header=[mode,n,p,len(query),int(cfg['weighted']),cfg['seed'],cfg['count'],depth,split,leaf,features,criterion,splitter,
                rate,impurity]
        payload=' '.join(map(str,header))+'\n'+'\n'.join(' '.join(format(v,'.17g') for v in a.ravel(order='F'))
                    for a in [x,query,y,w,score_w])+'\n'
        proc=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
        if model is None:
            assert proc.returncode!=0 and 'no better than random guessing' in proc.stderr+proc.stdout,(index,proc.stderr)
            checks.append(dict(fixture=index,field='source_fit_rejection',max_abs_error=0.))
            inventories.append(dict(fixture=index,**cfg,fields=['source_fit_rejection']));continue
        assert proc.returncode==0,(index,cfg,proc.stderr)
        actual=fields(proc.stdout);assert len(actual)==len(expected),(index,len(actual),len(expected))
        names=[]
        for value,(field,reference) in zip(actual,expected):
            error=compare(value,reference,index,field)
            checks.append(dict(fixture=index,field=field,max_abs_error=error));names.append(field)
            if np.issubdtype(np.asarray(reference).dtype,np.floating) and np.any(~np.isfinite(reference)):
                nonfinite.append(dict(fixture=index,field=field,nan=int(np.isnan(reference).sum()),
                                      positive_inf=int(np.isposinf(reference).sum()),negative_inf=int(np.isneginf(reference).sum())))
        checks.append(dict(fixture=index,field='input_preservation',max_abs_error=0.));names.append('input_preservation')
        inventories.append(dict(fixture=index,**cfg,retained=retained,mode=mode,fields=names))
        if index%100==0:print(f'{index}/{len(cases)} fixtures passed',flush=True)
    for name,message in REJECTIONS.items():
        proc=subprocess.run([str(exe),name],text=True,capture_output=True)
        assert proc.returncode!=0 and message in proc.stdout+proc.stderr,(name,proc.stdout,proc.stderr)
        assert 'SIGFPE' not in proc.stderr and 'Fortran runtime error' not in proc.stderr,(name,proc.stderr)
    source=Path(sklearn.__file__).parent
    report=dict(profile=args.profile,sklearn_version=sklearn.__version__,numpy_version=np.__version__,
        fixtures=len(cases),start_fixture=args.start_fixture,checks=checks,inventories=inventories,
        rejections=list(REJECTIONS),source_fit_errors=fit_errors,single_class_stage_errors=stage_errors,
        normalization_corrections=normalization,
        nonfinite_reference_results=nonfinite,source_warnings=dict(warnings_seen),
        executable=str(exe),executable_sha256=digest(exe),validator_sha256=digest(__file__),
        parser_sha256=digest(Path(__file__).with_name('compare_image_graph.py')),
        tree_runtime=tree_runtime_provenance(),
        source_sha256={s:digest(source/s) for s in ['ensemble/_weight_boosting.py','ensemble/_base.py','tree/_classes.py',
            'tree/_splitter.pyx','tree/_criterion.pyx','tree/_partitioner.pyx','tree/_utils.pxd']})
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'PASS: {len(cases)} fixtures, {len(checks)} groups, {len(REJECTIONS)} diagnostics')


if __name__=='__main__':main()
