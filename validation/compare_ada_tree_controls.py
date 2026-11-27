"""AdaBoost tree parameters, bootstrap rounds and complete learned tree state."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, subprocess, warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn_tree_runtime import tree_runtime_provenance


def fixtures(kind,smoke):
    criteria=['squared_error','friedman_mse','absolute_error','poisson'] if kind=='regression' else ['gini','entropy','log_loss']
    variants=['linear','square','exponential'] if kind=='regression' else [2,3,5]
    for criterion,splitter,seed,weighted,profile,variant in itertools.product(
            criteria,['best','random'],[0,7],[0,1],range(8),variants):
        if smoke and (splitter!='best' or seed or not weighted or profile!=1):continue
        n=37 if seed==0 else 53;p=3 if seed==0 else 5;rng=np.random.RandomState(761+n)
        x=rng.normal(size=(n,p));w=np.resize([1.,0.,.5,2.,3.],n)
        if kind=='regression':
            y=x[:,0]**2+.5*x[:,1]+.2*rng.normal(size=n)
            if criterion=='poisson':y=np.exp(.3*x[:,0])+.1
        else:
            y=(np.argmax(x@rng.normal(size=(p,variant))+.4*rng.normal(size=(n,variant)),axis=1)*17-9).astype(int)
        leaf,split,features,budget,ccp,fraction=[
            (1,2,'none',0,0.,0.),(.1,.3,.7,4,.005,0.),(2,.6,'sqrt',3,.01,.1),
            (.2,3,'log2',5,.03,.05),(.49,1.,1.,0,.1,.3),(1,2,p+2,2,0.,.2),
            (.01,.01,1,0,0.,.5),(1,2,'none',0,1.e3,0.)][profile]
        yield dict(kind=kind,n=n,p=p,seed=seed,nt=6,depth=3 if seed==0 else 5,criterion=criterion,splitter=splitter,
            leaf=leaf,split=split,features=features,budget=budget,ccp=ccp,fraction=fraction,decrease=0.,
            rate=.5 if seed==0 else 1.,loss=variant if kind=='regression' else 'linear',weighted=weighted,profile=profile,variant=variant),x,y,w


def source_case(c,x,y,w):
    regression=c['kind']=='regression';tree_cls=DecisionTreeRegressor if regression else DecisionTreeClassifier
    tree=tree_cls(max_depth=c['depth'],criterion=c['criterion'],splitter=c['splitter'],
        min_samples_leaf=c['leaf'],min_samples_split=c['split'],max_features=None if c['features']=='none' else c['features'],
        max_leaf_nodes=c['budget'] or None,ccp_alpha=c['ccp'],min_weight_fraction_leaf=c['fraction'],
        min_impurity_decrease=c['decrease'],random_state=91)
    params=dict(estimator=tree,n_estimators=c['nt'],learning_rate=c['rate'],random_state=c['seed'])
    if regression:params['loss']=c['loss']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model=(AdaBoostRegressor if regression else AdaBoostClassifier)(**params).fit(x,y,sample_weight=w if c['weighted'] else None)
        query=x+.137;trees=model.estimators_;nt=len(trees)
        importance=model.feature_importances_
        expected=dict(metadata=[model.n_features_in_,nt,0 if regression else model.n_classes_],weights=model.estimator_weights_.tolist(),
            errors=model.estimator_errors_.tolist(),importance=importance.tolist(),train=model.predict(x).tolist(),query=model.predict(query).tolist(),
            staged=np.stack(list(model.staged_predict(query)),axis=1).ravel().tolist(),score=[model.score(x,y),model.score(x,y,sample_weight=w)],
            staged_score=list(model.staged_score(x,y))+list(model.staged_score(x,y,sample_weight=w)),
            tree_metadata=np.array([[e.tree_.node_count,e.get_depth(),e.get_n_leaves(),e.n_features_in_,e.n_outputs_,e.max_features_] for e in trees]).ravel().tolist(),
            seeds=[e.random_state for e in trees],values=np.concatenate([e.tree_.value.ravel() for e in trees]).tolist())
        expected.update({key:np.concatenate([getattr(e.tree_,attr) for e in trees]).tolist() for key,attr in [
            ('impurity','impurity'),('mass','weighted_n_node_samples'),('count','n_node_samples')]})
        expected.update(left=np.concatenate([e.tree_.children_left+1 for e in trees]).tolist(),right=np.concatenate([e.tree_.children_right+1 for e in trees]).tolist(),
            feature=np.concatenate([np.where(e.tree_.children_left<0,0,e.tree_.feature+1) for e in trees]).tolist(),
            threshold=np.concatenate([np.where(e.tree_.children_left<0,0,e.tree_.threshold) for e in trees]).tolist(),
            component_importance=np.concatenate([e.feature_importances_ for e in trees]).tolist(),
            decision=[] if regression else model.decision_function(query).ravel().tolist(),
            probabilities=[] if regression else model.predict_proba(query).ravel().tolist(),prototype=[0,0,model.estimator_.random_state])
    adaptations=[]
    if regression and not np.all(np.isfinite(importance)):
        assert np.all(np.isnan(importance)) and np.sum(model.estimator_weights_)==0
        adaptations.append('existing native zero importance for zero ensemble weight; Source property is undefined')
    return expected,adaptations


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--kind',choices=['regression','classification'],required=True)
    ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true');a=ap.parse_args();records=[];payload=[]
    for c,x,y,w in fixtures(a.kind,a.smoke):
        fields=[int(a.kind=='classification')]+[c[k] for k in ['n','p','seed','nt','depth','criterion','splitter','leaf','split','features','budget','ccp','fraction','decrease','rate','loss','weighted']]
        text=' '.join(map(str,fields))+'\n'+'\n'.join(' '.join(map(str,row)) for row in x)+'\n'
        text+=' '.join(map(str,y))+'\n'+' '.join(map(str,w))+'\n'
        error=None;expected=None;adaptations=[]
        try:expected,adaptations=source_case(c,x,y,w)
        except ValueError as e:
            assert a.kind=='classification' and 'worse than random' in str(e),(c,str(e))
            error=dict(type=type(e).__name__,message=str(e))
        record=dict(case=c,source=expected,source_error=error,adaptations=adaptations,input=text,differences=[])
        records.append(record)
        if error is None:payload.append(text)
    text=''.join(payload);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.with_suffix('.in').write_text(text)
    p=subprocess.run([str(a.executable)],input=text,text=True,capture_output=True,timeout=180)
    a.report.with_suffix('.stdout').write_text(p.stdout);a.report.with_suffix('.stderr').write_text(p.stderr)
    lines=p.stdout.splitlines();valid=[r for r in records if r['source'] is not None];width=23
    integer={'metadata','tree_metadata','seeds','count','left','right','feature','prototype'}
    if a.kind=='classification':integer.update(['train','query','staged'])
    for i,r in enumerate(valid):
        chunk=lines[i*width:(i+1)*width];r['native']={}
        if len(chunk)!=width:r['differences'].append('missing native output');continue
        for (key,want),line in zip(r['source'].items(),chunk):
            got=np.fromstring(line,sep=' ');want=np.asarray(want);r['native'][key]=got.tolist()
            if key=='importance' and r['adaptations']:want=np.zeros_like(want)
            ok=got.shape==want.shape and (np.array_equal(got,want) if key in integer else np.allclose(got,want,rtol=2.e-11,atol=2.e-12,equal_nan=True))
            if not ok:r['differences'].append(key)
    for r in records:
        if r['source_error'] is None:continue
        rejected=subprocess.run([str(a.executable)],input=r['input'],text=True,capture_output=True,timeout=30)
        output=rejected.stdout+rejected.stderr;r['native_error']=dict(returncode=rejected.returncode,output=output)
        if not(rejected.returncode!=0 and 'no better than random guessing' in output and not any(v in output for v in [
                'SIGFPE','SIGSEGV','Fortran runtime error'])):r['differences'].append('expected controlled Source weak-learner rejection')
    files=[Path(__file__),a.executable,Path('app/reference_ada_tree_controls_driver.f90')]+[Path('src')/name for name in [
        'sklearn_adaboost.f90','sklearn_adaboost_regression.f90','sklearn_tree.f90','sklearn_tree_features.f90','sklearn_search_adapters.f90']]
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report=dict(kind=a.kind,configurations=len(records),successful_fits=len(valid),source_rejections=len(records)-len(valid),
        failed_configurations=sum(bool(r['differences']) for r in records),native_returncode=p.returncode,records=records,
        input_sha256=hashlib.sha256(text.encode()).hexdigest(),sklearn=sklearn.__version__,runtime=tree_runtime_provenance(),
        sha256={str(p):sha(p) for p in files})
    a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['records','runtime','sha256']})
    assert p.returncode==0 and len(lines)==len(valid)*width and report['failed_configurations']==0


if __name__=='__main__':main()
