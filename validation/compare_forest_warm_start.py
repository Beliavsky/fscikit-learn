"""Actual sklearn staged forests: retained trees, seed skipping, OOB and resets."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,RandomForestRegressor,ExtraTreesRegressor
from sklearn_tree_runtime import tree_runtime_provenance
from sklearn_feature_fixtures import forest_feature_fixtures

def fixtures(kind):
    criteria=['gini','entropy','log_loss'] if kind=='classification' else ['squared_error','friedman_mse','absolute_error','poisson']
    paths=range(7 if kind=='classification' else 6)
    for sequence,(mode,criterion,q,seed,boot,path) in enumerate(itertools.product([0,1],criteria,[1,2,3],[0,7],[0,1],paths)):
        for step in range(4):
            nt=[[2,2,5,8],[2,2,5,7],[2,5,3,6],[2,4,4,7],[3,3,5,5],[2,4,6,6],[2,4,6,6]][path][step]
            n=17 if path==0 else [17,11,19,13][step]
            rng=np.random.RandomState(1131+q+(0 if path==0 else step))
            x=rng.normal(size=(n,3));w=np.resize([1.,.5,2.,3.],n)*(1 if step%2==0 else .75)
            if kind=='regression':y=np.abs(rng.normal(size=(n,q)))+np.arange(q)*.2+.1
            elif path==5 and step==0:y=np.full((n,q),-9,dtype=int)
            elif path==6:y=np.column_stack([rng.randint(k%3+1 if step else 1,size=n)*17-9+k for k in range(q)])
            else:y=rng.randint(2,size=(n,q))*(1 if step<2 else 17)-(0 if step<2 else 9)
            current_boot=boot
            if path==1 and step>0:current_boot=1
            if path==3:current_boot=[1,0,1,1][step]
            if path==4:current_boot=1
            oob=bool(current_boot)
            if path==2 and step in [1,2]:oob=False
            if path==4 and step in [0,2]:oob=False
            if path==6:oob=False
            cw=0 if path==0 else (step+seed+q)%4
            yield dict(sequence=sequence,step=step,reset=int(step==0),warm=int(not(path==2 and step==2)),
                       mode=mode,criterion=criterion if path==0 else criteria[(criteria.index(criterion)+step)%len(criteria)],
                       seed=seed if path==0 else [seed,42,7,0][step],q=q,n=n,p=3,nt=nt,boot=current_boot,
                       ns=6 if current_boot and path in [1,4] and step>0 else 0,oob=int(oob),weighted=path!=0 or seed==7,
                       depth=3 if path==0 else 2+step%3,budget=0 if path==0 or step%2==0 else 5,nf=0 if path==0 or step%2==0 else 2,
                       leaf=1,alpha=.01 if step%2 else 0.,decrease=0.,cw=cw,path=path),x,y,w

def fractional_fixtures(kind):
    criteria=['gini','entropy','log_loss'] if kind=='classification' else ['squared_error','friedman_mse','absolute_error','poisson']
    for sequence,(mode,criterion,q,seed,path) in enumerate(itertools.product([0,1],criteria,[1,2],[0,7],range(4))):
        for step in range(4):
            n=[5,7,9,11][step] if path==2 else 17
            rng=np.random.RandomState(1131+q+step)
            x=rng.normal(size=(n,3));w=np.resize([1.,.5,2.,3.],n)
            y=np.abs(rng.normal(size=(n,q)))+.1 if kind=='regression' else rng.randint(2,size=(n,q))
            # Keep label spaces stable: changing class widths is covered by
            # the separate warm guard grid and can fail sklearn prediction.
            if kind=='classification':y[0,:]=0;y[1,:]=1
            ns=[[.5,.3,.001,1.0],[1,1.0,0,.7],[.5,.5,.5,.5],
                [np.nextafter(.5,0),.5,np.nextafter(.5,1),.9]][path][step]
            yield dict(sequence=sequence,step=step,reset=int(step==0),warm=1,mode=mode,criterion=criterion,
                       seed=seed,q=q,n=n,p=3,nt=[2,2,5,8][step],boot=1,ns=ns,oob=1,
                       weighted=seed==7,depth=3,budget=0,nf=0,leaf=1,alpha=0.,decrease=0.,
                       cw=2 if path%2 else 0,path=path),x,y,w

def snapshot(model,x,y,w,c,coverage,kind):
    q=c['q'];trees=[e.tree_ for e in model.estimators_];bags=model.estimators_samples_
    expected=dict(metadata=[model.n_features_in_,model.n_outputs_,len(trees)])
    if kind=='classification':
        classes=[model.classes_] if q==1 else model.classes_
        probs=model.predict_proba(x);logs=model.predict_log_proba(x)
        if q==1:probs=[probs];logs=[logs]
        expected.update(class_counts=[len(v) for v in classes],classes=np.concatenate(classes).tolist())
    expected['prediction']=model.predict(x).ravel().tolist()
    if kind=='classification':
        expected.update(probabilities=np.concatenate([v.ravel() for v in probs]).tolist(),log_probabilities=np.concatenate([v.ravel() for v in logs]).tolist())
    expected.update(leaves=(model.apply(x)+1).ravel().tolist(),importance=model.feature_importances_.tolist(),
                    seeds=[e.random_state for e in model.estimators_],samples=(np.stack(bags,axis=1)+1).ravel().tolist())
    if kind=='classification':
        oob=getattr(model,'oob_decision_function_',None)
        flat=[] if oob is None else (oob.ravel() if oob.ndim==2 else np.concatenate([oob[:,:,k].ravel() for k in range(oob.shape[2])])).tolist()
        expected['oob_probabilities']=flat
    else:expected['oob_prediction']=getattr(model,'oob_prediction_',np.array([])).ravel().tolist()
    expected.update(oob_score=[model.oob_score_] if hasattr(model,'oob_score_') else [],oob_counts=coverage.tolist() if coverage is not None else [],
                    tree_metadata=np.array([[t.node_count,e.get_depth(),e.get_n_leaves(),e.n_features_in_,e.n_outputs_] for e,t in zip(model.estimators_,trees)],int).ravel().tolist())
    if kind=='classification':
        counts=[np.atleast_1d(e.n_classes_).astype(int) for e in model.estimators_]
        expected.update(tree_class_counts=np.concatenate(counts).tolist(),
                        tree_classes=np.concatenate([np.concatenate([e.classes_] if q==1 else e.classes_) for e in model.estimators_]).tolist(),
                        values=np.concatenate([np.concatenate([t.value[:,k,:counts[i][k]] for k in range(q)],axis=1).ravel() for i,t in enumerate(trees)]).tolist())
    else:expected['values']=np.concatenate([t.value[:,:,0].ravel() for t in trees]).tolist()
    expected.update({k:np.concatenate([getattr(t,attr) for t in trees]).tolist() for k,attr in [('impurity','impurity'),('mass','weighted_n_node_samples'),('count','n_node_samples')]})
    expected.update(left=np.concatenate([t.children_left+1 for t in trees]).tolist(),right=np.concatenate([t.children_right+1 for t in trees]).tolist(),
                    feature=np.concatenate([np.where(t.children_left<0,0,t.feature+1) for t in trees]).tolist(),
                    threshold=np.concatenate([np.where(t.children_left<0,0,t.threshold) for t in trees]).tolist(),
                    tree_importance=np.concatenate([e.feature_importances_ for e in model.estimators_]).tolist())
    try:expected['score']=[model.score(x,y,sample_weight=w)];error=None
    except (ValueError,ZeroDivisionError) as e:expected['score']=[];error=str(e)
    return expected,error

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kind',choices=['classification','regression'],required=True)
    ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--reuse-report',type=Path)
    ap.add_argument('--features',action='store_true')
    ap.add_argument('--fractional',action='store_true',help='Fractional counts, ties, typed resets and warm OOB state')
    a=ap.parse_args();records=[];payloads=[];coverage=None
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.reuse_report:
        old=json.loads(a.reuse_report.read_text());assert old['kind']==a.kind and old['sklearn']==sklearn.__version__
        assert old.get('fractional',False)==a.fractional and old.get('features',False)==a.features
        assert old['validator_sha256']==sha(__file__)
        records=old['records'];payload=a.reuse_report.with_suffix('.in').read_text()
        assert hashlib.sha256(payload.encode()).hexdigest()==old['input_sha256']
        for r in records:r['differences']=[];r.pop('native',None)
    else:
        for c,x,y,w in (forest_feature_fixtures(a.kind) if a.features else fractional_fixtures(a.kind) if a.fractional else fixtures(a.kind)):
            if c['reset']:
                cls=([RandomForestClassifier,ExtraTreesClassifier] if a.kind=='classification' else [RandomForestRegressor,ExtraTreesRegressor])[c['mode']]
                model=cls(n_jobs=1);coverage=None
            params=dict(n_estimators=c['nt'],warm_start=bool(c['warm']),criterion=c['criterion'],random_state=c['seed'],bootstrap=bool(c['boot']),
                        max_samples=c['ns'] or None,oob_score=bool(c['oob']),max_depth=c['depth'],max_leaf_nodes=c['budget'] or None,
                        max_features=None if c['nf']=='none' else c['nf'] or ('sqrt' if a.kind=='classification' else 1.),min_samples_leaf=c['leaf'],ccp_alpha=c['alpha'],min_impurity_decrease=c['decrease'])
            if a.kind=='classification':
                maps=[{v:i+1.25 for i,v in enumerate(np.unique(y[:,k]))} for k in range(c['q'])]
                params['class_weight']=[None,'balanced','balanced_subsample',maps[0] if c['q']==1 else maps][c['cw']]
            old_count=len(model.estimators_) if c['warm'] and hasattr(model,'estimators_') else 0
            recompute=bool(c['oob'] and (c['nt']>old_count or not hasattr(model,'oob_score_')))
            model.set_params(**params);fit_weights=w if c['weighted'] else None
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');model.fit(x,y,sample_weight=fit_weights)
                if recompute:coverage=np.sum([np.bincount(b,minlength=c['n'])==0 for b in model.estimators_samples_],axis=0)
                expected,error=snapshot(model,x,y,fit_weights,c,coverage,a.kind)
            if a.features:expected['resolved_features']=[e.max_features_ for e in model.estimators_]
            if a.kind=='regression':assert error is None,(c,error)
            records.append(dict(case=c,source=expected,score_error=error,differences=[]))
            keys=['n','p','q','mode','criterion','seed','nt','boot','ns','oob','weighted','depth','budget','nf','leaf','alpha','decrease']
            text=' '.join(str(int(c[k])) if k=='weighted' else str(c[k]) for k in keys)
            if a.kind=='classification':text+=f" {c['cw']} {int(error is None)}"
            text+=f" {c['warm']} {c['reset']}\n"
            for arr in [x,y]:text+='\n'.join(' '.join(map(str,row)) for row in arr)+'\n'
            text+=' '.join(map(str,w))+'\n';payloads.append(text)
        payload=''.join(payloads)
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.with_suffix('.in').write_text(payload)
    native=subprocess.run([str(a.executable)]+(['--features'] if a.features else []),input=payload,text=True,capture_output=True)
    a.report.with_suffix('.stdout').write_text(native.stdout);a.report.with_suffix('.stderr').write_text(native.stderr)
    lines=native.stdout.splitlines();fields=list(records[0]['source']);width=len(fields);assert width==(26 if a.kind=='classification' else 20)+int(a.features)
    integer={'resolved_features','metadata','class_counts','classes','prediction','leaves','seeds','samples','oob_counts','tree_metadata','tree_class_counts','tree_classes','count','left','right','feature'}
    if a.kind=='regression':integer.remove('prediction')
    for i,r in enumerate(records):
        chunk=lines[width*i:width*(i+1)];r['native']={}
        if len(chunk)!=width:r['differences'].append('missing native output');continue
        for key,line in zip(fields,chunk):
            got=np.fromstring(line,sep=' ');want=np.asarray(r['source'][key]);r['native'][key]=got.tolist()
            passed=got.shape==want.shape and (np.array_equal(got,want) if key in integer else np.allclose(got,want,rtol=2e-10,atol=2e-12,equal_nan=True))
            if not passed:r['differences'].append(key)
    report=dict(kind=a.kind,features=a.features,fractional=a.fractional,configurations=len(records),sequences=sum(r['case']['reset'] for r in records),records=records,
                failed_configurations=sum(bool(r['differences']) for r in records),native_returncode=native.returncode,
                input_sha256=hashlib.sha256(payload.encode()).hexdigest(),sklearn=sklearn.__version__,runtime=tree_runtime_provenance(),validator_sha256=sha(__file__),
                sha256={str(p):sha(p) for p in [a.executable,Path('src/sklearn_ensemble.f90'),Path('src/sklearn_tree.f90'),Path('src/sklearn_tree_features.f90'),Path('validation/sklearn_feature_fixtures.py')]})
    a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['records','sha256','runtime','validator_sha256']})
    assert not native.returncode and not report['failed_configurations']

if __name__=='__main__':main()
