"""Classic boosting warm-start sequences against installed sklearn."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn_tree_runtime import tree_runtime_provenance

def groups(kind,smoke):
    pairs=[('squared_error',1),('absolute_error',1),('quantile',1),('huber',1)] if kind=='regression' else [('log_loss',2),('log_loss',3),('log_loss',5),('log_loss',9),('exponential',2)]
    for (loss,classes),seed,weighted,subsample,zero in itertools.product(pairs,[0,7],[0,1],[1.,.65],[0,1]):
        if smoke and (seed or weighted!=1 or subsample!=.65 or zero):continue
        yield dict(loss=loss,classes=classes,seed=seed,weighted=weighted,subsample=subsample,zero=zero,profile=0)
    if smoke:return
    for (loss,classes),seed,profile in itertools.product(pairs,[0,42],range(1,7)):
        yield dict(loss=loss,classes=classes,seed=seed,weighted=1,subsample=.65,zero=0,profile=profile)
    for (loss,classes),seed in itertools.product(pairs,[0,42]):
        yield dict(loss=loss,classes=classes,seed=seed,weighted=1,subsample=.65,zero=0,profile=7)
    for seed in [0,42]:
        yield dict(loss='squared_error' if kind=='regression' else 'log_loss',classes=1 if kind=='regression' else 2,seed=seed,weighted=1,subsample=.65,zero=0,profile=8)
    if kind=='regression':
        for seed in [0,42]:yield dict(loss='squared_error',classes=1,seed=seed,weighted=1,subsample=1.,zero=0,profile=9)
    for seed,zero in itertools.product([0,42],[1] if kind=='regression' else [0,1]):
        yield dict(loss='squared_error' if kind=='regression' else 'log_loss',classes=1 if kind=='regression' else 2,seed=seed,weighted=1,subsample=.65,zero=zero,profile=10)

def snapshot(model,x,y,w,kind,prior,delta):
    n=len(y);query=x+.137;trees=list(model.estimators_.ravel());q=model.n_trees_per_iteration_
    meta=[model.n_features_in_,model.n_estimators_,q,model.max_features_]
    if kind=='classification':
        out=dict(metadata=meta+[model.n_classes_],classes=model.classes_.tolist(),prior=prior.tolist(),init=model._raw_predict_init(x)[0].tolist(),
            loss=model.train_score_.tolist(),importance=model.feature_importances_.tolist(),train=model.decision_function(x).ravel().tolist(),
            query=model.decision_function(query).ravel().tolist(),proba=model.predict_proba(query).ravel().tolist(),prediction=model.predict(query).tolist(),
            staged=np.stack([v.reshape(n,q) for v in model.staged_decision_function(query)],axis=1).ravel().tolist(),leaves=(model.apply(query)+1).ravel().tolist())
    else:
        out=dict(metadata=meta,init=model._raw_predict_init(x)[0].tolist(),loss=model.train_score_.tolist(),importance=model.feature_importances_.tolist(),
            train=model.predict(x).tolist(),query=model.predict(query).tolist(),staged=np.stack(list(model.staged_predict(query)),axis=1).ravel().tolist(),leaves=(model.apply(query)+1).ravel().tolist())
    out.update(tree_metadata=np.array([[e.tree_.node_count,e.get_depth(),e.get_n_leaves(),e.max_features_] for e in trees]).ravel().tolist(),values=np.concatenate([e.tree_.value.ravel() for e in trees]).tolist())
    out.update({key:np.concatenate([getattr(e.tree_,attr) for e in trees]).tolist() for key,attr in [('impurity','impurity'),('mass','weighted_n_node_samples'),('count','n_node_samples')]})
    out.update(left=np.concatenate([e.tree_.children_left+1 for e in trees]).tolist(),right=np.concatenate([e.tree_.children_right+1 for e in trees]).tolist(),
        feature=np.concatenate([np.where(e.tree_.children_left<0,0,e.tree_.feature+1) for e in trees]).tolist(),threshold=np.concatenate([np.where(e.tree_.children_left<0,0,e.tree_.threshold) for e in trees]).tolist(),component_importance=np.concatenate([e.feature_importances_ for e in trees]).tolist())
    if kind=='regression':out['delta']=delta.tolist()
    try:out['score']=[model.score(x,y,sample_weight=w)];score_error=None
    except (ZeroDivisionError,ValueError) as e:out['score']=[];score_error=str(e)
    out.update(oob_scores=getattr(model,'oob_scores_',np.empty(0)).tolist(),oob_improvement=getattr(model,'oob_improvement_',np.empty(0)).tolist(),oob_score=[getattr(model,'oob_score_',0.)])
    return out,score_error

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--kind',choices=['regression','classification'],required=True)
    ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    records=[];payload=[];rejections=[];all_inputs=[];cls=GradientBoostingRegressor if a.kind=='regression' else GradientBoostingClassifier
    for group,g in enumerate(groups(a.kind,a.smoke)):
        model=cls();delta=np.empty(0);prior=None;history=[]
        for step,nt in enumerate([2,2,4,6]):
            n=19 if not(g['profile']==2 and step>1) else 23
            if g['profile'] in [8,10]:n=20
            p=3;rng=np.random.RandomState(119+n)
            x=rng.normal(size=(n,p));y=2*x[:,0]+x[:,1]*x[:,2] if a.kind=='regression' else np.arange(n)%g['classes']*17-9
            if a.kind=='classification':rng.shuffle(y)
            w=np.resize([1.,.5,2.,3.],n);weights=w if g['weighted'] else None
            sub=g['subsample'];warm=int(step>0);loss=g['loss'];rate=.15;seed=g['seed'];zero=g['zero'];alpha=.9
            if g['profile']==1:sub=[1.,1.,.65,1.][step]
            if g['profile']==2 and step>1:
                if a.kind=='regression':y=y+3
                else:y=y+100
                w=w[::-1];weights=w
            if g['profile']==3 and step>1:rate=.07;seed=91;zero=1;alpha=.3
            if g['profile']==4 and step>1:loss='huber' if a.kind=='regression' else 'log_loss'
            if g['profile']==5 and step>1:warm=0;nt=3
            if g['profile']==6:sub=[.65,.65,1.,.6][step]
            if g['profile']==7 and step>1:
                if a.kind=='regression':loss={'squared_error':'absolute_error','absolute_error':'squared_error','quantile':'huber','huber':'quantile'}[loss]
                elif g['classes']==2:loss='log_loss' if loss=='exponential' else 'exponential'
            if g['profile']==8 and step>1:w=np.where(np.arange(n)%2,1.,-1.);weights=w
            if g['profile']==9:
                if step==0:warm=1
                if step==1:w[:]=0;weights=w
            if g['profile']==10:
                w=np.where(np.arange(n)%2,1.,-1.) if a.kind=='regression' else np.where(y==min(y),-1.,1.)
                weights=w
            c=dict(n=n,p=p,loss=loss,criterion='friedman_mse' if seed%2 else 'squared_error',seed=seed,nt=nt,depth=2 if step<2 else 3,feature='sqrt' if step<2 else .7,
                leaf=1,split=2,budget=0 if step<2 else 4,rate=rate,alpha=alpha,decrease=0.,ccp=0.,fraction=0.,weighted=int(weights is not None),zero=zero,subsample=sub,warm=warm,fresh=int(step==0),profile=g['profile'],step=step,group=group)
            fresh=not warm or not hasattr(model,'estimators_');delta=np.zeros(nt) if fresh else np.resize(delta,nt)
            if fresh and a.kind=='classification':
                with warnings.catch_warnings():warnings.simplefilter('ignore');prior=DummyClassifier(strategy='prior').fit(x,y,sample_weight=weights).class_prior_
            params=dict(n_estimators=nt,warm_start=bool(warm),loss=loss,criterion=c['criterion'],random_state=seed,max_depth=c['depth'],max_features=c['feature'],min_samples_leaf=1,min_samples_split=2,max_leaf_nodes=c['budget'] or None,
                learning_rate=rate,min_impurity_decrease=0.,ccp_alpha=0.,min_weight_fraction_leaf=0.,init='zero' if zero else None,subsample=sub)
            if a.kind=='regression':params['alpha']=alpha
            model.set_params(**params)
            def monitor(i,est,local):
                delta[i]=est._loss.closs.delta if loss=='huber' else 0.
            source_error=None;score_error=None;expected=None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');model.fit(x,y,sample_weight=weights,monitor=monitor if a.kind=='regression' else None)
                    expected,score_error=snapshot(model,x,y,weights,a.kind,prior,delta)
            except (ValueError,ZeroDivisionError,IndexError) as e:
                source_error=dict(type=type(e).__name__,message=str(e))
            c['score_ok']=int(score_error is None)
            keys=['n','p','loss','criterion','seed','nt','depth','feature','leaf','split']
            keys+=['budget','rate','alpha','decrease','ccp','fraction','weighted','zero'] if a.kind=='regression' else ['rate','decrease','zero','weighted','budget','ccp','fraction']
            keys+=['subsample','warm','fresh','score_ok']
            text=' '.join(str(c[k]) for k in keys)+'\n'+'\n'.join(' '.join(map(str,row)) for row in x)+'\n'+' '.join(map(str,y))+'\n'+' '.join(map(str,w))+'\n'
            all_inputs.append(text)
            record=dict(case=c,source=expected,score_error=score_error,differences=[])
            if source_error:
                record['source_error']=source_error;record['input']=''.join(history)+text;rejections.append(record)
            else:
                payload.append(text);history.append(text)
            records.append(record)
    text=''.join(payload);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.with_suffix('.in').write_text(text)
    native=subprocess.run([str(a.executable)],input=text,text=True,capture_output=True,timeout=180)
    a.report.with_suffix('.stdout').write_text(native.stdout);a.report.with_suffix('.stderr').write_text(native.stderr)
    lines=native.stdout.splitlines();width=23 if a.kind=='regression' else 26
    integers={'metadata','classes','prediction','leaves','tree_metadata','count','left','right','feature'}
    valid=[r for r in records if r['source'] is not None]
    for i,r in enumerate(valid):
        chunk=lines[i*width:(i+1)*width];r['native']={}
        if len(chunk)!=width:r['differences'].append('missing native output');continue
        for (key,want),line in zip(r['source'].items(),chunk):
            got=np.fromstring(line,sep=' ');want=np.asarray(want);r['native'][key]=got.tolist()
            ok=got.shape==want.shape and (np.array_equal(got,want) if key in integers else np.allclose(got,want,rtol=2e-10,atol=2e-12,equal_nan=True))
            if not ok:r['differences'].append(key)
    for record in rejections:
        rejected=subprocess.run([str(a.executable)],input=record['input'],text=True,capture_output=True,timeout=30)
        output=rejected.stdout+rejected.stderr;record['native_error']=dict(returncode=rejected.returncode,output=output)
        if not (rejected.returncode!=0 and 'fscikit-learn:' in output and not any(v in output for v in ['SIGFPE','SIGSEGV','Fortran runtime error'])):
            record['differences'].append('expected controlled fit rejection')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    driver='gbr' if a.kind=='regression' else 'gbc'
    files=[Path(__file__),Path(f'app/reference_{driver}_warm_driver.f90'),a.executable,Path('src/sklearn_classic_gradient_boosting.f90'),Path('src/sklearn_tree.f90')]
    report=dict(kind=a.kind,configurations=len(records),successful_fits=len(valid),source_rejections=len(rejections),groups=group+1,failed_configurations=sum(bool(r['differences']) for r in records),native_returncode=native.returncode,records=records,input_sha256=hashlib.sha256(text.encode()).hexdigest(),all_input_sha256=hashlib.sha256(''.join(all_inputs).encode()).hexdigest(),sklearn=sklearn.__version__,runtime=tree_runtime_provenance(),sha256={str(p):sha(p) for p in files})
    a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['records','runtime','sha256']})
    assert native.returncode==0 and len(lines)==len(valid)*width and report['failed_configurations']==0
if __name__=='__main__':main()
