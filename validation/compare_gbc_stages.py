"""Strict sklearn classic classification boosting, including complete trees."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn_tree_runtime import tree_runtime_provenance

def fixtures():
    for (loss,classes),criterion,seed,n,zero,pattern in itertools.product(
        [('log_loss',2),('log_loss',3),('exponential',2)],['friedman_mse','squared_error'],[0,7],[12,19],[0,1],[0,1]):
        p=3 if n==12 else 8;rng=np.random.RandomState(919+n+seed)
        x=rng.normal(size=(n,p));labels=(np.arange(n)%classes)*17-9
        if pattern:x[:]=1
        else:rng.shuffle(labels)
        feature=['none',.5,'sqrt','log2'][seed%2+2*zero]
        yield dict(n=n,p=p,loss=loss,criterion=criterion,seed=seed,nt=5,depth=3,feature=feature,
                   leaf=1,split=2,rate=.15,decrease=0.,zero=zero,pattern=pattern),x,labels
    for classes,criterion,seed,zero,pattern in itertools.product([5,9],['friedman_mse','squared_error'],[0,42],[0,1],[0,1]):
        n=29;p=4;rng=np.random.RandomState(3917+seed);x=rng.normal(size=(n,p));labels=(np.arange(n)%classes)*17-9
        if pattern:x[:]=1
        else:rng.shuffle(labels)
        yield dict(n=n,p=p,loss='log_loss',criterion=criterion,seed=seed,nt=4,depth=3,
                   feature=1. if seed==0 else p+3,leaf=2,split=4,rate=.2,decrease=.01 if pattern else 0.,zero=zero,pattern=pattern),x,labels
    for (loss,classes),criterion,zero in itertools.product([('log_loss',2),('log_loss',3),('exponential',2)],['friedman_mse','squared_error'],[0,1]):
        n=19;p=3;rng=np.random.RandomState(119);x=rng.normal(size=(n,p));labels=(np.arange(n)%classes)*17-9;rng.shuffle(labels)
        yield dict(n=n,p=p,loss=loss,criterion=criterion,seed=7,nt=4,depth=3,feature='none',leaf=1,split=2,rate=0.,decrease=0.,zero=zero,pattern=2),x,labels

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    a=ap.parse_args();records=[];payload=[]
    for c,x,y in fixtures():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model=GradientBoostingClassifier(loss=c['loss'],criterion=c['criterion'],random_state=c['seed'],n_estimators=c['nt'],
                max_depth=c['depth'],max_features=None if c['feature']=='none' else c['feature'],min_samples_leaf=c['leaf'],
                min_samples_split=c['split'],learning_rate=c['rate'],min_impurity_decrease=c['decrease'],init='zero' if c['zero'] else None).fit(x,y)
            query=x+.137;trees=list(model.estimators_.ravel());q=model.n_trees_per_iteration_
            raw=np.stack([v.reshape(len(y),q) for v in model.staged_decision_function(query)],axis=1)
            expected=dict(metadata=[model.n_features_in_,model.n_estimators_,q,model.max_features_],classes=model.classes_.tolist(),
                prior=[np.mean(y==k) for k in model.classes_],init=model._raw_predict_init(x)[0].tolist(),loss=model.train_score_.tolist(),
                importance=model.feature_importances_.tolist(),train=model.decision_function(x).ravel().tolist(),
                query=model.decision_function(query).ravel().tolist(),proba=model.predict_proba(query).ravel().tolist(),prediction=model.predict(query).tolist(),
                staged=raw.ravel().tolist(),leaves=(model.apply(query)+1).ravel().tolist(),
                tree_metadata=np.array([[e.tree_.node_count,e.get_depth(),e.get_n_leaves(),e.max_features_] for e in trees]).ravel().tolist(),
                values=np.concatenate([e.tree_.value.ravel() for e in trees]).tolist())
            expected.update({key:np.concatenate([getattr(e.tree_,attr) for e in trees]).tolist() for key,attr in [('impurity','impurity'),('mass','weighted_n_node_samples'),('count','n_node_samples')]})
            expected.update(left=np.concatenate([e.tree_.children_left+1 for e in trees]).tolist(),right=np.concatenate([e.tree_.children_right+1 for e in trees]).tolist(),
                feature=np.concatenate([np.where(e.tree_.children_left<0,0,e.tree_.feature+1) for e in trees]).tolist(),
                threshold=np.concatenate([np.where(e.tree_.children_left<0,0,e.tree_.threshold) for e in trees]).tolist(),
                component_importance=np.concatenate([e.feature_importances_ for e in trees]).tolist(),score=[model.score(x,y)])
        records.append(dict(case=c,source=expected,differences=[]))
        text=' '.join(str(c[k]) for k in ['n','p','loss','criterion','seed','nt','depth','feature','leaf','split','rate','decrease','zero'])+'\n'
        text+='\n'.join(' '.join(map(str,row)) for row in x)+'\n'+' '.join(map(str,y))+'\n';payload.append(text)
    text=''.join(payload);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.with_suffix('.in').write_text(text)
    p=subprocess.run([str(a.executable)],input=text,text=True,capture_output=True)
    a.report.with_suffix('.stdout').write_text(p.stdout);a.report.with_suffix('.stderr').write_text(p.stderr)
    lines=p.stdout.splitlines();fields=list(records[0]['source']);width=len(fields);assert width==23
    integer={'metadata','classes','prediction','leaves','tree_metadata','count','left','right','feature'}
    for i,r in enumerate(records):
        chunk=lines[width*i:width*(i+1)];r['native']={}
        if len(chunk)!=width:r['differences'].append('missing native output');continue
        for key,line in zip(fields,chunk):
            got=np.fromstring(line,sep=' ');want=np.asarray(r['source'][key]);r['native'][key]=got.tolist()
            ok=got.shape==want.shape and (np.array_equal(got,want) if key in integer else np.allclose(got,want,rtol=2e-10,atol=2e-12,equal_nan=True))
            if not ok:r['differences'].append(key)
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    report=dict(configurations=len(records),records=records,failed_configurations=sum(bool(r['differences']) for r in records),native_returncode=p.returncode,
        input_sha256=hashlib.sha256(text.encode()).hexdigest(),sklearn=sklearn.__version__,runtime=tree_runtime_provenance(),
        sha256={str(f):sha(f) for f in [Path(__file__),a.executable,Path('src/sklearn_tree.f90'),Path('src/sklearn_classic_gradient_boosting.f90'),Path('src/sklearn_ensemble.f90'),Path('src/sklearn_boosting_loss.f90')]})
    a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['records','runtime','sha256']})
    assert p.returncode==0 and report['failed_configurations']==0
if __name__=='__main__':main()
