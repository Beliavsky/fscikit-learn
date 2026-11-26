"""Actual sklearn classic regression boosting: all losses and complete stages."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.stats import _weighted_percentile
from sklearn_tree_runtime import tree_runtime_provenance

def fixtures(smoke=False):
    for loss,criterion,seed,n,weighted,zero in itertools.product(
        ['squared_error','absolute_error','quantile','huber'],['friedman_mse','squared_error'],[0,7],[12,17],[0,1],[0,1]):
        if smoke and (criterion!='friedman_mse' or seed!=7 or n!=12 or zero):continue
        rng=np.random.RandomState(2117+n);p=3;x=rng.normal(size=(n,p));y=2*x[:,0]+x[:,1]*x[:,2]
        y[-1]+=5;w=np.resize([1.,0.,.5,2.,3.],n)
        yield dict(n=n,p=p,loss=loss,criterion=criterion,seed=seed,nt=5,depth=3,feature=.5 if seed==7 else 'none',
                   leaf=1,split=2,budget=0,rate=.15,alpha=.3 if n==12 else .9,decrease=0.,ccp=0.,fraction=0.,weighted=weighted,zero=zero),x,y,w
    if smoke:return
    for loss,seed,profile,zero in itertools.product(['squared_error','absolute_error','quantile','huber'],[0,42],range(6),[0,1]):
        n=[9,13,21][profile%3];p=[1,3,8][profile%3];rng=np.random.RandomState(4117+profile)
        x=rng.normal(size=(n,p));y=np.round(rng.normal(size=n),1);w=np.resize([1.,0.,.5,2.,3.],n)
        if profile==0:y[:]=.25
        if profile==1:x[:]=1
        if profile==2:x[:,:p//2]=0;y[-1]+=20
        if profile==4:w[0]=-.1
        if profile==5:y[::3]=0;w*=.75
        yield dict(n=n,p=p,loss=loss,criterion='friedman_mse' if profile%2==0 else 'squared_error',seed=seed,
                   nt=7,depth=3,feature=[1,1.,.001,'log2',p+3,0][profile],leaf=1 if profile<3 else 2,split=2,
                   budget=5 if profile%2 else 0,rate=.2,alpha=[.1,.5,.9][profile%3],decrease=.03 if profile==3 else 0.,
                   ccp=.02 if profile==5 else 0.,fraction=.05 if profile==4 else 0.,weighted=1,zero=zero,profile=profile),x,y,w
    for loss,weighted,zero in itertools.product(['squared_error','absolute_error','quantile','huber'],[0,1],[0,1]):
        n=11;p=3;rng=np.random.RandomState(919);x=rng.normal(size=(n,p));y=rng.normal(size=n);w=np.resize([1.,0.,.5,2.],n)
        yield dict(n=n,p=p,loss=loss,criterion='friedman_mse',seed=7,nt=4,depth=3,feature='none',leaf=1,split=2,budget=0,
                   rate=0.,alpha=.9,decrease=0.,ccp=0.,fraction=0.,weighted=weighted,zero=zero,profile=6),x,y,w

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--executable',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();records=[];payload=[]
    for c,x,y,w in fixtures(a.smoke):
        params=dict(loss=c['loss'],criterion=c['criterion'],random_state=c['seed'],n_estimators=c['nt'],max_depth=c['depth'],
                    max_features=None if c['feature']=='none' else c['feature'] or None,min_samples_leaf=c['leaf'],min_samples_split=c['split'],
                    max_leaf_nodes=c['budget'] or None,learning_rate=c['rate'],alpha=c['alpha'],min_impurity_decrease=c['decrease'],
                    ccp_alpha=c['ccp'],min_weight_fraction_leaf=c['fraction'],init='zero' if c['zero'] else None)
        weights=w if c['weighted'] else None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');model=GradientBoostingRegressor(**params).fit(x,y,sample_weight=weights)
            query=x+.137;trees=[v[0] for v in model.estimators_];stages=list(model.staged_predict(query))
            delta=np.zeros(c['nt'])
            if c['loss']=='huber':
                raw=model._raw_predict_init(x)[:,0]
                for i,pred in enumerate(model.staged_predict(x)):
                    delta[i]=_weighted_percentile(np.abs(y-raw),np.ones(len(y)) if weights is None else weights,100*c['alpha']);raw=pred
            expected=dict(metadata=[model.n_features_in_,model.n_estimators_,model.n_trees_per_iteration_,model.max_features_],
                          init=model._raw_predict_init(x)[0].tolist(),loss=model.train_score_.tolist(),importance=model.feature_importances_.tolist(),
                          train=model.predict(x).tolist(),query=model.predict(query).tolist(),staged=np.stack(stages,axis=1).ravel().tolist(),
                          leaves=(model.apply(query)+1).ravel().tolist(),
                          tree_metadata=np.array([[e.tree_.node_count,e.get_depth(),e.get_n_leaves(),e.max_features_] for e in trees]).ravel().tolist(),
                          values=np.concatenate([e.tree_.value.ravel() for e in trees]).tolist())
            expected.update({key:np.concatenate([getattr(e.tree_,attr) for e in trees]).tolist() for key,attr in [('impurity','impurity'),('mass','weighted_n_node_samples'),('count','n_node_samples')]})
            expected.update(left=np.concatenate([e.tree_.children_left+1 for e in trees]).tolist(),right=np.concatenate([e.tree_.children_right+1 for e in trees]).tolist(),
                            feature=np.concatenate([np.where(e.tree_.children_left<0,0,e.tree_.feature+1) for e in trees]).tolist(),
                            threshold=np.concatenate([np.where(e.tree_.children_left<0,0,e.tree_.threshold) for e in trees]).tolist(),
                            component_importance=np.concatenate([e.feature_importances_ for e in trees]).tolist(),delta=delta.tolist(),score=[model.score(x,y,sample_weight=weights)])
        records.append(dict(case=c,source=expected,differences=[]))
        text=' '.join(str(c[k]) for k in ['n','p','loss','criterion','seed','nt','depth','feature','leaf','split','budget','rate','alpha','decrease','ccp','fraction','weighted','zero'])+'\n'
        text+='\n'.join(' '.join(map(str,row)) for row in x)+'\n'+' '.join(map(str,y))+'\n'+' '.join(map(str,w))+'\n';payload.append(text)
    text=''.join(payload);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.with_suffix('.in').write_text(text)
    p=subprocess.run([str(a.executable)],input=text,text=True,capture_output=True)
    a.report.with_suffix('.stdout').write_text(p.stdout);a.report.with_suffix('.stderr').write_text(p.stderr)
    lines=p.stdout.splitlines();fields=list(records[0]['source']);width=len(fields);assert width==20
    integer={'metadata','leaves','tree_metadata','count','left','right','feature'}
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
                sha256={str(f):sha(f) for f in [Path(__file__),a.executable,Path('src/sklearn_classic_gradient_boosting.f90'),Path('src/sklearn_ensemble.f90'),Path('src/sklearn_tree.f90'),Path('src/sklearn_boosting_loss.f90')]})
    a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['records','runtime','sha256']})
    assert p.returncode==0 and report['failed_configurations']==0
if __name__=='__main__':main()
