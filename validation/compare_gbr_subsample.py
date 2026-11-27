"""Sklearn stochastic regression boosting: complete trees and OOB histories."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.stats import _weighted_percentile
from sklearn_tree_runtime import tree_runtime_provenance

def fixtures(smoke=False):
    for loss,criterion,seed,subsample,weighted,zero in itertools.product(
        ['squared_error','absolute_error','quantile','huber'],['friedman_mse','squared_error'],[0,7],[.3,.65,.999,1.],[0,1,2],[0,1]):
        if smoke and (criterion!='friedman_mse' or seed or weighted!=2 or zero or subsample!=.65):continue
        n=19 if seed==0 else 31;p=3 if seed==0 else 5;rng=np.random.RandomState(781+n);x=rng.normal(size=(n,p))
        y=2*x[:,0]+x[:,1]*x[:,2];y[-1]+=5;w=np.ones(n) if weighted!=2 else np.resize([1.,0.,.5,2.,3.],n)
        yield dict(n=n,p=p,loss=loss,criterion=criterion,seed=seed,nt=6,depth=3,feature='sqrt' if seed else .7,
            leaf=1,split=2,budget=4 if seed else 0,rate=.15,alpha=.3 if seed else .9,decrease=0.,ccp=.01 if seed else 0.,
            fraction=.05 if seed else 0.,weighted=int(weighted>0),weight_mode=weighted,zero=zero,subsample=subsample),x,y,w
    if smoke:return
    for loss,seed,weighted in itertools.product(['squared_error','absolute_error','quantile','huber'],[0,42],[0,1]):
        n=13;p=3;rng=np.random.RandomState(119);x=rng.normal(size=(n,p));y=rng.normal(size=n)
        w=np.resize([1.,.5,2.,3.],n)
        yield dict(n=n,p=p,loss=loss,criterion='friedman_mse',seed=seed,nt=5,depth=2,feature='none',leaf=1,split=2,
            budget=0,rate=.15,alpha=.9,decrease=0.,ccp=0.,fraction=0.,weighted=weighted,zero=0,subsample=.001),x,y,w

    for loss,seed,weighted in itertools.product(['squared_error','absolute_error','quantile','huber'],[0,42],[0,1]):
        n=17;p=3;rng=np.random.RandomState(119);x=rng.normal(size=(n,p));y=rng.normal(size=n)
        w=np.resize([1.,.5,2.,3.],n)
        yield dict(n=n,p=p,loss=loss,criterion='friedman_mse',seed=seed,nt=5,depth=2,feature='none',leaf=1,split=2,
            budget=0,rate=0.,alpha=.9,decrease=0.,ccp=0.,fraction=0.,weighted=weighted,zero=0,subsample=.65),x,y,w

    for loss,seed,negative in itertools.product(['squared_error','absolute_error','quantile','huber'],[0,42],[False,True]):
        n=17;p=3;rng=np.random.RandomState(391);x=rng.normal(size=(n,p));y=rng.normal(size=n)
        w=np.resize([1.,.5,2.,3.],n);w[0]=-.1
        if negative:w[:]=-1
        yield dict(n=n,p=p,loss=loss,criterion='friedman_mse',seed=seed,nt=5,depth=3,feature='none',leaf=1,split=2,
            budget=0,rate=.15,alpha=.9,decrease=0.,ccp=0.,fraction=0.,weighted=1,weight_mode=4 if negative else 3,
            zero=0,subsample=.65),x,y,w

def source_case(c,x,y,w):
    params=dict(loss=c['loss'],criterion=c['criterion'],random_state=c['seed'],n_estimators=c['nt'],max_depth=c['depth'],
                max_features=None if c['feature']=='none' else c['feature'] or None,min_samples_leaf=c['leaf'],min_samples_split=c['split'],
                max_leaf_nodes=c['budget'] or None,learning_rate=c['rate'],alpha=c['alpha'],min_impurity_decrease=c['decrease'],
                ccp_alpha=c['ccp'],min_weight_fraction_leaf=c['fraction'],init='zero' if c['zero'] else None,subsample=c['subsample'])
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
    expected.update(oob_scores=getattr(model,'oob_scores_',np.empty(0)).tolist(),
        oob_improvement=getattr(model,'oob_improvement_',np.empty(0)).tolist(),oob_score=[getattr(model,'oob_score_',0.)])
    return expected

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--executable',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();records=[];payload=[];rejections=[];all_inputs=[]
    for c,x,y,w in fixtures(a.smoke):
        text=' '.join(str(c[k]) for k in ['n','p','loss','criterion','seed','nt','depth','feature','leaf','split','budget','rate','alpha','decrease','ccp','fraction','weighted','zero','subsample'])+'\n'
        text+='\n'.join(' '.join(map(str,row)) for row in x)+'\n'+' '.join(map(str,y))+'\n'+' '.join(map(str,w))+'\n'
        all_inputs.append(text)
        try:expected=source_case(c,x,y,w)
        except ZeroDivisionError as error:
            assert str(error)=="Weights sum to zero, can't be normalized",(c,str(error))
            record=dict(case=c,source=None,source_error=dict(type=type(error).__name__,message=str(error)),input=text,differences=[])
            rejections.append(record)
        else:
            record=dict(case=c,source=expected,differences=[]);payload.append(text)
        records.append(record)
    text=''.join(payload);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.with_suffix('.in').write_text(text)
    p=subprocess.run([str(a.executable)],input=text,text=True,capture_output=True,timeout=180)
    a.report.with_suffix('.stdout').write_text(p.stdout);a.report.with_suffix('.stderr').write_text(p.stderr)
    lines=p.stdout.splitlines();valid=[r for r in records if r['source'] is not None];fields=list(valid[0]['source']);width=len(fields);assert width==23
    integer={'metadata','leaves','tree_metadata','count','left','right','feature'}
    for i,r in enumerate(valid):
        chunk=lines[width*i:width*(i+1)];r['native']={}
        if len(chunk)!=width:r['differences'].append('missing native output');continue
        for key,line in zip(fields,chunk):
            got=np.fromstring(line,sep=' ');want=np.asarray(r['source'][key]);r['native'][key]=got.tolist()
            ok=got.shape==want.shape and (np.array_equal(got,want) if key in integer else np.allclose(got,want,rtol=2e-10,atol=2e-12,equal_nan=True))
            if not ok:r['differences'].append(key)
    for record in rejections:
        rejected=subprocess.run([str(a.executable)],input=record['input'],text=True,capture_output=True,timeout=30)
        output=rejected.stdout+rejected.stderr
        record['native_error']=dict(returncode=rejected.returncode,output=output)
        if not (rejected.returncode!=0 and 'fscikit-learn: gradient boosting weights must have nonzero total' in output and not any(v in output for v in ['SIGFPE','SIGSEGV','Fortran runtime error'])):
            record['differences'].append('expected controlled zero-weight rejection')
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    report=dict(configurations=len(records),successful_fits=len(valid),source_rejections=len(rejections),all_input_sha256=hashlib.sha256(''.join(all_inputs).encode()).hexdigest(),records=records,failed_configurations=sum(bool(r['differences']) for r in records),native_returncode=p.returncode,
                input_sha256=hashlib.sha256(text.encode()).hexdigest(),sklearn=sklearn.__version__,runtime=tree_runtime_provenance(),
                sha256={str(f):sha(f) for f in [Path(__file__),Path('app/reference_gbr_subsample_driver.f90'),a.executable,Path('src/sklearn_classic_gradient_boosting.f90'),Path('src/sklearn_ensemble.f90'),Path('src/sklearn_tree.f90'),Path('src/sklearn_boosting_loss.f90')]})
    a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['records','runtime','sha256']})
    assert p.returncode==0 and len(lines)==len(valid)*width and report['failed_configurations']==0
if __name__=='__main__':main()
