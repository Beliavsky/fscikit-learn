"""Shared multioutput classification forests, ragged probabilities and OOB."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, subprocess, warnings
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn_tree_runtime import tree_runtime_provenance

def fixtures():
    for mode, criterion, seed, q, pattern, setting, cw, weighted in itertools.product(
            [0, 1], ['gini', 'entropy', 'log_loss'], [0, 7], [1, 2, 3], range(2), range(3), range(4), [False, True]):
        n=19; p=4; rng=np.random.RandomState(831+q)
        x=rng.normal(size=(n,p))
        if pattern==0:
            y=np.column_stack([rng.randint(k%3+1 if q>1 else 3,size=n)*17-9+k for k in range(q)])
        else:
            y=rng.randint(2,size=(n,q))
        w=None if not weighted else np.resize([1.,0.,.5,2.,3.],n)
        yield dict(mode=mode,criterion=criterion,seed=seed,q=q,n=n,p=p,nt=1 if setting==1 else 5,
                   boot=int(setting!=0),ns=7 if setting==2 else 0,oob=int(setting==2 and (pattern==1 or q==1)),
                   weighted=weighted,depth=4,budget=0 if setting==0 else 5,nf=0,leaf=1,alpha=.01,
                   decrease=0.,cw=cw,pattern=pattern),x,y,w
    for mode,criterion,q,n,pattern in itertools.product([0,1],['gini','entropy','log_loss'],[1,2,4],[1,5,11],range(3)):
        t=np.arange(n);x=np.column_stack([t,t%3]).astype(float)
        y=np.column_stack([t%2 for k in range(q)])
        w=np.resize([1.,.5,2.],n)
        if pattern==0:y[:]=np.arange(q)*17-9
        if pattern==2:
            y=np.column_stack([t%(k%3+1)*17-9+k for k in range(q)])
            w[0]=-.1
            if n>1:y[-1,-1]=99;w[-1]=0
        yield dict(mode=mode,criterion=criterion,seed=42,q=q,n=n,p=2,nt=3,boot=1,
                   ns=1 if pattern==0 else 0,oob=int(pattern!=2 or q==1),weighted=True,depth=3,
                   budget=0,nf=0,leaf=1,alpha=0.,decrease=0.,cw=2,pattern=pattern+2),x,y,w
    for mode,criterion,q in itertools.product([0,1],['gini','entropy','log_loss'],[1,2,4]):
        n=7;t=np.arange(n);x=np.column_stack([t,t%2]).astype(float)
        y=np.column_stack([t%(k+2) for k in range(q)])
        yield dict(mode=mode,criterion=criterion,seed=7,q=q,n=n,p=2,nt=3,boot=1,ns=0,oob=0,
                   weighted=True,depth=3,budget=0,nf=0,leaf=1,alpha=0.,decrease=0.,cw=0,pattern=5),x,y,np.zeros(n)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    a=ap.parse_args();records=[];payloads=[]
    for c,x,y,w in fixtures():
        cls=RandomForestClassifier if c['mode']==0 else ExtraTreesClassifier
        maps=[{v:i+1.25 for i,v in enumerate(np.unique(y[:,k]))} for k in range(c['q'])]
        cw=[None,'balanced','balanced_subsample',maps[0] if c['q']==1 else maps][c['cw']]
        model=cls(n_estimators=c['nt'],criterion=c['criterion'],random_state=c['seed'],bootstrap=bool(c['boot']),
                  max_samples=c['ns'] or None,oob_score=bool(c['oob']),max_depth=c['depth'],max_leaf_nodes=c['budget'] or None,
                  max_features=c['nf'] or 'sqrt',min_samples_leaf=c['leaf'],ccp_alpha=c['alpha'],
                  min_impurity_decrease=c['decrease'],class_weight=cw,n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');model.fit(x,y,sample_weight=w)
            probabilities=model.predict_proba(x);logs=model.predict_log_proba(x)
            try:score=[model.score(x,y,sample_weight=w)];score_error=None
            except (ValueError,ZeroDivisionError) as error:score=[];score_error=str(error)
        classes=[model.classes_] if c['q']==1 else model.classes_
        if c['q']==1:probabilities=[probabilities];logs=[logs]
        counts=[len(v) for v in classes];trees=[e.tree_ for e in model.estimators_];bags=model.estimators_samples_
        coverage=np.sum([np.bincount(b,minlength=c['n'])==0 for b in bags],axis=0)
        oob=model.oob_decision_function_ if c['oob'] else None
        expected=dict(metadata=[model.n_features_in_,model.n_outputs_,len(trees)],class_counts=counts,
                      classes=np.concatenate(classes).tolist(),prediction=model.predict(x).ravel().tolist(),
                      probabilities=np.concatenate([v.ravel() for v in probabilities]).tolist(),
                      log_probabilities=np.concatenate([v.ravel() for v in logs]).tolist(),
                      leaves=(model.apply(x)+1).ravel().tolist(),importance=model.feature_importances_.tolist(),
                      seeds=[e.random_state for e in model.estimators_],samples=(np.stack(bags,axis=1)+1).ravel().tolist(),
                      oob_probabilities=(oob.ravel() if c['q']==1 else np.concatenate([oob[:,:,k].ravel() for k in range(c['q'])])).tolist() if c['oob'] else [],
                      oob_score=[model.oob_score_] if c['oob'] else [],oob_counts=coverage.tolist() if c['oob'] else [],
                      tree_metadata=np.array([[t.node_count,e.get_depth(),e.get_n_leaves(),e.n_features_in_,e.n_outputs_] for e,t in zip(model.estimators_,trees)],int).ravel().tolist(),
                      tree_class_counts=np.tile(counts,len(trees)).tolist(),
                      tree_classes=np.concatenate([np.concatenate([e.classes_] if c['q']==1 else e.classes_) for e in model.estimators_]).tolist(),
                      values=np.concatenate([np.concatenate([t.value[:,k,:counts[k]] for k in range(c['q'])],axis=1).ravel() for t in trees]).tolist())
        expected.update({k:np.concatenate([getattr(t,attr) for t in trees]).tolist() for k,attr in [('impurity','impurity'),('mass','weighted_n_node_samples'),('count','n_node_samples')]})
        expected.update(left=np.concatenate([t.children_left+1 for t in trees]).tolist(),right=np.concatenate([t.children_right+1 for t in trees]).tolist(),
                        feature=np.concatenate([np.where(t.children_left<0,0,t.feature+1) for t in trees]).tolist(),
                        threshold=np.concatenate([np.where(t.children_left<0,0,t.threshold) for t in trees]).tolist(),
                        tree_importance=np.concatenate([e.feature_importances_ for e in model.estimators_]).tolist(),score=score)
        records.append(dict(case=c,source=expected,score_error=score_error,differences=[]))
        text=' '.join(str(int(c[k])) if k=='weighted' else str(c[k]) for k in ['n','p','q','mode','criterion','seed','nt','boot','ns','oob','weighted','depth','budget','nf','leaf','alpha','decrease','cw'])+f' {int(score_error is None)}\n'
        for arr in [x,y]:text+='\n'.join(' '.join(map(str,row)) for row in arr)+'\n'
        text+=' '.join(map(str,np.ones(len(x)) if w is None else w))+'\n';payloads.append(text)
    a.report.parent.mkdir(parents=True,exist_ok=True);payload=''.join(payloads);a.report.with_suffix('.in').write_text(payload)
    native=subprocess.run([str(a.executable)],input=payload,text=True,capture_output=True)
    a.report.with_suffix('.stdout').write_text(native.stdout);a.report.with_suffix('.stderr').write_text(native.stderr)
    lines=native.stdout.splitlines();fields=list(records[0]['source']);assert len(fields)==26
    integer={'metadata','class_counts','classes','prediction','leaves','seeds','samples','oob_counts','tree_metadata','tree_class_counts','tree_classes','count','left','right','feature'}
    for i,r in enumerate(records):
        chunk=lines[26*i:26*i+26];r['native']={}
        if len(chunk)!=26:r['differences'].append('missing native output');continue
        for key,line in zip(fields,chunk):
            got=np.fromstring(line,sep=' ');r['native'][key]=got.tolist()
            try:
                if key in integer:np.testing.assert_array_equal(got,r['source'][key])
                else:np.testing.assert_allclose(got,r['source'][key],rtol=2e-10,atol=2e-12,equal_nan=True)
            except AssertionError:r['differences'].append(key)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report=dict(configurations=len(records),records=records,failed_configurations=sum(bool(r['differences']) for r in records),
                native_returncode=native.returncode,input_sha256=hashlib.sha256(payload.encode()).hexdigest(),sklearn=sklearn.__version__,runtime=tree_runtime_provenance(),
                sha256={str(p):sha(p) for p in [Path(__file__),a.executable,Path('src/sklearn_ensemble.f90')]})
    a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['records','sha256','runtime']})
    assert not native.returncode and not report['failed_configurations']

if __name__=='__main__':main()
