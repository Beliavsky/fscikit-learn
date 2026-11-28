"""Weighted dense SVC metadata, decisions and independent dual certificates."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel,rbf_kernel,polynomial_kernel,sigmoid_kernel


def fixtures():
    for seed,kernel,classes,cw,weighted,shape in itertools.product([0,7,42],['linear','rbf','poly','sigmoid'],[2,3,4],range(3),range(2),range(3)):
        yield dict(seed=seed,kernel=kernel,classes=classes,cw=cw,weighted=weighted,shape=shape,n=25,p=4,c=.7,
            gamma=.4,coef0=.2,degree=2,tol=1e-8,max_iter=10000)
    for kernel,rule,cost,cap in itertools.product(['linear','rbf','poly','sigmoid'],range(2),[.05,1.3,30.],[1,3,31,67,10000]):
        yield dict(seed=42,kernel=kernel,classes=4,cw=1,weighted=1,shape=2,n=25,p=5,c=cost,
            gamma=-1.,gamma_rule=rule,coef0=.2,degree=3,tol=1e-8,max_iter=cap)
    for classes,shape in itertools.product([2,3,4],range(3)):
        yield dict(seed=0,kernel='linear',classes=classes,cw=0,weighted=0,shape=shape,n=12,p=3,c=1.,
            gamma=-1.,coef0=0.,degree=3,tol=1e-8,max_iter=10000,zero=True)
    # Larger free sets plus caps beyond the first shrinking sweep.
    for kernel,cap in itertools.product(['linear','rbf','poly','sigmoid'],[31,67,10000]):
        yield dict(seed=19,kernel=kernel,classes=3,cw=1,weighted=1,shape=2,n=80,p=8,c=30.,
            gamma=.4,coef0=.2,degree=2,tol=1e-8,max_iter=cap)


def arrays(c):
    rng=np.random.RandomState(c['seed']);labels=np.array([-4,7,21,33])[:c['classes']];y=np.resize(labels,c['n'])
    centers=rng.normal(size=(c['classes'],c['p']));x=rng.normal(scale=.7,size=(c['n'],c['p']))+centers[np.searchsorted(labels,y)]
    q=rng.normal(size=(9,c['p']));w=rng.uniform(.3,2,c['n']);w[::7]=0
    if not c['weighted']:w[:]=1
    if c.get('zero'):x[:]=0;q[:]=0
    return x,q,y,w,np.linspace(.5,2,c['classes'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--diagnostic',action='store_true');parser.add_argument('--no-shrinking',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),args.executable,root/'src/sklearn_svm.f90',root/'app/reference_svc_weighted_driver.f90',
        *[Path(sklearn.__file__).parent/'svm'/p for p in ['_base.py','_classes.py','src/libsvm/svm.cpp']]]
    hashes={str(p):sha(p) for p in files};cases=list(fixtures());data=[arrays(c) for c in cases];expected=[];iterations=[];source_status=[];unshrunk_iterations=[]
    payload=str(len(cases))+'\n'
    for c,(x,q,y,w,cw) in zip(cases,data):
        shape='ovo' if c['shape']==0 else 'ovr';gamma=('auto' if c.get('gamma_rule',0) else 'scale') if c['gamma']==-1 else c['gamma']
        model=SVC(shrinking=not args.no_shrinking,C=c['c'],kernel=c['kernel'],gamma=gamma,coef0=c['coef0'],degree=c['degree'],tol=c['tol'],
            max_iter=c['max_iter'],decision_function_shape=shape,break_ties=c['shape']==2,
            class_weight=None if c['cw']==0 else ('balanced' if c['cw']==1 else dict(zip(np.unique(y),cw))))
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');model.fit(x,y,sample_weight=w)
        source_status.append(model.fit_status_);iterations.append(model.n_iter_.tolist())
        unshrunk=SVC(**dict(model.get_params(),shrinking=False))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');unshrunk.fit(x,y,sample_weight=w)
        unshrunk_iterations.append(unshrunk.n_iter_.tolist())
        v=dict(classes=model.classes_,support=model.support_,n_support=model.n_support_,support_vectors=model.support_vectors_,
            dual_coef=model.dual_coef_,intercept=model.intercept_,coef=model.coef_ if c['kernel']=='linear' else np.array([]),
            gamma=[model._gamma],class_weight=model.class_weight_,train=model.decision_function(x),query=model.decision_function(q),
            predict=model.predict(q),status=[model.fit_status_],score=[model.score(x,y),model.score(x,y,sample_weight=w)])
        expected.append({k:np.asarray(a).ravel(order='F').tolist() for k,a in v.items()})
        header=[c['n'],c['p'],len(q),c['kernel'],c['c'],c['gamma'],c['coef0'],c['degree'],c['tol'],c['max_iter'],c['cw'],int(shape=='ovr'),int(c['shape']==2),c.get('gamma_rule',0)]
        payload+=' '.join(map(str,header))+'\n'
        for a in [x,q,y,w,cw]:payload+=' '.join(format(v,'.17g') for v in np.asarray(a).ravel(order='F'))+'\n'
    run=subprocess.run([str(args.executable),*(["--no-shrinking"] if args.no_shrinking else [])],input=payload,capture_output=True,text=True);assert run.returncode==0,(run.stdout[-1000:],run.stderr[-2000:])
    lines=iter(run.stdout.splitlines());actual=[];checks=[];index=-1
    for line in lines:
        parts=line.split()
        if not parts:continue
        if parts[0]=='BEGIN':index+=1;actual.append({})
        elif parts[0]=='END':assert actual[index].keys()==expected[index].keys()|{'native_iterations','native_kkt','native_support_original'}
        else:
            field=parts[0];size=int(parts[1]);values=np.fromstring(next(lines),sep=' ');assert len(values)==size;actual[index][field]=values.tolist()
            if not field.startswith('native_'):
                want=np.asarray(expected[index][field]);exact=field in ['classes','support','n_support','predict','status']
                passed=values.shape==want.shape and (np.array_equal(values,want) if exact else np.allclose(values,want,atol=3e-6,rtol=3e-6))
                checks.append(dict(case=index,field=field,passed=bool(passed),max_abs_error=float(np.max(np.abs(values-want),initial=0)) if values.shape==want.shape else None))
    assert len(actual)==len(cases);certificates=[]
    for case_id,(c,a,v) in enumerate(zip(cases,data,actual)):
        x,q,y,w,cw=a;k=c['classes'];support=np.asarray(v['native_support_original'],int);counts=np.asarray(v['n_support'],int)
        source_support=np.flatnonzero(w>0)[np.asarray(expected[case_id]['support'],int)]
        source_dual=np.asarray(expected[case_id]['dual_coef']).reshape((k-1,len(source_support)),order='F')
        dual=np.asarray(v['dual_coef']).reshape((k-1,len(support)),order='F');intercepts=np.asarray(v['intercept']);gamma=v['gamma'][0]
        weights=np.ones(k) if c['cw']==0 else (np.array([len(y)/(k*np.sum(y==label)) for label in np.unique(y)]) if c['cw']==1 else cw)
        cost=w*(c['c']*weights[np.searchsorted(np.unique(y),y)])
        if c['kernel']=='linear':gram=linear_kernel(x)
        elif c['kernel']=='rbf':gram=rbf_kernel(x,gamma=gamma)
        elif c['kernel']=='poly':gram=polynomial_kernel(x,gamma=gamma,coef0=c['coef0'],degree=c['degree'])
        else:gram=sigmoid_kernel(x,gamma=gamma,coef0=c['coef0'])
        pair=0;labels=np.unique(y);records=[]
        for first in range(k-1):
            for last in range(first+1,k):
                beta=np.zeros(len(y));lower=y[support]==labels[first];upper=y[support]==labels[last]
                beta[support[lower]]=dual[last-1,lower];beta[support[upper]]=dual[first,upper]
                if k>2:beta=-beta
                signed=np.where(y==labels[last],1.,-1.);mask=((y==labels[first])|(y==labels[last]))&(cost>0)
                alpha=beta*signed;margin=gram@beta+(intercepts[pair] if k==2 else -intercepts[pair])
                assert np.all(alpha>=-1e-10) and np.all(alpha<=cost+1e-10)
                assert abs(np.sum(beta))<1e-8
                pg=np.zeros(len(y));g=signed*margin-1
                pg[mask&(alpha<cost-1e-10)]=np.minimum(g[mask&(alpha<cost-1e-10)],0)
                pg[mask&(alpha>1e-10)]=np.maximum(np.abs(pg[mask&(alpha>1e-10)]),np.maximum(g[mask&(alpha>1e-10)],0))
                violation=float(np.max(np.abs(pg),initial=0));np.testing.assert_allclose(violation,v['native_kkt'][pair],atol=1e-8,rtol=1e-6)
                penalty=.5*beta@gram@beta;primal=penalty+np.sum(cost[mask]*np.maximum(0,1-signed[mask]*margin[mask]));dual_objective=np.sum(alpha)-penalty
                source_beta=np.zeros(len(y));source_lower=y[source_support]==labels[first];source_upper=y[source_support]==labels[last]
                source_beta[source_support[source_lower]]=source_dual[last-1,source_lower]
                source_beta[source_support[source_upper]]=source_dual[first,source_upper]
                if k>2:source_beta=-source_beta
                source_alpha=source_beta*signed;source_bias=expected[case_id]['intercept'][pair]*(1 if k==2 else -1)
                source_gradient=signed*(gram@source_beta+source_bias)-1
                source_violation=max(float(np.max(-source_gradient[mask&(source_alpha<cost-1e-10)],initial=0)),
                    float(np.max(source_gradient[mask&(source_alpha>1e-10)],initial=0)))
                records.append(dict(kkt=violation,source_kkt=source_violation,gap=float(primal-dual_objective),passed=True,
                    small_kkt=bool(violation<2e-6),source_small_kkt=bool(source_violation<2e-6),capped=bool(source_status[case_id])));pair+=1
        certificates.append(records)
    assert hashes=={p:sha(p) for p in hashes};gaps=[c for c in checks if not c['passed']]
    report=dict(cases=cases,checks=checks,gaps=gaps,certificates=certificates,actual=actual,expected=expected,
        shrinking=not args.no_shrinking,iteration_disagreements=[i for i,(a,b) in enumerate(zip(actual,iterations)) if a["native_iterations"]!=b],source_iterations=iterations,source_status=source_status,source_nonshrinking_iterations=unshrunk_iterations,hashes=hashes,strict_parity=not gaps,sklearn=sklearn.__version__)
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(len(cases),'fits;',len(checks),'groups;',len(gaps),'gaps;',sum(map(len,certificates)),'certificates')
    if gaps:print(gaps[:12])
    raise SystemExit(bool(gaps) and not args.diagnostic)

if __name__=='__main__':main()
