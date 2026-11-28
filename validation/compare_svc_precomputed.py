"""Precomputed C-SVC matrices, support indexing and Source query semantics."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel,rbf_kernel,polynomial_kernel,sigmoid_kernel


def fixtures():
    for seed,base,classes,cw,weighted,shape,cap in itertools.product([0,7,42],['linear','rbf','poly','sigmoid','indefinite','asymmetric'],
            [2,3,4],range(3),range(2),[0,2],[1,3,10000]):
        yield dict(seed=seed,kernel='precomputed',base=base,classes=classes,cw=cw,weighted=weighted,shape=shape,n=13,p=3,c=.7,
            gamma=.4,coef0=.2,degree=2,tol=1e-8,max_iter=cap)


def arrays(c):
    rng=np.random.RandomState(c['seed']);labels=np.array([-4,7,21,33])[:c['classes']];y=np.resize(labels,c['n'])
    centers=rng.normal(size=(c['classes'],c['p']));x=rng.normal(scale=.7,size=(c['n'],c['p']))+centers[np.searchsorted(labels,y)]
    q=rng.normal(size=(9,c['p']));w=rng.uniform(.3,2,c['n']);w[::7]=0
    if not c['weighted']:w[:]=1
    if c.get('zero'):x[:]=0;q[:]=0
    if c['base']=='linear':train=linear_kernel(x);query=linear_kernel(q,x)
    elif c['base']=='rbf':train=rbf_kernel(x,gamma=.4);query=rbf_kernel(q,x,gamma=.4)
    elif c['base']=='poly':train=polynomial_kernel(x,gamma=.4,coef0=.2,degree=2);query=polynomial_kernel(q,x,gamma=.4,coef0=.2,degree=2)
    elif c['base']=='sigmoid':train=sigmoid_kernel(x,gamma=.4,coef0=.2);query=sigmoid_kernel(q,x,gamma=.4,coef0=.2)
    else:
        train=rng.normal(size=(c['n'],c['n']));query=rng.normal(size=(9,c['n']))
        if c['base']=='indefinite':train=(train+train.T)/2
    return train,query,y,w,np.linspace(.5,2,c['classes'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--diagnostic',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),args.executable,root/'src/sklearn_svm.f90',root/'app/reference_svc_precomputed_driver.f90',
        *[Path(sklearn.__file__).parent/'svm'/p for p in ['_base.py','_classes.py','src/libsvm/svm.cpp']]]
    hashes={str(p):sha(p) for p in files};cases=list(fixtures());data=[arrays(c) for c in cases];expected=[];iterations=[];source_status=[];unshrunk_iterations=[]
    payload=str(len(cases))+'\n'
    for c,(x,q,y,w,cw) in zip(cases,data):
        shape='ovo' if c['shape']==0 else 'ovr';gamma=('auto' if c.get('gamma_rule',0) else 'scale') if c['gamma']==-1 else c['gamma']
        model=SVC(C=c['c'],kernel=c['kernel'],gamma=gamma,coef0=c['coef0'],degree=c['degree'],tol=c['tol'],
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
        header=[c['n'],x.shape[1],len(q),c['kernel'],c['c'],c['gamma'],c['coef0'],c['degree'],c['tol'],c['max_iter'],c['cw'],int(shape=='ovr'),int(c['shape']==2),c.get('gamma_rule',0)]
        payload+=' '.join(map(str,header))+'\n'
        for a in [x,q,y,w,cw]:payload+=' '.join(format(v,'.17g') for v in np.asarray(a).ravel(order='F'))+'\n'
    run=subprocess.run([str(args.executable)],input=payload,capture_output=True,text=True);assert run.returncode==0,(run.stdout[-1000:],run.stderr[-2000:])
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
    for c,(x,q,y,w,cw),v in zip(cases,data,actual):
        assert v['support_vectors']==[] and v['coef']==[] and v['gamma']==[0.]
        k=c['classes'];supports=np.asarray(v['support'],int);original=np.asarray(v['native_support_original'],int)
        np.testing.assert_array_equal(np.flatnonzero(w>0)[supports],original)
        dual=np.asarray(v['dual_coef']).reshape((k-1,len(supports)),order='F');pair=0;records=[]
        weights=np.ones(k) if c['cw']==0 else (np.array([len(y)/(k*np.sum(y==label)) for label in np.unique(y)]) if c['cw']==1 else cw)
        cost=w*(c['c']*weights[np.searchsorted(np.unique(y),y)])
        for first in range(k-1):
            for last in range(first+1,k):
                beta=np.zeros(len(y));lower=y[original]==np.unique(y)[first];upper=y[original]==np.unique(y)[last]
                beta[original[lower]]=dual[last-1,lower];beta[original[upper]]=dual[first,upper]
                if k>2:beta=-beta
                signed=np.where(y==np.unique(y)[last],1.,-1.);alpha=beta*signed
                assert np.all(alpha>=-1e-10) and np.all(alpha<=cost+1e-10) and abs(np.sum(beta))<1e-8
                # Kernel entries need not be symmetric: replay the reported original-row margin residual.
                margin=x@beta+v['intercept'][pair]*(1 if k==2 else -1);g=signed*margin-1
                mask=(y==np.unique(y)[first])|(y==np.unique(y)[last]);mask&=cost>0
                violation=max(float(np.max(-g[mask&(alpha<cost-1e-10)],initial=0)),float(np.max(g[mask&(alpha>1e-10)],initial=0)))
                np.testing.assert_allclose(v['native_kkt'][pair],violation,atol=1e-8,rtol=1e-6)
                records.append(dict(passed=True,kkt=violation));pair+=1
        certificates.append(records)
    assert hashes=={p:sha(p) for p in hashes};gaps=[c for c in checks if not c['passed']]
    allowed_fields={'support','n_support','dual_coef','intercept','train','query','predict','score'}
    retained=[g for g in gaps if cases[g['case']]['base']=='asymmetric' and cases[g['case']]['max_iter']==10000
        and source_status[g['case']]==1 and actual[g['case']]['status']==[1.] and g['field'] in allowed_fields]
    unexpected=[g for g in gaps if g not in retained]
    report=dict(cases=cases,checks=checks,gaps=gaps,certificates=certificates,actual=actual,expected=expected,
        source_iterations=iterations,source_status=source_status,source_nonshrinking_iterations=unshrunk_iterations,hashes=hashes,strict_parity=not gaps,supported_kernel_parity=not unexpected,retained_asymmetric_cap_gaps=retained,unexpected_gaps=unexpected,sklearn=sklearn.__version__)
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(len(cases),'fits;',len(checks),'groups;',len(gaps),'gaps;',sum(map(len,certificates)),'certificates')
    if gaps:print(gaps[:12])
    raise SystemExit(bool(unexpected) or (bool(gaps) and not args.diagnostic))

if __name__=='__main__':main()
