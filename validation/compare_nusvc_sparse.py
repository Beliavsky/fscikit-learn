"""CSR NuSVC LIBSVM outputs, raw nodes and independent dual certificates."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, subprocess, warnings
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.svm import NuSVC
import scipy
from compare_kernel_svm_sparse import matrix,csr_payload
from compare_svc_sparse import raw_gram


def fixtures():
    for seed,kernel,classes,nu,weighted,layout,shrinking in itertools.product(
            [7,42],['linear','rbf','poly','sigmoid'],[2,3,4],[.2,.55],[0,1],range(4),[False,True]):
        yield dict(seed=seed,kernel=kernel,classes=classes,nu=nu,weighted=weighted,layout=layout,
                   shrinking=shrinking,shape=(seed+classes+int(nu*10))%3,n=25,p=7,max_iter=10000,gamma_rule=0,gamma=-1.)
    for seed,kernel,p,rule,shrinking in itertools.product([7,42],['linear','rbf','poly','sigmoid'],[17,43],[1,2],[False,True]):
        yield dict(seed=seed,kernel=kernel,classes=3,nu=.55,weighted=1,layout=2,shrinking=shrinking,
                   shape=2,n=25,p=p,max_iter=10000,gamma_rule=rule,gamma=-1. if rule==1 else .4)
    for kernel,layout,nu,cap,shrinking in itertools.product(['linear','rbf','poly','sigmoid'],[0,2],[.2,.55],[1,31],[False,True]):
        yield dict(seed=19,kernel=kernel,classes=3,nu=nu,weighted=1,layout=layout,shrinking=shrinking,
                   shape=2,n=80,p=8,max_iter=cap,gamma_rule=2,gamma=.4)


def arrays(c):
    rng=np.random.RandomState(c['seed']);n=c['n'];p=c['p'];k=c['classes'];labels=np.array([-5,7,21,40])[:k]
    y=np.resize(labels,n);centers=rng.normal(size=(k,p));x=rng.normal(scale=.8,size=(n,p))+centers[np.searchsorted(labels,y)]
    query=rng.normal(size=(7,p));weight=rng.uniform(.8,1.2,n) if c['weighted'] else np.ones(n)
    if c['weighted']:weight[0]=0
    x[rng.uniform(size=x.shape)<.5]=0;query[rng.uniform(size=query.shape)<.4]=0;x[0]=0
    return matrix(x,c['layout']),matrix(query,c['layout']),y,weight


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--diagnostic',action='store_true');parser.add_argument('--limit',type=int)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),args.executable,root/'src/sklearn_nu_classification.f90',root/'app/reference_nusvc_sparse_driver.f90',root/'src/sklearn_libsvm_sparse.f90',
           root/'validation/compare_kernel_svm_sparse.py',root/'validation/compare_svc_sparse.py',
           *[Path(sklearn.__file__).parent/'svm'/p for p in ['_base.py','_classes.py','src/libsvm/svm.cpp']]]
    hashes={str(p):sha(p) for p in files};cases=list(fixtures())[:args.limit];expected=[];source_iterations=[]
    data=[arrays(c) for c in cases];payload=str(len(cases))+'\n'
    for c,(x,q,y,w) in zip(cases,data):
        kernel=c['kernel'];positive=np.flatnonzero(w>0)
        gamma='scale' if c['gamma_rule']==0 else ('auto' if c['gamma_rule']==1 else .4)
        model=NuSVC(kernel=kernel,nu=c['nu'],gamma=gamma,coef0=.2,degree=2,tol=1e-8,max_iter=c['max_iter'],
                    shrinking=c['shrinking'],decision_function_shape='ovo' if c['shape']==0 else 'ovr',break_ties=c['shape']==2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');model.fit(x.copy(),y,sample_weight=w)
        v=dict(classes=model.classes_,support=positive[model.support_],n_support=model.n_support_,
               sv_indptr=model.support_vectors_.indptr+1,sv_indices=model.support_vectors_.indices+1,sv_data=model.support_vectors_.data,
               dual_coef=model.dual_coef_.toarray(),intercept=model.intercept_,coef=model.coef_.toarray() if kernel=='linear' else [],
               status=[model.fit_status_],gamma=[model._gamma],train=model.decision_function(x.copy()),
               query=model.decision_function(q.copy()),predict=model.predict(q.copy()),
               dense_query=model.decision_function(q.toarray()),dense_predict=model.predict(q.toarray()),
               score=[model.score(x.copy(),y),model.score(x.copy(),y,sample_weight=w)])
        expected.append({key:np.asarray(value).ravel(order='F').tolist() for key,value in v.items()})
        source_iterations.append(model.n_iter_.tolist())
        header=[len(y),x.shape[1],q.shape[0],kernel,c['nu'],c['gamma'],.2,2,1e-8,c['max_iter'],0,int(c['shape']!=0),int(c['shape']==2),int(c['gamma_rule']==1),int(c['shrinking'])]
        payload+=' '.join(map(str,header))+'\n'+csr_payload(x)+csr_payload(q)
        for value in [y,w,np.ones(c['classes'])]:
            payload+=' '.join(format(v,'.17g') for v in np.asarray(value).ravel(order='F'))+'\n'
    run=subprocess.run([str(args.executable)],input=payload,capture_output=True,text=True)
    assert run.returncode==0,(run.stdout[-1000:],run.stderr[-2000:])
    actual=[];lines=iter(run.stdout.splitlines());checks=[];index=-1
    native={'native_support_original','native_iterations','native_kkt','native_alpha','native_scales','native_storage'}
    for line in lines:
        parts=line.split()
        if not parts:continue
        if parts[0]=='BEGIN':index+=1;actual.append({})
        elif parts[0]=='END':assert actual[index].keys()==expected[index].keys()|native
        else:
            key=parts[0];values=np.fromstring(next(lines),sep=' ');assert len(values)==int(parts[1]);actual[index][key]=values.tolist()
            if key in native:continue
            want=np.asarray(expected[index][key]);exact=key in ['classes','support','n_support','predict','dense_predict','status','sv_indptr','sv_indices']
            passed=values.shape==want.shape and (np.array_equal(values,want) if exact else np.allclose(values,want,atol=3e-6,rtol=3e-6))
            checks.append(dict(case=index,field=key,passed=bool(passed),max_abs_error=float(np.max(np.abs(values-want),initial=0)) if values.shape==want.shape else None))
    assert len(actual)==len(cases);certificates=[]
    for case_id,(c,values,v) in enumerate(zip(cases,data,actual)):
        x,q,y,w=values;k=c['classes'];gram=raw_gram(x,c['kernel'],v['gamma'][0],.2,2)
        assert v['native_storage']==[0,0]
        alpha=np.asarray(v['native_alpha']).reshape((len(y),-1),order='F')
        labels=np.unique(y);pair=0;records=[];support=np.asarray(v['support'],int)
        dual=np.asarray(v['dual_coef']).reshape((k-1,len(support)),order='F')
        assert np.all(alpha>=-1e-10) and np.all(alpha<=w[:,None]+1e-10)
        for a in range(k-1):
            for b in range(a+1,k):
                members=(y==labels[a])|(y==labels[b]);mass=c['nu']*sum(w[members])/2
                for label in [labels[a],labels[b]]:np.testing.assert_allclose(sum(alpha[y==label,pair]),mass,atol=1e-10,rtol=1e-10)
                assert np.all(alpha[~members,pair]==0)
                signs=np.where(y==labels[a],1.,-1.);unscaled=alpha[:,pair]*signs
                gradient=signs*(gram@unscaled);gap=0.
                for label in [labels[a],labels[b]]:
                    above=(y==label)&(w>0)&(alpha[:,pair]>0);below=(y==label)&(w>0)&(alpha[:,pair]<w)
                    if np.any(above) and np.any(below):gap=max(gap,float(gradient[above].max()-gradient[below].min()))
                np.testing.assert_allclose(gap,v['native_kkt'][pair],atol=1e-8,rtol=1e-6)
                cache_error=float(np.max(np.abs(gram-gram.astype(np.float32).astype(float))@alpha[:,pair]))
                if v['native_iterations'][pair]<c['max_iter']:assert gap<=1e-8+2*cache_error+1e-8,(case_id,pair,gap,cache_error)
                public=np.zeros(len(y))
                lower=y[support]==labels[a];upper=y[support]==labels[b]
                public[support[lower]]=dual[b-1,lower];public[support[upper]]=dual[a,upper]
                np.testing.assert_allclose(public,unscaled/v['native_scales'][pair]*(-1 if k==2 else 1),atol=1e-10,rtol=1e-10)
                records.append(dict(passed=True,kkt=gap,capped=bool(v['status'][0])));pair+=1
        certificates.append(records)
    gaps=[check for check in checks if not check['passed']]
    assert hashes=={p:sha(p) for p in hashes}
    report=dict(cases=cases,checks=checks,gaps=gaps,actual=actual,expected=expected,certificates=certificates,
                source_iterations=source_iterations,iteration_disagreements=[i for i,(v,s) in enumerate(zip(actual,source_iterations)) if v['native_iterations']!=s],
                hashes=hashes,strict_parity=not gaps,sklearn=sklearn.__version__,scipy=scipy.__version__)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(len(cases),'fits;',len(checks),'groups;',len(gaps),'gaps;',sum(map(len,certificates)),'certificates')
    if gaps:print(gaps[:15])
    raise SystemExit(bool(gaps) and not args.diagnostic)


if __name__=='__main__':main()
