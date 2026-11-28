"""NuSVC LIBSVM outputs and independent two-group feasibility/residual replays."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, subprocess, warnings
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.svm import NuSVC
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel


def fixtures():
    kinds=['linear','rbf','poly','sigmoid','pre_linear','pre_rbf','pre_poly','pre_sigmoid','pre_indefinite','pre_asymmetric','pre_near_asymmetric','pre_small_asymmetric']
    for seed,kind,classes,nu,weighted,cap,shrinking in itertools.product(
            [7,42],kinds,[2,3,4],[.2,.55],[0,1],[1,31,10000],[False,True]):
        yield dict(seed=seed,kind=kind,classes=classes,nu=nu,weighted=weighted,max_iter=cap,
                   shrinking=shrinking,shape=(seed+classes+int(nu*10))%3)


def arrays(c):
    rng=np.random.RandomState(c['seed']);n=25;p=4;k=c['classes'];labels=np.array([-5,7,21,40])[:k]
    y=np.resize(labels,n);centers=rng.normal(size=(k,p));x=rng.normal(scale=.8,size=(n,p))+centers[np.searchsorted(labels,y)]
    query=rng.normal(size=(7,p));weight=rng.uniform(.8,1.2,n) if c['weighted'] else np.ones(n)
    if c['weighted']:weight[0]=0
    base=c['kind'].removeprefix('pre_');args=dict(gamma=.4,coef0=.2,degree=2)
    if base=='linear':gram=linear_kernel(x);qgram=linear_kernel(query,x)
    elif base=='rbf':gram=rbf_kernel(x,gamma=.4);qgram=rbf_kernel(query,x,gamma=.4)
    elif base=='poly':gram=polynomial_kernel(x,**args);qgram=polynomial_kernel(query,x,**args)
    elif base=='sigmoid':gram=sigmoid_kernel(x,gamma=.4,coef0=.2);qgram=sigmoid_kernel(query,x,gamma=.4,coef0=.2)
    else:
        raw=rng.normal(size=(n,n));qgram=rng.normal(size=(len(query),n))
        if base=='indefinite':gram=(raw+raw.T)/2
        elif base=='near_asymmetric':gram=(raw+raw.T)/2+1e-8*(raw-raw.T)
        elif base=='small_asymmetric':gram=raw*1e-12;qgram*=1e-12
        else:gram=raw
    precomputed=c['kind'].startswith('pre_')
    return (gram if precomputed else x),(qgram if precomputed else query),y,weight,gram


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--diagnostic',action='store_true');parser.add_argument('--limit',type=int)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),args.executable,root/'src/sklearn_nu_classification.f90',root/'app/reference_nusvc_libsvm_driver.f90',
           *[Path(sklearn.__file__).parent/'svm'/p for p in ['_base.py','_classes.py','src/libsvm/svm.cpp']]]
    hashes={str(p):sha(p) for p in files};cases=list(fixtures())[:args.limit];expected=[];source_iterations=[]
    data=[arrays(c) for c in cases];payload=str(len(cases))+'\n'
    for c,(x,q,y,w,gram) in zip(cases,data):
        pre=c['kind'].startswith('pre_');kernel='precomputed' if pre else c['kind'];positive=np.flatnonzero(w>0)
        fit_x=x[np.ix_(positive,positive)] if pre else x
        fit_y=y[positive] if pre else y;fit_w=w[positive] if pre else w
        train=x[:,positive] if pre else x;query=q[:,positive] if pre else q
        model=NuSVC(kernel=kernel,nu=c['nu'],gamma=.4,coef0=.2,degree=2,tol=1e-8,max_iter=c['max_iter'],
                    shrinking=c['shrinking'],decision_function_shape='ovo' if c['shape']==0 else 'ovr',break_ties=c['shape']==2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');model.fit(fit_x,fit_y,sample_weight=fit_w)
        v=dict(classes=model.classes_,support=positive[model.support_],n_support=model.n_support_,
               support_vectors=model.support_vectors_,dual_coef=model.dual_coef_,intercept=model.intercept_,
               coef=model.coef_ if kernel=='linear' else [],status=[model.fit_status_],gamma=[model._gamma],
               train=model.decision_function(train),query=model.decision_function(query),predict=model.predict(query),
               score=[model.score(train,y),model.score(train,y,sample_weight=w)])
        expected.append({key:np.asarray(value).ravel(order='F').tolist() for key,value in v.items()})
        source_iterations.append(model.n_iter_.tolist())
        header=[len(y),x.shape[1],len(q),kernel,c['nu'],.4,.2,2,1e-8,c['max_iter'],0,int(c['shape']!=0),int(c['shape']==2),0,int(c['shrinking'])]
        payload+=' '.join(map(str,header))+'\n'
        for value in [x,q,y,w,np.ones(c['classes'])]:
            payload+=' '.join(format(v,'.17g') for v in np.asarray(value).ravel(order='F'))+'\n'
    run=subprocess.run([str(args.executable)],input=payload,capture_output=True,text=True)
    assert run.returncode==0,(run.stdout[-1000:],run.stderr[-2000:])
    actual=[];lines=iter(run.stdout.splitlines());checks=[];index=-1
    native={'native_support_original','native_iterations','native_kkt','native_alpha','native_scales'}
    for line in lines:
        parts=line.split()
        if not parts:continue
        if parts[0]=='BEGIN':index+=1;actual.append({})
        elif parts[0]=='END':assert actual[index].keys()==expected[index].keys()|native
        else:
            key=parts[0];values=np.fromstring(next(lines),sep=' ');assert len(values)==int(parts[1]);actual[index][key]=values.tolist()
            if key in native:continue
            want=np.asarray(expected[index][key]);exact=key in ['classes','support','n_support','predict','status']
            passed=values.shape==want.shape and (np.array_equal(values,want) if exact else np.allclose(values,want,atol=3e-6,rtol=3e-6))
            checks.append(dict(case=index,field=key,passed=bool(passed),max_abs_error=float(np.max(np.abs(values-want),initial=0)) if values.shape==want.shape else None))
    assert len(actual)==len(cases);certificates=[]
    for case_id,(c,values,v) in enumerate(zip(cases,data,actual)):
        x,q,y,w,gram=values;k=c['classes'];alpha=np.asarray(v['native_alpha']).reshape((len(y),-1),order='F')
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
                if not c['kind'].endswith('asymmetric') and v['native_iterations'][pair]<c['max_iter']:assert gap<=1e-8+2*cache_error+1e-8,(case_id,pair,gap,cache_error)
                public=np.zeros(len(y))
                lower=y[support]==labels[a];upper=y[support]==labels[b]
                public[support[lower]]=dual[b-1,lower];public[support[upper]]=dual[a,upper]
                np.testing.assert_allclose(public,unscaled/v['native_scales'][pair]*(-1 if k==2 else 1),atol=1e-10,rtol=1e-10)
                records.append(dict(passed=True,kkt=gap,capped=bool(v['status'][0]),
                                    symmetric_qp_bound_applicable=not c['kind'].endswith('asymmetric'),
                                    cache_rounding_error=cache_error));pair+=1
        certificates.append(records)
    gaps=[check for check in checks if not check['passed']]
    assert hashes=={p:sha(p) for p in hashes}
    report=dict(cases=cases,checks=checks,gaps=gaps,actual=actual,expected=expected,certificates=certificates,
                source_iterations=source_iterations,iteration_disagreements=[i for i,(v,s) in enumerate(zip(actual,source_iterations)) if v['native_iterations']!=s],
                hashes=hashes,strict_parity=not gaps,sklearn=sklearn.__version__)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(len(cases),'fits;',len(checks),'groups;',len(gaps),'gaps;',sum(map(len,certificates)),'certificates')
    if gaps:print(gaps[:15])
    raise SystemExit(bool(gaps) and not args.diagnostic)


if __name__=='__main__':main()
