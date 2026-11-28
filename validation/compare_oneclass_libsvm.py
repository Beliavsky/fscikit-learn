"""Full OneClassSVM Source outputs and independent weighted-dual certificates."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn,scipy
from sklearn.svm import OneClassSVM
from sklearn.metrics.pairwise import linear_kernel,rbf_kernel,polynomial_kernel,sigmoid_kernel
from compare_kernel_svm_sparse import matrix,csr_payload
from compare_svc_sparse import raw_gram


def fixtures():
    for fmt,kernel,seed,nu,weighted,shrinking,cap in itertools.product([0,1,2],['linear','rbf','poly','sigmoid'],[7,42],[.2,.6],[0,1],[False,True],[1,31,10000]):
        for layout in (range(4) if fmt==1 else [0]):
            yield dict(fmt=fmt,kernel=kernel,seed=seed,nu=nu,weighted=weighted,shrinking=shrinking,max_iter=cap,
                       layout=layout,n=18,p=4,tol=1e-8,gamma=.4,coef0=-.2 if kernel=='sigmoid' else .2,kind='ordinary')
    for kind,seed,nu,weighted,shrinking,cap in itertools.product(['asymmetric','near_asymmetric','small_asymmetric'],[7,42],[.2,.6],[0,1],[False,True],[1,31,10000]):
        yield dict(fmt=2,kernel='rbf',seed=seed,nu=nu,weighted=weighted,shrinking=shrinking,max_iter=cap,
                   layout=0,n=18,p=4,tol=1e-8,gamma=.4,coef0=.2,kind=kind)
    for fmt,nu,weighted,shrinking in itertools.product([0,1],[.2,.6],[0,1],[False,True]):
        yield dict(fmt=fmt,kernel='rbf',seed=7,nu=nu,weighted=weighted,shrinking=shrinking,max_iter=10000,
                   layout=0,n=18,p=4,tol=1e-8,gamma=0.,coef0=.2,kind='constant')
    for tol,shrinking in itertools.product([1e-5,1e-7],[False,True]):
        yield dict(fmt=0,kernel='rbf',seed=7,nu=.25,weighted=0,shrinking=shrinking,max_iter=100000,
                   layout=0,n=40,p=2,tol=tol,gamma=1.,coef0=.2,kind='circle')


def arrays(c):
    rng=np.random.RandomState(c['seed']);n=c['n'];p=c['p'];x=rng.normal(size=(n,p));q=rng.normal(size=(7,p))
    if c['seed']==42:x=np.abs(x)+.3
    w=rng.uniform(.3,2,n) if c['weighted'] else np.ones(n)
    if c['weighted']:w[::7]=0
    if c['kind']=='circle':
        t=2*np.pi*np.arange(1,n+1)/n;x=np.c_[.5*np.cos(t),.5*np.sin(t)];q=np.array([[0.,0.],[5.,5.]])
    if c['fmt']==1:
        x[rng.uniform(size=x.shape)<.45]=0;q[rng.uniform(size=q.shape)<.4]=0;x[0]=0
        return matrix(x,c['layout']),matrix(q,c['layout']),w
    if c['fmt']==2:
        if c['kind'].endswith('asymmetric'):
            a=rng.normal(size=(n,n));b=rng.normal(size=(len(q),n))
            if c['kind']=='near_asymmetric':a=(a+a.T)/2+1e-8*(a-a.T)
            if c['kind']=='small_asymmetric':a*=1e-12;b*=1e-12
            return a,b,w
        if c['kernel']=='linear':return linear_kernel(x),linear_kernel(q,x),w
        if c['kernel']=='rbf':return rbf_kernel(x,gamma=c['gamma']),rbf_kernel(q,x,gamma=c['gamma']),w
        if c['kernel']=='poly':
            args=dict(gamma=c['gamma'],coef0=c['coef0'],degree=2)
            return polynomial_kernel(x,**args),polynomial_kernel(q,x,**args),w
        return sigmoid_kernel(x,gamma=c['gamma'],coef0=c['coef0']),sigmoid_kernel(q,x,gamma=c['gamma'],coef0=c['coef0']),w
    return x,q,w


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--diagnostic',action='store_true');parser.add_argument('--limit',type=int)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),args.executable,*[root/p for p in ['src/sklearn_kernel_svm.f90','src/sklearn_libsvm_solver.f90',
           'src/sklearn_libsvm_sparse.f90','app/reference_oneclass_libsvm_driver.f90','validation/compare_svc_sparse.py','validation/compare_kernel_svm_sparse.py']],
           *[Path(sklearn.__file__).parent/'svm'/p for p in ['_base.py','_classes.py','src/libsvm/svm.cpp']]]
    hashes={str(p):sha(p) for p in files};cases=list(fixtures())[:args.limit];data=[arrays(c) for c in cases]
    payload=str(len(cases))+'\n';expected=[];source_iterations=[]
    for c,(x,q,w) in zip(cases,data):
        pre=c['fmt']==2;sparse=c['fmt']==1;positive=np.flatnonzero(w>0)
        fx=x[np.ix_(positive,positive)] if pre else x.copy();fw=w[positive] if pre else w
        tx=x[:,positive] if pre else x.copy();qx=q[:,positive] if pre else q.copy()
        model=OneClassSVM(kernel='precomputed' if pre else c['kernel'],gamma=c['gamma'],coef0=c['coef0'],degree=2,
                          nu=c['nu'],tol=c['tol'],max_iter=c['max_iter'],shrinking=c['shrinking'])
        with warnings.catch_warnings():warnings.simplefilter('ignore');model.fit(fx,sample_weight=fw)
        original=positive[model.support_];source_iterations.append([int(model.n_iter_)])
        v=dict(support=model.support_ if sparse else original,support_original=original,n_support=model.n_support_,
               dual_coef=model.dual_coef_.toarray() if sparse else model.dual_coef_,intercept=model.intercept_,offset=model.offset_,gamma=[model._gamma],
               coef=(model.coef_.toarray() if sparse else model.coef_) if c['kernel']=='linear' and not pre else [],
               vectors=[] if sparse else model.support_vectors_,sv_indptr=model.support_vectors_.indptr+1 if sparse else [],
               sv_indices=model.support_vectors_.indices+1 if sparse else [],sv_data=model.support_vectors_.data if sparse else [],
               train=model.decision_function(tx),query=model.decision_function(qx),train_raw=model.score_samples(tx),query_raw=model.score_samples(qx),
               predict=model.predict(qx),fit_predict=model.predict(tx),status=[model.fit_status_])
        if sparse:v.update(dense_query=model.decision_function(q.toarray()),dense_predict=model.predict(q.toarray()))
        expected.append({k:np.asarray(a).ravel(order='F').tolist() for k,a in v.items()})
        header=[c['fmt'],x.shape[0],x.shape[1],q.shape[0],'precomputed' if pre else c['kernel'],c['nu'],c['gamma'],c['coef0'],2,c['tol'],c['max_iter'],0,int(c['shrinking'])]
        payload+=' '.join(map(str,header))+'\n'
        if sparse:payload+=csr_payload(x)+csr_payload(q)
        else:
            for a in [x,q]:payload+=' '.join(format(v,'.17g') for v in np.asarray(a).ravel(order='F'))+'\n'
        payload+=' '.join(format(v,'.17g') for v in w)+'\n'
    run=subprocess.run([str(args.executable)],input=payload,capture_output=True,text=True)
    assert run.returncode==0,(run.stdout[-1200:],run.stderr[-2000:])
    actual=[];checks=[];lines=iter(run.stdout.splitlines());index=-1
    for line in lines:
        p=line.split()
        if not p:continue
        if p[0]=='BEGIN':index+=1;actual.append({})
        elif p[0]=='END':assert actual[index].keys()==expected[index].keys()|{'native_iterations','native_kkt'}
        else:
            field=p[0];values=np.fromstring(next(lines),sep=' ');assert len(values)==int(p[1]);actual[index][field]=values.tolist()
            if field.startswith('native_'):continue
            want=np.asarray(expected[index][field]);exact=field in ['support','support_original','n_support','sv_indptr','sv_indices','predict','fit_predict','dense_predict','status']
            passed=values.shape==want.shape and (np.array_equal(values,want) if exact else np.allclose(values,want,atol=3e-6,rtol=3e-6))
            checks.append(dict(case=index,field=field,passed=bool(passed),max_abs_error=float(np.max(np.abs(values-want),initial=0)) if values.shape==want.shape else None))
    assert len(actual)==len(cases);certificates=[]
    for i,(c,(x,q,w),v) in enumerate(zip(cases,data,actual)):
        support=np.asarray(v['support_original'],int);alpha=np.zeros(len(w));alpha[support]=v['dual_coef']
        assert np.all(alpha>=-1e-10) and np.all(alpha<=w+1e-10)
        np.testing.assert_allclose(sum(alpha),c['nu']*sum(w),atol=1e-9,rtol=1e-10)
        if c['fmt']==1:gram=raw_gram(x,c['kernel'],v['gamma'][0],c['coef0'],2)
        elif c['fmt']==2:gram=x
        elif c['kernel']=='linear':gram=linear_kernel(x)
        elif c['kernel']=='rbf':gram=rbf_kernel(x,gamma=v['gamma'][0])
        elif c['kernel']=='poly':gram=polynomial_kernel(x,gamma=v['gamma'][0],coef0=c['coef0'],degree=2)
        else:gram=sigmoid_kernel(x,gamma=v['gamma'][0],coef0=c['coef0'])
        g=gram@alpha;below=(w>0)&(alpha<w);above=(w>0)&(alpha>0)
        gap=max(0.,float(g[above].max()-g[below].min())) if np.any(above) and np.any(below) else 0.
        np.testing.assert_allclose(gap,v['native_kkt'][0],atol=1e-8,rtol=1e-6)
        error=float(np.max(np.abs(gram-gram.astype(np.float32).astype(float))@alpha))
        applicable=not c['kind'].endswith('asymmetric')
        if applicable and not v['status'][0]:assert gap<=c['tol']+2*error+1e-8,(i,gap,error)
        np.testing.assert_allclose(np.asarray(v['query_raw'])-v['offset'][0],v['query'],atol=1e-9,rtol=1e-9)
        certificates.append(dict(passed=True,mass=float(sum(alpha)),physical_gap=gap,cache_error=error,symmetric_qp_bound_applicable=applicable,capped=bool(v['status'][0])))
    gaps=[v for v in checks if not v['passed']];assert hashes=={p:sha(p) for p in hashes}
    result=dict(cases=cases,checks=checks,gaps=gaps,actual=actual,expected=expected,certificates=certificates,source_iterations=source_iterations,
                iteration_disagreements=[i for i,(a,b) in enumerate(zip(actual,source_iterations)) if a['native_iterations']!=b],hashes=hashes,
                strict_parity=not gaps,sklearn=sklearn.__version__,scipy=scipy.__version__)
    args.report.write_text(json.dumps(result,indent=2)+'\n');print(len(cases),'fits;',len(checks),'groups;',len(gaps),'gaps;',len(certificates),'certificates')
    if gaps:print(gaps[:20])
    raise SystemExit(bool(gaps) and not args.diagnostic)


if __name__=='__main__':main()
