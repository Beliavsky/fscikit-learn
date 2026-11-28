"""Raw CSR C-SVC storage, weighting, kernel and query comparisons."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.svm import SVC
import scipy
import scipy.sparse as sp
from compare_kernel_svm_sparse import matrix,csr_payload
from sklearn.metrics.pairwise import linear_kernel,rbf_kernel,polynomial_kernel,sigmoid_kernel


def fixtures():
    for seed,kernel,classes,cw,weighted,shape,layout in itertools.product([7,42],['linear','rbf','poly','sigmoid'],[2,3,4],range(3),range(2),[0,2],range(4)):
        yield dict(seed=seed,kernel=kernel,classes=classes,cw=cw,weighted=weighted,shape=shape,layout=layout,n=25,p=7,c=.7,
            gamma=-1.,coef0=.2,degree=2,tol=1e-8,max_iter=10000)
    for seed,kernel,p,rule in itertools.product([7,42],['linear','rbf','poly','sigmoid'],[17,43],range(2)):
        yield dict(seed=seed,kernel=kernel,classes=3,cw=2,weighted=1,shape=2,layout=2,n=25,p=p,c=1.3,
            gamma=-1.,gamma_rule=rule,coef0=.2,degree=2,tol=1e-8,max_iter=10000)
    for kernel,layout,cap in itertools.product(['linear','rbf','poly','sigmoid'],[0,2],[1,3,31,67,10000]):
        yield dict(seed=19,kernel=kernel,classes=3,cw=1,weighted=1,shape=2,layout=layout,n=80,p=8,c=30.,
            gamma=.4,coef0=.2,degree=2,tol=1e-8,max_iter=cap)


def arrays(c):
    rng=np.random.RandomState(c['seed']);labels=np.array([-4,7,21,33])[:c['classes']];y=np.resize(labels,c['n'])
    centers=rng.normal(size=(c['classes'],c['p']));x=rng.normal(scale=.7,size=(c['n'],c['p']))+centers[np.searchsorted(labels,y)]
    q=rng.normal(size=(9,c['p']));w=rng.uniform(.3,2,c['n']);w[::7]=0
    if not c['weighted']:w[:]=1
    x[rng.uniform(size=x.shape)<.5]=0;q[rng.uniform(size=q.shape)<.4]=0;x[0]=0
    if c.get('zero'):x[:]=0;q[:]=0
    return matrix(x,c['layout']),matrix(q,c['layout']),y,w,np.linspace(.5,2,c['classes'])


def raw_gram(x,kernel,gamma,coef0,degree):
    x=x.copy();x.sort_indices();n=x.shape[0];result=np.zeros((n,n))
    def dot(i,j):
        a,b=x.indptr[i],x.indptr[j];ae,be=x.indptr[i+1],x.indptr[j+1];value=0.
        while a<ae and b<be:
            if x.indices[a]==x.indices[b]:value+=x.data[a]*x.data[b];a+=1;b+=1
            elif x.indices[a]<x.indices[b]:a+=1
            else:b+=1
        return value
    diagonal=[dot(i,i) for i in range(n)]
    for i in range(n):
        for j in range(n):
            value=dot(i,j)
            if kernel=='rbf':value=np.exp(-gamma*(diagonal[i]+diagonal[j]-2*value))
            elif kernel=='poly':value=(gamma*value+coef0)**degree
            elif kernel=='sigmoid':value=np.tanh(gamma*value+coef0)
            result[i,j]=value
    return result


def certificate(c,a,v):
    x,q,y,w,cw=a;k=c['classes'];support=np.asarray(v['native_support_original'],int)
    dual=np.asarray(v['dual_coef']).reshape((k-1,len(support)),order='F');labels=np.unique(y)
    weights=np.ones(k) if c['cw']==0 else (np.array([len(y)/(k*np.sum(y==label)) for label in labels]) if c['cw']==1 else cw)
    cost=w*(c['c']*weights[np.searchsorted(labels,y)]);gram=raw_gram(x,c['kernel'],v['gamma'][0],c['coef0'],c['degree']);records=[];pair=0
    for first in range(k-1):
        for last in range(first+1,k):
            beta=np.zeros(len(y));lower=y[support]==labels[first];upper=y[support]==labels[last]
            beta[support[lower]]=dual[last-1,lower];beta[support[upper]]=dual[first,upper]
            if k>2:beta=-beta
            signed=np.where(y==labels[last],1.,-1.);mask=((y==labels[first])|(y==labels[last]))&(cost>0);alpha=beta*signed
            assert np.all(alpha>=-1e-10) and np.all(alpha<=cost+1e-10) and abs(np.sum(beta))<1e-8
            margin=gram@beta+v['intercept'][pair]*(1 if k==2 else -1);g=signed*margin-1
            violation=max(float(np.max(-g[mask&(alpha<cost-1e-10)],initial=0)),float(np.max(g[mask&(alpha>1e-10)],initial=0)))
            np.testing.assert_allclose(v['native_kkt'][pair],violation,atol=1e-8,rtol=1e-6)
            penalty=.5*beta@gram@beta;primal=penalty+np.sum(cost[mask]*np.maximum(0,1-signed[mask]*margin[mask]));dual_value=np.sum(alpha)-penalty
            if c['kernel']!='sigmoid':assert primal-dual_value>=-1e-8
            records.append(dict(passed=True,kkt=violation,gap=float(primal-dual_value),small_kkt=bool(violation<2e-6)));pair+=1
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--diagnostic',action='store_true');parser.add_argument('--no-shrinking',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),args.executable,root/'src/sklearn_svm.f90',root/'src/sklearn_libsvm_sparse.f90',root/'app/reference_svc_sparse_driver.f90',
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
            warnings.simplefilter('ignore');model.fit(x.copy(),y,sample_weight=w)
        source_status.append(model.fit_status_);iterations.append(model.n_iter_.tolist())
        unshrunk=SVC(**dict(model.get_params(),shrinking=False))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');unshrunk.fit(x.copy(),y,sample_weight=w)
        unshrunk_iterations.append(unshrunk.n_iter_.tolist())
        v=dict(classes=model.classes_,support=model.support_,n_support=model.n_support_,sv_indptr=model.support_vectors_.indptr+1,sv_indices=model.support_vectors_.indices+1,sv_data=model.support_vectors_.data,
            dual_coef=model.dual_coef_.toarray(),intercept=model.intercept_,coef=model.coef_.toarray() if c['kernel']=='linear' else np.array([]),
            gamma=[model._gamma],class_weight=model.class_weight_,train=model.decision_function(x.copy()),query=model.decision_function(q.copy()),dense_query=model.decision_function(q.toarray()),
            predict=model.predict(q.copy()),dense_predict=model.predict(q.toarray()),status=[model.fit_status_],score=[model.score(x.copy(),y),model.score(x.copy(),y,sample_weight=w)])
        expected.append({k:np.asarray(a).ravel(order='F').tolist() for k,a in v.items()})
        header=[c['n'],c['p'],q.shape[0],c['kernel'],c['c'],c['gamma'],c['coef0'],c['degree'],c['tol'],c['max_iter'],c['cw'],int(shape=='ovr'),int(c['shape']==2),c.get('gamma_rule',0)]
        payload+=' '.join(map(str,header))+'\n'
        payload+=csr_payload(x)+csr_payload(q)
        for a in [y,w,cw]:payload+=' '.join(format(v,'.17g') for v in np.asarray(a).ravel(order='F'))+'\n'
    run=subprocess.run([str(args.executable),*(["--no-shrinking"] if args.no_shrinking else [])],input=payload,capture_output=True,text=True);assert run.returncode==0,(run.stdout[-1000:],run.stderr[-2000:])
    lines=iter(run.stdout.splitlines());actual=[];checks=[];index=-1
    for line in lines:
        parts=line.split()
        if not parts:continue
        if parts[0]=='BEGIN':index+=1;actual.append({})
        elif parts[0]=='END':assert actual[index].keys()==expected[index].keys()|{'native_iterations','native_kkt','native_support_original','native_storage'}
        else:
            field=parts[0];size=int(parts[1]);values=np.fromstring(next(lines),sep=' ');assert len(values)==size;actual[index][field]=values.tolist()
            if not field.startswith('native_'):
                want=np.asarray(expected[index][field]);exact=field in ['classes','support','n_support','predict','dense_predict','status','sv_indptr','sv_indices']
                passed=values.shape==want.shape and (np.array_equal(values,want) if exact else np.allclose(values,want,atol=3e-6,rtol=3e-6))
                checks.append(dict(case=index,field=field,passed=bool(passed),max_abs_error=float(np.max(np.abs(values-want),initial=0)) if values.shape==want.shape else None))
    assert len(actual)==len(cases)
    for v in actual:assert v['native_storage']==[0,0,0]
    certificates=[certificate(c,a,v) for c,a,v in zip(cases,data,actual)]
    assert hashes=={p:sha(p) for p in hashes};gaps=[c for c in checks if not c['passed']]
    report=dict(cases=cases,checks=checks,gaps=gaps,certificates=certificates,actual=actual,expected=expected,
        shrinking=not args.no_shrinking,iteration_disagreements=[i for i,(a,b) in enumerate(zip(actual,iterations)) if a["native_iterations"]!=b],source_iterations=iterations,source_status=source_status,source_nonshrinking_iterations=unshrunk_iterations,hashes=hashes,strict_parity=not gaps,sklearn=sklearn.__version__,scipy=scipy.__version__,numpy=np.__version__)
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(len(cases),'fits;',len(checks),'groups;',len(gaps),'gaps;',sum(map(len,certificates)),'certificates')
    if gaps:print(gaps[:12])
    raise SystemExit(bool(gaps) and not args.diagnostic)

if __name__=='__main__':main()
