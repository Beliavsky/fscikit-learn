"""Crammer-Singer outputs and independent joint primal/dual certificates."""
from pathlib import Path
import argparse,hashlib,itertools,json,subprocess,warnings
import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.svm import LinearSVC
import sklearn.svm._liblinear as backend
from compare_linear_svc_dual import arrays,payload


def fixtures():
    for seed,(n,p),classes,intercept,cw,weighted,layout in itertools.product([0,7,42],
            [(18,5),(7,13)],[2,3,5],range(2),range(3),range(2),['dense','csr','reverse','duplicates','zeros']):
        yield dict(seed=seed,n=n,p=p,classes=classes,intercept=intercept,cw=cw,weighted=weighted,layout=layout,
            loss=1,dual=2,max_iter=1000,scaling=1.7,tol=1e-6,c=.7)
    for loss,dual,cap in itertools.product(range(2),range(3),[1,3,1000]):
        yield dict(seed=2147483647,n=13,p=7,classes=3,intercept=1,cw=2,layout='csr',
            loss=loss,dual=dual,max_iter=cap,scaling=.7,tol=1e-8,c=1.3)
    for classes,intercept,layout in itertools.product([2,3,5],range(2),['dense','csr']):
        yield dict(seed=7,n=12,p=4,classes=classes,intercept=intercept,cw=0,layout=layout,
            loss=1,dual=2,max_iter=1000,scaling=1.,tol=1e-6,c=1.,zero=True)
    for active in [1,2]:
        yield dict(seed=42,n=9,p=4,classes=3,intercept=1,cw=0,layout='csr',
            loss=1,dual=2,max_iter=1000,scaling=1.,tol=1e-6,c=1.,active=active)


def inputs(c):
    # Reuse the established CSR fixtures, with five-class labels when requested.
    base=dict(c,classes=min(c['classes'],3));x,q,y,w,cw=arrays(base)
    if c['classes']==5:
        y=np.resize(np.array([-4,7,21,33,105]),c['n']);cw=np.linspace(.5,2,5)
    if c.get('zero'):x=sp.csr_matrix(x.shape);q=sp.csr_matrix(q.shape)
    return x,q,y,w,cw


def oracle(c,a):
    x,q,y,w,cw=a
    if c['layout']=='dense':x=x.toarray();q=q.toarray()
    model=LinearSVC(multi_class='crammer_singer',loss=['hinge','squared_hinge'][c['loss']],
        dual={0:False,1:True,2:'auto'}[c['dual']],random_state=c['seed'],fit_intercept=bool(c['intercept']),
        intercept_scaling=c['scaling'],C=c['c'],tol=c['tol'],max_iter=c['max_iter'],
        class_weight=None if c['cw']==0 else ('balanced' if c['cw']==1 else dict(zip(np.unique(y),cw))))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always');model.fit(x,y,sample_weight=w)
    weights=np.ones(len(np.unique(y))) if c['cw']==0 else (np.array([len(y)/(len(np.unique(y))*np.sum(y==label)) for label in np.unique(y)]) if c['cw']==1 else cw)
    # linear.cpp allocates one CS counter, but get_n_iter copies nr_class counters.
    # Only element zero is defined; the public multiclass max can include garbage.
    raw,counts=backend.train_wrap(x,np.searchsorted(np.unique(y),y).astype(np.float64),sp.issparse(x),
        4,c['tol'],c['scaling'] if c['intercept'] else -1.,c['c'],weights,c['max_iter'],
        int(np.random.RandomState(c['seed']).randint(np.iinfo('i').max)),0.,w)
    recorded_messages=[str(w.message) for w in recorded]
    recorded_messages.append(dict(public_iterations=int(model.n_iter_),defined_backend_iterations=int(counts[0]),
        public_counter_defined=len(np.unique(y[w>0]))<=2))
    result=dict(classes=model.classes_,coef=model.coef_.ravel(order='F'),
        intercept=np.broadcast_to(model.intercept_,(model.coef_.shape[0],)),iterations=[int(counts[0])],
        train=model.decision_function(x).reshape(len(y),-1).ravel(order='F'),
        query=model.decision_function(q).reshape(8,-1).ravel(order='F'),
        dense_query=model.decision_function(q.toarray() if sp.issparse(q) else q).reshape(8,-1).ravel(order='F'),
        predict=model.predict(q),score=[model.score(x,y)])
    return {k:np.asarray(v).tolist() for k,v in result.items()},recorded_messages


def certificate(c,a,v):
    x,q,y,w,cw=a;classes=np.unique(y);active=np.unique(y[w>0]);n=len(y);k=len(active)
    alpha=np.asarray(v['alpha']).reshape((n,k),order='F');assert np.all(alpha[w==0]==0)
    aug=sp.hstack([x,np.full((n,1),c['scaling'])],format='csr') if c['intercept'] else x
    raw=np.asarray(aug.T@alpha).T
    coef=raw[:,:c['p']];intercept=raw[:,-1]*c['scaling'] if c['intercept'] else np.zeros(k)
    if len(classes)==2:coef=(coef[1]-coef[0])[None,:];intercept=np.array([intercept[1]-intercept[0]])
    np.testing.assert_allclose(v['coef'],coef.ravel(order='F'),rtol=2e-10,atol=2e-10)
    np.testing.assert_allclose(v['intercept'],intercept,rtol=2e-10,atol=2e-10)
    np.testing.assert_allclose(np.sum(alpha,axis=1),0,atol=2e-10)
    cw=np.ones(len(classes)) if c['cw']==0 else (np.array([n/(len(classes)*np.sum(y==label)) for label in classes]) if c['cw']==1 else cw)
    cost=w*(c['c']*cw[np.searchsorted(classes,y)]);scores=np.asarray(aug@raw.T)
    norm=.5*np.sum(raw**2);primal=norm;dual=-norm;kkt=0
    for i in np.flatnonzero(w>0):
        true=int(np.searchsorted(active,y[i]));upper=np.zeros(k);upper[true]=cost[i]
        assert np.all(alpha[i]<=upper+2e-10)
        loss=0
        if k>1:
            margins=1+scores[i]-scores[i,true];margins[true]=0;loss=np.max(margins)
        primal+=cost[i]*loss;dual+=alpha[i,true]
        g=1+scores[i];g[true]=scores[i,true];free=alpha[i]<upper
        if np.any(free):kkt=max(kkt,float(np.max(g)-np.min(g[free])))
    np.testing.assert_allclose([v['objective'][0],v['dual_objective'][0],v['gap'][0],v['kkt'][0],v['gradient'][0]],
        [primal,dual,primal-dual,kkt,kkt],rtol=2e-9,atol=2e-10)
    assert primal-dual>=-2e-10*max(1,abs(primal))
    return dict(passed=True,gap=float(primal-dual),kkt=kkt,classes=k)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--diagnostic',action='store_true');args=parser.parse_args()
    cases=list(fixtures());data=[inputs(c) for c in cases];expected=[oracle(c,a) for c,a in zip(cases,data)]
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),args.executable,root/'src/sklearn_linear_svm.f90',root/'app/reference_linear_svc_cs_driver.f90',
        root/'src/sklearn_legacy_rng.f90',root/'validation/compare_linear_svc_dual.py',Path(backend.__file__),
        *[Path(sklearn.__file__).parent/'svm'/name for name in ['_classes.py','_base.py','_liblinear.pyx',
            'src/liblinear/linear.cpp','src/newrand/newrand.h']]]
    hashes={str(p):sha(p) for p in files}
    run=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,data)),capture_output=True,text=True)
    assert run.returncode==0,(run.stdout[-1000:],run.stderr[-2000:])
    lines=iter(run.stdout.splitlines());actual=[];index=-1;checks=[]
    native={'objective','dual_objective','gap','kkt','gradient','alpha'}
    for line in lines:
        parts=line.split()
        if not parts:continue
        if parts[0]=='BEGIN':index+=1;actual.append({});assert int(parts[1])==index+1
        elif parts[0]=='END':assert actual[index].keys()==expected[index][0].keys()|native
        else:
            field=parts[0];size=int(parts[1]);assert field not in actual[index]
            values=np.fromstring(next(lines),sep=' ');assert len(values)==size;actual[index][field]=values.tolist()
            if field not in native:
                want=np.asarray(expected[index][0][field]).ravel()
                passed=values.shape==want.shape and (np.array_equal(values,want) if field in ['classes','iterations','predict'] else
                    np.allclose(values,want,rtol=2e-10,atol=2e-10,equal_nan=True))
                checks.append(dict(case=index,field=field,passed=bool(passed),oracle='source_backend_first_counter' if field=='iterations' else 'source_public',max_abs_error=float(np.max(np.abs(values-want),initial=0)) if values.shape==want.shape else None))
    assert index+1==len(cases)
    certificates=[certificate(c,a,v) for c,a,v in zip(cases,data,actual)]
    assert hashes=={p:sha(p) for p in hashes};gaps=[c for c in checks if not c['passed']]
    result=dict(cases=cases,checks=checks,gaps=gaps,certificates=certificates,actual=actual,
        expected=[e for e,_ in expected],source_warnings=[w for _,w in expected],hashes=hashes,
        strict_parity=not gaps,versions=dict(sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__))
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(len(cases),'fits;',len(checks),'groups;',len(gaps),'gaps;',len(certificates),'joint certificates')
    if gaps:print(gaps[:12])
    raise SystemExit(bool(gaps) and not args.diagnostic)


if __name__=='__main__':main()
