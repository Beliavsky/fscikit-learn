"""LIBLINEAR classification outputs plus independent channel-wise certificates."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.svm import LinearSVC
import sklearn.svm._liblinear as backend
from compare_linear_svm import sparse_input
from compare_linear_svr_dual import csr_payload


def fixtures():
    for loss,seed,(n,p),classes,intercept,cw,weighted,layout in itertools.product(range(2),[0,7,42],
            [(18,5),(7,13)],[2,3],range(2),range(3),range(2),['dense','csr','reverse','duplicates','zeros']):
        yield dict(loss=loss,seed=seed,n=n,p=p,classes=classes,intercept=intercept,cw=cw,weighted=weighted,layout=layout,
                   dual=1,max_iter=10000,scaling=1.7,tol=1e-8,c=.7)
    for loss,(n,p),dual,cap in itertools.product(range(2),[(17,4),(6,13),(7,7)],[1,2],[1,3,1000]):
        yield dict(loss=loss,seed=2147483647,n=n,p=p,classes=3,intercept=1,cw=2,layout='csr',
                   dual=dual,max_iter=cap,scaling=.7,tol=1e-6,c=1.3)
    for loss,dual,active in itertools.product(range(2),[1,2],[1,2]):
        yield dict(loss=loss,seed=42,n=9,p=4,classes=3,intercept=1,cw=0,layout='csr',
                   dual=dual,max_iter=1000,scaling=1.,tol=1e-6,c=1.,active=active)


def arrays(c):
    rng=np.random.RandomState(c['seed']);x=rng.normal(size=(c['n'],c['p']));x[rng.uniform(size=x.shape)<.5]=0
    q=rng.normal(size=(8,c['p']));q[rng.uniform(size=q.shape)<.4]=0
    labels=np.array([-4,7,21])[:c['classes']];y=np.resize(labels,c['n'])
    w=rng.uniform(.3,2,c['n']);w[::7]=0
    if not c.get('weighted',1):w[:]=1
    if 'active' in c:w[~np.isin(y,labels[-c['active']:])]=0
    cw=np.linspace(.5,2,c['classes'])
    layout='csr' if c['layout']=='dense' else c['layout']
    return sparse_input(x,layout),sparse_input(q,layout),y,w,cw


def payload(c,a):
    x,q,y,w,cw=a
    control=[int(c['layout']!='dense'),c['loss'],c['dual'],c['seed'],c['intercept'],c['cw'],
             c['max_iter'],c['scaling'],c['tol'],c['c']]
    return ' '.join(map(str,control))+'\n'+csr_payload(x)+csr_payload(q)+\
        ' '.join(map(str,y))+'\n'+' '.join(format(v,'.17g') for v in w)+'\n'+' '.join(map(str,cw))+'\n'


def oracle(c,a):
    x,q,y,w,cw=a
    if c['layout']=='dense':x=x.toarray();q=q.toarray()
    model=LinearSVC(loss=['hinge','squared_hinge'][c['loss']],dual={0:False,1:True,2:'auto'}[c['dual']],
        random_state=c['seed'],fit_intercept=bool(c['intercept']),intercept_scaling=c['scaling'],
        C=c['c'],tol=c['tol'],max_iter=c['max_iter'],
        class_weight=None if c['cw']==0 else ('balanced' if c['cw']==1 else dict(zip(np.unique(y),cw))))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always');model.fit(x,y,sample_weight=w)
    result=dict(classes=model.classes_,coef=model.coef_.ravel(order='F'),
        intercept=np.broadcast_to(model.intercept_,(model.coef_.shape[0],)),iterations=[model.n_iter_],
        train=model.decision_function(x).reshape(len(y),-1).ravel(order='F'),
        query=model.decision_function(q).reshape(8,-1).ravel(order='F'),
        dense_query=model.decision_function(q.toarray() if sp.issparse(q) else q).reshape(8,-1).ravel(order='F'),
        predict=model.predict(q),score=[model.score(x,y)])
    return result,[str(w.message) for w in recorded]


def certificate(c,a,v):
    x,q,y,w,cw=a;classes=np.unique(y);active=np.unique(y[w>0]);channels=1 if len(active)<=2 else len(active)
    coef=np.asarray(v['coef']).reshape((channels,c['p']),order='F')
    theta=np.column_stack([coef,np.asarray(v['intercept'])/c['scaling']]) if c['intercept'] else coef
    aug=sp.hstack([x,np.full((x.shape[0],1),c['scaling'])],format='csr') if c['intercept'] else x
    is_dual=c['dual']==1 or (c['dual']==2 and (not c['loss'] or c['n']<c['p']))
    alpha=np.asarray(v['alpha']);assert bool(alpha.size)==is_dual
    if is_dual:alpha=alpha.reshape((len(y),channels),order='F')
    weights=np.ones(len(classes)) if c['cw']==0 else (np.array([len(y)/(len(classes)*np.sum(y==k)) for k in classes]) if c['cw']==1 else cw)
    records=[]
    for k in range(channels):
        positive=active[1] if len(active)==2 else active[k]
        signed=np.where(y==positive,1.,-1.)
        cost=w*(c['c']*(weights[np.searchsorted(classes,y)] if len(active)==2 else np.where(y==positive,weights[np.searchsorted(classes,positive)],1.)))
        margin=np.asarray(aug@theta[k]).ravel();excess=np.maximum(0,1-signed*margin)
        primal=.5*theta[k]@theta[k]+np.sum(cost*(excess**2 if c['loss'] else excess))
        np.testing.assert_allclose(v['objective'][k],primal,rtol=2e-10,atol=2e-10)
        if not is_dual:
            gradient=theta[k]-2*np.asarray(aug.T@(cost*signed*excess)).ravel()
            np.testing.assert_allclose(v['gradient'][k],np.linalg.norm(gradient),rtol=1e-8,atol=1e-10)
            records.append(dict(kind='primal',passed=True));continue
        ak=alpha[:,k];assert np.all(ak>=0) and np.all(ak[w==0]==0)
        if not c['loss']:assert np.all(ak<=cost+1e-12)
        np.testing.assert_allclose(aug.T@(signed*ak),theta[k],rtol=2e-10,atol=2e-10)
        penalty=np.sum(ak[w>0]**2/(4*cost[w>0])) if c['loss'] else 0.
        dual=np.sum(ak)-.5*theta[k]@theta[k]-penalty;gap=primal-dual
        np.testing.assert_allclose([v['dual_objective'][k],v['gap'][k]],[dual,gap],rtol=2e-10,atol=2e-10)
        assert gap>=-2e-10*max(1,abs(primal))
        g=signed*margin-1
        if c['loss']:g[w>0]+=ak[w>0]/(2*cost[w>0])
        pg=g.copy();pg[ak==0]=np.minimum(pg[ak==0],0)
        if not c['loss']:pg[ak==cost]=np.maximum(pg[ak==cost],0)
        kkt=float(np.max(np.abs(pg[w>0]),initial=0))
        np.testing.assert_allclose(v['kkt'][k],kkt,rtol=2e-10,atol=2e-10)
        records.append(dict(kind='dual',gap=float(gap),kkt=kkt,passed=True))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--diagnostic',action='store_true');args=parser.parse_args()
    cases=list(fixtures());inputs=[arrays(c) for c in cases];expected=[oracle(c,a) for c,a in zip(cases,inputs)]
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),args.executable,root/'src/sklearn_linear_svm.f90',root/'src/sklearn_legacy_rng.f90',
        root/'app/reference_linear_svc_dual_driver.f90',Path(backend.__file__),
        Path(sklearn.__file__).parent/'svm/src/liblinear/linear.cpp',Path(sklearn.__file__).parent/'svm/src/newrand/newrand.h']
    hashes={str(p):sha(p) for p in files}
    run=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,inputs)),capture_output=True,text=True)
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
                checks.append(dict(case=index,field=field,passed=bool(passed),max_abs_error=float(np.max(np.abs(values-want),initial=0)) if values.shape==want.shape else None))
    assert index+1==len(cases)
    certificates=[certificate(c,a,v) for c,a,v in zip(cases,inputs,actual)]
    assert hashes=={p:sha(p) for p in hashes};gaps=[c for c in checks if not c['passed']]
    result=dict(cases=cases,checks=checks,gaps=gaps,certificates=certificates,actual=actual,
        expected=[{k:np.asarray(v).tolist() for k,v in e.items()} for e,_ in expected],source_warnings=[w for _,w in expected],
        hashes=hashes,strict_parity=not gaps,versions=dict(sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__))
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(len(cases),'fits;',len(checks),'groups;',len(gaps),'gaps;',sum(map(len,certificates)),'channel certificates')
    if gaps:print(gaps[:12])
    raise SystemExit(bool(gaps) and not args.diagnostic)


if __name__=='__main__':main()
