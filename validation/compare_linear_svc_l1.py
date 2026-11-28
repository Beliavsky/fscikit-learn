"""L1 squared-hinge LinearSVC outputs and minimum-norm subgradient replay."""
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
    for seed,(n,p),classes,intercept,cw,weighted,layout,dual in itertools.product([0,7,42],
            [(18,5),(7,13)],[2,3],range(2),range(3),range(2),['dense','csr','reverse','duplicates','zeros'],[0,2]):
        yield dict(loss=1,seed=seed,n=n,p=p,classes=classes,intercept=intercept,cw=cw,weighted=weighted,layout=layout,
                   dual=dual,max_iter=10000,scaling=1.7,tol=1e-8,c=.7)
    for (n,p),dual,cap in itertools.product([(17,4),(6,13),(7,7)],[0,2],[1,3,1000]):
        yield dict(loss=1,seed=2147483647,n=n,p=p,classes=3,intercept=1,cw=2,layout='csr',
                   dual=dual,max_iter=cap,scaling=.7,tol=1e-6,c=1.3)
    for classes,intercept,dual,layout in itertools.product([2,3],range(2),[0,2],['dense','csr']):
        yield dict(loss=1,seed=7,n=12,p=4,classes=classes,intercept=intercept,cw=0,layout=layout,
                   dual=dual,max_iter=1000,scaling=1.,tol=1e-8,c=1.,zero=True)
    for seed,cost in itertools.product([0,7,42],[.01,10.]):
        yield dict(loss=1,seed=seed,n=17,p=7,classes=3,intercept=1,cw=2,layout='csr',
                   dual=2,max_iter=10000,scaling=.7,tol=1e-10,c=cost)
    for active in [1,2]:
        yield dict(loss=1,seed=42,n=9,p=4,classes=3,intercept=1,cw=0,layout='csr',
                   dual=2,max_iter=1000,scaling=1.,tol=1e-6,c=1.,active=active)


def arrays(c):
    rng=np.random.RandomState(c['seed']);x=rng.normal(size=(c['n'],c['p']));x[rng.uniform(size=x.shape)<.5]=0
    q=rng.normal(size=(8,c['p']));q[rng.uniform(size=q.shape)<.4]=0
    labels=np.array([-4,7,21])[:c['classes']];y=np.resize(labels,c['n'])
    w=rng.uniform(.3,2,c['n']);w[::7]=0
    if not c.get('weighted',1):w[:]=1
    if 'active' in c:w[~np.isin(y,labels[-c['active']:])]=0
    cw=np.linspace(.5,2,c['classes'])
    if c.get('zero'):x[:]=0;q[:]=0
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
    model=LinearSVC(penalty='l1',loss=['hinge','squared_hinge'][c['loss']],dual={0:False,1:True,2:'auto'}[c['dual']],
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
    assert not v['alpha'] and np.all(np.asarray(v['gap'])==0) and np.all(np.asarray(v['dual_objective'])==0)
    weights=np.ones(len(classes)) if c['cw']==0 else (np.array([len(y)/(len(classes)*np.sum(y==k)) for k in classes]) if c['cw']==1 else cw)
    records=[]
    for k in range(channels):
        positive=active[1] if len(active)==2 else active[k];signed=np.where(y==positive,1.,-1.)
        cost=w*(c['c']*(weights[np.searchsorted(classes,y)] if len(active)==2 else np.where(y==positive,weights[np.searchsorted(classes,positive)],1.)))
        margin=np.asarray(aug@theta[k]).ravel();excess=np.maximum(0,1-signed*margin)
        objective=np.sum(np.abs(theta[k]))+np.sum(cost*excess**2)
        g=-2*np.asarray(aug.T@(cost*signed*excess)).ravel()
        subgradient=g+np.sign(theta[k]);zero=theta[k]==0
        subgradient[zero]=np.sign(g[zero])*np.maximum(np.abs(g[zero])-1,0)
        np.testing.assert_allclose([v['objective'][k],v['gradient'][k],v['kkt'][k]],
            [objective,np.linalg.norm(subgradient),np.max(np.abs(subgradient),initial=0)],rtol=1e-8,atol=2e-10)
        records.append(dict(kind='l1_primal',passed=True,nonzero=int(np.count_nonzero(theta[k]))))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--diagnostic',action='store_true');args=parser.parse_args()
    cases=list(fixtures());inputs=[arrays(c) for c in cases];expected=[oracle(c,a) for c,a in zip(cases,inputs)]
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),args.executable,root/'src/sklearn_linear_svm.f90',root/'src/sklearn_legacy_rng.f90',
        root/'app/reference_linear_svc_l1_driver.f90',Path(backend.__file__),
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
