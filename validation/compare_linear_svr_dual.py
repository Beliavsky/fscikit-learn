"""Seeded LIBLINEAR dual regression and independent primal/dual certificates."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import subprocess
import warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.svm import LinearSVR
import sklearn.svm._liblinear as backend
from compare_linear_svm import sparse_input


def fixtures():
    for loss,seed,(n,p),intercept,weighted,layout in itertools.product(range(2),[0,7,42,2026,2147483647],
            [(19,5),(7,17),(11,1)],range(2),range(2),['dense','csr','reverse','duplicates','zeros']):
        yield dict(loss=loss,seed=seed,n=n,p=p,intercept=intercept,weighted=weighted,layout=layout,
                   dual=1,max_iter=10000,epsilon=.15,scaling=2.,tol=1e-8,c=.7)
    for loss,(n,p),dual,cap,tube in itertools.product(range(2),[(17,4),(6,13),(7,7)],['auto',True],[1,3,1000],[0.,10.]):
        yield dict(loss=loss,seed=42,n=n,p=p,intercept=1,weighted=1,layout='csr',
                   dual=2 if dual=='auto' else 1,max_iter=cap,epsilon=tube,scaling=.7,tol=1e-6,c=1.3)


def arrays(c):
    rng=np.random.RandomState(c['seed']);x=rng.normal(size=(c['n'],c['p']));x[rng.uniform(size=x.shape)<.5]=0
    q=rng.normal(size=(8,c['p']));q[rng.uniform(size=q.shape)<.4]=0
    y=np.sin(x[:,0])+.2*rng.normal(size=c['n'])+.3
    w=rng.uniform(.3,2,c['n']);w[::7]=0
    if not c['weighted']:w[:]=1
    layout='csr' if c['layout']=='dense' else c['layout']
    return sparse_input(x,layout),sparse_input(q,layout),y,w


def csr_payload(a):
    return f'{a.shape[0]} {a.shape[1]} {a.nnz}\n'+'\n'.join([
        ' '.join(map(str,a.indptr+1)),' '.join(map(str,a.indices+1)),
        ' '.join(format(v,'.17g') for v in a.data)])+'\n'


def payload(c,a):
    x,q,y,w=a
    control=[int(c['layout']!='dense'),c['loss'],c['dual'],c['seed'],c['intercept'],c['weighted'],
             c['max_iter'],c['epsilon'],c['scaling'],c['tol'],c['c']]
    return ' '.join(map(str,control))+'\n'+csr_payload(x)+csr_payload(q)+\
        ' '.join(format(v,'.17g') for v in y)+'\n'+' '.join(format(v,'.17g') for v in w)+'\n'


def oracle(c,a):
    x,q,y,w=a
    if c['layout']=='dense':x=x.toarray();q=q.toarray()
    dual={0:False,1:True,2:'auto'}[c['dual']]
    model=LinearSVR(loss=['epsilon_insensitive','squared_epsilon_insensitive'][c['loss']],dual=dual,
        random_state=c['seed'],fit_intercept=bool(c['intercept']),epsilon=c['epsilon'],
        intercept_scaling=c['scaling'],C=c['c'],tol=c['tol'],max_iter=c['max_iter'])
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always');model.fit(x,y,sample_weight=w if c['weighted'] else None)
    return dict(coef=model.coef_,intercept=np.atleast_1d(model.intercept_),iterations=[model.n_iter_],
        train=model.predict(x),query=model.predict(q),dense_query=model.predict(q.toarray() if sp.issparse(q) else q),
        score=[model.score(x,y)],converged=[int(model.n_iter_<c['max_iter'])]),[str(w.message) for w in recorded]


def certificate(c,a,actual):
    x,q,y,w=a;theta=np.asarray(actual['coef']);beta=np.asarray(actual['beta'])
    expected_dual=c['dual']==1 or (c['dual']==2 and (not c['loss'] or c['n']<c['p']))
    assert (beta.size>0)==expected_dual
    if expected_dual:assert beta.shape==(c['n'],)
    if c['intercept']:theta=np.r_[theta,actual['intercept'][0]/c['scaling']]
    aug=sp.hstack([x,np.full((x.shape[0],1),c['scaling'])],format='csr') if c['intercept'] else x
    excess=np.maximum(0,np.abs(aug@theta-y)-c['epsilon'])
    primal=.5*theta@theta+c['c']*np.sum(w*(excess**2 if c['loss'] else excess))
    diagnostic=actual['diagnostics']
    np.testing.assert_allclose(diagnostic[0],primal,rtol=2e-11,atol=2e-11)
    if beta.size==0:
        residual=aug@theta-y
        grad=theta+2*c['c']*np.asarray(aug.T@(w*np.sign(residual)*excess)).ravel()
        initial=2*c['c']*np.asarray(aug.T@(-w*np.sign(y)*np.maximum(0,np.abs(y)-c['epsilon']))).ravel()
        bound=c['tol']*max(1,np.linalg.norm(initial))
        if actual['converged'][0]:assert np.linalg.norm(grad)<=max(1e-10,2*bound)
        return dict(primal=float(primal),gradient_norm=float(np.linalg.norm(grad)),bound=float(bound),passed=True,kind='primal')
    np.testing.assert_allclose(aug.T@beta,theta,rtol=2e-10,atol=2e-10)
    assert np.all(beta[w==0]==0)
    penalty=np.sum(beta[w>0]**2/(4*c['c']*w[w>0])) if c['loss'] else 0.
    dual=y@beta-c['epsilon']*np.sum(np.abs(beta))-.5*theta@theta-penalty
    np.testing.assert_allclose(diagnostic[1:3],[dual,primal-dual],rtol=2e-10,atol=2e-10)
    assert primal-dual>=-2e-10*max(1,abs(primal))
    if not c['loss']:assert np.all(np.abs(beta)<=c['c']*w+1e-12)
    g=np.asarray(aug@theta-y).ravel()
    if c['loss']:g[w>0]+=beta[w>0]/(2*c['c']*w[w>0])
    violation=[]
    for i in np.flatnonzero(w>0):
        if beta[i]==0:v=max(0,-g[i]-c['epsilon'],g[i]-c['epsilon'])
        elif not c['loss'] and beta[i]>=c['c']*w[i]:v=max(0,g[i]+c['epsilon'])
        elif not c['loss'] and beta[i]<=-c['c']*w[i]:v=max(0,-g[i]+c['epsilon'])
        else:v=abs(g[i]+np.sign(beta[i])*c['epsilon'])
        violation.append(v)
    kkt=max(violation,default=0.)
    np.testing.assert_allclose(diagnostic[3],kkt,rtol=2e-10,atol=2e-10)
    return dict(primal=float(primal),dual=float(dual),gap=float(primal-dual),kkt=kkt,passed=True,kind='dual')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--diagnostic',action='store_true');args=parser.parse_args()
    cases=list(fixtures());inputs=[arrays(c) for c in cases];expected=[oracle(c,a) for c,a in zip(cases,inputs)]
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),args.executable,root/'src/sklearn_linear_svm.f90',root/'src/sklearn_legacy_rng.f90',
        root/'app/reference_linear_svr_dual_driver.f90',Path(backend.__file__),
        Path(sklearn.__file__).parent/'svm/src/liblinear/linear.cpp',
        Path(sklearn.__file__).parent/'svm/src/newrand/newrand.h']
    hashes={str(p):sha(p) for p in files}
    run=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,inputs)),
        capture_output=True,text=True)
    assert run.returncode==0,(run.stdout[-1000:],run.stderr[-2000:])
    lines=iter(run.stdout.splitlines());actual=[];index=-1;checks=[]
    for line in lines:
        parts=line.split()
        if not parts:continue
        if parts[0]=='BEGIN':index+=1;actual.append({});assert int(parts[1])==index+1
        elif parts[0]=='END':
            assert actual[index].keys()==expected[index][0].keys()|{'diagnostics','beta'}
        else:
            field=parts[0];size=int(parts[1]);assert field not in actual[index]
            values=np.fromstring(next(lines),sep=' ');assert len(values)==size;actual[index][field]=values.tolist()
            if field not in ['diagnostics','beta']:
                want=np.asarray(expected[index][0][field]).ravel()
                passed=values.shape==want.shape and (np.array_equal(values,want) if field in ['iterations','converged'] else
                    np.allclose(values,want,rtol=2e-10,atol=2e-10,equal_nan=True))
                checks.append(dict(case=index,field=field,passed=bool(passed),
                    oracle='source_iteration_cap_status' if field=='converged' else 'source',
                    max_abs_error=float(np.max(np.abs(values-want),initial=0)) if values.shape==want.shape else None))
    assert index+1==len(cases)
    certificates=[certificate(c,a,v) for c,a,v in zip(cases,inputs,actual)]
    assert hashes=={p:sha(p) for p in hashes};gaps=[c for c in checks if not c['passed']]
    dual_gaps=[g for g in gaps if certificates[g['case']]['kind']=='dual']
    result=dict(cases=cases,checks=checks,gaps=gaps,dual_gaps=dual_gaps,certificates=certificates,actual=actual,
        expected=[{k:np.asarray(v).tolist() for k,v in e.items()} for e,_ in expected],
        source_warnings=[w for _,w in expected],hashes=hashes,strict_parity=not gaps,
        versions=dict(sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__))
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(len(cases),'fits;',len(checks),'groups;',len(gaps),'gaps;',len(certificates),'certificates')
    if gaps:print(gaps[:12])
    raise SystemExit(bool(gaps) and not args.diagnostic)


if __name__=='__main__':main()
