"""PA-I/PA-II raw CSR fitting, sparse queries, averaging and weighted scoring."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.linear_model import PassiveAggressiveRegressor,PassiveAggressiveClassifier
from compare_pa_seeded import fixtures,arrays,vector
from compare_kernel_svm_sparse import matrix,csr_payload


def oracle(c,x,q,y,weights):
    options=dict(C=c['c'],tol=None if c['tol']<0 else c['tol'],max_iter=c['limit'],n_iter_no_change=3,
                 shuffle=bool(c['shuffle']),fit_intercept=bool(c['bias']),average=c['average'] or False,random_state=c['seed'])
    before=[a.copy() for a in [x.indptr,x.indices,x.data,q.indptr,q.indices,q.data]]
    if c['family']==0:
        model=PassiveAggressiveRegressor(**options,epsilon=c['epsilon'],loss='squared_epsilon_insensitive' if c['mode'] else 'epsilon_insensitive').fit(x,y)
        values=dict(coef=model.coef_,intercept=model.intercept_,train=model.predict(x),query=model.predict(q))
    else:
        weight=None
        if c['weights']==1:weight='balanced'
        if c['weights']>=2:weight=dict(zip(np.unique(y),weights))
        model=PassiveAggressiveClassifier(**options,loss='squared_hinge' if c['mode'] else 'hinge',class_weight=weight).fit(x,y)
        values=dict(coef=model.coef_,intercept=model.intercept_,train=model.decision_function(x),query=model.decision_function(q),
                    classes=model.classes_,prediction=model.predict(q))
    values.update(iterations=[model.n_iter_],t=[model.t_],features=[model.n_features_in_],
                  score=[model.score(x,y),model.score(x,y,sample_weight=np.arange(len(y))/len(y))],
                  dense_query=model.predict(q.toarray()),preserved=[1])
    assert all(np.array_equal(a,b) for a,b in zip(before,[x.indptr,x.indices,x.data,q.indptr,q.indices,q.data]))
    return {k:np.asarray(v).ravel(order='F').tolist() for k,v in values.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--limit',type=int)
    parser.add_argument('--diagnostic',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    paths=[Path(__file__),args.executable,root/'src/sklearn_linear_model.f90',root/'src/sklearn_sparse.f90',
           root/'app/reference_pa_csr_driver.f90',root/'validation/compare_pa_seeded.py',root/'validation/compare_kernel_svm_sparse.py',
           *[Path(sklearn.__file__).parent/p for p in ['linear_model/_passive_aggressive.py','linear_model/_stochastic_gradient.py',
               'linear_model/_sgd_fast.pyx.tp','utils/_seq_dataset.pyx.tp','utils/_weight_vector.pyx.tp']]]
    hashes={str(p):sha(p) for p in paths};cases=[dict(c,layout=layout) for c in fixtures() for layout in range(4)][:args.limit]
    for family,mode,layout,kind in itertools.product([0,2,3],[0,1],range(4),['overflow_norm','overflow_sum']):
        cases.append(dict(family=family,mode=mode,layout=layout,kind=kind,n=12,p=4,c=1.,epsilon=.1,tol=-1.,
                          average=1,limit=1,shuffle=True,bias=True,seed=7,weights=0))
    if args.limit is not None:cases=cases[:args.limit]
    expected=[];inputs=[]
    for c in cases:
        x,q,y,w=arrays(c)
        if c['kind'].startswith('overflow_'):
            x[:]=1e160 if c['kind']=='overflow_norm' else 1e154;q[:]=x[0]
        x=matrix(x,c['layout']);q=matrix(q,c['layout'])
        with warnings.catch_warnings():warnings.simplefilter('ignore');expected.append(oracle(c,x,q,y,w))
        header=[c['n'],c['p'],q.shape[0],*[c[k] for k in ['family','mode','shuffle','bias','average','seed','weights','c','epsilon','tol','limit']],3]
        text=' '.join(str(int(v)) if isinstance(v,(bool,np.bool_)) else str(v) for v in header)+'\n'+csr_payload(x)+csr_payload(q)+vector(y)
        if c['family'] and c['weights']>=2:text+=vector(w)
        inputs.append(text)
    run=subprocess.run([str(args.executable.resolve())],input=str(len(cases))+'\n'+''.join(inputs),capture_output=True,text=True)
    assert run.returncode==0,(run.stdout[-1000:],run.stderr[-1800:])
    lines=iter(run.stdout.splitlines());actual=[];checks=[];case=-1
    for line in lines:
        p=line.split()
        if not p:continue
        if p[0]=='BEGIN':actual.append({});case+=1
        elif p[0]=='END':assert actual[case].keys()==expected[case].keys()
        else:
            values=np.fromstring(next(lines),sep=' ');assert len(values)==int(p[1]);actual[case][p[0]]=values.tolist()
    assert len(actual)==len(cases)
    for i,(a,e) in enumerate(zip(actual,expected)):
        for field,want in e.items():
            value=np.asarray(a[field]);want=np.asarray(want)
            exact=field in ['classes','prediction','iterations','t','features','preserved'] or (field=='dense_query' and cases[i]['family']!=0)
            passed=value.shape==want.shape and (np.array_equal(value,want) if exact else np.allclose(value,want,atol=3e-8,rtol=3e-8))
            checks.append(dict(case=i,field=field,oracle='native_storage_contract' if field=='preserved' else 'source',passed=bool(passed),
                max_abs_error=float(np.max(np.abs(value-want),initial=0)) if value.shape==want.shape else None))
    assert hashes=={p:sha(p) for p in hashes};gaps=[c for c in checks if not c['passed']]
    args.report.write_text(json.dumps(dict(cases=cases,expected=expected,actual=actual,checks=checks,gaps=gaps,
                                         hashes=hashes,sklearn=sklearn.__version__,strict_parity=not gaps),indent=2)+'\n')
    print(len(cases),'CSR fits;',len(checks),'groups;',len(gaps),'gaps',flush=True)
    if gaps:print(gaps[:20])
    raise SystemExit(bool(gaps) and not args.diagnostic)


if __name__=='__main__':main()
