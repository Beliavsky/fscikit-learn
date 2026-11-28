"""Dense/CSR mixed SGDOneClassSVM sequences against installed sklearn."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import scipy
import sklearn
from sklearn.linear_model import SGDOneClassSVM
from compare_linear_svm import sparse_input
from compare_linear_svr_dual import csr_payload


def fixtures():
    for seed,layout,schedule,average,operation in itertools.product([0,7,42],
            ['csr','reverse','duplicates','zeros'],['constant','optimal','invscaling','adaptive'],[0,1,19],range(4)):
        yield dict(seed=seed,layout=layout,schedule=schedule,average=average,operation=operation)
    for seed,layout,edge in itertools.product([0,7,42],['csr','reverse','duplicates','zeros'],range(10)):
        yield dict(seed=seed,layout=layout,schedule='constant',average=1,operation=edge%4,edge=edge)


def sequence(c):
    rng=np.random.RandomState(c['seed']);p=5;n=21
    x=rng.normal(.4,.5,(n,p));x[rng.uniform(size=x.shape)<.5]=0;x[0]=0
    q=rng.normal(.4,.5,(9,p));q[rng.uniform(size=q.shape)<.5]=0
    w=rng.uniform(.2,2,n);w[::7]=0
    options=dict(nu=.35,eta0=.02,power_t=.3,tol=None,max_iter=8,n_iter_no_change=3,
        average=c['average'],random_state=c['seed'],shuffle=True,fit_intercept=True,learning_rate=c['schedule'])
    if c['schedule']=='adaptive':options.update(tol=1e-3,max_iter=80)
    if 'edge' in c:
        edge=c['edge']
        if edge==0:options['fit_intercept']=False
        if edge==1:w[:]=0
        if edge==2:x[:]=0
        if edge==3:options.update(eta0=10.)
        if edge==4:options.update(learning_rate='invscaling',eta0=.001,power_t=-.2)
        if edge==5:options['average']=10000
        if edge==6:options.update(nu=1.,average=0)
        if edge==7:options.update(nu=.1,shuffle=False);w[:]=1
        if edge==8:options.update(shuffle=False,eta0=.01)
        if edge==9:options.update(learning_rate='adaptive',eta0=.1,max_iter=200,tol=1e-2,n_iter_no_change=2)
    if c['operation']==0:stages=[(0,1,x,w),(1,1,x[:7],w[:7])]
    elif c['operation']==1:stages=[(1,int(i!=1),x[i*7:(i+1)*7],w[i*7:(i+1)*7]) for i in range(3)]
    elif c['operation']==2:stages=[(3,1,x,w),(2,0,x+.02,w),(1,1,x,w)]
    else:stages=[(4,1,x,w),(1,0,x,w),(2,1,x,w)]
    stages=[(op,rep,sparse_input(a,c['layout']),weights) for op,rep,a,weights in stages]
    return options,stages,sparse_input(q,c['layout'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--diagnostic',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),args.executable,root/'src/sklearn_sgd_one_class.f90',root/'src/sklearn_legacy_rng.f90',
        root/'app/reference_sgd_one_class_sparse_driver.f90',*[Path(sklearn.__file__).parent/p for p in
        ['linear_model/_stochastic_gradient.py','linear_model/_sgd_fast.pyx.tp','utils/_weight_vector.pyx.tp','utils/_seq_dataset.pyx.tp']]]
    hashes={str(p):sha(p) for p in files};cases=list(fixtures());expected=[];payload=str(len(cases))+'\n';stage_cases=[]
    for c in cases:
        options,stages,q=sequence(c);model=SGDOneClassSVM(**{k:v for k,v in options.items() if k!='n_iter_no_change'});model.n_iter_no_change=options['n_iter_no_change']
        payload+=' '.join(map(str,[len(stages),options['learning_rate'],options['nu'],options['eta0'],options['power_t'],
            -1 if options['tol'] is None else options['tol'],options['max_iter'],options['n_iter_no_change'],
            options['average'],options['random_state'],int(options['shuffle']),int(options['fit_intercept'])]))+'\n'
        for op,rep,x,w in stages:
            payload+=f'{op} {rep}\n'+csr_payload(x)+csr_payload(q)+' '.join(format(v,'.17g') for v in w)+'\n'
            train=x if rep else x.toarray();model.warm_start=op==2
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if op==1:model.partial_fit(train,sample_weight=w)
                elif op==3:model.fit(train,sample_weight=w,coef_init=np.full(x.shape[1],.1),offset_init=.2)
                elif op==4:model.fit_predict(train,sample_weight=w)
                else:model.fit(train,sample_weight=w)
            result=dict(coef=model.coef_,offset=model.offset_,decision=model.decision_function(q),samples=model.score_samples(q),
                predict=model.predict(q),dense_decision=model.decision_function(q.toarray()),dense_samples=model.score_samples(q.toarray()),
                dense_predict=model.predict(q.toarray()),state=[model.t_,model.n_iter_,model.n_features_in_])
            expected.append({k:np.asarray(v).ravel().tolist() for k,v in result.items()});stage_cases.append(c)
    run=subprocess.run([str(args.executable)],input=payload,capture_output=True,text=True)
    assert run.returncode==0,(run.stdout[-1000:],run.stderr[-2000:])
    lines=iter(run.stdout.splitlines());actual=[];checks=[];index=-1
    for line in lines:
        parts=line.split()
        if not parts:continue
        if parts[0]=='BEGIN':index+=1;actual.append({})
        elif parts[0]=='END':assert actual[index].keys()==expected[index].keys()
        else:
            field=parts[0];size=int(parts[1]);values=np.fromstring(next(lines),sep=' ');assert len(values)==size
            actual[index][field]=values.tolist();want=np.asarray(expected[index][field]);exact=field in ['state','predict','dense_predict']
            passed=values.shape==want.shape and (np.array_equal(values,want) if exact else np.allclose(values,want,atol=3e-8,rtol=3e-8))
            checks.append(dict(stage=index,field=field,passed=bool(passed),max_abs_error=float(np.max(np.abs(values-want),initial=0))))
    assert len(actual)==len(expected);assert hashes=={p:sha(p) for p in hashes};gaps=[c for c in checks if not c['passed']]
    report=dict(cases=cases,stages=stage_cases,checks=checks,gaps=gaps,actual=actual,expected=expected,hashes=hashes,
        strict_parity=not gaps,versions=dict(sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__))
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(len(cases),'sequences;',len(actual),'stages;',len(checks),'groups;',len(gaps),'gaps')
    if gaps:print(gaps[:12])
    raise SystemExit(bool(gaps) and not args.diagnostic)


if __name__=='__main__':main()
