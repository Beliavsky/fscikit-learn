"""Passive-aggressive private seeding, class weights, averaging and stopping."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.linear_model import PassiveAggressiveRegressor,PassiveAggressiveClassifier


def fixtures():
    for family,mode,shuffle,bias,average,seed,tol in itertools.product([0,2,3],[0,1],[0,1],[0,1],[0,1,7,500],[0,42,2026],[-1.,.003]):
        for weights in ([0] if family==0 else [0,1,2]):
            yield dict(family=family,mode=mode,shuffle=shuffle,bias=bias,average=average,seed=seed,tol=tol,
                       weights=weights,n=23,p=5,c=.3,epsilon=.1,limit=9 if tol<0 else 100,kind='ordinary')
    for family,mode,seed,p,weights in itertools.product([0,2,3],[0,1],[7,42],[1,17,43],[0,3,4]):
        if family==0 and weights:continue
        yield dict(family=family,mode=mode,shuffle=True,bias=True,average=7,seed=seed,tol=.001,
                   weights=weights,n=27,p=p,c=.7,epsilon=0.,limit=100,kind='wide')
    for family,mode,shuffle,average in itertools.product([0,2,3],[0,1],[0,1],[0,1,7,500]):
        yield dict(family=family,mode=mode,shuffle=shuffle,bias=True,average=average,seed=0,tol=.003,
                   weights=0,n=12,p=3,c=.4,epsilon=0.,limit=40,kind='zero')


def arrays(c):
    rng=np.random.RandomState(c['seed']);x=rng.normal(size=(c['n'],c['p']))*.5;q=rng.normal(size=(7,c['p']))*.5
    x[::5]=0
    if c['kind']=='zero':x[:]=0
    if c['family']==0:y=x@np.linspace(-.4,1.3,c['p'])+.2+rng.normal(size=c['n'])*.03
    else:
        labels=np.arange(c['n'])%c['family'];labels[1:5]=0;rng.shuffle(labels);y=np.array([-5,7,21])[labels]
    weights=np.array([.5,2.,1.3])[:c['family']]
    if c['weights']==3:weights[0]=0
    if c['weights']==4:weights[0]=-.5
    return x,q,y,weights


def vector(a):return ' '.join(format(v,'.17g') for v in np.asarray(a).ravel(order='F'))+'\n'


def payload(c,x,q,y,weights):
    header=[c['n'],c['p'],len(q),*[c[k] for k in ['family','mode','shuffle','bias','average','seed','weights','c','epsilon','tol','limit']],3]
    result=' '.join(str(int(v)) if isinstance(v,(bool,np.bool_)) else str(v) for v in header)+'\n'+vector(x)+vector(q)+vector(y)
    if c['family'] and c['weights']>=2:result+=vector(weights)
    return result


def oracle(c,x,q,y,weights):
    options=dict(C=c['c'],tol=None if c['tol']<0 else c['tol'],max_iter=c['limit'],n_iter_no_change=3,
                 shuffle=bool(c['shuffle']),fit_intercept=bool(c['bias']),average=c['average'] or False,random_state=c['seed'])
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
    values.update(iterations=[model.n_iter_],t=[model.t_],features=[model.n_features_in_],score=[model.score(x,y)])
    return {k:np.asarray(v).ravel(order='F').tolist() for k,v in values.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--limit',type=int)
    parser.add_argument('--diagnostic',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    paths=[Path(__file__),args.executable,root/'src/sklearn_linear_model.f90',root/'src/sklearn_legacy_rng.f90',root/'app/reference_pa_seeded_driver.f90',
           *[Path(sklearn.__file__).parent/p for p in ['linear_model/_passive_aggressive.py','linear_model/_stochastic_gradient.py',
               'linear_model/_sgd_fast.pyx.tp','utils/_seq_dataset.pyx.tp','utils/_random.pxd','utils/_weight_vector.pyx.tp']]]
    hashes={str(p):sha(p) for p in paths};cases=list(fixtures())[:args.limit];expected=[];inputs=[]
    for c in cases:
        data=arrays(c);inputs.append(payload(c,*data))
        with warnings.catch_warnings():warnings.simplefilter('ignore');expected.append(oracle(c,*data))
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
            value=np.asarray(a[field]);want=np.asarray(want);exact=field in ['classes','prediction','iterations','t','features']
            passed=value.shape==want.shape and (np.array_equal(value,want) if exact else np.allclose(value,want,atol=3e-8,rtol=3e-8))
            checks.append(dict(case=i,field=field,passed=bool(passed),max_abs_error=float(np.max(np.abs(value-want),initial=0)) if value.shape==want.shape else None))
    assert hashes=={p:sha(p) for p in hashes};gaps=[c for c in checks if not c['passed']]
    args.report.write_text(json.dumps(dict(cases=cases,expected=expected,actual=actual,checks=checks,gaps=gaps,
                                         hashes=hashes,sklearn=sklearn.__version__,strict_parity=not gaps),indent=2)+'\n')
    print(len(cases),'fits;',len(checks),'groups;',len(gaps),'gaps',flush=True)
    if gaps:print(gaps[:20])
    raise SystemExit(bool(gaps) and not args.diagnostic)


if __name__=='__main__':main()
