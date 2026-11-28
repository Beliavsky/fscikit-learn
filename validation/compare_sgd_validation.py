"""Dense/CSR SGD validation splits, weighted stopping, averaging and warm fits."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, subprocess, warnings
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.linear_model import SGDRegressor, SGDClassifier


def fixtures():
    for family,sparse,seed,rate,shuffle,average,tol in itertools.product(
            [0,2,3],[0,1],[7,42],['constant','invscaling','optimal','adaptive'],[0,1],[0,1,100],[0.,.003,-1.]):
        yield dict(family=family,sparse=sparse,seed=seed,rate=rate,shuffle=shuffle,average=average,tol=tol,
                   n=39,p=5,loss=['squared_error','huber'][seed%2] if family==0 else ['hinge','log_loss'][seed%2],
                   penalty=['l2','l1','elasticnet'][average%3],alpha=.03,eta=.04,fraction=.3,
                   weighted=seed==42,bias=average!=100,repeats=1,early=True,kind='ordinary',limit=60)
    for family,sparse,seed,kind in itertools.product([0,2,3],[0,1],[7,42],['warm','large_alpha','large_validation','disabled']):
        yield dict(family=family,sparse=sparse,seed=seed,rate='adaptive' if kind=='warm' else 'constant',
                   shuffle=True,average=1 if kind=='warm' else 0,tol=.001,n=45,p=7,
                   loss='squared_error' if family==0 else 'log_loss',penalty='l2',
                   alpha=1.3 if kind=='large_alpha' else .1,eta=.05,fraction=.8 if kind=='large_validation' else .2,
                   weighted=True,bias=True,repeats=2 if kind=='warm' else 1,early=kind!='disabled',kind=kind,limit=75)
    for sparse,seed,kind in itertools.product([0,1],[7,42],['constant_target','single_validation','late_average']):
        yield dict(family=0,sparse=sparse,seed=seed,rate='constant',shuffle=True,average=400 if kind=='late_average' else 0,
                   tol=.01,n=24,p=3,loss='squared_error',penalty='l2',alpha=.1,eta=.03,
                   fraction=.9 if kind=='late_average' else .01 if kind=='single_validation' else .3,
                   weighted=True,bias=True,repeats=1,early=True,kind=kind,limit=30)

    for family,sparse,seed,loss in itertools.product([0,2,3],[0,1],[7,42],
            ['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive','hinge','squared_hinge','perceptron','log_loss','modified_huber']):
        if family==0 and loss not in ['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']:continue
        for weight_mode in ([0] if family==0 else [1,2]):
            yield dict(family=family,sparse=sparse,seed=seed,rate='adaptive',shuffle=True,average=1,tol=.003,
                       n=45,p=5,loss=loss,penalty='elasticnet',alpha=.05,eta=.03,fraction=.35,weighted=True,
                       bias=True,repeats=1,early=True,kind='loss_and_class_weights',limit=65,class_weight=weight_mode)

    for sparse,loss in itertools.product([0,1],['squared_error','squared_epsilon_insensitive']):
        yield dict(family=0,sparse=sparse,seed=7,rate='constant',shuffle=True,average=0,tol=.001,n=39,p=5,
                   loss=loss,penalty='l2',alpha=.03,eta=.04,fraction=.3,weighted=False,bias=True,
                   repeats=1,early=True,kind='huge_target',limit=5)


def arrays(c):
    rng=np.random.RandomState(c['seed']);x=rng.normal(size=(c['n'],c['p']))*.5
    x[rng.uniform(size=x.shape)<.3]=0
    if c['family']==0:
        y=x[:,0]+.7*x[:,1]-.2*x[:,2]+rng.normal(size=c['n'])*.2
        if c['kind']=='constant_target':y[:]=2
    else:
        y=np.arange(c['n'])%c['family'];rng.shuffle(y);y=np.array([-5,7,21])[y]
    if c['kind']=='huge_target':y*=1e160
    w=rng.uniform(.2,2,c['n']) if c['weighted'] else np.ones(c['n'])
    if c['weighted']:w[::11]=0
    return x,y,w


def vector(a):
    return ' '.join(format(v,'.17g') for v in np.asarray(a).ravel(order='F'))+'\n'


def payload(c,x,y,w):
    header=[c.get(k,0) for k in ['n','p','family','sparse','seed','average','shuffle','bias','repeats','early','class_weight',
                             'loss','penalty','rate','alpha','eta','tol','fraction','limit']]+[3]
    return ' '.join(str(int(v)) if isinstance(v,(bool,np.bool_)) else str(v) for v in header)+'\n'+vector(x)+vector(y)+vector(w)


def oracle(c,x,y,w):
    model=(SGDRegressor if c['family']==0 else SGDClassifier)(loss=c['loss'],penalty=c['penalty'],learning_rate=c['rate'],
            alpha=c['alpha'],eta0=c['eta'],power_t=.25,epsilon=.1,l1_ratio=.15,tol=None if c['tol']<0 else c['tol'],
            max_iter=c['limit'],n_iter_no_change=3,early_stopping=c['early'],validation_fraction=c['fraction'],
            random_state=c['seed'],average=c['average'] or False,fit_intercept=bool(c['bias']),shuffle=bool(c['shuffle']),warm_start=c['repeats']>1)
    if c['family']!=0 and c.get('class_weight',0):
        model.class_weight='balanced' if c['class_weight']==1 else dict(zip(np.unique(y),[.5,2.,1.3]))
    design=sp.csr_matrix(x) if c['sparse'] else x
    result=[]
    for _ in range(c['repeats']):
        model.fit(design,y,sample_weight=w)
        # Binary fit_binary stratifies signed labels; multiclass uses original
        # labels. Their sorted class membership is identical for these inputs.
        mask=model._make_validation_split(y,sample_mask=w>0)
        result.append({k:np.asarray(v).ravel(order='F').tolist() for k,v in dict(
            coef=model.coef_,intercept=model.intercept_,iterations=[model.n_iter_],t=[model.t_],
            mask=mask.astype(int),prediction=model.predict(design)).items()})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--limit',type=int);parser.add_argument('--diagnostic',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),args.executable,root/'src/sklearn_sgd.f90',root/'src/sklearn_resampling.f90',
           root/'app/reference_sgd_validation_driver.f90',*[Path(sklearn.__file__).parent/p for p in
           ['linear_model/_stochastic_gradient.py','linear_model/_sgd_fast.pyx.tp','utils/_weight_vector.pyx.tp','model_selection/_split.py']]]
    hashes={str(p):sha(p) for p in files};cases=list(fixtures())[:args.limit];expected=[];inputs=[]
    for c in cases:
        x,y,w=arrays(c);inputs.append(payload(c,x,y,w))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');expected.append(oracle(c,x,y,w))
    run=subprocess.run([str(args.executable.resolve())],input=str(len(cases))+'\n'+''.join(inputs),capture_output=True,text=True)
    assert run.returncode==0,(run.stdout[-1500:],run.stderr[-2500:])
    lines=iter(run.stdout.splitlines());actual=[];current=None;case=-1;checks=[]
    for line in lines:
        parts=line.split()
        if not parts:continue
        if parts[0]=='BEGIN':actual.append([]);case+=1;current={}
        elif parts[0]=='STEP':actual[case].append(current);current={}
        elif parts[0]=='END':assert len(actual[case])==len(expected[case])
        else:
            values=np.fromstring(next(lines),sep=' ');assert len(values)==int(parts[1]);current[parts[0]]=values.tolist()
    assert len(actual)==len(cases)
    for i,(left,right) in enumerate(zip(actual,expected)):
        for step,(a,e) in enumerate(zip(left,right)):
            assert a.keys()==e.keys()
            for field in e:
                value=np.asarray(a[field]);want=np.asarray(e[field])
                exact=field in ['iterations','t','mask'] or (field=='prediction' and cases[i]['family']!=0)
                passed=value.shape==want.shape and (np.array_equal(value,want) if exact else np.allclose(value,want,atol=3e-9,rtol=3e-9))
                checks.append(dict(case=i,step=step,field=field,passed=bool(passed),
                    max_abs_error=float(np.max(np.abs(value-want),initial=0)) if value.shape==want.shape else None))
    gaps=[c for c in checks if not c['passed']];assert hashes=={p:sha(p) for p in hashes}
    result=dict(cases=cases,checks=checks,gaps=gaps,actual=actual,expected=expected,hashes=hashes,
                strict_parity=not gaps,sklearn=sklearn.__version__,project_strict_source_parity=False)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(len(cases),'cases;',len(checks),'checks;',len(gaps),'gaps',flush=True)
    if gaps:print(gaps[:20])
    raise SystemExit(bool(gaps) and not args.diagnostic)


if __name__=='__main__':main()
