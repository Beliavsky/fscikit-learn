"""Perceptron dense/CSR fits, penalties, warm/partial calls and validation stopping."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, subprocess, warnings
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.linear_model import Perceptron


def fixtures():
    for family,sparse,seed,penalty,shuffle,bias,workflow in itertools.product(
            [2,3],[0,1],[0,42],['none','l1','l2','elasticnet'],[0,1],[0,1],range(5)):
        if workflow==4 and family!=2:continue
        yield dict(family=family,sparse=sparse,seed=seed,rate='constant',shuffle=shuffle,average=workflow,
            tol=.001 if seed else -1.,n=39,p=5,loss='perceptron',penalty=penalty,alpha=.03,eta=.04,fraction=.3,
            weighted=seed==42,bias=bias,repeats=3 if workflow in [1,2] else 1,early=False,kind='ordinary',limit=40,
            class_weight=0 if workflow==2 else (seed+len(penalty))%3)
    for family,sparse,seed,penalty,bias,tol in itertools.product([2,3],[0,1],[0,42],
            ['none','l1','l2','elasticnet'],[0,1],[0.,.003,-1.]):
        yield dict(family=family,sparse=sparse,seed=seed,rate='constant',shuffle=1,average=1,tol=tol,
            n=45,p=5,loss='perceptron',penalty=penalty,alpha=.03,eta=.04,fraction=.3,weighted=True,
            bias=bias,repeats=2,early=True,kind='early',limit=40,class_weight=(seed+len(penalty))%3)
    for family,sparse in itertools.product([2,3],[0,1]):
        yield dict(family=family,sparse=sparse,seed=0,rate='constant',shuffle=1,average=0,tol=.001,
            n=45,p=5,loss='perceptron',penalty='none',alpha=.0001,eta=1.,fraction=.1,weighted=False,
            bias=True,repeats=1,early=False,kind='defaults',limit=1000,class_weight=0,patience=5)


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
                             'loss','penalty','rate','alpha','eta','tol','fraction','limit']]+[c.get('patience',3)]
    return ' '.join(str(int(v)) if isinstance(v,(bool,np.bool_)) else str(v) for v in header)+'\n'+vector(x)+vector(y)+vector(w)


def oracle(c,x,y,w):
    model=Perceptron(penalty=None if c['penalty']=='none' else c['penalty'],alpha=c['alpha'],eta0=c['eta'],
            l1_ratio=.15,tol=None if c['tol']<0 else c['tol'],max_iter=c['limit'],n_iter_no_change=c.get('patience',3),
            early_stopping=c['early'],validation_fraction=c['fraction'],random_state=c['seed'],
            fit_intercept=bool(c['bias']),shuffle=bool(c['shuffle']),warm_start=c['average']==1)
    if c['family']!=0 and c.get('class_weight',0):
        model.class_weight='balanced' if c['class_weight']==1 else dict(zip(np.unique(y),[.5,2.,1.3]))
    design=sp.csr_matrix(x) if c['sparse'] else x
    result=[]
    for _ in range(c['repeats']):
        if c['average']==2:model.partial_fit(design,y,classes=np.unique(y),sample_weight=w)
        else:
            extra={}
            if c['average'] in [3,4]:
                heads=1 if c['family']==2 else c['family']
                initial=.03*(np.arange(1,heads+1)[:,None]+np.arange(1,c['p']+1)[None,:])
                extra=dict(coef_init=initial if c['average']==3 else initial[0],intercept_init=np.full(heads,.1))
            model.fit(design,y,sample_weight=w,**extra)
        # Binary fit_binary stratifies signed labels; multiclass uses original
        # labels. Their sorted class membership is identical for these inputs.
        mask=model._make_validation_split(y,sample_mask=w>0)
        result.append({k:np.asarray(v).ravel(order='F').tolist() for k,v in dict(
            coef=model.coef_,intercept=model.intercept_,iterations=[model.n_iter_],t=[model.t_],
            mask=mask.astype(int),prediction=model.predict(design),classes=model.classes_,features=[model.n_features_in_],
            scores=model.decision_function(x),score=[model.score(sp.csr_matrix(x),y,sample_weight=w)]).items()})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--limit',type=int);parser.add_argument('--diagnostic',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),args.executable,root/'src/sklearn_sgd.f90',root/'src/sklearn_resampling.f90',
           root/'app/reference_perceptron_driver.f90',root/'src/sklearn_perceptron.f90',*[Path(sklearn.__file__).parent/p for p in
           ['linear_model/_perceptron.py','linear_model/_stochastic_gradient.py','linear_model/_sgd_fast.pyx.tp','utils/_weight_vector.pyx.tp','model_selection/_split.py']]]
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
                exact=field in ['iterations','t','mask','features','classes'] or (field=='prediction' and cases[i]['family']!=0)
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
