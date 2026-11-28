"""Signed sample/class weights in SGD and Perceptron dense/CSR fitting and streaming."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, subprocess, warnings
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.linear_model import SGDRegressor, SGDClassifier, Perceptron


def fixtures():
    for kind,family in [('reg',0),('clf',2),('clf',3),('pc',2),('pc',3)]:
        losses=['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'] if kind=='reg' else ['perceptron'] if kind=='pc' else [
            'hinge','squared_hinge','perceptron','log_loss','modified_huber','squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']
        for sparse,seed,loss,penalty,weight,workflow in itertools.product([0,1],[0,42],losses,
                ['none','l1','l2','elasticnet'],['mixed','negative','zeros','zero_mass'],range(3)):
            average=0 if kind=='pc' else [0,1,7][(len(loss)+len(weight)+seed)%3]
            cw=0 if kind=='reg' else (len(loss)+len(weight)+workflow)%5
            yield dict(kind=kind,family=family,sparse=sparse,seed=seed,loss=loss,penalty=penalty,
                rate='fixed' if kind=='pc' else ['constant','invscaling','optimal','adaptive'][['none','l1','l2','elasticnet'].index(penalty)],
                shuffle=True,average=average,tol=-1.,n=40,p=5,alpha=.02,eta=.005,fraction=.3,
                weighted=True,bias=not(workflow==2 and family and average>0),repeats=3 if workflow else 1,
                early=False,weight=weight,workflow=workflow,cw=cw,class_weight=2 if family else 0,limit=7)
        for sparse,seed,loss,weight,cw in itertools.product([0,1],[0,42],losses,
                ['mixed','held_negative','training_zero'],[0,2,4] if family else [0]):
            yield dict(kind=kind,family=family,sparse=sparse,seed=seed,loss=loss,penalty='elasticnet',
                rate='fixed' if kind=='pc' else 'adaptive',shuffle=True,average=0 if kind=='pc' else 1,
                tol=.003,n=40,p=5,alpha=.02,eta=.005,fraction=.3,weighted=True,bias=True,
                repeats=2,early=True,weight=weight,workflow=1,cw=cw,class_weight=2 if family else 0,limit=25)


def class_weights(c):
    return np.array([[1.,1.,1.],[0.,1.,1.],[-.5,1.,0.],[-1.,-2.,-3.],[0.,0.,0.]][c['cw']])[:c['family']]


def arrays(c):
    rng=np.random.RandomState(c['seed']);x=rng.normal(size=(c['n'],c['p']))*.15
    x[rng.uniform(size=x.shape)<.3]=0
    y=x[:,0]+.7*x[:,1]-.2*x[:,2]+rng.normal(size=c['n'])*.03 if not c['family'] else np.array([-5,7,21])[np.arange(c['n'])%c['family']]
    w=rng.uniform(.25,1.5,c['n'])
    if c['weight']=='mixed':w[::4]*=-.5;w[::11]=0
    elif c['weight']=='negative':w*=-1
    elif c['weight']=='zeros':w[:]=0
    elif c['weight']=='zero_mass':w[:]=np.tile([1.,-1.],c['n']//2)
    else:
        model=(SGDClassifier if c['family'] else SGDRegressor)(early_stopping=True,validation_fraction=c['fraction'],random_state=c['seed'])
        mask=model._make_validation_split(y,np.ones(len(y),dtype=bool))
        if c['weight']=='held_negative':w[mask]=-2.;w[np.flatnonzero(mask)[0]]=1.
        elif c['weight']=='training_zero':w[~mask]=0
        else:raise AssertionError(c)
    return x,y,w


def vector(a):
    return ' '.join(format(v,'.17g') for v in np.asarray(a).ravel(order='F'))+'\n'


def payload(c,x,y,w):
    header=[c.get(k,0) for k in ['n','p','family','sparse','seed','average','shuffle','bias','repeats','early','class_weight',
                             'loss','penalty','rate','alpha','eta','tol','fraction','limit']]+[3,c['workflow']]
    return ' '.join(str(int(v)) if isinstance(v,(bool,np.bool_)) else str(v) for v in header)+'\n'+vector(x)+vector(y)+vector(w)+(vector(class_weights(c)) if c['family'] else '')


def oracle(c,x,y,w):
    options=dict(penalty=None if c['penalty']=='none' else c['penalty'],alpha=c['alpha'],eta0=c['eta'],
            l1_ratio=.15,tol=None if c['tol']<0 else c['tol'],max_iter=c['limit'],n_iter_no_change=3,
            early_stopping=c['early'],validation_fraction=c['fraction'],random_state=c['seed'],
            fit_intercept=bool(c['bias']),shuffle=bool(c['shuffle']),warm_start=c['workflow']==1)
    if c['kind']=='pc':model=Perceptron(**options)
    else:model=(SGDRegressor if not c['family'] else SGDClassifier)(loss=c['loss'],learning_rate=c['rate'],
            power_t=.25,epsilon=.1,average=c['average'] or False,**options)
    if c['family']:model.class_weight=dict(zip(np.unique(y),class_weights(c)))
    design=sp.csr_matrix(x) if c['sparse'] else x
    result=[]
    for _ in range(c['repeats']):
        if c['workflow']==2:
            model.partial_fit(design,y,sample_weight=w,**({'classes':np.unique(y)} if c['family'] else {}))
        else:model.fit(design,y,sample_weight=w)
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
           root/'app/reference_sgd_weights_driver.f90',root/'src/sklearn_perceptron.f90',root/'src/sklearn_subset_metrics.f90',root/'src/sklearn_variance_scores.f90',*[Path(sklearn.__file__).parent/p for p in
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
