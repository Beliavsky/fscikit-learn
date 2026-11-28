"""Dense/CSR passive-aggressive streaming traces against installed sklearn."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.linear_model import PassiveAggressiveRegressor,PassiveAggressiveClassifier
from compare_kernel_svm_sparse import matrix,csr_payload
from compare_pa_seeded import vector


def fixtures():
    for family,mode,pattern,bias,average,seed,shuffle in itertools.product([0,2,3],[0,1],range(3),[0,1],[0,1,7,50],[0,42],[0,1]):
        for weights in ([0] if family==0 else [0,2,4]):
            yield dict(family=family,mode=mode,pattern=pattern,bias=bias,average=average,seed=seed,shuffle=shuffle,
                       weights=weights,layout=(mode+average+shuffle+seed)%4,p=5,workflow='stream')
    for family,mode,pattern,workflow,average in itertools.product([0,2,3],[0,1],range(3),
            ['fit_then_stream','refit','threshold_changes','toggle_average'],[1,7]):
        yield dict(family=family,mode=mode,pattern=pattern,bias=1,average=average,seed=7,shuffle=1,
                   weights=2 if family else 0,layout=2,p=7,workflow=workflow)
    for family,mode,pattern in itertools.product([2,3],[0,1],range(3)):
        yield dict(family=family,mode=mode,pattern=pattern,bias=1,average=7,seed=42,shuffle=1,
                   weights=1,layout=3,p=3,workflow='balanced_after_fit')


def sequence(c):
    rng=np.random.RandomState(c['seed']);p=c['p'];q=rng.normal(size=(7,p))*.3;sq=matrix(q,c['layout'])
    sizes=[1,6,2,11,3]
    if c['workflow'] in ['fit_then_stream','balanced_after_fit']:sizes=[12,7,6,9,6]
    if c['workflow']=='refit':sizes=[1,6,12,3,6]
    steps=[];offset=0
    for step,n in enumerate(sizes):
        x=rng.normal(size=(n,p))*.3;x[rng.uniform(size=x.shape)<.35]=0;x[::5]=0
        y=x@np.linspace(-.4,1.3,p)+.2 if not c['family'] else np.array([-5,7,21])[(np.arange(n)+offset)%c['family']]
        offset+=n
        fmt=c['pattern'] if c['pattern']<2 else step%2
        action=int((step==0 and c['workflow'] in ['fit_then_stream','balanced_after_fit']) or (step==2 and c['workflow']=='refit'))
        avg=c['average']
        if c['workflow']=='threshold_changes':avg=[1,100,3,60,1][step]
        if c['workflow']=='toggle_average':avg=[1,0,0,7,1][step]
        steps.append(dict(x=matrix(x,(c['layout']+step)%4) if fmt else x,y=y,fmt=fmt,action=action,average=avg,classes=step%2==0))
    declared=np.array([-5,7,21])[:c['family']]
    if c['family']:declared=np.r_[declared[::-1],declared[0]]
    return q,sq,declared,steps


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--limit',type=int)
    parser.add_argument('--diagnostic',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    paths=[Path(__file__),args.executable,root/'src/sklearn_linear_model.f90',root/'app/reference_pa_partial_driver.f90',
           root/'validation/compare_pa_seeded.py',root/'validation/compare_kernel_svm_sparse.py',
           *[Path(sklearn.__file__).parent/p for p in ['linear_model/_passive_aggressive.py','linear_model/_stochastic_gradient.py',
               'linear_model/_sgd_fast.pyx.tp','utils/_seq_dataset.pyx.tp','utils/_weight_vector.pyx.tp','utils/multiclass.py']]]
    hashes={str(p):sha(p) for p in paths};cases=list(fixtures())[:args.limit];expected=[];inputs=[]
    for case,c in enumerate(cases):
        q,sq,classes,steps=sequence(c)
        options=dict(C=.3,loss=('squared_epsilon_insensitive' if c['mode'] else 'epsilon_insensitive') if not c['family'] else
            ('squared_hinge' if c['mode'] else 'hinge'),tol=None,max_iter=5,shuffle=bool(c['shuffle']),
            fit_intercept=bool(c['bias']),average=c['average'] or False,random_state=c['seed'])
        if not c['family']:options['epsilon']=.1
        weights=np.array([.5,2.,1.3])[:c['family']]
        if c['weights']==4:weights[0]=-.5
        if c['family']:
            if c['weights']==1:options['class_weight']='balanced'
            elif c['weights']>=2:options['class_weight']=dict(zip(np.unique(classes),weights))
        model=(PassiveAggressiveClassifier if c['family'] else PassiveAggressiveRegressor)(**options)
        header=[*[c[k] for k in ['family','mode','shuffle','bias','average','seed','weights']],.3,.1,len(steps),c['p'],len(q),len(classes)]
        payload=' '.join(map(str,header))+'\n'+vector(classes)+vector(q)+csr_payload(sq)
        if c['weights']>=2:payload+=vector(weights)
        outputs=[]
        for step,a in enumerate(steps):
            x,y=a['x'],a['y'];before=[t.copy() for t in [x.data,x.indices,x.indptr]] if a['fmt'] else [x.copy()]
            model.average=a['average'] or False
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if a['action']:model.fit(x,y)
                elif c['family']:model.partial_fit(x,y,**({'classes':classes} if a['classes'] else {}))
                else:model.partial_fit(x,y)
                v=dict(coef=model.coef_,intercept=model.intercept_,iterations=[model.n_iter_],t=[model.t_],
                       features=[model.n_features_in_],preserved=[1])
                if c['family']:
                    v.update(train=model.decision_function(x),query_dense=model.decision_function(q),query_csr=model.decision_function(sq),
                             classes=model.classes_,prediction=model.predict(sq))
                else:v.update(train=model.predict(x),query_dense=model.predict(q),query_csr=model.predict(sq))
            after=[x.data,x.indices,x.indptr] if a['fmt'] else [x]
            assert all(np.array_equal(a,b) for a,b in zip(before,after))
            outputs.append({k:np.asarray(v).ravel(order='F').tolist() for k,v in v.items()})
            payload+=' '.join(map(str,[a['fmt'],len(y),a['action'],a['average'],int(a['classes'])]))+'\n'
            payload+=csr_payload(x) if a['fmt'] else vector(x)
            payload+=vector(y)
        expected.append(outputs);inputs.append(payload)
    run=subprocess.run([str(args.executable.resolve())],input=str(len(cases))+'\n'+''.join(inputs),capture_output=True,text=True)
    assert run.returncode==0,(run.stdout[-1600:],run.stderr[-2500:])
    lines=iter(run.stdout.splitlines());actual=[];current={};case=-1
    for line in lines:
        p=line.split()
        if not p:continue
        if p[0]=='BEGIN':case+=1;actual.append([]);current={}
        elif p[0]=='STEP':actual[case].append(current);current={}
        elif p[0]=='END':assert len(actual[case])==len(expected[case])
        else:
            values=np.fromstring(next(lines),sep=' ');assert len(values)==int(p[1]);current[p[0]]=values.tolist()
    assert len(actual)==len(cases);checks=[]
    for case,(a,e) in enumerate(zip(actual,expected)):
        for step,(left,right) in enumerate(zip(a,e)):
            assert left.keys()==right.keys()
            for field,want in right.items():
                value=np.asarray(left[field]);want=np.asarray(want);exact=field in ['iterations','t','classes','prediction','features','preserved']
                passed=value.shape==want.shape and (np.array_equal(value,want) if exact else np.allclose(value,want,atol=3e-8,rtol=3e-8))
                checks.append(dict(case=case,step=step,field=field,passed=bool(passed),
                    max_abs_error=float(np.max(np.abs(value-want),initial=0)) if value.shape==want.shape else None))
    assert hashes=={p:sha(p) for p in hashes};gaps=[c for c in checks if not c['passed']]
    args.report.write_text(json.dumps(dict(cases=cases,expected=expected,actual=actual,checks=checks,gaps=gaps,
                                         hashes=hashes,sklearn=sklearn.__version__,strict_parity=not gaps),indent=2)+'\n')
    print(len(cases),'sequences;',sum(map(len,expected)),'stages;',len(checks),'groups;',len(gaps),'gaps',flush=True)
    if gaps:print(gaps[:20])
    raise SystemExit(bool(gaps) and not args.diagnostic)


if __name__=='__main__':main()
