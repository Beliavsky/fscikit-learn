"""MLP L-BFGS-B fits and pointwise loss/gradient contracts against sklearn 1.6.1."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
from unittest.mock import patch
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np
import scipy,sklearn
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.neural_network import _multilayer_perceptron as mlp
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def query_fields(model,q,kind):
    fields=dict(prediction=model.predict(q))
    if kind:
        fields['classes']=model.classes_.copy()
        fields['probability']=model.predict_proba(q).ravel(order='F')
        with np.errstate(divide='ignore'):fields['log_probability']=model.predict_log_proba(q).ravel(order='F')
    return fields

class InitializationProbe:
    direction=1
    def _init_coef(self,*args):
        w,b=super()._init_coef(*args)
        return np.nextafter(w,self.direction*np.inf),np.nextafter(b,self.direction*np.inf)
class PerturbedRegressor(InitializationProbe,MLPRegressor):pass
class PerturbedClassifier(InitializationProbe,MLPClassifier):pass

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--profile',default='checked');parser.add_argument('--report',type=Path)
    parser.add_argument('--diagnose',action='store_true');parser.add_argument('--fixture',type=int)
    args=parser.parse_args();exe=args.executable.resolve();records=[];inputs=[]
    limits=[(1,15000,1e-8),(2,15000,1e-8),(7,15000,1e-8),(200,15000,1e-6),
            (200,1,1e-8),(200,3,1e-8),(200,15,1e-8),(200,15000,1000.),(200,15000,0.),(200,15000,1e-9)]
    for index,(seed,kind,activation,hidden,limit) in enumerate(itertools.product(
            [0,7,42],range(3),['identity','logistic','tanh','relu'],[(),(4,),(5,3)],range(len(limits)))):
        if args.fixture is not None and index!=args.fixture:continue
        n=19;p=3;nq=9;rng=np.random.RandomState(seed+81)
        x=rng.normal(0,.5,(n,p));q=rng.normal(0,.6,(nq,p))
        if index%19==0:x[:]=0
        elif index%17==0:x[:,2]=0
        elif index%13==0:x[:,2]=x[:,0]
        signal=np.sin(x[:,0])+x[:,1]*.3-x[:,2]**2*.2
        if kind==0:y=signal
        elif kind==1:y=3+7*(np.arange(n)%2 if np.all(x==0) else signal>np.median(signal))
        else:y=3+7*(np.arange(n)%3 if np.all(x==0) else np.argmax(np.column_stack([x[:,0],x[:,1],-x[:,0]-x[:,1]]),axis=1))
        # Ensure all declared classes even in rank-degenerate datasets.
        if kind:y[:kind+1]=3+7*np.arange(kind+1)
        epochs,maxfun,tol=limits[limit];alpha=[0.,.001,.3][index%3]
        controls=dict(hidden_layer_sizes=hidden,activation=activation,solver='lbfgs',random_state=seed,
                      max_iter=epochs,max_fun=maxfun,tol=tol,alpha=alpha)
        model=(MLPRegressor if kind==0 else MLPClassifier)(**controls)
        capture=[];minimize=scipy.optimize.minimize
        def optimize(*a,**kw):
            result=minimize(*a,**kw);capture.append(result);return result
        with patch.object(mlp.scipy.optimize,'minimize',side_effect=optimize),warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning);model.fit(x,y)
        assert len(capture)==1
        result=capture[0];fields=query_fields(model,q,kind)
        for l,(w,b) in enumerate(zip(model.coefs_,model.intercepts_),1):
            fields[f'weight_{l}']=w.ravel(order='F').copy();fields[f'bias_{l}']=b.copy()
        fields['metadata']=np.array([model.n_iter_,model.t_,model.n_features_in_,model.n_outputs_,model.n_layers_,model.loss_,result.nfev,float(result.success)])
        target=model._label_binarizer.transform(y) if kind else y[:,None]
        records.append(dict(index=index,seed=seed,kind=kind,activation=activation,hidden=hidden,limit=limit,
                            controls=controls,model=model,x=x,y=y,q=q,target=target,fields=fields))
        lines=[' '.join(map(str,[n,p,kind,len(hidden),nq,epochs,0,seed,1,1,10,maxfun])),
               f'lbfgs {activation} constant',f'.001 {alpha:.17g} {tol:.17g} .9 .5 .9 .999 1e-8']
        if hidden:lines.append(' '.join(map(str,hidden)))
        lines.extend(' '.join(format(v,'.17g') for v in row) for row in x)
        lines.append(' '.join(format(v,'.17g') for v in y))
        lines.extend(' '.join(format(v,'.17g') for v in row) for row in q);inputs.extend(lines)
    run=subprocess.run([str(exe)],input=str(len(records))+'\n'+'\n'.join(inputs)+'\n',text=True,capture_output=True,cwd=ROOT)
    assert run.returncode==0,(run.returncode,run.stdout[-2000:],run.stderr[-2000:])
    checks=[];trajectory_checks=[];case=-1;actual={};certificates=[];source_probes=[]
    def check(field,got,want,rtol=2e-6,atol=2e-7,trajectory=False):
        got=np.asarray(got);want=np.asarray(want);assert got.shape==want.shape,(case,field,got.shape,want.shape)
        passed=True
        try:np.testing.assert_allclose(got,want,rtol=rtol,atol=atol,err_msg=str((case,field,records[case]['index'])))
        except AssertionError:
            if not trajectory and not args.diagnose:raise
            passed=False
        finite=np.isfinite(got)&np.isfinite(want)
        (trajectory_checks if trajectory else checks).append(dict(fixture=case,field=field,passed=passed,
                           max_abs_error=float(np.max(np.abs(got[finite]-want[finite]),initial=0))))
    for line in run.stdout.splitlines():
        parts=line.split()
        if not parts:continue
        if parts[0]=='BEGIN':case=int(parts[1])-1;actual={}
        elif parts[0]=='END':
            record=records[case];model=record['model'];fields=record['fields']
            assert set(actual)==set(fields)|{'gradient','previous_parameters','certificate'},(case,set(actual)^set(fields))
            for name,want in fields.items():check(name,actual[name],want,trajectory=record['limit'] in [3,8,9])
            # Evaluate the unmodified sklearn objective at the native returned point.
            # This checks backpropagation and queries independently of optimizer rounding.
            point=np.concatenate([actual[f'weight_{l}'].reshape(w.shape,order='F').ravel() for l,w in enumerate(model.coefs_,1)]+
                                 [actual[f'bias_{l}'] for l in range(1,len(model.coefs_)+1)])
            activations=[record['x']]+[None]*(model.n_layers_-1);deltas=[None]*(model.n_layers_-1)
            coef_grad=[np.empty_like(w) for w in model.coefs_];bias_grad=[np.empty_like(b) for b in model.intercepts_]
            loss,grad=model._loss_grad_lbfgs(point,record['x'],record['target'],activations,deltas,coef_grad,bias_grad)
            check('objective_at_native',np.array([actual['metadata'][5]]),np.array([loss]),5e-10,5e-11)
            check('gradient_at_native',actual['gradient'],grad,5e-9,5e-10)
            for name,want in query_fields(model,record['q'],record['kind']).items():check(name+'_at_native',actual[name],want,5e-9,5e-10)
            meta=actual['metadata'];assert meta[0]>=0 and meta[0]<=record['controls']['max_iter'] and meta[6]>=1
            check('structural_metadata',meta[[1,2,3,4]],fields['metadata'][[1,2,3,4]],0,0)
            previous,unused=model._loss_grad_lbfgs(actual['previous_parameters'],record['x'],record['target'],
                                                  activations,deltas,coef_grad,bias_grad)
            check('objective_at_previous',actual['certificate'][:1],np.array([previous]),5e-10,5e-11)
            reason=int(actual['certificate'][1]);tol=record['controls']['tol']
            roundoff=abs(previous-actual['certificate'][0])+abs(loss-meta[5])
            infinity_norm=float(np.max(np.abs(grad)))
            delta=previous-loss;bound=1e7*np.finfo(float).eps*max(1.,abs(previous),abs(loss))
            if reason==1:
                assert meta[7]==1 and infinity_norm<=tol+np.max(np.abs(grad-actual['gradient']))+1e-12
            elif reason==2:
                assert meta[7]==1 and delta<=bound+roundoff+1e-15
            elif reason==3:assert meta[7]==0 and meta[0]==record['controls']['max_iter']
            elif reason==4:assert meta[7]==0 and meta[6]>record['controls']['max_fun']
            elif reason==5:assert meta[7]==0
            else:raise AssertionError((case,'unknown termination',reason))
            checks.append(dict(fixture=case,field='termination',passed=True,max_abs_error=0.))
            certificates.append(dict(fixture=case,reason=reason,gradient_inf=infinity_norm,
                                     relative_decrease=delta/max(1.,abs(previous),abs(loss)),
                                     objective_roundoff=roundoff,source_loss=float(fields['metadata'][5]),native_loss=float(loss),
                                     native_iterations=int(meta[0]),source_iterations=int(fields['metadata'][0]),
                                     native_evaluations=int(meta[6]),source_evaluations=int(fields['metadata'][6])))
            if record['index'] in [43,1079]:
                for direction in [-1,1]:
                    cls=PerturbedRegressor if record['kind']==0 else PerturbedClassifier
                    probe=cls(**record['controls']);probe.direction=direction
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',ConvergenceWarning);probe.fit(record['x'],record['y'])
                    coefficient_error=max(float(np.max(np.abs(w.ravel(order='F')-fields[f'weight_{l}']))) for l,w in enumerate(probe.coefs_,1))
                    source_probes.append(dict(fixture=case,direction=direction,coefficient_change=coefficient_error,
                                              loss_change=abs(float(probe.loss_)-fields['metadata'][5])))

        else:
            assert parts[0] not in actual;actual[parts[0]]=np.array(list(map(float,parts[1:])))
    assert case+1==len(records)
    invalid={
        'solver':'unknown MLP solver','activation':'unknown MLP activation','rate':'unknown MLP learning rate',
        'epochs':'invalid MLP training controls','eta':'invalid MLP training controls','alpha':'invalid MLP training controls',
        'tol':'invalid MLP training controls','patience':'invalid MLP iteration or batch controls',
        'batch':'invalid MLP iteration or batch controls','seed':'MLP random_state must be nonnegative',
        'momentum':'MLP momentum must be in [0,1]','power':'invalid MLP optimizer controls',
        'beta1':'invalid MLP optimizer controls','beta2':'invalid MLP optimizer controls','epsilon':'invalid MLP optimizer controls',
        'nonfinite':'nonfinite MLP controls','hidden':'hidden layer sizes must be positive',
        'data':'NaN or infinity','target':'NaN or infinity','target_count':'sample', 'one_class':'MLP classification needs at least two classes',
        'reg_unfitted':'MLP is not fitted','cls_unfitted':'MLP is not fitted', 'reg_width':'sample','cls_width':'sample'}
    invalid.update(max_fun='MLP max_fun must be positive',max_fun_negative='MLP max_fun must be positive')
    rejections=[]
    for mode,message in invalid.items():
        run=subprocess.run([str(exe),mode],text=True,capture_output=True,cwd=ROOT)
        text=run.stdout+run.stderr
        assert run.returncode!=0 and message in text,(mode,text)
        assert not any(v in text for v in ['SIGFPE','Fortran runtime error']), (mode,text)
        rejections.append(dict(case=mode,passed=True))
    failures=[c for c in checks if not c['passed']]
    for c in failures:print('FAIL',c,{k:records[c['fixture']][k] for k in ['index','seed','kind','activation','hidden','limit']},flush=True)
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,
                fixtures=len(records),checks=checks,rejections=rejections,trajectory_comparisons=trajectory_checks,termination_certificates=certificates,
                source_roundoff_probes=source_probes,executable_sha256=sha(exe),validator_sha256=sha(__file__),
                native_source_sha256=sha(ROOT/'src/sklearn_neural_network.f90'),
                source_sha256={r:sha(Path(sklearn.__file__).parent/r) for r in ['neural_network/_multilayer_perceptron.py','neural_network/_base.py']},
                scipy_lbfgsb_source_sha256=sha(Path(scipy.__file__).parent/'optimize/_lbfgsb_py.py'),
                optimizer_source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'vendor/lbfgsb/src').glob('*.[fF]90')},
                configurations=[{k:r[k] for k in ['index','seed','kind','activation','hidden','limit','controls']} for r in records])
    output=args.report or ROOT/f'validation/results/mlp_lbfgs_{args.profile}.json'
    output.parent.mkdir(exist_ok=True,parents=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(len(checks),'required groups across',len(records),'fixtures;',len(failures),'failures;',len(rejections),'diagnostics passed')
    print(len(trajectory_checks),'long-fit trajectory diagnostics;',sum(not c['passed'] for c in trajectory_checks),'differ from source')
    print('Termination reasons:',{i:sum(c['reason']==i for c in certificates) for i in range(1,6)})
    assert not failures or args.diagnose
if __name__=='__main__':main()
