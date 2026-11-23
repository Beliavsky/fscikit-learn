"""MLP validation splits, checkpoint restoration and stopping vs installed sklearn."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np
import scipy,sklearn
from sklearn.neural_network import MLPRegressor,MLPClassifier
import sklearn.neural_network._multilayer_perceptron as mlp_source
from compare_mlp_warm import fields as model_fields
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--profile',default='checked');parser.add_argument('--report',type=Path)
    parser.add_argument('--workflow',type=int);parser.add_argument('--diagnose',action='store_true')
    args=parser.parse_args();records=[];inputs=[];workflows=[];captured={}
    original_split=mlp_source.train_test_split
    def capture_split(x,y,**kw):
        result=original_split(x,y,np.arange(len(x)),**kw)
        captured['training_indices']=result[4]+1;captured['validation_indices']=result[5]+1
        return result[:4]
    mlp_source.train_test_split=capture_split
    optimizers=[dict(solver='adam'),dict(solver='adam',beta_1=.6,beta_2=.83,epsilon=.003,learning_rate_init=.01),
                dict(solver='sgd',momentum=0.,nesterovs_momentum=False,learning_rate_init=.03),
                dict(solver='sgd',momentum=.81,learning_rate_init=.01),
                dict(solver='sgd',learning_rate='invscaling',power_t=.37,learning_rate_init=.02),
                dict(solver='sgd',learning_rate='adaptive',tol=2.,n_iter_no_change=1,learning_rate_init=.001)]
    for index,(seed,spec,activation,hidden,opt) in enumerate(itertools.product(
            [0,7,42],[(0,1),(0,3),(1,1),(2,1),(3,2),(3,4)],
            ['identity','logistic','tanh','relu'],[(),(4,),(5,3)],range(6))):
        if args.workflow is not None and index!=args.workflow:continue
        kind,width=spec;n=24 if index%2 else 23;p=3;nq=7;rng=np.random.RandomState(seed+81)
        x=rng.normal(0,.4,(n,p));q=rng.normal(0,.5,(nq,p))
        if index%19==0:x[:]=0
        elif index%17==0:x[:,2]=x[:,0]
        def targets(data):
            if kind==0:return np.column_stack([np.sin((j+1)*data[:,0])+.2*(j+1)*data[:,1] for j in range(width)])
            if kind in [1,2]:return (3+7*(np.arange(len(data))%(kind+1)))[:,None]
            return np.column_stack([(data[:,0]+(.3*j-.5)*data[:,1]>.15*j).astype(int) for j in range(width)])
        y=targets(x);qy=targets(q)
        if kind==0 and index%11==0:y[:,-1]=2.;qy[:,-1]=2.
        if kind==3 and index%13==0:y[:,0]=0
        control=dict(hidden_layer_sizes=hidden,activation=activation,random_state=seed,max_iter=12,
                     batch_size=[1,6,'auto'][index%3],shuffle=bool(index%2),learning_rate_init=.001,
                     alpha=[0.,.001,.3][index%3],tol=1e-6,n_iter_no_change=2,momentum=.9,
                     nesterovs_momentum=True,learning_rate='constant',power_t=.5,beta_1=.9,beta_2=.999,
                     epsilon=1e-8,warm_start=False,early_stopping=True,validation_fraction=[.2,.3,.5][index%3])
        control.update(optimizers[opt]);base=control.copy()
        model=(MLPRegressor if kind==0 else MLPClassifier)(**control)
        workflows.append(dict(index=index,seed=seed,kind=kind,width=width,activation=activation,hidden=hidden,optimizer=opt))
        for stage in range(5):
            control=base.copy();control.update(warm_start=stage in [1,4],random_state=seed+stage)
            control['max_iter']=[12,7,5,1,6][stage]
            control['validation_fraction']=[.2,.5,.3][(index+stage)%3]
            if kind==0 and index%23==0:control['validation_fraction']=.01 # one-row R2 is NaN
            if stage==3:control.update(solver='lbfgs') # cold reset retains validation history, resets checkpoint
            if stage==4:control.update(solver='sgd',learning_rate_init=.001)
            model.set_params(**control);order=np.roll(np.arange(n),stage);data=x[order];target=y[order]
            captured.clear()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');model.fit(data,target[:,0] if width==1 else target)
                result=model_fields(model,q,kind);score=model.score(q,qy[:,0] if width==1 else qy)
            result['training_state']=np.array([np.nan if model.best_loss_ is None else model.best_loss_,model._no_improvement_count])
            result['prediction']=result['prediction'].reshape(nq,width).ravel(order='F')
            result['metadata']=np.r_[result['metadata'],float(kind==3)]
            result['score']=np.array([score]);result['gradient']=np.array([])
            result['validation_scores']=np.array(model.validation_scores_)
            result['best_validation_score']=np.array([model.best_validation_score_])
            # L-BFGS does not touch these native last-stochastic-split diagnostics.
            if captured:last_split={key:val.copy() for key,val in captured.items()}
            result.update(last_split)
            for l,(w,b) in enumerate(zip(model._best_coefs,model._best_intercepts),1):
                result[f'best_weight_{l}']=w.ravel(order='F');result[f'best_bias_{l}']=b
            optimizer=model._optimizer
            result['optimizer']=np.array([optimizer.learning_rate,getattr(optimizer,'t',0)])
            encoded=model._label_binarizer.transform(target[:,0] if width==1 else target).astype(bool) if kind else target
            records.append(dict(workflow=index,stage=stage,kind=kind,width=width,controls=control.copy(),
                                fields={k:np.asarray(v,dtype=float).copy() for k,v in result.items()},
                                model=model if stage==3 else None,x=data.copy(),target=encoded.copy()))
            c=control;batch=0 if c['batch_size']=='auto' else c['batch_size']
            lines=[' '.join(map(str,[n,p,kind,width,len(hidden),nq,c['max_iter'],batch,c['random_state'],int(c['shuffle']),
                       c['n_iter_no_change'],int(c['warm_start']),int(stage==0),int(c['nesterovs_momentum']),1])),
                   f"{c['solver']} {activation} {c['learning_rate']}",
                   ' '.join(format(c[a],'.17g') for a in ['learning_rate_init','alpha','tol','momentum','power_t','beta_1','beta_2','epsilon','validation_fraction'])]
            if hidden:lines.append(' '.join(map(str,hidden)))
            for array in [data,target,q,qy]:lines.extend(' '.join(format(v,'.17g') for v in row) for row in array)
            inputs.extend(lines)
    mlp_source.train_test_split=original_split
    exe=args.executable.resolve();run=subprocess.run([str(exe)],input=str(len(records))+'\n'+'\n'.join(inputs)+'\n',text=True,capture_output=True,cwd=ROOT)
    assert run.returncode==0,(run.returncode,run.stdout[-2500:],run.stderr[-2500:])
    checks=[];actual={};case=-1
    def check(name,got,want,rtol=5e-8,atol=5e-9):
        got=np.asarray(got);want=np.asarray(want);assert got.shape==want.shape,(case,name,got.shape,want.shape)
        passed=True
        try:np.testing.assert_allclose(got,want,rtol=rtol,atol=atol,err_msg=str((case,name,records[case]['workflow'],records[case]['stage'])))
        except AssertionError:
            if not args.diagnose:raise
            passed=False
        finite=np.isfinite(got)&np.isfinite(want)
        checks.append(dict(fixture=case,field=name,passed=passed,max_abs_error=float(np.max(np.abs(got[finite]-want[finite]),initial=0))))
    for line in run.stdout.splitlines():
        parts=line.split()
        if not parts:continue
        if parts[0]=='BEGIN':case=int(parts[1])-1;actual={}
        elif parts[0]=='END':
            r=records[case];expected=r['fields'];assert set(actual)==set(expected),(case,set(actual)^set(expected))
            for name,want in expected.items():
                if name!='gradient':check(name,actual[name],want,0 if name.endswith('indices') else 5e-8,0 if name.endswith('indices') else 5e-9)
            if r['controls']['solver']=='lbfgs':
                model=r['model'];model.set_params(**r['controls'])
                point=np.concatenate([actual[f'weight_{l}'].reshape(w.shape,order='F').ravel() for l,w in enumerate(model.coefs_,1)]+
                                     [actual[f'bias_{l}'] for l in range(1,len(model.coefs_)+1)])
                activations=[r['x']]+[None]*(model.n_layers_-1);deltas=[None]*(model.n_layers_-1)
                loss,gradient=model._loss_grad_lbfgs(point,r['x'],r['target'],activations,deltas,
                                                    [np.empty_like(w) for w in model.coefs_],[np.empty_like(b) for b in model.intercepts_])
                check('objective_at_native',actual['metadata'][5:6],np.array([loss]),5e-10,5e-11)
                check('gradient_at_native',actual['gradient'],gradient,5e-9,5e-10)
            else:check('gradient',actual['gradient'],np.array([]),0,0)
        else:
            assert parts[0] not in actual;actual[parts[0]]=np.array(list(map(float,parts[1:])))
    assert case+1==len(records)
    invalid=dict(negative_fraction='MLP validation fraction',unit_fraction='MLP validation fraction',
                 nan_fraction='nonfinite MLP validation fraction',zero_fraction='positive validation fraction',
                 empty_train='no training samples',singleton_class='two members per class',
                 small_test='two samples per partition',small_train='two samples per partition',
                 partial_reg='partial_fit does not support early stopping',partial_cls='partial_fit does not support early stopping',
                 warm_on='validation history unavailable',warm_off='loss history unavailable after early stopping')
    rejections=[]
    for mode,message in invalid.items():
        run=subprocess.run([str(exe),mode],text=True,capture_output=True,cwd=ROOT);output=run.stdout+run.stderr
        assert run.returncode!=0 and message in output,(mode,output)
        assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(mode,output)
        rejections.append(dict(case=mode,passed=True))
    failures=[c for c in checks if not c['passed']]
    for c in failures:print('FAIL',c,{k:records[c['fixture']][k] for k in ['workflow','stage','kind','width','controls']},flush=True)
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,workflows=len(workflows),stages=len(records),
                checks=checks,rejections=rejections,workflow_configurations=workflows,
                configurations=[{k:r[k] for k in ['workflow','stage','kind','width','controls']} for r in records],
                nonfinite_reference_values=sum(int(np.count_nonzero(~np.isfinite(a))) for r in records for a in r['fields'].values()),
                validator_sha256=sha(__file__),helper_sha256=sha(ROOT/'validation/compare_mlp_warm.py'),executable_sha256=sha(exe),
                native_source_sha256=sha(ROOT/'src/sklearn_neural_network.f90'),
                source_sha256={r:sha(Path(sklearn.__file__).parent/r) for r in ['neural_network/_multilayer_perceptron.py','neural_network/_base.py',
                  'neural_network/_stochastic_optimizers.py','model_selection/_split.py','utils/extmath.py']})
    output=args.report or ROOT/f'validation/results/mlp_early_{args.profile}.json';output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(len(checks),'groups;',len(workflows),'workflows;',len(records),'fits;',len(rejections),'diagnostics;',len(failures),'failures')
    assert not failures or args.diagnose
if __name__=='__main__':main()
