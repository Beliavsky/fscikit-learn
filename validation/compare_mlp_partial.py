"""Persistent MLP partial_fit optimizers and declared classes vs sklearn 1.6.1."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np
import scipy,sklearn
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer,SGDOptimizer
from compare_mlp_warm import fields as model_fields
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def optimizer_fields(model):
    opt=getattr(model,'_optimizer',None)
    if isinstance(opt,AdamOptimizer):return np.array([1.,opt.learning_rate,opt.learning_rate_init,opt.beta_1,opt.beta_2,opt.epsilon,opt.t])
    if isinstance(opt,SGDOptimizer):return np.array([2.,opt.learning_rate,opt.learning_rate_init,opt.momentum,opt.power_t,float(opt.nesterov),['constant','invscaling','adaptive'].index(opt.lr_schedule)])
    return np.array([])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True,type=Path)
    parser.add_argument('--profile',default='checked');parser.add_argument('--report',type=Path)
    parser.add_argument('--workflow',type=int);parser.add_argument('--diagnose',action='store_true')
    args=parser.parse_args();records=[];inputs=[];workflows=[]
    specs=[(0,1,0),(0,3,0),(1,1,2),(1,1,3),(2,2,0),(2,4,0)]
    optimizers=[dict(solver='adam'),dict(solver='adam',learning_rate_init=.01,beta_1=.6,beta_2=.83,epsilon=.003),
                dict(solver='sgd',momentum=0.,nesterovs_momentum=False,learning_rate_init=.02),
                dict(solver='sgd',momentum=.8,nesterovs_momentum=True,learning_rate_init=.02),
                dict(solver='sgd',learning_rate='invscaling',power_t=.37,learning_rate_init=.02),
                dict(solver='sgd',learning_rate='adaptive',tol=2.,n_iter_no_change=1,learning_rate_init=.02),dict(solver='lbfgs',max_iter=1)]
    index=-1
    for seed,spec,activation,hidden,opt in itertools.product([0,7,42],specs,['identity','logistic','tanh','relu'],[(),(4,),(5,3)],range(7)):
        kind,width,classcount=spec
        if opt==6 and kind!=0:continue
        index+=1
        if args.workflow is not None and index!=args.workflow:continue
        n=23;p=3;nq=7;rng=np.random.RandomState(seed+83);x=rng.normal(0,.4,(n,p));q=rng.normal(0,.5,(nq,p))
        if kind==0:
            y=np.column_stack([np.sin((j+1)*x[:,0])+.2*x[:,1]*(j+1) for j in range(width)])
            qy=np.column_stack([np.sin((j+1)*q[:,0])+.2*q[:,1]*(j+1) for j in range(width)]);classes=[]
        elif kind==1:
            y=(3+7*(np.arange(n)%classcount))[:,None];qy=(3+7*(np.arange(nq)%classcount))[:,None]
            classes=[int(v) for v in 3+7*np.arange(classcount)]
        else:
            y=np.column_stack([(x[:,0]+.3*j*x[:,1]>0).astype(int) for j in range(width)])
            qy=np.column_stack([(q[:,0]+.3*j*q[:,1]>0).astype(int) for j in range(width)])
            y[0]=0;y[1]=1;classes=list(range(width))
            if index%37==0:classes.append(width)
        c=dict(hidden_layer_sizes=hidden,activation=activation,random_state=seed,solver='adam',max_iter=2,
               batch_size=[1,5,'auto'][index%3],shuffle=bool(index%2),learning_rate_init=.001,alpha=[0.,.001,.3][index%3],
               tol=1e-6,n_iter_no_change=4,momentum=.9,nesterovs_momentum=True,power_t=.5,beta_1=.9,beta_2=.999,epsilon=1e-8,
               learning_rate='constant',warm_start=False,max_fun=15000)
        c.update(optimizers[opt]);base_solver=c['solver'];alternate='sgd' if base_solver=='adam' else 'adam'
        model=(MLPRegressor if kind==0 else MLPClassifier)(**c)
        workflows.append(dict(index=index,seed=seed,kind=kind,width=width,classcount=classcount,activation=activation,hidden=hidden,optimizer=opt))
        for stage in range(9):
            op=1;count=[1,5,11,n,7,n,3,n,5][stage];c.update(warm_start=False,random_state=seed+stage)
            declared=[]
            if stage==0 and kind:declared=list(reversed(classes))+classes[:1]
            if stage==2:
                c.update(learning_rate_init=c['learning_rate_init']*3,beta_1=.45,beta_2=.7,epsilon=.02,momentum=.1,power_t=.8,
                         learning_rate='constant' if c['learning_rate']!='constant' else 'invscaling',shuffle=not c['shuffle'],alpha=c['alpha']*2)
                if base_solver=='lbfgs':op=0;c.update(solver='adam',max_iter=2)
                else:c['solver']=alternate
                if kind:declared=classes
            if stage==3:
                op=0;c.update(warm_start=True,solver=alternate,max_iter=2)
                if kind==2 and len(classes)!=width:c['warm_start']=False
            if stage==4:c.update(solver='adam' if alternate=='sgd' else 'sgd',learning_rate_init=.007,beta_1=.3)
            if stage==5:op=0;c.update(solver='adam' if base_solver=='lbfgs' else base_solver,max_iter=2)
            if stage==7:
                if kind==0:op=0;c.update(solver='lbfgs',max_iter=1)
                else:c.update(solver=alternate)
            if stage==8:c.update(solver='sgd' if base_solver=='adam' else 'adam')
            if c['solver']=='lbfgs':c['max_iter']=1
            model.set_params(**c);order=np.roll(np.arange(n),stage)[:count];data=x[order];target=y[order]
            form=(index+stage)%2 if width==1 else 1
            actual_target=target[:,0] if kind==1 or (kind==0 and form==0) else target
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if op==0:model.fit(data,actual_target)
                elif kind:model.partial_fit(data,actual_target,**({'classes':declared} if declared else {}))
                else:model.partial_fit(data,actual_target)
                result=model_fields(model,q,int(kind>0));score=model.score(q,qy[:,0] if kind==1 else qy)
            result['prediction']=np.asarray(result['prediction']).reshape(nq,width).ravel(order='F')
            result['metadata']=np.r_[result['metadata'],float(kind==2 and width>1)]
            result['optimizer']=optimizer_fields(model);result['score']=np.array([score]);result['gradient']=np.array([])
            encoded=model._label_binarizer.transform(target[:,0] if kind==1 or width==1 else target).astype(bool) if kind else target
            records.append(dict(workflow=index,stage=stage,kind=kind,width=width,hidden=hidden,controls=c.copy(),op=op,
                                form=form,declared=list(declared),fields=result,model=model,x=data,target=encoded))
            batch=0 if c['batch_size']=='auto' else c['batch_size']
            lines=[' '.join(map(str,[count,p,kind,width,len(hidden),nq,c['max_iter'],batch,c['random_state'],int(c['shuffle']),c['n_iter_no_change'],int(c['nesterovs_momentum']),int(c['warm_start']),int(stage==0),op,len(declared),form])),
                   f"{c['solver']} {activation} {c['learning_rate']}",
                   ' '.join(format(c[a],'.17g') for a in ['learning_rate_init','alpha','tol','momentum','power_t','beta_1','beta_2','epsilon'])]
            if hidden:lines.append(' '.join(map(str,hidden)))
            if declared:lines.append(' '.join(map(str,declared)))
            for array in [data,target,q,qy]:lines.extend(' '.join(format(v,'.17g') for v in row) for row in array)
            inputs.extend(lines)
    exe=args.executable.resolve();run=subprocess.run([str(exe)],input=str(len(records))+'\n'+'\n'.join(inputs)+'\n',text=True,capture_output=True,cwd=ROOT)
    assert run.returncode==0,(run.returncode,run.stdout[-2500:],run.stderr[-2500:])
    checks=[];case=-1;actual={}
    def check(name,got,want,rtol=5e-8,atol=5e-9):
        got=np.asarray(got);want=np.asarray(want);assert got.shape==want.shape,(case,name,got.shape,want.shape,records[case]['workflow'],records[case]['stage'])
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
                if name!='gradient':check(name,actual[name],want)
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
    invalid=dict(missing_classes='MLP first partial_fit requires classes',empty_classes='MLP partial_fit classes must be nonempty',
                 empty_batch='X must be nonempty',unknown_class='MLP partial_fit contains an undeclared class',
                 changed_classes='MLP partial_fit class declaration mismatch',changed_class_count='MLP partial_fit class declaration mismatch',
                 warm_subset='MLP warm fit class set mismatch',cls_solver='MLP classifier partial_fit requires a stochastic solver',
                 reg_width='MLP warm fit feature count mismatch',cls_width='MLP warm fit feature count mismatch',
                 reg_outputs='MLP warm fit output count mismatch',indicator_width='MLP warm fit indicator width mismatch',
                 indicator_values='MLP indicator targets must be zero or one',indicator_classes='MLP partial_fit contains an undeclared class',
                 indicator_mode='MLP partial_fit target kind mismatch',vector_mode='MLP partial_fit target kind mismatch',
                 reg_nan='NaN or infinity',reg_count='sample',cls_count='sample',controls='invalid MLP training controls',
                 history='MLP stochastic history unavailable')
    rejections=[]
    for mode,message in invalid.items():
        run=subprocess.run([str(exe),mode],text=True,capture_output=True,cwd=ROOT);output=run.stdout+run.stderr
        assert run.returncode!=0 and message in output,(mode,output)
        assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(mode,output)
        rejections.append(dict(case=mode,passed=True))
    failures=[c for c in checks if not c['passed']]
    for c in failures:print('FAIL',c,{k:records[c['fixture']][k] for k in ['workflow','stage','kind','width','controls','op']},flush=True)
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,workflows=len(workflows),stages=len(records),
                checks=checks,rejections=rejections,workflow_configurations=workflows,
                configurations=[{k:r[k] for k in ['workflow','stage','kind','width','controls','op','form','declared']} for r in records],
                nonfinite_reference_values=sum(int(np.count_nonzero(~np.isfinite(a))) for r in records for a in r['fields'].values()),
                validator_sha256=sha(__file__),helper_sha256=sha(ROOT/'validation/compare_mlp_warm.py'),executable_sha256=sha(exe),
                native_source_sha256=sha(ROOT/'src/sklearn_neural_network.f90'),
                source_sha256={r:sha(Path(sklearn.__file__).parent/r) for r in ['neural_network/_multilayer_perceptron.py','neural_network/_stochastic_optimizers.py','neural_network/_base.py','utils/multiclass.py']})
    output=args.report or ROOT/f'validation/results/mlp_partial_{args.profile}.json';output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(len(checks),'groups;',len(workflows),'workflows;',len(records),'fits;',len(rejections),'diagnostics;',len(failures),'failures')
    assert not failures or args.diagnose
if __name__=='__main__':main()
