"""Stateful MLP warm fits, cold resets and solver transitions vs sklearn 1.6.1."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings,copy
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np
import sklearn,scipy
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer,SGDOptimizer
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fields(model,q,kind):
    data={};optimizer=getattr(model,'_optimizer',None);layers=len(model.coefs_)
    for l,(w,b) in enumerate(zip(model.coefs_,model.intercepts_),1):
        data[f'weight_{l}']=w.ravel(order='F');data[f'bias_{l}']=b
        if isinstance(optimizer,AdamOptimizer):
            data[f'moment_{l}']=optimizer.ms[l-1].ravel(order='F');data[f'variance_{l}']=optimizer.vs[l-1].ravel(order='F')
            data[f'bias_moment_{l}']=optimizer.ms[layers+l-1];data[f'bias_variance_{l}']=optimizer.vs[layers+l-1]
        elif isinstance(optimizer,SGDOptimizer):
            data[f'moment_{l}']=optimizer.velocities[l-1].ravel(order='F');data[f'variance_{l}']=np.zeros(w.size)
            data[f'bias_moment_{l}']=optimizer.velocities[layers+l-1];data[f'bias_variance_{l}']=np.zeros(b.size)
        else:
            data[f'moment_{l}']=np.zeros(w.size);data[f'variance_{l}']=np.zeros(w.size)
            data[f'bias_moment_{l}']=np.zeros(b.size);data[f'bias_variance_{l}']=np.zeros(b.size)
    data['loss_curve']=np.array(getattr(model,'loss_curve_',[]))
    data['training_state']=np.array([model.best_loss_,model._no_improvement_count]) if hasattr(model,'loss_curve_') else np.array([])
    data['metadata']=np.array([model.n_iter_,model.t_,model.n_features_in_,model.n_outputs_,model.n_layers_,model.loss_])
    data['prediction']=model.predict(q)
    if kind:
        data['classes']=model.classes_;data['probability']=model.predict_proba(q).ravel(order='F')
        with np.errstate(divide='ignore'):data['log_probability']=model.predict_log_proba(q).ravel(order='F')
    return {name:np.asarray(value).copy() for name,value in data.items()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True,type=Path)
    parser.add_argument('--profile',default='checked');parser.add_argument('--report',type=Path)
    parser.add_argument('--workflow',type=int);parser.add_argument('--diagnose',action='store_true')
    args=parser.parse_args();inputs=[];references=[];configurations=[];workflows=[];roundoff={};source_probes=[]
    optimizers=[dict(solver='adam'),dict(solver='adam',learning_rate_init=.01,beta_1=.6,beta_2=.83,epsilon=.003),
                dict(solver='sgd',learning_rate_init=.03,momentum=0.,nesterovs_momentum=False),
                dict(solver='sgd',learning_rate_init=.01,momentum=.81,nesterovs_momentum=True),
                dict(solver='sgd',learning_rate='invscaling',learning_rate_init=.01,power_t=.37),
                dict(solver='sgd',learning_rate='adaptive',learning_rate_init=.01,tol=2.,n_iter_no_change=1),
                dict(solver='lbfgs',max_iter=1)]
    for index,(seed,kind,activation,hidden,opt) in enumerate(itertools.product(
            [0,7,42],range(3),['identity','logistic','tanh','relu'],[(),(4,),(5,3)],range(7))):
        if args.workflow is not None and index!=args.workflow:continue
        n=19;p=3;nq=9;rng=np.random.RandomState(seed+61)
        x=rng.normal(0,.4,(n,p));q=rng.normal(0,.5,(nq,p))
        if index%13==0:x[:]=0
        elif index%11==0:x[:,2]=0
        elif index%17==0:x[:,2]=x[:,0]
        if kind:q[0]=[-10000.,0.,0.];q[1]=[10000.,0.,0.]
        y=np.sin(.9*x[:,0])+.3*x[:,1] if kind==0 else 3+7*(np.arange(n)%(kind+1))
        control=dict(hidden_layer_sizes=hidden,activation=activation,random_state=seed,max_iter=3,batch_size=[1,6,'auto'][index%3],
                     shuffle=bool(index%2),learning_rate_init=.001,alpha=[0.,.001,.3][index%3],tol=1e-6,n_iter_no_change=4,
                     momentum=.9,nesterovs_momentum=True,power_t=.5,beta_1=.9,beta_2=.999,epsilon=1e-8,
                     learning_rate='constant',max_fun=15000,warm_start=False)
        control.update(optimizers[opt]);base_solver=control['solver']
        model=(MLPRegressor if kind==0 else MLPClassifier)(**control)
        workflows.append(dict(index=index,seed=seed,kind=kind,activation=activation,hidden=hidden,optimizer=opt))
        for stage in range(6):
            mutate=int(stage==2 and bool(index%2))
            if index==4 and stage==1:mutate=2
            if index==7 and stage==2:mutate=3
            control.update(warm_start=stage not in [0,3],random_state=seed+stage)
            control['max_iter']=1 if control['solver']=='lbfgs' else [3,4,5,2,1,3][stage]
            if stage==2:
                control.update(alpha=control['alpha']*2,learning_rate_init=control['learning_rate_init']*.5,shuffle=not control['shuffle'])
                if base_solver!='lbfgs':control['hidden_layer_sizes']=tuple(v+2 for v in hidden)
            if stage==4:
                if base_solver=='lbfgs':control.update(tol=1000.,max_iter=1)
                else:control.update(solver='lbfgs',max_iter=1)
            if stage==5:
                if base_solver=='lbfgs':control.update(solver='sgd',warm_start=False,tol=1e-6,max_iter=3)
                else:control.update(solver='sgd' if base_solver=='adam' else 'adam',max_iter=3)
            model.set_params(**control)
            if mutate==1:
                for l in range(len(model.coefs_)):
                    model.coefs_[l]=.9*model.coefs_[l]+.002;model.intercepts_[l]=model.intercepts_[l]-.015*(l+1)
            if mutate==2:model.t_=2147483640
            if mutate==3:
                width=control['hidden_layer_sizes'][0]
                model.coefs_[0]=np.full((p,width),.1);model.intercepts_[0]=np.full(width,.05)
                model.coefs_[1]=np.full((width,len(model.intercepts_[1])),-.1)
            order=np.roll(np.arange(n),stage);data=x[order];target=y[order]
            if index==585 and stage==4:
                probe_start=copy.deepcopy(model)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore',ConvergenceWarning);warnings.simplefilter('ignore',UserWarning);model.fit(data,target)
            expected=fields(model,q,kind)
            if index==585 and stage==4:probe_checkpoint=copy.deepcopy(model)
            if index==585 and stage==5:
                radii={name:np.zeros_like(a,dtype=float) for name,a in expected.items()}
                working_scale=max(1.,max(float(np.max(np.abs(a))) for a in probe_start.coefs_+probe_start.intercepts_))
                for units in [sign*2.**exponent for exponent in range(-12,4) for sign in [-1,1]]:
                    probe=copy.deepcopy(probe_checkpoint)
                    perturbation=units*np.finfo(float).eps*working_scale
                    probe.intercepts_[-1]=probe.intercepts_[-1]+perturbation
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',ConvergenceWarning)
                        probe.set_params(**control);probe.fit(data,target)
                    altered=fields(probe,q,kind)
                    for name,a in expected.items():
                        assert altered[name].shape==a.shape
                        radii[name]=np.maximum(radii[name],np.abs(altered[name]-a))
                    source_probes.append(dict(workflow=index,units=units,perturbation=perturbation,
                                               maximum_bias_change=max(float(np.max(np.abs(altered[f'bias_{l}']-expected[f'bias_{l}']))) for l in range(1,len(probe.coefs_)+1))))
                roundoff[len(references)]=radii
            references.append(expected);configurations.append(dict(workflow=index,stage=stage,controls=control.copy(),mutate=mutate))
            c=control;h=c['hidden_layer_sizes'];batch=0 if c['batch_size']=='auto' else c['batch_size']
            lines=[' '.join(map(str,[n,p,kind,len(h),nq,c['max_iter'],batch,c['random_state'],int(c['shuffle']),
                                    int(c['nesterovs_momentum']),c['n_iter_no_change'],c['max_fun'],int(stage==0),int(c['warm_start']),int(mutate)])),
                   f"{c['solver']} {c['activation']} {c['learning_rate']}",
                   ' '.join(format(c[a],'.17g') for a in ['learning_rate_init','alpha','tol','momentum','power_t','beta_1','beta_2','epsilon'])]
            if h:lines.append(' '.join(map(str,h)))
            lines.extend(' '.join(format(v,'.17g') for v in row) for row in data)
            lines.append(' '.join(format(v,'.17g') for v in target))
            lines.extend(' '.join(format(v,'.17g') for v in row) for row in q);inputs.extend(lines)
    run=subprocess.run([str(args.executable.resolve())],input=str(len(references))+'\n'+'\n'.join(inputs)+'\n',text=True,capture_output=True,cwd=ROOT)
    assert run.returncode==0,(run.returncode,run.stdout[-2500:],run.stderr[-2500:])
    checks=[];actual={};case=-1
    for line in run.stdout.splitlines():
        parts=line.split()
        if not parts:continue
        if parts[0]=='BEGIN':case=int(parts[1])-1;actual={}
        elif parts[0]=='END':
            expected=references[case];assert set(actual)==set(expected),(case,set(actual)^set(expected))
            for name,want in expected.items():
                got=actual[name];assert got.shape==want.shape,(case,name,got.shape,want.shape,configurations[case])
                passed=True;strict=True
                try:np.testing.assert_allclose(got,want,rtol=5e-9,atol=5e-10,err_msg=str((case,name,configurations[case])))
                except AssertionError:
                    strict=False
                    if case in roundoff:
                        passed=bool(np.all(np.abs(got-want)<=roundoff[case][name]+5e-10+5e-9*np.abs(want)))
                    else:passed=False
                    if not passed and not args.diagnose:raise
                finite=np.isfinite(got)&np.isfinite(want)
                checks.append(dict(fixture=case,field=name,passed=passed,strict_passed=strict,max_abs_error=float(np.max(np.abs(got[finite]-want[finite]),initial=0))))
        else:
            assert parts[0] not in actual;actual[parts[0]]=np.array(list(map(float,parts[1:])))
    assert case+1==len(references)
    invalid=dict(reg_warm_width='MLP warm fit feature count mismatch',cls_warm_width='MLP warm fit feature count mismatch',
                 warm_classes='MLP warm fit class set mismatch',warm_class_count='MLP warm fit class set mismatch',
                 warm_layers='MLP warm fit layer count mismatch',warm_lbfgs_width='MLP warm L-BFGS layer width mismatch',
                 warm_history='MLP stochastic history unavailable',warm_controls='invalid MLP training controls')
    rejections=[]
    for mode,message in invalid.items():
        run=subprocess.run([str(args.executable.resolve()),mode],text=True,capture_output=True,cwd=ROOT);output=run.stdout+run.stderr
        assert run.returncode!=0 and message in output,(mode,output)
        assert not any(v in output for v in ['SIGFPE','Fortran runtime error']),(mode,output)
        rejections.append(dict(case=mode,passed=True))
    failures=[c for c in checks if not c['passed']]
    for c in failures:print('FAIL',c,configurations[c['fixture']],flush=True)
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,
                workflows=len(workflows),stages=len(references),checks=checks,rejections=rejections,
                configurations=configurations,workflow_configurations=workflows,
                rounding_envelopes={str(k):{name:a.tolist() for name,a in value.items()} for k,value in roundoff.items()},
                source_roundoff_probes=source_probes,
                nonfinite_reference_values=sum(int(np.count_nonzero(~np.isfinite(a))) for fields in references for a in fields.values()),
                executable_sha256=sha(args.executable),validator_sha256=sha(__file__),native_source_sha256=sha(ROOT/'src/sklearn_neural_network.f90'),
                source_sha256={r:sha(Path(sklearn.__file__).parent/r) for r in ['neural_network/_multilayer_perceptron.py','neural_network/_stochastic_optimizers.py','neural_network/_base.py']})
    output=args.report or ROOT/f'validation/results/mlp_warm_{args.profile}.json';output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(len(checks),'groups;',len(workflows),'workflows;',len(references),'stages;',len(rejections),'diagnostics;',len(failures),'failures')
    print('Rounding-envelope groups:',sum(c['passed'] and not c['strict_passed'] for c in checks))
    assert not failures or args.diagnose
if __name__=='__main__':main()
