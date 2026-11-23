"""Single-class MLP encodings, histories and class-layout changes vs sklearn."""
from pathlib import Path
import argparse,copy,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np
import scipy,sklearn
from scipy.sparse import csr_matrix
from sklearn.neural_network import MLPRegressor,MLPClassifier
import sklearn.neural_network._multilayer_perceptron as mlp_source
from compare_mlp_warm import fields as model_fields
from compare_mlp_partial import optimizer_fields
from compare_mlp_sparse import storage,serialized
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--profile',default='checked');parser.add_argument('--report',type=Path)
    parser.add_argument('--workflow',type=int);parser.add_argument('--diagnose',action='store_true')
    args=parser.parse_args();records=[];inputs=[];workflows=[];captured={}
    original_split=mlp_source.train_test_split
    def capture_split(x,y,**kw):
        result=original_split(x,y,np.arange(x.shape[0]),**kw)
        captured['training_indices']=result[4]+1;captured['validation_indices']=result[5]+1
        return result[:4]
    mlp_source.train_test_split=capture_split
    optimizers=[dict(solver='adam'),dict(solver='adam',beta_1=.6,beta_2=.83,epsilon=.003,learning_rate_init=.01),
                dict(solver='sgd',momentum=0.,nesterovs_momentum=False,learning_rate_init=.03),
                dict(solver='sgd',momentum=.81,learning_rate_init=.01),
                dict(solver='sgd',learning_rate='invscaling',power_t=.37,learning_rate_init=.02),
                dict(solver='sgd',learning_rate='adaptive',tol=2.,n_iter_no_change=1,learning_rate_init=.001),
                dict(solver='lbfgs',max_iter=1)]
    for index,(seed,label,activation,hidden,opt) in enumerate(itertools.product(
            [0,7,42],[-11,0,7],['identity','logistic','tanh','relu'],[(),(4,),(5,3)],range(7))):
        if args.workflow is not None and index!=args.workflow:continue
        n=[2,7,19][index%3];p=3;nq=7;rng=np.random.RandomState(seed+101)
        x=rng.normal(0,.4,(n,p));q=rng.normal(0,.5,(nq,p))
        x[rng.uniform(size=x.shape)<.5]=0;q[rng.uniform(size=q.shape)<.5]=0;x[-1]=0
        if index%29==0:x[:]=0
        q[0]=[-10000.,0.,0.];q[1]=[10000.,0.,0.];q[-1]=0
        base=dict(hidden_layer_sizes=hidden,activation=activation,random_state=seed,max_iter=3,
                     batch_size=[1,6,'auto'][index%3],shuffle=bool(index%2),learning_rate_init=.001,
                     alpha=[0.,.001,.3][index%3],tol=1e-6,n_iter_no_change=2,momentum=.9,
                     nesterovs_momentum=True,learning_rate='constant',power_t=.5,beta_1=.9,beta_2=.999,
                     epsilon=1e-8,warm_start=False,early_stopping=False,validation_fraction=.3)
        base.update(optimizers[opt]);model=MLPClassifier(**base)
        workflows.append(dict(index=index,seed=seed,label=label,samples=n,activation=activation,
                              hidden=hidden,optimizer=opt,storage=index%6))
        last_split={name:np.array([]) for name in ['training_indices','validation_indices']}
        for stage in range(10):
            control=base.copy();control.update(warm_start=stage in [1,4],random_state=seed+stage)
            op=int(opt!=6 and (stage in [2,9] or (stage==0 and index%2==0)))
            count=min(n,3) if stage==2 else (1 if op else (max(n,6) if stage in [5,6,7] else n))
            if stage in [3,4]:
                control.update(early_stopping=True,solver='adam' if opt==6 else base['solver'],
                               max_iter=12 if stage==3 else 5,validation_fraction=[.1,.3,.5][(index+stage)%3])
                if stage==4 and index%2==0:control['tol']=0.
            if stage==9 and opt==6:control['warm_start']=True
            sparse_train=(index+stage)%2;sparse_query=(index+stage//2)%2
            kind=2 if stage==7 else 1;width=2 if stage==7 else 1;form=stage%2
            def targets(count):
                if stage==5:return (label+17*(np.arange(count)%2))[:,None]
                if stage==6:return (label+np.array([-13,0,17])[np.arange(count)%3])[:,None]
                if stage==7:return np.column_stack([np.arange(count)%2,(np.arange(count)//2)%2])
                return np.full((count,1),label)
            declared=[label,label] if stage==0 and op else ([label] if op and index%3==0 else [])
            data=x[(np.arange(count)+stage)%n];target=targets(count);qy=targets(nq)
            if kind==1 and index%5==0:qy[-1]=label+101
            sx=storage(data,index%6);sq=storage(q,(index+stage)%6)
            data=sx if sparse_train else data;query=sq if sparse_query else q
            model.set_params(**control);captured.clear()
            source_target=target[:,0] if width==1 and form==0 else target
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if op:model.partial_fit(data,source_target,**({'classes':declared} if declared else {}))
                else:model.fit(data,source_target)
                result=model_fields(model,query,kind);score=model.score(query,qy[:,0] if width==1 else qy)
            result['training_state']=np.array([np.nan if model.best_loss_ is None else model.best_loss_,model._no_improvement_count]) if hasattr(model,'loss_curve_') else np.array([])
            result['prediction']=result['prediction'].reshape(nq,width).ravel(order='F')
            result['metadata']=np.r_[result['metadata'],float(kind==2)]
            result['single_class']=np.array([float(len(model.classes_)==1)])
            result['score']=np.array([score]);result['gradient']=np.array([])
            result['validation_scores']=np.array(getattr(model,'validation_scores_',None) or [])
            best=getattr(model,'best_validation_score_',None)
            result['best_validation_score']=np.array([np.nan if best is None else best])
            if captured:last_split={key:val.copy() for key,val in captured.items()}
            elif control['solver']!='lbfgs':last_split={name:np.array([]) for name in last_split}
            result.update(last_split)
            for layer,(w,b) in enumerate(zip(model._best_coefs,model._best_intercepts),1):
                result[f'best_weight_{layer}']=w.ravel(order='F');result[f'best_bias_{layer}']=b
            result['optimizer']=optimizer_fields(model)
            encoded=model._label_binarizer.transform(target[:,0] if width==1 else target).astype(bool)
            records.append(dict(workflow=index,stage=stage,kind=kind,width=width,controls=control.copy(),op=op,
                                sparse_train=sparse_train,sparse_query=sparse_query,
                                fields={key:np.asarray(val,dtype=float).copy() for key,val in result.items()},
                                model=copy.deepcopy(model) if control['solver']=='lbfgs' else None,x=data.copy(),target=encoded.copy()))
            c=control;batch=0 if c['batch_size']=='auto' else c['batch_size'];form=stage%2
            lines=[' '.join(map(str,[count,p,kind,width,len(hidden),nq,c['max_iter'],batch,c['random_state'],int(c['shuffle']),
                       c['n_iter_no_change'],int(c['nesterovs_momentum']),int(c['warm_start']),int(stage==0),op,len(declared),form,
                       sparse_train,sparse_query,int(c['early_stopping'])])),f"{c['solver']} {activation} {c['learning_rate']}",
                   ' '.join(format(c[key],'.17g') for key in ['learning_rate_init','alpha','tol','momentum','power_t','beta_1','beta_2','epsilon','validation_fraction'])]
            if hidden:lines.append(' '.join(map(str,hidden)))
            if declared:lines.append(' '.join(map(str,declared)))
            lines.extend(serialized(sx) if sparse_train else [' '.join(format(val,'.17g') for val in row) for row in data])
            lines.extend(' '.join(format(val,'.17g') for val in row) for row in target)
            lines.extend(serialized(sq) if sparse_query else [' '.join(format(val,'.17g') for val in row) for row in q])
            lines.extend(' '.join(format(val,'.17g') for val in row) for row in qy);inputs.extend(lines)
        if (index+1)%100==0:print('Prepared',index+1,'source workflows',flush=True)
    mlp_source.train_test_split=original_split
    print('Comparing',len(records),'native fits',flush=True)
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
    invalid=dict(empty_classes='MLP partial_fit classes must be nonempty',unknown_class='MLP partial_fit contains an undeclared class',
                 changed_count='MLP partial_fit class declaration mismatch',changed_value='MLP partial_fit class declaration mismatch',
                 warm_classes='MLP warm fit class set mismatch',warm_kind='MLP warm fit target kind mismatch',
                 partial_early='partial_fit does not support early stopping',empty_partition='no training samples',
                 huge_classifier_query='CSR pointer length mismatch',huge_regressor_query='CSR pointer length mismatch')
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
                configurations=[{k:r[k] for k in ['workflow','stage','kind','width','controls','op','sparse_train','sparse_query']} for r in records],
                nonfinite_reference_values=sum(int(np.count_nonzero(~np.isfinite(a))) for r in records for a in r['fields'].values()),
                validator_sha256=sha(__file__),helper_sha256={f:sha(ROOT/'validation'/f) for f in ['compare_mlp_warm.py','compare_mlp_partial.py','compare_mlp_sparse.py']},executable_sha256=sha(exe),
                native_source_sha256=sha(ROOT/'src/sklearn_neural_network.f90'),csr_source_sha256=sha(ROOT/'src/sklearn_sparse.f90'),
                source_sha256={r:sha(Path(sklearn.__file__).parent/r) for r in ['neural_network/_multilayer_perceptron.py','neural_network/_base.py',
                  'neural_network/_stochastic_optimizers.py','model_selection/_split.py','utils/extmath.py','preprocessing/_label.py','utils/multiclass.py']},
                scipy_sparse_source_sha256={r:sha(Path(scipy.__file__).parent/r) for r in ['sparse/_csr.py','sparse/_compressed.py','sparse/_base.py']})
    output=args.report or ROOT/f'validation/results/mlp_single_{args.profile}.json';output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(len(checks),'groups;',len(workflows),'workflows;',len(records),'fits;',len(rejections),'diagnostics;',len(failures),'failures')
    assert not failures or args.diagnose
if __name__=='__main__':main()
