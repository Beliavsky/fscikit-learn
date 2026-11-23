"""CSR MLP kernels, dense/CSR transitions and stochastic histories vs sklearn."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np
import scipy,sklearn
from scipy.sparse import csr_matrix
from sklearn.neural_network import MLPRegressor,MLPClassifier
import sklearn.neural_network._multilayer_perceptron as mlp_source
from compare_mlp_warm import fields as model_fields
from compare_mlp_partial import optimizer_fields
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def storage(x,variant):
    data=[];indices=[];pointer=[0]
    for row in x:
        for col in (range(len(row)-1,-1,-1) if variant%2 else range(len(row))):
            value=row[col]
            if value==0 and variant not in [2,3]:continue
            if variant in [3,4] and value!=0:
                data.extend([value*.25,value*.75]);indices.extend([col,col])
            else:data.append(value);indices.append(col)
        pointer.append(len(data))
    return csr_matrix((np.array(data),np.array(indices,dtype=np.int32),np.array(pointer,dtype=np.int32)),shape=x.shape)

def serialized(x):
    lines=[str(x.nnz),' '.join(map(str,x.indptr+1))]
    if x.nnz:lines+=[' '.join(map(str,x.indices+1)),' '.join(format(v,'.17g') for v in x.data)]
    return lines

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
    index=-1
    for seed,spec,activation,hidden,opt in itertools.product(
            [0,7,42],[(0,1,0),(0,3,0),(1,1,2),(1,1,3),(2,2,0),(2,4,0)],
            ['identity','logistic','tanh','relu'],[(),(4,),(5,3)],range(7)):
        kind,width,classcount=spec
        if opt==6 and kind!=0:continue
        index+=1
        if args.workflow is not None and index!=args.workflow:continue
        n=23;p=7;nq=7;rng=np.random.RandomState(seed+91)
        x=rng.normal(0,.4,(n,p));q=rng.normal(0,.5,(nq,p));x[rng.uniform(size=x.shape)<.7]=0;q[rng.uniform(size=q.shape)<.65]=0
        x[2]=0;q[2]=0
        if index%29==0:x[:]=0
        if index%17==0:x[:,-1]=x[:,0]
        def targets(data):
            if kind==0:return np.column_stack([np.sin((j+1)*data[:,0])+.2*(j+1)*data[:,1] for j in range(width)])
            if kind==1:return (3+7*(np.arange(len(data))%classcount))[:,None]
            return np.column_stack([(data[:,0]+(.3*j-.5)*data[:,1]>.15*j).astype(int) for j in range(width)])
        y=targets(x);qy=targets(q)
        if kind==2:y[0]=0;y[1]=1
        if kind==0 and index%11==0:y[:,-1]=2.;qy[:,-1]=2.
        classes=list(map(int,np.unique(y))) if kind==1 else list(range(width))
        base=dict(hidden_layer_sizes=hidden,activation=activation,random_state=seed,max_iter=3,
                     batch_size=[1,6,'auto'][index%3],shuffle=bool(index%2),learning_rate_init=.001,
                     alpha=[0.,.001,.3][index%3],tol=1e-6,n_iter_no_change=2,momentum=.9,
                     nesterovs_momentum=True,learning_rate='constant',power_t=.5,beta_1=.9,beta_2=.999,
                     epsilon=1e-8,warm_start=False,early_stopping=False,validation_fraction=.3)
        base.update(optimizers[opt]);model=(MLPRegressor if kind==0 else MLPClassifier)(**base)
        workflows.append(dict(index=index,seed=seed,kind=kind,width=width,classcount=classcount,activation=activation,hidden=hidden,optimizer=opt,storage=index%6))
        last_split={name:np.array([]) for name in ['training_indices','validation_indices']}
        for stage in range(6):
            control=base.copy();control.update(warm_start=stage in [2,4],random_state=seed+stage)
            op=int(stage==1 or (stage==0 and index%2==0));count=5 if stage==1 else (1 if op else n)
            if stage==2:control.update(learning_rate_init=.007,alpha=base['alpha']*2)
            if stage in [3,4]:control.update(early_stopping=True,solver='adam' if opt==6 else base['solver'],max_iter=5 if stage==3 else 3)
            if kind==0 and index%31==0:control['validation_fraction']=.01
            sparse_train=int(stage!=2 and (stage!=1 or index%2==0) and (stage!=4 or index%2==1))
            sparse_query=(index+stage)%2
            declared=classes if stage==0 and op and kind else []
            order=np.roll(np.arange(n),stage)[:count];data=x[order];target=y[order]
            sx=storage(data,index%6);sq=storage(q,(index+stage)%6)
            data=sx if sparse_train else data;query=sq if sparse_query else q
            control['warm_start']=stage in [2,4];model.set_params(**control);captured.clear()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if op:model.partial_fit(data,target[:,0] if width==1 else target,**({'classes':declared} if declared else {}))
                else:model.fit(data,target[:,0] if width==1 else target)
                result=model_fields(model,query,kind);score=model.score(query,qy[:,0] if width==1 else qy)
            result['training_state']=np.array([np.nan if model.best_loss_ is None else model.best_loss_,model._no_improvement_count]) if hasattr(model,'loss_curve_') else np.array([])
            result['prediction']=result['prediction'].reshape(nq,width).ravel(order='F')
            result['metadata']=np.r_[result['metadata'],float(kind==2)]
            result['score']=np.array([score]);result['gradient']=np.array([])
            result['validation_scores']=np.array(getattr(model,'validation_scores_',None) or [])
            best=getattr(model,'best_validation_score_',None)
            result['best_validation_score']=np.array([np.nan if best is None else best])
            if captured:last_split={key:val.copy() for key,val in captured.items()}
            elif control['solver']!='lbfgs':last_split={name:np.array([]) for name in last_split}
            result.update(last_split)
            for l,(w,b) in enumerate(zip(model._best_coefs,model._best_intercepts),1):
                result[f'best_weight_{l}']=w.ravel(order='F');result[f'best_bias_{l}']=b
            result['optimizer']=optimizer_fields(model)
            encoded=model._label_binarizer.transform(target[:,0] if width==1 else target).astype(bool) if kind else target
            records.append(dict(workflow=index,stage=stage,kind=kind,width=width,controls=control.copy(),op=op,
                                sparse_train=sparse_train,sparse_query=sparse_query,
                                fields={k:np.asarray(v,dtype=float).copy() for k,v in result.items()},
                                model=model if control['solver']=='lbfgs' else None,x=data.copy(),target=encoded.copy()))
            c=control;batch=0 if c['batch_size']=='auto' else c['batch_size'];form=stage%2
            lines=[' '.join(map(str,[count,p,kind,width,len(hidden),nq,c['max_iter'],batch,c['random_state'],int(c['shuffle']),
                       c['n_iter_no_change'],int(c['nesterovs_momentum']),int(c['warm_start']),int(stage==0),op,len(declared),form,
                       sparse_train,sparse_query,int(c['early_stopping'])])),f"{c['solver']} {activation} {c['learning_rate']}",
                   ' '.join(format(c[a],'.17g') for a in ['learning_rate_init','alpha','tol','momentum','power_t','beta_1','beta_2','epsilon','validation_fraction'])]
            if hidden:lines.append(' '.join(map(str,hidden)))
            if declared:lines.append(' '.join(map(str,declared)))
            lines.extend(serialized(sx) if sparse_train else [' '.join(format(v,'.17g') for v in row) for row in data])
            lines.extend(' '.join(format(v,'.17g') for v in row) for row in target)
            lines.extend(serialized(sq) if sparse_query else [' '.join(format(v,'.17g') for v in row) for row in q])
            lines.extend(' '.join(format(v,'.17g') for v in row) for row in qy);inputs.extend(lines)
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
    invalid=dict(unallocated='CSR arrays are unallocated',pointer_length='CSR pointer length mismatch',
                 pointer_end='CSR pointer endpoints',pointer_order='CSR pointers must be nondecreasing',column='CSR column index',
                 data_length='CSR data/index lengths',nan='MLP sparse input contains NaN or infinity',empty_rows='MLP sparse input must be nonempty',
                 empty_columns='MLP sparse input must be nonempty',sample_count='MLP sparse sample count mismatch',class_count='MLP sparse sample count mismatch',
                 warm_width='MLP warm fit feature count mismatch',partial_width='MLP warm fit feature count mismatch',query_width='MLP sparse feature count mismatch',
                 query_nan='MLP sparse input contains NaN or infinity',unfitted='MLP is not fitted',partial_early='partial_fit does not support early stopping')
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
                validator_sha256=sha(__file__),helper_sha256={f:sha(ROOT/'validation'/f) for f in ['compare_mlp_warm.py','compare_mlp_partial.py']},executable_sha256=sha(exe),
                native_source_sha256=sha(ROOT/'src/sklearn_neural_network.f90'),csr_source_sha256=sha(ROOT/'src/sklearn_sparse.f90'),
                source_sha256={r:sha(Path(sklearn.__file__).parent/r) for r in ['neural_network/_multilayer_perceptron.py','neural_network/_base.py',
                  'neural_network/_stochastic_optimizers.py','model_selection/_split.py','utils/extmath.py']},
                scipy_sparse_source_sha256={r:sha(Path(scipy.__file__).parent/r) for r in ['sparse/_csr.py','sparse/_compressed.py','sparse/_base.py']})
    output=args.report or ROOT/f'validation/results/mlp_sparse_{args.profile}.json';output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(len(checks),'groups;',len(workflows),'workflows;',len(records),'fits;',len(rejections),'diagnostics;',len(failures),'failures')
    assert not failures or args.diagnose
if __name__=='__main__':main()
