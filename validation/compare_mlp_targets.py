"""Joint MLP responses and multilabel indicators against installed sklearn."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np
import scipy,sklearn
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from compare_mlp_warm import fields as model_fields
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def targets(x,kind,width):
    if kind:
        y=np.column_stack([(x[:,0]+(.3*j-.5)*x[:,1]>.15*j).astype(int) for j in range(width)])
    else:y=np.column_stack([np.sin((j+1)*x[:,0])+.2*(j+1)*x[:,1] for j in range(width)])
    return y

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--profile',default='checked');parser.add_argument('--report',type=Path)
    parser.add_argument('--workflow',type=int);parser.add_argument('--diagnose',action='store_true')
    args=parser.parse_args();records=[];inputs=[];workflows=[]
    optimizers=[dict(solver='adam'),dict(solver='adam',beta_1=.6,beta_2=.83,epsilon=.003,learning_rate_init=.01),
                dict(solver='sgd',momentum=.8,learning_rate_init=.02),
                dict(solver='sgd',learning_rate='invscaling',power_t=.37,learning_rate_init=.02),
                dict(solver='sgd',learning_rate='adaptive',tol=2.,n_iter_no_change=1,learning_rate_init=.02),
                dict(solver='lbfgs',max_iter=1)]
    for index,(seed,kind,width,activation,hidden,opt) in enumerate(itertools.product(
            [0,7,42],range(2),[1,2,4],['identity','logistic','tanh','relu'],[(),(4,),(5,3)],range(6))):
        if args.workflow is not None and index!=args.workflow:continue
        n=17;p=3;nq=1 if index%23==0 else 7;rng=np.random.RandomState(seed+71)
        x=rng.normal(0,.4,(n,p));q=rng.normal(0,.5,(nq,p))
        if index%19==0:x[:]=0
        elif index%17==0:x[:,2]=x[:,0]
        y=targets(x,kind,width);qy=targets(q,kind,width)
        if kind:
            y[0]=0;y[1]=1
            if width>1 and index%11==0:y[:,0]=0
            if width>1 and index%13==0:y[:,-1]=1
            if nq>1:q[0]=[-10000.,0.,0.];q[1]=[10000.,0.,0.]
        elif width>1 and index%11==0:y[:,-1]=2.;qy[:,-1]=2.
        controls=dict(hidden_layer_sizes=hidden,activation=activation,random_state=seed,solver='adam',
                      max_iter=3,batch_size=[1,6,'auto'][index%3],shuffle=bool(index%2),
                      learning_rate_init=.001,alpha=[0.,.001,.3][index%3],tol=1e-6,n_iter_no_change=4,
                      momentum=.9,nesterovs_momentum=True,learning_rate='constant',power_t=.5,
                      beta_1=.9,beta_2=.999,epsilon=1e-8,warm_start=False)
        controls.update(optimizers[opt]);base_alpha=controls['alpha']
        model=(MLPRegressor if not kind else MLPClassifier)(**controls)
        workflows.append(dict(index=index,seed=seed,kind=kind,width=width,activation=activation,hidden=hidden,optimizer=opt))
        for stage in range(3):
            controls.update(warm_start=stage==1,random_state=seed+stage,alpha=base_alpha*(stage+1),
                            max_iter=1 if controls['solver']=='lbfgs' else [3,4,2][stage])
            model.set_params(**controls);order=np.roll(np.arange(n),stage);data=x[order];target=y[order]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');model.fit(data,target)
                result=model_fields(model,q,kind);score=model.score(q,qy)
            result['prediction']=np.asarray(result['prediction']).reshape(nq,width).ravel(order='F')
            result['metadata']=np.r_[result['metadata'],float(kind and width>1)]
            result['score']=np.array([score]);result['gradient']=np.array([])
            encoded=model._label_binarizer.transform(target[:,0] if width==1 else target).astype(bool) if kind else target
            records.append(dict(workflow=index,stage=stage,kind=kind,width=width,hidden=hidden,controls=controls.copy(),
                                fields=result,model=model,x=data,q=q,target=encoded))
            c=controls;batch=0 if c['batch_size']=='auto' else c['batch_size']
            lines=[' '.join(map(str,[n,p,kind,width,len(hidden),nq,c['max_iter'],batch,c['random_state'],int(c['shuffle']),c['n_iter_no_change'],int(c['warm_start']),int(stage==0)])),
                   f"{c['solver']} {activation} {c['learning_rate']}",
                   ' '.join(format(c[a],'.17g') for a in ['learning_rate_init','alpha','tol','momentum','power_t','beta_1','beta_2','epsilon'])]
            if hidden:lines.append(' '.join(map(str,hidden)))
            for array in [data,target,q,qy]:lines.extend(' '.join(format(v,'.17g') for v in row) for row in array)
            inputs.extend(lines)
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
    invalid=dict(reg_empty='nonempty',reg_count='sample',reg_nan='NaN or infinity',reg_unfitted='MLP is not fitted',
                 reg_vector='use predict_multioutput',reg_score_width='MLP response score width mismatch',reg_query_width='sample',
                 reg_warm_width='MLP warm fit output count mismatch',cls_empty='MLP indicator target width must be positive',
                 cls_count='sample',cls_values='MLP indicator targets must be zero or one',
                 cls_unfitted='MLP indicator prediction requires a multilabel model',cls_vector='use predict_indicator',
                 cls_scalar_indicator='MLP indicator prediction requires a multilabel model',cls_score_width='MLP indicator score width mismatch',
                 cls_score_values='MLP indicator targets must be zero or one',cls_warm_width='MLP warm fit indicator width mismatch',
                 cls_warm_kind='MLP warm fit target kind mismatch',cls_warm_vector='MLP warm fit target kind mismatch')
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
                source_sha256={r:sha(Path(sklearn.__file__).parent/r) for r in ['neural_network/_multilayer_perceptron.py','neural_network/_base.py','neural_network/_stochastic_optimizers.py']})
    output=args.report or ROOT/f'validation/results/mlp_targets_{args.profile}.json';output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(len(checks),'groups;',len(workflows),'workflows;',len(records),'fits;',len(rejections),'diagnostics;',len(failures),'failures')
    assert not failures or args.diagnose
if __name__=='__main__':main()
