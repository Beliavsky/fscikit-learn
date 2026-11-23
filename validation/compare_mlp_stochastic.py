"""Strict minibatch MLP coefficients, optimizer states and histories vs sklearn 1.6.1."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
import scipy
from scipy.special import expit
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def reference_fields(model,q,kind,solver):
    fields={}
    for l,(w,b) in enumerate(zip(model.coefs_,model.intercepts_),1):
        fields[f'weight_{l}']=w.ravel(order='F');fields[f'bias_{l}']=b
        state=model._optimizer;layers=len(model.coefs_)
        if solver=='adam':
            fields[f'moment_{l}']=state.ms[l-1].ravel(order='F')
            fields[f'variance_{l}']=state.vs[l-1].ravel(order='F')
            fields[f'bias_moment_{l}']=state.ms[layers+l-1]
            fields[f'bias_variance_{l}']=state.vs[layers+l-1]
        else:
            fields[f'moment_{l}']=state.velocities[l-1].ravel(order='F')
            fields[f'variance_{l}']=np.zeros(w.size)
            fields[f'bias_moment_{l}']=state.velocities[layers+l-1]
            fields[f'bias_variance_{l}']=np.zeros(b.size)
    fields['loss_curve']=np.array(model.loss_curve_)
    fields['metadata']=np.array([model.n_iter_,model.t_,model.n_features_in_,model.n_outputs_,model.n_layers_,model.loss_,model.best_loss_])
    if kind==0:fields['prediction']=model.predict(q)
    else:
        fields['prediction']=model.predict(q);fields['classes']=model.classes_
        fields['probability']=model.predict_proba(q).ravel(order='F')
        with np.errstate(divide='ignore'):fields['log_probability']=model.predict_log_proba(q).ravel(order='F')
    return fields

class PerturbedRegressor(MLPRegressor):
    """Diagnostic only: move specified initialization values by one ULP."""
    direction=1
    perturb_mode='all'
    draw_layer=0
    def _init_coef(self,*args):
        w,b=super()._init_coef(*args);self.draw_layer+=1
        direction=self.direction*np.inf
        if self.perturb_mode=='all':w=np.nextafter(w,direction);b=np.nextafter(b,direction)
        elif self.perturb_mode=='first' and self.draw_layer==1:b[0]=np.nextafter(b[0],direction)
        elif self.perturb_mode=='last' and self.draw_layer==len(self.hidden_layer_sizes)+1:b[0]=np.nextafter(b[0],direction)
        return w,b

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True,type=Path)
    parser.add_argument('--profile',default='checked');parser.add_argument('--report',type=Path)
    parser.add_argument('--fixture',type=int);parser.add_argument('--epochs',type=int)
    args=parser.parse_args();exe=args.executable.resolve()
    assert expit(-710.)==0 and expit(-709.)>0
    fixtures=[];inputs=[];references=[];roundoff={}
    # Adam defaults/custom moments; SGD without momentum, classical/Nesterov,
    # inverse scaling and adaptive reductions (including eventual rate stopping).
    optimizers=[dict(solver='adam'),dict(solver='adam',beta_1=.6,beta_2=.83,epsilon=.003),
                dict(solver='sgd',momentum=0,nesterovs_momentum=False),
                dict(solver='sgd',momentum=.73,nesterovs_momentum=False),
                dict(solver='sgd',momentum=.81,nesterovs_momentum=True),
                dict(solver='sgd',learning_rate='invscaling',power_t=.37),
                dict(solver='sgd',learning_rate='adaptive'),
                dict(solver='sgd',learning_rate='adaptive',tol=2.,n_iter_no_change=1,max_iter=65)]
    for index,(seed,kind,activation,opt,batch,shuffle) in enumerate(itertools.product(
            [0,7,42],range(3),['identity','logistic','tanh','relu'],range(len(optimizers)),[0,1,6,17,29],[False,True])):
        if args.fixture is not None and index!=args.fixture:continue
        n=203 if batch==0 else 17;p=3;nq=9
        rng=np.random.RandomState(seed+91);x=rng.normal(0,.4,(n,p));q=rng.normal(0,.5,(nq,p))
        if kind>0:
            q[0]=[-10000.,0.,0.];q[1]=[10000.,0.,0.]
        if index%11==0:x[:]=0
        elif index%11==1:x[:,2]=0
        elif index%11==2:x[::3]=x[0]
        y=np.sin(x[:,0])+x[:,1]*.3 if kind==0 else 3+7*(np.arange(n)% (kind+1))
        hidden=[(),(4,),(5,3)][index%3]
        control=dict(hidden_layer_sizes=hidden,activation=activation,random_state=seed,
                     max_iter=13,batch_size='auto' if batch==0 else batch,shuffle=shuffle,
                     learning_rate_init=.025,alpha=[0.,.001,.4][index%3],tol=0.,n_iter_no_change=4,
                     momentum=.9,nesterovs_momentum=True,power_t=.5,beta_1=.9,beta_2=.999,
                     epsilon=1e-8,learning_rate='constant')
        control.update(optimizers[opt])
        if index%13==0:control.update(tol=.1,n_iter_no_change=2)
        if args.epochs is not None:control['max_iter']=args.epochs
        model=(MLPRegressor if kind==0 else MLPClassifier)(**control)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning);warnings.simplefilter('ignore',UserWarning)
            model.fit(x,y)
        fields=reference_fields(model,q,kind,control['solver'])
        if kind==0 and opt==1 and batch==1 and np.all(x==0):
            # Degenerate custom-Adam trajectories can amplify roundoff. Compare
            # to the unchanged baseline and retain its independently measured
            # +/- one-ULP initialization sensitivity, without widening other cases.
            radii={name:np.zeros_like(value,dtype=float) for name,value in fields.items()}
            for mode,direction in itertools.product(['all','first','last'],[-1,1]):
                probe=PerturbedRegressor(**control);probe.perturb_mode=mode;probe.direction=direction
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',ConvergenceWarning);probe.fit(x,y)
                altered=reference_fields(probe,q,kind,control['solver'])
                for name,value in fields.items():radii[name]=np.maximum(radii[name],np.abs(altered[name]-value))
            if index==253:
                assert max(np.max(a) for name,a in radii.items() if name.startswith('weight_'))>1e-8
            roundoff[len(fixtures)]=radii
        c=control
        header=[n,p,kind,len(hidden),nq,c['max_iter'],batch,seed,int(shuffle),int(c['nesterovs_momentum']),c['n_iter_no_change']]
        lines=[' '.join(map(str,header)),f"{c['solver']} {activation} {c['learning_rate']}",
               ' '.join(format(c[a],'.17g') for a in ['learning_rate_init','alpha','tol','momentum','power_t','beta_1','beta_2','epsilon'])]
        if hidden:lines.append(' '.join(map(str,hidden)))
        lines.extend(' '.join(format(v,'.17g') for v in row) for row in x)
        lines.append(' '.join(format(v,'.17g') for v in y))
        lines.extend(' '.join(format(v,'.17g') for v in row) for row in q)
        inputs.extend(lines);references.append(fields)
        fixtures.append(dict(index=index,seed=seed,kind=kind,activation=activation,optimizer=opt,batch=batch,shuffle=shuffle,hidden=hidden))
    run=subprocess.run([str(exe)],input=str(len(fixtures))+'\n'+'\n'.join(inputs)+'\n',text=True,capture_output=True,cwd=ROOT)
    assert run.returncode==0,(run.returncode,run.stdout[-3000:],run.stderr[-3000:])
    checks=[];actual={};case=-1
    for line in run.stdout.splitlines():
        parts=line.split()
        if not parts:continue
        if parts[0]=='BEGIN':case=int(parts[1])-1;actual={}
        elif parts[0]=='END':
            ref=references[case];assert set(actual)==set(ref),(case,set(actual)^set(ref))
            for name,want in ref.items():
                got=actual[name];assert got.shape==want.shape,(case,name,got.shape,want.shape,fixtures[case])
                np.testing.assert_array_equal(np.isfinite(got),np.isfinite(want),err_msg=str((case,name)))
                strict_passed=True
                try:np.testing.assert_allclose(got,want,rtol=5e-9,atol=5e-11,err_msg=str((case,name,fixtures[case])))
                except AssertionError:
                    if case not in roundoff:raise
                    strict_passed=False
                    radius=roundoff[case][name]
                    assert np.all(np.abs(got-want)<=radius+5e-11+5e-9*np.abs(want)),(case,name,got,want,radius)
                finite=np.isfinite(want);error=float(np.max(np.abs(got[finite]-want[finite]),initial=0))
                checks.append(dict(fixture=case,field=name,passed=True,strict_passed=strict_passed,max_abs_error=error))
        else:
            assert parts[0] not in actual;actual[parts[0]]=np.array(list(map(float,parts[1:])))
    assert case+1==len(fixtures)
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
    rejections=[]
    for mode,message in invalid.items():
        run=subprocess.run([str(exe),mode],text=True,capture_output=True,cwd=ROOT)
        text=run.stdout+run.stderr
        assert run.returncode!=0 and message in text,(mode,text)
        assert not any(v in text for v in ['SIGFPE','Fortran runtime error']), (mode,text)
        rejections.append(dict(case=mode,passed=True))
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,fixtures=len(fixtures),
                nonfinite_reference_values=sum(int(np.count_nonzero(~np.isfinite(a))) for fields in references for a in fields.values()),
                checks=checks,rejections=rejections,configurations=fixtures,
                rounding_envelopes={str(k):{name:a.tolist() for name,a in fields.items()} for k,fields in roundoff.items()},
                executable_sha256=sha(exe),validator_sha256=sha(__file__),
                native_source_sha256=sha(ROOT/'src/sklearn_neural_network.f90'),
                source_sha256={r:sha(Path(sklearn.__file__).parent/r) for r in
                    ['neural_network/_multilayer_perceptron.py','neural_network/_stochastic_optimizers.py','neural_network/_base.py']})
    output=args.report or ROOT/f'validation/results/mlp_stochastic_{args.profile}.json'
    output.parent.mkdir(exist_ok=True,parents=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} MLP result groups across {len(fixtures)} fixtures; {len(rejections)} diagnostics passed')
    print('Groups within measured sklearn one-ULP propagation envelope:',sum(not c['strict_passed'] for c in checks))
    print('Maximum coefficient error:',max(c['max_abs_error'] for c in checks if c['field'].startswith('weight_')))

if __name__=='__main__':main()
