"""NCA source objectives, gradients, initialization geometry and fitted transforms."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import warnings
import numpy as np
import sklearn
import scipy
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA

ROOT=Path(__file__).resolve().parents[1]
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile):
    options=['--profile','release'] if profile=='release' else ['--flag',FLAGS]
    r=subprocess.run(['fpm','run',*options,'--target','reference_nca_driver','--runner'],cwd=ROOT,
                     text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/s.strip().strip('"') for s in r.stdout.splitlines()
           if s.strip().strip('"').endswith(('reference_nca_driver','reference_nca_driver.exe'))]
    paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,r.stdout
    return paths[0]


def encode(c,x,y,q,a):
    header=[c['kind'],len(x),x.shape[1],c['k'],len(q),c['seed'],c['maxiter'],c['tol'],c['matrix'],c['warm']]
    return '\n'.join([' '.join(map(str,header)),c['init'],
                      ' '.join(format(v,'.17g') for v in x.ravel(order='F')),
                      ' '.join(map(str,y)),
                      ' '.join(format(v,'.17g') for v in q.ravel(order='F')),
                      ' '.join(format(v,'.17g') for v in a.ravel(order='F'))])+'\n'


def fixtures():
    for seed in [0,42,2026]:
        rng=np.random.RandomState(seed)
        for method in ['identity','random','pca','lda','auto','matrix']:
            for case in range(10):
                n,p=[(18,3),(31,5),(24,4),(40,3),(17,2)][case%5]
                k=min(2,p);y=np.arange(n)%3
                x=rng.normal(size=(n,p));q=rng.normal(size=(7,p))
                x[:,0]+=y*.5
                if case==2:x[:3]=x[0]
                if case==3:x*=.1
                if case==4:x+=4
                if case==5 and method not in ['lda','auto']:y[:]=7
                if case==6 and method not in ['lda','auto']:y=np.arange(n)
                if case==7:k=1
                if case==8:k=0
                # LDA cannot request p components with only three classes.
                if method=='lda' and k==0:k=2
                if method=='lda' and p==2:k=1
                a=rng.normal(size=(k or p,p))*.4
                c=dict(kind=0,n=n,p=p,seed=seed,init=method,k=k,matrix=int(method=='matrix'),case=case,
                       maxiter=1 if case==0 else (5 if case==1 else 35),tol=1e-7,warm=3 if case==9 else 0)
                yield c,x,y.astype(int),q,a
        for p in [1,2,4,7]:
            for case in range(8):
                n=[2,7,18,31][case%4];k=min(3,p)
                x=rng.normal(size=(n,p));a=rng.normal(size=(k,p))*.6;y=np.arange(n)%3
                if case==2:x[:]=0
                if case==3:y[:]=1
                if case==4:y=np.arange(n)
                if case==5:x+=20
                if case==6:a[:]=0
                if case==7:x*=.01
                c=dict(kind=1,n=n,p=p,seed=seed,init='identity',k=k,matrix=1,case=case,maxiter=1,tol=1e-7,warm=0)
                yield c,x,y.astype(int),rng.normal(size=(4,p)),a


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);ap.add_argument('--cases',type=int,nargs='*')
    args=ap.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile)
    vendor=json.loads((ROOT/'vendor/lbfgsb/PROVENANCE.json').read_text())
    expected_hashes=dict(vendor['normalized_sha256'])
    expected_hashes.update({patch['file']:patch['normalized_sha256']for patch in vendor.get('local_patches',[])})
    for file,digest in expected_hashes.items():
        data=(ROOT/'vendor/lbfgsb'/file).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(data).hexdigest()==digest,file
    checks=[];count=0;errors={};iterations=[];fit_comparisons=[];certificates=[]
    for c,x,y,q,a in fixtures():
        if args.cases is not None and c['case'] not in args.cases:continue
        r=subprocess.run([str(exe)],input=encode(c,x,y,q,a),cwd=ROOT,text=True,
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert r.returncode==0,(c,r.stdout)
        lines=r.stdout.splitlines()
        def arr(i):return np.fromstring(lines[i],sep=' ')
        def check(name,actual,expected,tol=3e-7):
            np.testing.assert_allclose(actual,expected,rtol=tol,atol=tol,err_msg=f'{c} {name}')
            errors[name]=max(errors.get(name,0.),float(np.max(abs(np.asarray(actual)-expected),initial=0)))
            checks.append(dict(**c,name=name,passed=True))
        oracle=NCA(n_components=c['k'] or None,random_state=c['seed'],max_iter=c['maxiter'],tol=c['tol'])
        if c['kind']==1:
            oracle.n_iter_=0
            f,g=oracle._loss_grad_lbfgs(a.ravel(),x,y[:,None]==y[None,:])
            check('objective',arr(0),f,3e-11);check('gradient',arr(1).reshape(a.shape,order='F'),g.reshape(a.shape),3e-10)
        else:
            meta=arr(0).astype(int);k=meta[0];initial=arr(1).reshape((k,x.shape[1]),order='F')
            final=arr(2).reshape(initial.shape,order='F')
            oracle.init=a if c['matrix'] else c['init'];oracle.random_state_=np.random.RandomState(c['seed'])
            expected_init=oracle._initialize(x,y,oracle.init)
            check('initial_metric',initial.T@initial,expected_init.T@expected_init,3e-9)
            if c['init'] not in ['pca','lda','auto']:check('initial_components',initial,expected_init,3e-12)
            # SVD row signs/bases may differ. Fit sklearn from the native matrix after
            # independently checking that its metric equals source initialization.
            oracle.init=initial.copy()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if c['warm']:oracle.max_iter=c['warm']
                oracle.fit(x,y)
                if c['warm']:
                    source_warm_start=oracle.components_.copy()
                    oracle.warm_start=True;oracle.max_iter=c['maxiter'];oracle.fit(x,y)
            if c['warm']:check('warm_start_source_prefix',arr(7).reshape(initial.shape,order='F'),source_warm_start,3e-9)
            source_f=oracle._loss_grad_lbfgs(oracle.components_.ravel(),x,y[:,None]==y[None,:])[0]
            native_values=[final,arr(3).reshape((len(x),k),order='F'),arr(4).reshape((len(q),k),order='F'),arr(5)]
            source_values=[oracle.components_,oracle.transform(x),oracle.transform(q),source_f]
            direct=all(np.allclose(v,w,rtol=3e-7,atol=3e-7) for v,w in zip(native_values,source_values))
            fit_comparisons.append(dict(**c,direct=direct,
                max_component_difference=float(np.max(abs(final-oracle.components_))),
                source_objective=float(source_f),native_objective=float(arr(5)[0])))
            if direct:
                for name,v,w in zip(['components','training_transform','query_transform','fitted_objective'],
                                    native_values,source_values):check(name,v,w)
            else:
                # Do not count numerically different fitted matrices as direct matches.
                # Validate source transforms at native parameters and certify its steps.
                mapped=NCA();mapped.components_=final;mapped.n_features_in_=x.shape[1]
                trace=arr(8).reshape((meta[6],final.size),order='F')
                terminal=trace[-1].reshape(final.shape) if len(trace) else arr(7).reshape(final.shape,order='F')
                check('components_from_accepted_trace',final,terminal,3e-12)
                check('training_transform_source_formula',native_values[1],mapped.transform(x),3e-11)
                check('query_transform_source_formula',native_values[2],mapped.transform(q),3e-11)
                f_native=oracle._loss_grad_lbfgs(final.ravel(),x,y[:,None]==y[None,:])[0]
                check('terminal_objective_source_formula',arr(5),f_native,3e-11)
            previous=arr(7).reshape(final.shape,order='F').ravel()
            history=arr(8).reshape((meta[6],final.size),order='F')
            prev_f,prev_g=oracle._loss_grad_lbfgs(previous,x,y[:,None]==y[None,:],-1.)
            initial_f=prev_f;steps=[];last_drop=0.
            searches=arr(10).reshape((meta[6],4),order='F')
            for step_index,point in enumerate(history):
                new_f,new_g=oracle._loss_grad_lbfgs(point,x,y[:,None]==y[None,:],-1.)
                step=point-previous;old_slope=float(prev_g@step);new_slope=float(new_g@step)
                roundoff=3e-9*max(1.,abs(prev_f),abs(new_f),np.linalg.norm(step)*np.linalg.norm(prev_g))
                assert old_slope<=roundoff,(c,'not descent',old_slope)
                assert new_f<=prev_f+1e-3*old_slope+roundoff,(c,'Armijo',new_f,prev_f,old_slope)
                status=lines[11+step_index].strip()
                curvature=abs(new_slope)<=.9*abs(old_slope)+roundoff
                if not curvature:
                    lower,upper,alpha,bracketed=searches[step_index]
                    if 'XTOL TEST SATISFIED' in status:
                        assert bracketed and upper-lower<=.1*upper+1e-12,(c,status,searches[step_index])
                    elif 'ROUNDING ERRORS PREVENT PROGRESS' in status:
                        assert bracketed and (alpha<=lower+1e-12 or alpha>=upper-1e-12),(c,status)
                    else:raise AssertionError((c,'Wolfe curvature',new_slope,old_slope,status))
                steps.append(dict(old_slope=old_slope,new_slope=new_slope,loss_before=float(prev_f),
                                  loss_after=float(new_f),roundoff=float(roundoff),line_search_status=status,
                                  strong_wolfe_curvature=bool(curvature),search=searches[step_index].tolist()))
                last_drop=prev_f-new_f;previous=point;prev_f=new_f;prev_g=new_g
            message=lines[9].strip()
            if 'CONVERGENCE' in message:
                if 'PROJECTED_GRADIENT' in message:
                    assert max(abs(prev_g),default=0)<=c['tol']+3e-9,(c,message,prev_g)
                else:
                    assert last_drop<=c['tol']*max(1.,abs(prev_f))+3e-9,(c,message,last_drop)
            else:
                assert meta[3]==c['maxiter'],(c,message,meta)
            assert prev_f<=initial_f+3e-9,(c,'loss did not improve')
            certificates.append(dict(**c,direct=direct,stopping_message=message,steps=steps))
            checks.append(dict(**c,name='source_line_search_certificates',passed=True))
            checks.append(dict(**c,name='source_stopping_certificate',passed=True))
            f,g=oracle._loss_grad_lbfgs(final.ravel(),x,y[:,None]==y[None,:])
            check('native_objective_source_formula',arr(5),f,3e-11)
            check('native_gradient_source_formula',arr(6).reshape(final.shape,order='F'),g.reshape(final.shape),3e-9)
            assert meta[1]==x.shape[1] and meta[2]==meta[3]+1 and meta[6]==meta[3],(c,meta)
            checks.append(dict(**c,name='callback_metadata',passed=True))
            iterations.append(dict(**c,native_iterations=int(meta[3]),source_iterations=oracle.n_iter_-1,
                                   native_evaluations=int(meta[4]),native_converged=bool(meta[5])))
        count+=1
        if count%50==0:print(f'{count} NCA fixtures passed',flush=True)
    source=Path(sklearn.__file__).parent
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,fixtures=count,
                cases_filter=args.cases,checks=checks,max_absolute_errors=errors,iterations=iterations,
                fit_comparisons=fit_comparisons,optimization_certificates=certificates,
                native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((source/p).read_bytes()).hexdigest()
                               for p in ['neighbors/_nca.py','discriminant_analysis.py','decomposition/_pca.py']},
                scipy_source_sha256={'optimize/_lbfgsb_py.py':hashlib.sha256(
                    (Path(scipy.__file__).parent/'optimize/_lbfgsb_py.py').read_bytes()).hexdigest()},
                vendor_provenance=vendor)
    (ROOT/'validation/results'/f'nca_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {count} NCA fixtures passed')


if __name__=='__main__':main()
