"""Dense/CSR GLM Newton-Cholesky versus installed sklearn; strict errors retained.

The native L-BFGS-B uses the installed sklearn loss contracts and maxls=50.
Source coefficients, predictions, scores and iteration counts are compared.
"""
from pathlib import Path
from unittest.mock import patch
import argparse, json, os, subprocess, warnings
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np
import sklearn
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._glm._newton_solver import NewtonCholeskySolver
from sklearn.linear_model import TweedieRegressor, PoissonRegressor, GammaRegressor


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--executable',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args();args.report.parent.mkdir(exist_ok=True,parents=True)
    records=[]
    configs=[]
    for power in (-2.,-1.,0.,.5,1.,1.5,2.,3.):
        for link in ('auto','identity','log'):
            for mode in range(5):
                configs.append(dict(family=0,power=power,link=link,mode=mode,intercept=1,steps=2,cap=300,alpha=.2,tol=1e-10))
    for family in (1,2):
        for mode in range(5):
            for intercept in (0,1):
                configs.append(dict(family=family,power=0.,link='auto',mode=mode,intercept=intercept,steps=2,cap=300,alpha=.2,tol=1e-10))
    for power,link in ((0.,'identity'),(0.,'log'),(-1.,'identity'),(.5,'log'),(3.,'log')):
        for cap in (0,1):
            configs.append(dict(family=0,power=power,link=link,mode=1,intercept=1,steps=1,cap=cap,alpha=.2,tol=1e-10))
    for power,link in ((1-1e-8,'log'),(1+1e-8,'log'),(2-1e-8,'log'),(2+1e-8,'log'),(-1.,'log'),(0.,'identity'),(1.,'log'),(2.,'identity')):
        configs.append(dict(family=0,power=power,link=link,mode=1,intercept=1,steps=2,cap=300,alpha=.2,tol=1e-10,signed=True))
    # Invalid controls, target domains, weight values and unsupported multiple
    # target columns are exercised separately by the native unit/guard driver.
    for variant in ('rank','illcondition','zero','curvature_log','curvature_identity'):
        configs.append(dict(family=0,power=(0. if variant=='curvature_log' else 2. if variant=='curvature_identity' else 1.),link=('identity' if variant=='curvature_identity' else 'log'),mode=1,intercept=(0 if variant=='curvature_log' else 1),steps=1,cap=100,alpha=0.,tol=1e-10,variant=variant))
    for scale in (1e3,1e4,1e5,1e6,1e7):
        configs.append(dict(family=0,power=1.5,link='identity',mode=1,intercept=1,steps=1,cap=3,alpha=.2,tol=1e-10,stress=scale))
    configs.append(dict(family=0,power=1.,link='log',mode=1,intercept=1,steps=1,cap=300,alpha=.2,tol=1e-10,constant=True))
    configs=[dict(c,repr=representation,base_index=index) for index,c in enumerate(configs) for representation in (0,1,2)]
    for index,c in enumerate(configs):
        rng=np.random.default_rng(100+c['base_index']//5);n,p=37,3
        x=rng.normal(size=(n,p))*.12
        y=np.exp(.3+x@np.array([.4,-.2,.6]))*(.8+.4*rng.random(n))
        if c['power']<=0 and c['family']==0:y[::9]*=-.1
        if c['power']>0 and c['power']<2 and c['family']==0:y[::11]=0
        weight=.2+rng.random(n)*2;weight[::7]=0
        if c.get('signed'):weight[::8]=-.05
        if c['mode'] in (2,4):weight[:]=1.7
        if c.get('stress'):
            n,p=5,1;x=np.array([-1.,-.5,0,.5,1.])[:,None]*c['stress'];y=np.array([.5,.7,1.,1.3,1.5]);weight=np.ones(n)
        input_path=args.report.with_name(args.report.stem+'_input.txt')
        output_path=args.report.with_name(args.report.stem+'_output.txt')
        if c.get('constant'):y=np.ones(n)
        if c.get('variant')=='rank':x[:,2]=2*x[:,1]
        if c.get('variant')=='illcondition':x[:,2]*=1e-10
        if c.get('variant')=='zero':x[:]=0
        if c.get('variant')=='curvature_log':x=np.abs(x)+.1;y=5+rng.random(n)
        if c.get('variant')=='curvature_identity':y[:20]=.01;y[20:]=10
        source_x=x
        if c['repr']==1:source_x=csr_matrix(x)
        if c['repr']==2:
            data=[];columns=[];pointers=[0]
            for row in x:
                for column in reversed(range(p)):
                    data.extend([.25*row[column],.75*row[column]]);columns.extend([column,column])
                data.append(0.);columns.append(0);pointers.append(len(data))
            source_x=csr_matrix((np.array(data),np.array(columns),np.array(pointers)),shape=x.shape)
        header=' '.join(str(c[k]) for k in ['family','power','link','alpha','tol','cap','intercept','mode','steps','repr'])
        with input_path.open('w') as f:
            f.write(f"{n} {p} {header}\n");np.savetxt(f,np.column_stack([x,y,weight]),fmt='%.17g')
        proc=subprocess.run([str(args.executable),str(input_path),str(output_path)],capture_output=True,text=True)
        options=dict(alpha=c['alpha'],tol=c['tol'],max_iter=c['cap'],fit_intercept=bool(c['intercept']),warm_start=c['steps']>1,solver='newton-cholesky')
        cls=(TweedieRegressor,PoissonRegressor,GammaRegressor)[c['family']]
        if c['family']==0:options.update(power=c['power'],link=c['link'])
        model=cls(**options);source=[];error=None;diagnostics=[]
        original=NewtonCholeskySolver.solve
        def tracked(solver,*args,**kwargs):
            result=original(solver,*args,**kwargs)
            diagnostics.append(dict(converged=bool(solver.converged),fallback=bool(solver.use_fallback_lbfgs_solve)))
            return result
        try:
            with patch.object(NewtonCholeskySolver,'solve',tracked),warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                for step in range(c['steps']):
                    w=None if c['mode']==0 else weight
                    model.fit(source_x,y if c['mode']<3 else y[:,None],sample_weight=w)
                    source.append(dict(coef=np.r_[model.coef_,model.intercept_].tolist(),prediction=model.predict(x+.017).tolist(),score=float(model.score(source_x,y,sample_weight=w)),n_iter=int(model.n_iter_),converged=diagnostics[-1]['converged'],fallback=diagnostics[-1]['fallback'],warnings=[str(w.message) for w in caught]))
                    caught.clear()
                    model.alpha*=.5;y=y*1.1
        except Exception as exc:error=f"{type(exc).__name__}: {exc}"
        differences=[];native=[]
        if error is None and proc.returncode==0:
            lines=output_path.read_text().splitlines()
            for step,expected in enumerate(source):
                values=[np.fromstring(lines[5*step+offset],sep=' ') for offset in (0,1,4,2)]
                actual=dict(coef=values[0].tolist(),prediction=values[1].tolist(),score=float(values[2][0]),metadata=values[3].tolist(),message=lines[5*step+3]);native.append(actual)
                if ('Newton->L-BFGS' in actual['message'])!=expected['fallback']:differences.append(f'{step}:fallback')
                for key in ('coef','prediction','score'):
                    if not np.allclose(actual[key],expected[key],atol=3e-8,rtol=3e-8,equal_nan=True):differences.append(f'{step}:{key}')
                if bool(actual['metadata'][2])!=expected['converged']:differences.append(f'{step}:converged')
                if actual['metadata'][1]!=expected['n_iter']:differences.append(f'{step}:n_iter')
                if actual['metadata'][0]!=p:differences.append(f'{step}:n_features')
        elif (error is None)!=(proc.returncode==0):differences.append('acceptance')
        if proc.returncode not in (0,1) or (proc.returncode==1 and 'fscikit-learn:' not in proc.stdout):differences.append('abnormal_native_exit')
        partial=None
        if proc.returncode and output_path.exists():
            parts=output_path.read_text().splitlines()
            if len(parts)>=4:
                partial=dict(coef=np.fromstring(parts[0],sep=' ').tolist(),prediction=np.fromstring(parts[1],sep=' ').tolist(),metadata=np.fromstring(parts[2],sep=' ').tolist(),message=parts[3])
        records.append(dict(index=index,config=c,input=input_path.read_text(),source=source,source_error=error,native=native,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,native_partial=partial,native_output=output_path.read_text() if output_path.exists() else None,differences=differences))
    report=dict(sklearn_version=sklearn.__version__,atol=3e-8,rtol=3e-8,records=records,failures=[r['index'] for r in records if r['differences']])
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(f"GLM Newton-Cholesky: {len(records)} configurations; strict failures {report['failures']}")
    raise SystemExit(bool(report['failures']))

if __name__=='__main__':main()
