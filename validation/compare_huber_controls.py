"""Compare real sklearn Huber fits, weights, warm transitions and CSR state."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, subprocess, warnings
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.linear_model import HuberRegressor
import sklearn.linear_model._huber as source

FIELDS=['coef','parameters','train','query','outliers','metadata','scores']
ROOT=Path(__file__).resolve().parents[1]

def matrix(x,style):
    if style==0:return x.copy()
    if style==1:return sp.csr_matrix(x)
    data=[];indices=[];ptr=[0]
    for row in x:
        for j in range(x.shape[1]-1,-1,-1):
            if row[j]==0:continue
            data.extend([.25*row[j],.75*row[j]]);indices.extend([j,j])
        data.append(0.);indices.append(0);ptr.append(len(data))
    return sp.csr_matrix((data,indices,ptr),shape=x.shape)

def fixtures():
    profiles=[(1.35,.0001,1e-5,100),(1.8,.2,1e-8,200),(2.,0.,1e-6,100),
              (1.,.01,1e-5,100),(1.35,.1,0.,100),(1.35,.1,1e-5,0),
              (1.35,.1,1e-5,1),(1.35,.1,1e-5,5)]
    for seed,representation,intercept,weights,profile in itertools.product([0,42],range(3),[0,1],range(4),profiles):
        yield dict(seed=seed,representation=representation,intercept=intercept,weights=weights,profile=profile,kind='cold',snapshots=1)
    for seed,representation,intercept,weights,kind in itertools.product([0,42],range(3),[0,1],[0,1,2],['warm','warm_toggle','warm_cold']):
        yield dict(seed=seed,representation=representation,intercept=intercept,weights=weights,profile=profiles[1],kind=kind,snapshots=3)
    for representation,intercept,scenario in itertools.product(range(3),[0,1],['zero_weights','one_weight','constant','rank_deficient','wide','zeros']):
        yield dict(seed=7,representation=representation,intercept=intercept,weights=1,profile=profiles[0],kind='cold',snapshots=1,scenario=scenario)

def arrays(c,step):
    n,p=(11,15) if c.get('scenario')=='wide' else (31,4)
    rng=np.random.RandomState(c['seed']);x=rng.normal(size=(n,p))*.6
    x[np.abs(x)<.2]=0
    query=rng.normal(size=(9,p))*.5
    y=x@np.linspace(-1.,2.,p)+.7+.08*rng.normal(size=n);y[::7]+=3
    w=.2+rng.uniform(size=n)*2
    if c['weights']==2:w[::5]=0
    if c['weights']==3:w.fill(2.5)
    scenario=c.get('scenario')
    if scenario=='zero_weights':w.fill(0.)
    if scenario=='one_weight':w.fill(0.);w[0]=1.
    if scenario=='constant':y.fill(2.)
    if scenario=='rank_deficient':x[:,1]=x[:,0];query[:,1]=query[:,0]
    if scenario=='zeros':x.fill(0.);query.fill(0.)
    if step:y=y*(1+.1*step)+.04*step
    return x,y,w,query

def snapshots(c):
    result=[]
    for step in range(c['snapshots']):
        x,y,w,q=arrays(c,step);n,p=x.shape;eps,alpha,tol,cap=c['profile']
        warm=int(c['kind']!='cold');intercept=c['intercept']
        if c['kind']=='warm_toggle' and step==1:intercept=1-intercept
        if c['kind']=='warm_cold' and step==2:warm=0
        kind=min(c['weights'],2)
        if c['weights']==2:kind=1
        if c['weights']==3:kind=2
        header=[n,p,len(q),c['representation'],(c['representation']+step)%3,step%2,kind,intercept,warm,eps,alpha,tol,cap]
        payload=' '.join(map(str,header))+'\n'+''.join(' '.join(format(v,'.17g') for v in a.ravel(order='F'))+'\n' for a in [x,y,w,q])
        result.append((payload,header,(x,y,w,q)))
    return result

def oracle(c,steps):
    model=HuberRegressor();records=[];optimizer=[];original=source.optimize.minimize
    def observe(*args,**kwargs):
        value=original(*args,**kwargs);optimizer.append(value);return value
    source.optimize.minimize=observe
    try:
        for payload,h,(x,y,w,q) in steps:
            _,_,_,representation,qrep,column,kind,intercept,warm,eps,alpha,tol,cap=h
            model.set_params(epsilon=eps,alpha=alpha,tol=tol,max_iter=cap,fit_intercept=bool(intercept),warm_start=bool(warm))
            xx=matrix(x,representation);qq=matrix(q,qrep);weight=None if kind==0 else w if kind==1 else w[0]
            with warnings.catch_warnings(record=True) as caught,np.errstate(all='ignore'):
                warnings.simplefilter('always');model.fit(xx,y[:,None] if column else y,sample_weight=weight)
                opt=optimizer[-1]
                fields=dict(coef=model.coef_.tolist(),parameters=[model.intercept_,model.scale_,opt.fun],
                    train=model.predict(xx).tolist(),query=model.predict(qq).tolist(),outliers=model.outliers_.astype(int).tolist(),
                    metadata=[model.n_features_in_,model.n_iter_,int(opt.status==0)],scores=[model.score(xx,y),model.score(xx,y,sample_weight=np.abs(w)+.5)])
            records.append(dict(source=fields,warnings=[str(v.message) for v in caught],optimizer_status=int(opt.status),optimizer_message=str(opt.message)))
    finally:source.optimize.minimize=original
    return records

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--limit',type=int)
    a=ap.parse_args();cases=list(fixtures());cases=cases[:a.limit] if a.limit else cases;records=[]
    for index,c in enumerate(cases):
        steps=snapshots(c);wire=str(len(steps))+'\n'+''.join(v[0] for v in steps)
        try:expected=oracle(c,steps);error=None
        except (ValueError,IndexError) as e:expected=None;error=dict(type=type(e).__name__,message=str(e))
        records.append(dict(case=c,input=wire,source=expected,source_error=error,differences=[]))
    valid=[r for r in records if r['source'] is not None];text=str(len(valid))+'\n'+''.join(r['input'] for r in valid)
    proc=subprocess.run([str(a.executable)],input=text,text=True,capture_output=True,timeout=300)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.with_suffix('.in').write_text(text);a.report.with_suffix('.stdout').write_text(proc.stdout);a.report.with_suffix('.stderr').write_text(proc.stderr)
    lines=iter(proc.stdout.splitlines())
    for r in valid:
        r['native']=[]
        for step in r['source']:
            actual={}
            for field in FIELDS:
                line=next(lines,'');got=np.fromstring(line,sep=' ');want=np.asarray(step['source'][field]);actual[field]=got.tolist()
                exact=field in ['outliers','metadata']
                same=got.shape==want.shape and (np.array_equal(got,want) if exact else np.allclose(got,want,rtol=3e-8,atol=3e-8,equal_nan=True))
                if not same:r['differences'].append(dict(step=len(r['native']),field=field))
            r['native'].append(actual)
    assert next(lines,None) is None,'unexpected extra native output'
    for r in records:
        if r['source'] is not None:continue
        p=subprocess.run([str(a.executable)],input='1\n'+r['input'],text=True,capture_output=True,timeout=60);output=p.stdout+p.stderr
        r['native_error']=dict(returncode=p.returncode,output=output)
        if not(p.returncode and 'fscikit-learn:' in output and not any(v in output for v in ['SIGFPE','SIGSEGV','Fortran runtime error'])):r['differences'].append(dict(field='expected Source fit rejection'))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),a.executable,ROOT/'src/sklearn_huber_solver.f90',ROOT/'src/sklearn_linear_model.f90',ROOT/'app/reference_huber_controls_driver.f90',Path(source.__file__)]
    d=dict(configurations=len(records),successful_sequences=len(valid),successful_fits=sum(len(r['source']) for r in valid),source_rejections=len(records)-len(valid),failed_configurations=sum(bool(r['differences']) for r in records),native_returncode=proc.returncode,records=records,input_sha256=hashlib.sha256(text.encode()).hexdigest(),sklearn=sklearn.__version__,rtol=3e-8,atol=3e-8,sha256={str(p):sha(p) for p in files})
    a.report.write_text(json.dumps(d,indent=2)+'\n');print({k:v for k,v in d.items() if k not in ['records','sha256']})
    print('first differences',[(i,r['case'],r['differences'][:3]) for i,r in enumerate(records) if r['differences']][:5])
    assert d['native_returncode']==d['failed_configurations']==0
if __name__=='__main__':main()
