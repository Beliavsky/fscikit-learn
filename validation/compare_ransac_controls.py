"""Actual sklearn OLS RANSAC fits, subset traces, callbacks and stopping."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.linear_model import LinearRegression,RANSACRegressor

def sparse(x,style):
    if style==0:return x
    if style==1:return sp.csr_matrix(x)
    values=[];indices=[];ptr=[0]
    for row in x:
        for j in range(x.shape[1]-1,-1,-1):
            if row[j]==0:continue
            values.extend([.25*row[j],.75*row[j]]);indices.extend([j,j])
        values.append(0.);indices.append(0);ptr.append(len(values))
    return sp.csr_matrix((values,indices,ptr),shape=x.shape)

def fixtures():
    profiles=[dict(),dict(min_kind=2,min_real=.25,threshold=.3,loss=1,probability=1.,trials=12),
              dict(min_kind=1,min_int=5,threshold=-1.,data=1,skips=2),
              dict(min_kind=2,min_real=1.,threshold=0.,trials=5),
              dict(min_kind=1,min_int=3,loss=2,model=1,stop_score=.9),
              dict(min_kind=2,min_real=.6,loss=3,data=1,model=1,skips=0)]
    for seed,style,(t,matrix),intercept,weighted,profile in itertools.product([0,42],[0,1],[(1,0),(1,1),(3,1)],[0,1],[0,1,2],profiles):
        yield dict(seed=seed,style=style,t=t,matrix=matrix,intercept=intercept,weighted=weighted)|profile
    for n,m in [(200,1),(200,2),(200,3),(200,198),(200,199),(200,200)]:
        for seed in [0,42]:yield dict(n=n,p=2,seed=seed,min_kind=1,min_int=m,trials=4,threshold=.15)
    for data,model,skips,trials in itertools.product([0,1,2],[0,1,2],[0,1],[0,1,7]):
        yield dict(data=data,model=model,skips=skips,trials=trials)
    # Append raw-CSR storage cases without moving the original 498 fixtures.
    for seed,(t,matrix),intercept,weighted,profile in itertools.product([0,42],[(1,0),(1,1),(3,1)],[0,1],[0,1,2],profiles):
        yield dict(seed=seed,style=2,t=t,matrix=matrix,intercept=intercept,weighted=weighted)|profile

def config(c):
    return dict(n=31,p=3,t=1,q=7,style=0,matrix=0,weighted=0,intercept=1,seed=0,min_kind=0,min_int=0,min_real=0.,trials=40,skips=2147483647,stop_n=2147483647,threshold=.15,probability=.99,stop_score=1e300,loss=0,data=0,model=0)|c

def arrays(c):
    rng=np.random.RandomState(2917+c['n']);x=rng.normal(size=(c['n'],c['p']));x[:,0]=np.arange(c['n'])
    coef=np.linspace(-.02,1.3,c['p']);y=np.column_stack([x@coef+.7+j*.3+.02*rng.normal(size=c['n']) for j in range(c['t'])])
    y[::6]+=2.;w=np.resize([.5,1.5,1.,2.,3.],c['n'])
    if c['weighted']==2:w.fill(2.)
    q=rng.normal(size=(c['q'],c['p']));q[:,0]=np.linspace(0,c['n'],c['q'])
    return x,y,w,q

def wire(c,a):
    keys=['n','p','t','q','style','matrix','weighted','intercept','seed','min_kind','min_int','min_real','trials','skips','stop_n','threshold','probability','stop_score','loss','data','model']
    return ' '.join(str(c[k]) for k in keys)+'\n'+''.join(' '.join(format(v,'.17g') for v in x.ravel(order='F'))+'\n' for x in a)

def oracle(c,a):
    x,y,w,q=a;trace=[];counts=[0,0,0]
    def data_valid(xx,yy):
        xx=xx.toarray() if sp.issparse(xx) else xx;counts[0]+=1;trace.append(np.rint(xx[:,0]).astype(int)+1)
        return False if c['data']==2 else int(round(xx[0,0]))%3!=0 if c['data']==1 else True
    def model_valid(model,xx,yy):
        counts[1]+=1
        return False if c['model']==2 else np.max(np.abs(model.coef_))<3 if c['model']==1 else True
    def loss(actual,predicted):
        counts[2]+=1;result=.75*np.abs(actual-predicted)
        if result.ndim==2:result=result.sum(axis=1)
        if c['loss']==3:result[0]=np.nan
        return result
    minimum=None if c['min_kind']==0 else c['min_int'] if c['min_kind']==1 else c['min_real']
    model=RANSACRegressor(estimator=LinearRegression(fit_intercept=bool(c['intercept'])),min_samples=minimum,
        residual_threshold=None if c['threshold']==-1 else c['threshold'],max_trials=c['trials'],max_skips=c['skips'],
        stop_n_inliers=c['stop_n'],stop_score=c['stop_score'],stop_probability=c['probability'],random_state=c['seed'],
        loss=loss if c['loss']>=2 else 'squared_error' if c['loss']==1 else 'absolute_error',
        is_data_valid=data_valid,is_model_valid=model_valid)
    xx=sparse(x,c['style']);qq=sparse(q,c['style']);target=y if c['matrix'] else y[:,0]
    weight=None if c['weighted']==0 else w if c['weighted']==1 else w[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(xx,target,sample_weight=weight)
        plain_score=model.score(xx,target)
        with sklearn.config_context(enable_metadata_routing=True):
            model.estimator.set_score_request(sample_weight=True)
            weighted_score=model.score(xx,target,sample_weight=np.abs(w)+.5)
        m=c['p']+1 if minimum is None else int(np.ceil(minimum*c['n'])) if 0<minimum<1 else int(minimum)
        fields=dict(metadata=[c['p'],c['t'],model.n_trials_,model.n_skips_no_inliers_,model.n_skips_invalid_data_,model.n_skips_invalid_model_,m,int(sum([model.n_skips_no_inliers_,model.n_skips_invalid_data_,model.n_skips_invalid_model_])>c['skips'])],
            coef=np.asarray(model.estimator_.coef_).ravel().tolist(),intercept=np.broadcast_to(model.estimator_.intercept_,(c['t'],)).tolist(),
            prediction=np.asarray(model.predict(qq)).ravel().tolist(),train=np.asarray(model.predict(xx)).ravel().tolist(),mask=model.inlier_mask_.astype(int).tolist(),
            threshold=[np.median(np.abs(target-np.median(target))) if c['threshold']==-1 else c['threshold']],
            scores=[plain_score,weighted_score],
            subsets=np.concatenate(trace).tolist(),callbacks=counts)
    return fields,[str(z.message) for z in caught]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--limit',type=int);a=ap.parse_args()
    cases=list(fixtures());cases=cases[:a.limit] if a.limit else cases;records=[]
    for c in map(config,cases):
        data=arrays(c);error=None;expected=None;seen=[]
        try:expected,seen=oracle(c,data)
        except (ValueError,ZeroDivisionError,UnboundLocalError) as e:error=dict(type=type(e).__name__,message=str(e))
        records.append(dict(case=c,input=wire(c,data),source=expected,source_error=error,warnings=seen,differences=[]))
    valid=[r for r in records if r['source'] is not None];payload=str(len(valid))+'\n'+''.join(r['input'] for r in valid)
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.with_suffix('.in').write_text(payload)
    failed_native=0;stdout=[];stderr=[]
    for r in valid:
        p=subprocess.run([str(a.executable)],input='1\n'+r['input'],text=True,capture_output=True,timeout=60)
        r['native_returncode']=p.returncode;r['native_stdout']=p.stdout;r['native_stderr']=p.stderr
        stdout.append(p.stdout);stderr.append(p.stderr);r['native']={}
        if p.returncode:
            failed_native+=1;r['differences'].append('unexpected native rejection');continue
        fields=list(r['source']);rows=p.stdout.splitlines();integer={'metadata','mask','subsets','callbacks'}
        if len(rows)!=len(fields):r['differences'].append('native field count')
        for k,field in enumerate(fields):
            got=np.fromstring(rows[k],sep=' ') if k<len(rows) else np.array([]);want=np.asarray(r['source'][field]);r['native'][field]=got.tolist()
            same=got.shape==want.shape and (np.array_equal(got,want) if field in integer else np.allclose(got,want,rtol=3e-8,atol=3e-8,equal_nan=True))
            if not same:r['differences'].append(field)
    a.report.with_suffix('.stdout').write_text(''.join(stdout));a.report.with_suffix('.stderr').write_text(''.join(stderr))
    for r in records:
        if r['source'] is not None:continue
        result=subprocess.run([str(a.executable)],input='1\n'+r['input'],text=True,capture_output=True,timeout=30);output=result.stdout+result.stderr
        r['native_error']=dict(returncode=result.returncode,output=output)
        if not(result.returncode and 'fscikit-learn:' in output and not any(v in output for v in ['SIGFPE','SIGSEGV','Fortran runtime error'])):r['differences'].append('expected controlled Source rejection')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),a.executable,Path('src/sklearn_linear_model.f90'),Path('src/sklearn_bagging_sampling.f90'),Path('app/reference_ransac_controls_driver.f90')]
    d=dict(configurations=len(records),successful_fits=len(valid),source_rejections=len(records)-len(valid),failed_configurations=sum(bool(x['differences']) for x in records),native_fit_failures=failed_native,records=records,input_sha256=hashlib.sha256(payload.encode()).hexdigest(),sklearn=sklearn.__version__,rtol=3e-8,atol=3e-8,index_origin='Native subset traces are one-based; Source indices are shifted by one',scalar_intercept_storage='Scalar Source intercepts broadcast to the native per-target vector',fit_metadata_routing=False,weighted_score_metadata_routing=True,sha256={str(p):sha(p) for p in files})
    a.report.write_text(json.dumps(d,indent=2)+'\n');print({k:v for k,v in d.items() if k not in ['records','sha256']});print('first failures',[(i,r['case'],r['differences']) for i,r in enumerate(records) if r['differences']][:8]);assert d['failed_configurations']==d['native_fit_failures']==0
if __name__=='__main__':main()
