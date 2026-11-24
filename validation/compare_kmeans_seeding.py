"""Public weighted k-means++ seeding versus installed sklearn 1.6.1."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
from scipy import sparse
import sklearn
import sklearn.cluster._kmeans as source
import sklearn.metrics.pairwise as pairwise
from sklearn.cluster import kmeans_plusplus
from sklearn.utils.extmath import row_norms,stable_cumsum

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def representation(c):
    x=c['x']
    if c['rep']==0:return x
    if c['rep']==1:return sparse.csr_matrix(x)
    n,p=x.shape;ptr=np.arange(n+1)*2*p
    columns=np.tile(np.repeat(np.arange(p-1,-1,-1),2),n)
    values=np.repeat(x[:,::-1].ravel()*.5,2)
    return sparse.csr_matrix((values,columns,ptr),shape=x.shape)


def payload(c):
    x=c['x'];w=c['weights'];norms=c['norms']
    lines=[' '.join(map(str,[*x.shape,c['k'],c['seed'],c['trials'],c['rep'],
                            -1 if w is None else len(w),-1 if norms is None else len(norms)])),
           ' '.join(map(str,x.ravel(order='F')))]
    if w is not None:lines+=[' '.join(map(str,w))]
    if norms is not None:lines+=[' '.join(map(str,norms))]
    return '\n'.join(lines)+'\n'


def oracle(c):
    x=representation(c)
    return kmeans_plusplus(x,c['k'],sample_weight=c['weights'],x_squared_norms=c['norms'],
                           random_state=c['seed'],n_local_trials=None if c['trials']==-1 else c['trials'])


def identical_distance_certificate(c,actual_ids):
    """Replay every draw; alternate choices require identical finite update vectors.

    BLAS may round different rows of a matrix-vector product differently even
    when those rows are identical. This certificate admits only that exact
    algebraic tie, rather than a tolerance on distinct candidate objectives.
    """
    x=representation(c);n,p=x.shape;w=np.ones(n) if c['weights'] is None else c['weights']
    norms=row_norms(x,squared=True) if c['norms'] is None else c['norms']
    trials=2+int(np.log(c['k'])) if c['trials']==-1 else c['trials']
    rng=np.random.RandomState(c['seed']);first=rng.choice(n,p=w/w.sum())
    assert actual_ids[0]==first+1,c['name']
    center=x[[first]] if c['rep']==0 else x[[first]].toarray()
    closest=source._euclidean_distances(center,x,Y_norm_squared=norms,squared=True)
    potential=closest@w;ties=[]
    for step in range(1,c['k']):
        keys=rng.uniform(size=trials)*potential
        candidates=np.searchsorted(stable_cumsum(w*closest),keys)
        np.clip(candidates,None,closest.size-1,out=candidates)
        d=source._euclidean_distances(x[candidates],x,Y_norm_squared=norms,squared=True)
        np.minimum(closest,d,out=d);costs=d@w.reshape(-1,1);best=int(np.argmin(costs))
        positions=np.flatnonzero(candidates==actual_ids[step]-1)
        assert len(positions)>0,(c['name'],step,actual_ids,candidates+1)
        chosen=int(positions[0])
        assert np.array_equal(d[chosen],d[best]),(c['name'],step,'distinct candidate distance updates')
        if candidates[chosen]!=candidates[best]:
            assert np.isfinite(d[chosen]).all() and np.isfinite(costs[[chosen,best]]).all()
            ties.append(dict(step=step+1,actual=int(candidates[chosen]+1),source_best=int(candidates[best]+1),
                             candidate_ids=(candidates+1).tolist(),updated_distances_equal=True,
                             distance_vector_sha256=hashlib.sha256(d[chosen].tobytes()).hexdigest(),
                             actual_cost=float(costs[chosen,0]),source_best_cost=float(costs[best,0])))
        closest=d[chosen];potential=costs[chosen]
    return ties


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path)
    ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path)
    ap.add_argument('--case');args=ap.parse_args()
    exe=args.executable.resolve()
    paths=[exe,Path(__file__),Path(source.__file__),Path(pairwise.__file__),ROOT/'src/sklearn_kmeans_seeding.f90',
           ROOT/'src/sklearn_cluster.f90',ROOT/'src/sklearn_legacy_rng.f90',ROOT/'src/sklearn_ieee_math.f90',
           ROOT/'app/reference_kmeans_seeding_driver.f90']
    before={str(p):sha(p) for p in paths};cases=[]
    def add(name,x,k,seed=0,trials=-1,rep=0,weights=None,norms=None):
        if args.case is not None and name!=args.case:return
        c=dict(name=name,x=np.asarray(x,float),k=k,seed=seed,trials=trials,rep=rep,
               weights=None if weights is None else np.asarray(weights,float),norms=norms)
        if isinstance(norms,str):
            base=row_norms(representation(c),squared=True)
            c['norms']={'exact':base,'offset':base+.25,'negative':-np.ones(len(base)),'zeros':np.zeros(len(base))}[norms]
        with warnings.catch_warnings(),np.errstate(all='ignore'):
            warnings.simplefilter('ignore');centers,indices=oracle(c)
        cases.append((c,centers,indices+1))
    for seed in [0,1,7,42,314159]:
        rng=np.random.RandomState(seed)
        for n,p,k in [(1,1,1),(2,4,2),(7,3,3),(31,11,7),(129,5,7),(33,129,5)]:
            x=rng.randint(-8,9,size=(n,p)).astype(float);x[rng.uniform(size=x.shape)<.5]=0
            for trials in [-1,1,2,7]:
                for scenario in range(6):
                    w=[None,np.ones(n),np.r_[np.zeros(n-1),1.],rng.uniform(.01,3,n),-np.ones(n),-rng.uniform(.01,3,n)][scenario]
                    norm=[None,'exact','zeros','offset','negative',None][scenario]
                    for rep in [0,1,2]:
                        add(f'seed_{seed}_{n}_{p}_{k}_{trials}_{scenario}_{rep}',x,k,seed,trials,rep,w,norm)
        for n,p,k in [(17,1,3),(31,7,5),(19,33,7)]:
            x=rng.normal(size=(n,p))
            for trials in [-1,1,5]:
                for rep in [0,1]:add(f'float_{seed}_{n}_{p}_{trials}_{rep}',x,k,seed,trials,rep,rng.uniform(.1,2,n))
    for seed in [0,7,42,2147483647]:
        for kind in ['zero','constant','duplicates','line']:
            x=np.zeros((9,3))
            if kind=='constant':x[:]=3
            if kind=='duplicates':x[:,0]=np.arange(9)%2
            if kind=='line':x[:,0]=np.arange(9)
            for rep in [0,1,2]:
                for trials in [-1,1,7]:add(f'degenerate_{seed}_{kind}_{rep}_{trials}',x,9,seed,trials,rep)
    for scale in [1.e-180,1.e154,1.e308]:
        x=np.array([[0,1],[1,0],[1,1],[0,0]],float)*scale
        for rep in [0,1,2]:add(f'scale_{scale}_{rep}',x,3,7,rep=rep)
    rng=np.random.RandomState(2026)
    x=rng.randint(-2,3,size=(10000,1)).astype(float)
    for rep in [0,1]:add(f'long_{rep}',x,5,42,rep=rep,weights=rng.uniform(.1,2,len(x)))
    x=np.zeros((5,100000));x[0,0]=1;x[1,100]=2;x[2,9999]=3;x[3,-1]=4
    for rep in [0,1]:add(f'wide_{rep}',x,3,7,rep=rep)
    assert cases,args.case
    assert len({c['name'] for c,_,_ in cases})==len(cases)
    process=subprocess.run([str(exe)],input=str(len(cases))+'\n'+''.join(payload(c) for c,_,_ in cases),text=True,capture_output=True)
    assert process.returncode==0,(process.stdout[-2000:],process.stderr[-2000:])
    tokens=iter(process.stdout.split());checks=[];configurations=[];certificates=[]
    for c,want,ids in cases:
        k,p=int(next(tokens)),int(next(tokens));got_ids=np.array([int(next(tokens)) for _ in range(k)])
        centers=np.array([float(next(tokens)) for _ in range(k*p)]).reshape((k,p),order='F')
        preserved=int(next(tokens));name=c['name']
        exact=np.array_equal(got_ids,ids)
        if exact:
            np.testing.assert_array_equal(centers,want,err_msg=name)
        else:
            ties=identical_distance_certificate(c,got_ids)
            assert ties,(name,'divergence without an identical-vector greedy tie')
            certificates.append(dict(fixture=name,public_source_ids=ids.tolist(),actual_ids=got_ids.tolist(),
                                     replayed_steps=k,ties=ties))
        actual_x=representation(c)
        selected=actual_x[got_ids-1] if c['rep']==0 else actual_x[got_ids-1].toarray()
        np.testing.assert_array_equal(centers,selected,err_msg=name)
        assert preserved==1,name
        for field in ['indices_exact' if exact else 'indices_tie_certificate',
                      'centers_exact' if exact else 'centers_tie_certificate',
                      'selected_row_identity','input_rng_preserved']:
            checks.append(dict(fixture=name,field=field,passed=True))
        configurations.append(dict(name=name,n=c['x'].shape[0],p=p,k=k,seed=c['seed'],trials=c['trials'],
                                   representation=c['rep'],weighted=c['weights'] is not None,supplied_norms=c['norms'] is not None,
                                   exact_source_indices=bool(exact)))
    assert next(tokens,None) is None
    invalid=[]
    def bad(name,**kw):
        c=dict(name=name,x=np.arange(6.).reshape((3,2)),k=2,seed=0,trials=-1,rep=0,weights=None,norms=None)
        c.update(kw);invalid.append(c)
    bad('empty_rows',x=np.zeros((0,2)));bad('empty_columns',x=np.zeros((3,0)))
    bad('zero_clusters',k=0);bad('too_many_clusters',k=4);bad('zero_trials',trials=0);bad('negative_trials',trials=-2)
    bad('negative_seed',seed=-1)
    for rep in [0,1,2]:
        for number,name in [(np.nan,'nan'),(np.inf,'inf')]:
            x=np.arange(6.).reshape((3,2));x[0,0]=number;bad(f'data_{name}_{rep}',x=x,rep=rep)
        bad(f'zero_weights_{rep}',rep=rep,weights=np.zeros(3))
        bad(f'mixed_sign_weights_{rep}',rep=rep,weights=np.array([-1.,2.,2.]))
        bad(f'overflow_weight_sum_{rep}',rep=rep,weights=np.full(3,1.e308))
        bad(f'weight_length_{rep}',rep=rep,weights=np.ones(2))
        bad(f'norm_length_{rep}',rep=rep,norms=np.ones(2))
        for number,name in [(np.nan,'nan'),(np.inf,'inf')]:
            bad(f'weight_{name}_{rep}',rep=rep,weights=np.array([1.,number,1.]))
            bad(f'norm_{name}_{rep}',rep=rep,norms=np.array([1.,number,1.]))
    rejections=[]
    for c in invalid:
        with warnings.catch_warnings(),np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            try:oracle(c)
            except ValueError as error:source_error=type(error).__name__
            else:raise AssertionError(('source accepts invalid fixture',c['name']))
        result=subprocess.run([str(exe)],input='1\n'+payload(c),text=True,capture_output=True)
        out=result.stdout+result.stderr
        assert result.returncode!=0 and 'fscikit-learn:' in out,(c['name'],out)
        assert not any(s in out for s in ['SIGFPE','Fortran runtime error']),(c['name'],out)
        rejections.append(dict(name=c['name'],source_error=source_error,passed=True))
    assert before=={str(p):sha(p) for p in paths}
    report=dict(profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,
                configurations=configurations,rejections=rejections,identical_distance_certificates=certificates,hashes=before)
    path=args.report or ROOT/'validation/results'/f'kmeans_seeding_{args.profile}.json'
    path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {len(cases)} fixtures / {len(rejections)} diagnostics: PASS')
    print(f'{len(cases)-len(certificates)} exact seed sequences / {len(certificates)} identical-distance replay certificates')


if __name__=='__main__':main()
