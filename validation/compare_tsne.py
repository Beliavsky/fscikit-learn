"""Dense exact t-SNE components, stable full trajectories and trustworthiness."""
from pathlib import Path
import argparse
import json
import subprocess
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
import numpy as np
import sklearn
from scipy.linalg import norm
from scipy.spatial.distance import pdist,squareform
from sklearn.manifold import TSNE, trustworthiness
from sklearn.manifold._t_sne import _joint_probabilities, _kl_divergence, _gradient_descent
from sklearn.metrics import pairwise_distances
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve():
    proc=subprocess.run(['fpm','run','--flag',FLAGS,'--target','reference_tsne_driver','--runner'],cwd=ROOT,
                        text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/line.strip().strip('"') for line in proc.stdout.splitlines()
           if line.strip().strip('"').endswith(('reference_tsne_driver','reference_tsne_driver.exe'))]
    paths=[p.resolve() for p in paths if p.is_file()]; assert len(paths)==1,proc.stdout
    return paths[0]


def fixtures(seed):
    rng=np.random.default_rng(seed)
    for n in [7,16,31]:
        for k in [1,2,4]:
            x=rng.normal(size=(n,6)); initial=rng.normal(scale=.02,size=(n,k))
            yield x,initial,dict(k=k,metric=['euclidean','manhattan','cosine'][k%3],init_mode=0,
                                perplexity=max(2,n/4),early=1.,rate=.5,min_norm=0.,steps=300,single=k!=2)
    for mode in [1,2]:
        for k in [1,2,4]:
            n=17; x=rng.normal(size=(n,6)); initial=rng.normal(scale=.02,size=(n,k)).astype(np.float32)
            yield x,initial,dict(k=k,metric='euclidean',init_mode=mode,perplexity=n-1.,early=1.,rate=-1.,
                                min_norm=1.e10,steps=350,single=True)
    for metric in ['euclidean','precomputed','manhattan']:
        x=rng.normal(size=(12,5)); initial=rng.normal(scale=.02,size=(12,2))
        if metric=='precomputed': x=pairwise_distances(x)
        yield x,initial,dict(k=2,metric=metric,init_mode=0,perplexity=3.5,early=12.,rate=.1,min_norm=0.,
                            steps=301,single=False)
    x=rng.normal(size=(9,3)); initial=rng.normal(scale=.02,size=(9,2))
    yield x,initial,dict(k=2,metric='euclidean',init_mode=0,perplexity=4.,early=1.,rate=.5,min_norm=0.,
                        steps=250,single=False)


def tie_certificate(x,z,metric,k,native,source):
    dx=x.copy() if metric=='precomputed' else pairwise_distances(x,metric=metric)
    np.fill_diagonal(dx,np.inf)
    for row in dx:
        if len(np.unique(row[np.isfinite(row)]))!=len(row)-1:return None
    ranks=np.argsort(np.argsort(dx,axis=1,kind='stable'),axis=1,kind='stable')+1
    dz=np.sqrt(np.sum((z[:,None,:]-z[None,:,:])**2,axis=2));np.fill_diagonal(dz,np.inf)
    stable=np.argsort(dz,axis=1,kind='stable')[:,:k]
    stable_penalty=np.maximum(0,np.take_along_axis(ranks,stable,axis=1)-k).sum()
    n=len(z);scale=2/(n*k*(2*n-3*k-1));stable_value=1-stable_penalty*scale
    if not np.isclose(native,stable_value,atol=1.e-12,rtol=0):return None
    low=high=0;tie_rows=[]
    for i in range(n):
        cutoff=np.sort(dz[i])[k-1];fixed=np.flatnonzero(dz[i]<cutoff);tied=np.flatnonzero(dz[i]==cutoff)
        needed=k-len(fixed);cost=np.sort(np.maximum(0,ranks[i,tied]-k));base=np.maximum(0,ranks[i,fixed]-k).sum()
        low+=base+cost[:needed].sum();high+=base+cost[-needed:].sum()
        if len(tied)>needed:tie_rows.append(i)
    lower=1-high*scale;upper=1-low*scale
    if not tie_rows or not lower-1.e-12<=source<=upper+1.e-12:return None
    return dict(native=float(native),source=float(source),lower=float(lower),upper=float(upper),
                stable_value=float(stable_value),tie_rows=tie_rows,input_ranks_unique=True)


def cancellation_bound(position,p,native_p,degrees,n,k):
    z=position.reshape(n,k);student=(1+pdist(z,'sqeuclidean')/degrees)**(-(degrees+1)/2)
    q=np.maximum(student/(2*student.sum()),np.finfo(float).eps)
    delta=np.abs(z[:,None,:]-z[None,:,:]);coefficient=2*(degrees+1)/degrees
    envelope=squareform((np.abs(p-q)+p+q)*student)
    affinity=squareform(np.abs(p-native_p)*student)
    gamma=64*(n+k+1)*np.finfo(float).eps
    budget=coefficient*(gamma*np.sum(envelope[:,:,None]*delta,axis=1)+
                        np.sum(affinity[:,:,None]*delta,axis=1))
    return budget.ravel()+256*n*np.nextafter(0.,1.)


def trajectory_certificate(path,initial,p,cfg,rate,actual):
    rows=np.loadtxt(path,ndmin=2);n,k=initial.shape;size=n*k;degrees=max(k-1,1)
    position=initial.ravel().copy();gains=np.ones(size);update=np.zeros(size);phase=0
    max_gradient=max_update=max_position=max_error=0.;evaluations=0;best_error=np.inf;best_it=0;ambiguous=[]
    for row in rows:
        stage,it=map(int,row[:2]);values=row[2:];pos,gain,up,grad=np.split(values[:-1],4);recorded_error=values[-1]
        if stage!=phase:
            gains=np.ones(size);update=np.zeros(size);phase=stage;best_error=np.inf;best_it=it
        active=p*cfg['early'] if phase==1 else (p*cfg['early'])/cfg['early']
        error,raw=_kl_divergence(position,active,degrees,n,k)
        inc=update*raw<0;expected_gain=np.maximum(np.where(inc,gains+.2,gains*.8),.01)
        mismatched=np.flatnonzero(gain!=expected_gain)
        if len(mismatched):
            native_active=actual['probabilities']*cfg['early'] if phase==1 else \
                (actual['probabilities']*cfg['early'])/cfg['early']
            budget=cancellation_bound(position,active,native_active,degrees,n,k)
            native_raw=grad/gain
            for index in mismatched:
                assert gain[index] in [max(gains[index]+.2,.01),max(gains[index]*.8,.01)]
                assert max(abs(raw[index]),abs(native_raw[index]))<=budget[index],(it,index,raw[index],native_raw[index],budget[index])
                ambiguous.append(dict(iteration=it,coordinate=int(index),source_gradient=float(raw[index]),
                    native_gradient=float(native_raw[index]),bound=float(budget[index])))
        expected_gradient=raw*gain;expected_update=(.5 if phase==1 else .8)*update-rate*expected_gradient
        expected_position=position+expected_update
        np.testing.assert_allclose(grad,expected_gradient,atol=3.e-12,rtol=3.e-11)
        np.testing.assert_allclose(up,expected_update,atol=3.e-12,rtol=3.e-11)
        np.testing.assert_allclose(pos,expected_position,atol=3.e-12,rtol=3.e-11)
        np.testing.assert_allclose(recorded_error,error,atol=3.e-11,rtol=3.e-11)
        max_gradient=max(max_gradient,float(np.max(abs(grad-expected_gradient))))
        max_update=max(max_update,float(np.max(abs(up-expected_update))))
        max_position=max(max_position,float(np.max(abs(pos-expected_position))))
        max_error=max(max_error,abs(float(recorded_error-error)))
        stop=False
        if (it+1)%50==0:
            if error<best_error:best_error=error;best_it=it
            elif it-best_it>(250 if phase==1 else 300):stop=True
            if norm(expected_gradient)<=cfg['min_norm']:stop=True
        last_in_phase=not np.any((rows[:,0]==stage)&(rows[:,1]>it))
        if stop:assert last_in_phase,('source would stop earlier',stage,it)
        if last_in_phase and not stop:assert it==(250 if stage==1 else cfg['steps'])-1
        position=pos;gains=gain;update=up;evaluations+=1
    np.testing.assert_array_equal(position,actual['embedding'])
    np.testing.assert_array_equal(actual['embedding'],actual['replay_embedding'])
    np.testing.assert_array_equal(actual['kl'],actual['replay_kl'])
    np.testing.assert_array_equal(actual['iteration'],actual['replay_iteration'])
    if actual['kl'][0]!=np.finfo(float).max:
        np.testing.assert_allclose(actual['kl'][0],rows[-1,-1],atol=3.e-11,rtol=3.e-11)
    assert np.array_equal(rows[:,1],np.arange(evaluations))
    empty_second=(rows[-1,0]==1 and rows[-1,1]==249 and cfg['steps']==250)
    np.testing.assert_array_equal(actual['iteration'],[250 if empty_second else rows[-1,1]])
    if empty_second:np.testing.assert_array_equal(actual['kl'],[np.finfo(float).max])
    return dict(evaluations=evaluations,max_scaled_gradient_error=max_gradient,max_update_error=max_update,
                max_position_error=max_position,max_objective_error=max_error,gain_decisions_exact=not ambiguous,roundoff_gain_certificates=ambiguous,
                reported_objective_certified=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path)
    parser.add_argument('--profile',default='checked')
    parser.add_argument('--report',type=Path,default=RESULTS/'tsne.json');args=parser.parse_args()
    executable=args.executable.resolve() if args.executable else resolve()
    RESULTS.mkdir(exist_ok=True);checks=[];ties=[];trajectories=[]
    for seed in [0,7,42,2026,314159]:
        for case,(x,initial,cfg) in enumerate(fixtures(seed)):
            n=len(x); k=cfg['k']; metric=cfg['metric']; nshort=13
            if cfg['single']:initial=initial.astype(np.float32).astype(np.float64)
            code=['euclidean','precomputed','manhattan','cosine'].index(metric)
            inp=RESULTS/'tsne_input.txt';out=RESULTS/'tsne_actual.txt'
            header=f"{n} {x.shape[1]} {k} {cfg['steps']} 300 {seed} {cfg['init_mode']} {code} {str(cfg['single'])[0]} {nshort}\n"
            header+=f"{cfg['perplexity']} {cfg['early']} {cfg['rate']} {cfg['min_norm']}\n"
            inp.write_text(header+'\n'.join(' '.join(map(str,row)) for row in x)+'\n'+
                           '\n'.join(' '.join(map(str,row)) for row in initial)+'\n')
            proc=subprocess.run([str(executable),str(inp),str(out)],cwd=ROOT,text=True,
                                stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            assert proc.returncode==0,(seed,case,proc.stdout)
            actual={}
            for line in out.read_text().splitlines():
                name,*values=line.split();actual[name]=np.array(values,float)
            distances=x.copy() if metric=='precomputed' else pairwise_distances(x,metric=metric,squared=True) if metric=='euclidean' else pairwise_distances(x,metric=metric)
            if metric!='euclidean':distances=distances**2
            p=_joint_probabilities(distances,cfg['perplexity'],0);d=max(k-1,1)
            params=initial.astype(np.float32 if cfg['single'] else np.float64).ravel()
            error,gradient=_kl_divergence(params,p,d,n,k)
            rate=np.float64(max(n/cfg['early']/4,50)) if cfg['rate']==-1 else cfg['rate']
            short,short_error,short_it=_gradient_descent(_kl_divergence,params,0,nshort,n_iter_check=50,
                n_iter_without_progress=300,momentum=.5,learning_rate=rate,min_grad_norm=cfg['min_norm'],
                args=[p*cfg['early'],d,n,k])
            init=initial if cfg['init_mode']==0 else 'random' if cfg['init_mode']==1 else 'pca'
            model=TSNE(n_components=k,method='exact',perplexity=cfg['perplexity'],early_exaggeration=cfg['early'],
                learning_rate='auto' if cfg['rate']==-1 else cfg['rate'],max_iter=cfg['steps'],
                n_iter_without_progress=300,min_grad_norm=cfg['min_norm'],random_state=seed,init=init,metric=metric)
            embedding=model.fit_transform(x.copy())
            native_embedding=actual['embedding'].reshape(n,k)
            final_error,final_gradient=_kl_divergence(native_embedding.ravel(),p,d,n,k)
            expected=dict(probabilities=p,objective=[error],gradient=gradient,short_embedding=short,
                short_error=[short_error],short_iteration=[short_it],embedding=embedding.ravel(),
                kl=[model.kl_divergence_],learning_rate=[model.learning_rate_],iteration=[model.n_iter_],
                features=[model.n_features_in_],trust=[trustworthiness(x,native_embedding,n_neighbors=min(3,(n-1)//2),metric=metric)],
                final_objective=[final_error],final_gradient=final_gradient,
                preserved=np.concatenate([x.ravel(),initial.ravel()]))
            certificate=None
            if cfg['init_mode']==0:
                certificate=trajectory_certificate(str(out)+'.trace',initial,p,cfg,rate,actual)
                trajectories.append(dict(seed=seed,case=case,**certificate))
                expected.update(replay_embedding=actual['embedding'],replay_kl=actual['kl'],replay_iteration=actual['iteration'])
            assert actual.keys()==expected.keys(),(actual.keys(),expected.keys())
            for name,want in expected.items():
                atol=rtol=3.e-8
                if name in ['gradient','short_embedding','short_error'] and cfg['single']:atol=rtol=3.e-5
                if name=='probabilities':atol=rtol=1.e-11
                if name=='gradient' and cfg['single']:atol=3.e-8;rtol=3.e-6
                oracle='input_identity' if name=='preserved' else 'trajectory_identity' if name.startswith('replay_') else 'sklearn'
                if oracle in ['input_identity','trajectory_identity']:
                    atol=rtol=0.;np.testing.assert_array_equal(actual[name],want)
                elif name in ['embedding','kl','iteration'] and certificate is not None and not np.allclose(actual[name],want,atol=atol,rtol=rtol):
                    oracle='sklearn_with_trajectory_certificate'
                elif name=='trust' and not np.allclose(actual[name],want,atol=atol,rtol=rtol):
                    certificate=tie_certificate(x,native_embedding,metric,min(3,(n-1)//2),actual[name][0],want[0])
                    assert certificate is not None,(seed,case,'uncertified trustworthiness difference')
                    ties.append(dict(seed=seed,case=case,**certificate))
                    oracle='sklearn_with_stable_tie_certificate'
                elif np.any(np.abs(want)==np.finfo(float).max):np.testing.assert_array_equal(actual[name],want)
                else:np.testing.assert_allclose(actual[name],want,atol=atol,rtol=rtol,
                    err_msg=f'{seed} {case} {name} {cfg}')
                checks.append(dict(seed=seed,case=case,name=name,atol=atol,rtol=rtol,max_abs_error=float(np.max(np.abs(actual[name]-want))),oracle=oracle,passed=True))
        print(f'seed {seed}: exact t-SNE and trustworthiness passed',flush=True)
    import hashlib
    args.report.write_text(json.dumps(dict(sklearn=sklearn.__version__,profile=args.profile,
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checks=checks,tie_certificates=ties,trajectory_certificates=trajectories),indent=2)+'\n')
    print(f'{len(checks)} t-SNE result groups passed')


if __name__=='__main__':main()
