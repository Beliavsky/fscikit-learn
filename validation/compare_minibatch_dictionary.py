"""Mini-batch dictionary/SparsePCA estimators and randomized SVD default backend."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import warnings
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import sklearn
from sklearn.decomposition import MiniBatchDictionaryLearning,MiniBatchSparsePCA
from sklearn.utils.extmath import randomized_svd
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def rows(x):return [' '.join(f'{v:.17g}' for v in row) for row in x]
def boolean(value):return 'T' if value else 'F'
def optional(value,sentinel):return sentinel if value is None else value


def run(executable,lines):
    inp=RESULTS/'minibatch_dictionary_input.txt';out=RESULTS/'minibatch_dictionary_output.txt'
    inp.write_text('\n'.join(lines)+'\n')
    proc=subprocess.run([str(executable),str(inp),str(out)],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert proc.returncode==0,proc.stdout
    result={}
    for line in out.read_text().splitlines():
        name,*values=line.split();result[name]=np.array(values,dtype=float)
    return result


def compare(actual,expected,checks,kind,seed,case):
    assert actual.keys()==expected.keys(),(kind,seed,case,actual.keys()-expected.keys(),expected.keys()-actual.keys())
    for name,value in expected.items():
        tol=0 if name.endswith(('meta','rng')) else 3.e-8
        np.testing.assert_allclose(np.asarray(actual[name]).ravel(order='F'),np.asarray(value).ravel(order='F'),atol=tol,rtol=tol,equal_nan=True,
                                   err_msg=f'{kind} seed={seed} case={case} {name}')
        checks.append(dict(kind=kind,seed=seed,case=case,name=name,passed=True))


def dictionary_cases(executable,checks):
    for seed,n,p in [(0,23,7),(7,31,5),(42,28,11),(2026,19,3)]:
        rng=np.random.RandomState(seed);base=rng.normal(size=(n,p));query=rng.normal(size=(8,p))
        for case in range(25):
            x=base.copy();q=query.copy();ops=[]
            config=dict(n_components=3,alpha=.6,max_iter=4,fit_algorithm='lars',batch_size=7,shuffle=True,
                        random_state=seed,tol=0.,max_no_improvement=None,transform_algorithm='lasso_cd',
                        transform_alpha=None,transform_n_nonzero_coefs=None,transform_max_iter=1000,
                        positive_code=False,positive_dict=False,split_sign=False,dict_init=None)
            def add(op,data,**changes):
                config.update(changes);ops.append((op,np.array(data,copy=True),config.copy()))
            if case==24:config.update(n_components=None,alpha=1.,max_iter=1000,batch_size=256,tol=.001,
                                      max_no_improvement=10,transform_algorithm='omp')
            if case==1:config.update(fit_algorithm='cd')
            if case==2:config.update(shuffle=False)
            if case==3:config.update(batch_size=1,max_iter=2)
            if case==4:config.update(batch_size=n+10)
            if case==5:config.update(max_iter=0)
            if case==6:config.update(n_components=None)
            if case==7:config.update(n_components=p+2,max_iter=2)
            if case==8:config.update(dict_init=rng.normal(size=(7,p)))
            if case==9:config.update(dict_init=rng.normal(size=(1,p)))
            if case==10:config.update(dict_init=np.zeros((3,p)),alpha=100.)
            if case==11:config.update(positive_code=True,positive_dict=True,fit_algorithm='cd')
            if case==12:config.update(positive_dict=True)
            if case==13:config.update(positive_code=True,fit_algorithm='cd')
            if case==14:config.update(tol=1.e9,max_iter=1000000000)
            if case==15:config.update(max_no_improvement=0,max_iter=1000000000)
            if case==16:x[:]=0;config.update(max_no_improvement=2,max_iter=1000000000)
            if case==17:config.update(alpha=0.,fit_algorithm='cd')
            if case==22:x=rng.normal(size=(41,37));q=rng.normal(size=(8,37));config.update(max_iter=2)
            if case==18:
                add(2,x[:5],max_iter=0,batch_size=2)
                add(2,x[5:13],batch_size=99,random_state=seed+10)
                add(2,x[13:])
            elif case==19:
                add(1,x);add(2,x[:2],fit_algorithm='cd');add(2,x[-6:])
            elif case==20:add(3,x,split_sign=True,transform_algorithm='threshold')
            elif case==23:
                add(2,x[:8]);x=np.column_stack([x,np.ones(len(x))]);q=np.column_stack([q,np.zeros(len(q))])
                add(1,x,n_components=4,random_state=seed+1)
            else:add(1,x)
            if case==13:add(2,x[:4],fit_algorithm='lars')
            if case==21:
                for method in ['lasso_lars','lasso_cd','lars','omp','threshold']:
                    add(4,q,transform_algorithm=method,transform_n_nonzero_coefs=2,transform_alpha=.3)
            for op,data in [(4,q),(2,x[:6]),(4,q)]:add(op,data)
            source=MiniBatchDictionaryLearning(n_jobs=1);lines=[f'1 {len(ops)}'];expected={}
            for i,(op,data,c) in enumerate(ops,1):
                actual_config=c.copy()
                if c['dict_init'] is not None:actual_config['dict_init']=c['dict_init'].copy()
                source.set_params(**actual_config);prefix=f'{i}_'
                ints=[optional(c['n_components'],0),c['max_iter'],c['batch_size'],c['random_state'],optional(c['max_no_improvement'],-1),
                      optional(c['transform_n_nonzero_coefs'],0),c['transform_max_iter']]
                lines.extend([f'{op} {data.shape[0]} {data.shape[1]}',' '.join(map(str,ints)),
                              f"{c['alpha']:.17g} {c['tol']:.17g} {optional(c['transform_alpha'],-1):.17g}",
                              ' '.join(boolean(c[k]) for k in ['shuffle','positive_code','positive_dict','split_sign']),
                              f"{c['fit_algorithm']} {c['transform_algorithm']}",str(0 if c['dict_init'] is None else len(c['dict_init']))])
                if c['dict_init'] is not None:lines.extend(rows(c['dict_init']))
                lines.extend(rows(data));result=None
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if op==1:source.fit(data)
                    elif op==2:source.partial_fit(data)
                    elif op==3:result=source.fit_transform(data)
                    elif op==4:result=source.transform(data)
                if result is not None:expected[prefix+'result']=result
                expected[prefix+'components']=source.components_.copy();expected[prefix+'a']=source._A.copy();expected[prefix+'b']=source._B.copy()
                expected[prefix+'meta']=[source.n_features_in_,source._n_components,source._batch_size,source.n_steps_,getattr(source,'n_iter_',0)]
                expected[prefix+'convergence']=[optional(getattr(source,'_ewa_cost',None),np.nan),
                    optional(getattr(source,'_ewa_cost_min',None),np.nan),getattr(source,'_no_improvement',0)]
            compare(run(executable,lines),expected,checks,'dictionary',seed,case)
        print(f'dictionary seed {seed}: 25 configurations passed',flush=True)


def sparse_pca_cases(executable,checks):
    for seed,n,p in [(0,25,7),(7,31,5),(42,28,11),(2026,19,3)]:
        rng=np.random.RandomState(seed);base=rng.normal(size=(n,p));query=rng.normal(size=(8,p))
        for case in range(17):
            x=base.copy();q=query.copy()
            config=dict(n_components=3,alpha=.5,ridge_alpha=.01,max_iter=4,batch_size=3,shuffle=True,
                        method='lars',random_state=seed,tol=0.,max_no_improvement=None)
            if case==16:config.update(n_components=None,alpha=1.,max_iter=1000,tol=.001,max_no_improvement=10)
            if case==1:config.update(method='cd')
            if case==2:config.update(shuffle=False)
            if case==3:config.update(max_iter=0)
            if case==4:config.update(n_components=None)
            if case==5:config.update(n_components=p+2,max_iter=2)
            if case==6:config.update(ridge_alpha=0.)
            if case==7:config.update(ridge_alpha=2.)
            if case==8:config.update(alpha=0.,method='cd')
            if case==9:config.update(alpha=1000.)
            if case==10:x[:]=1;q[:]=1
            if case==11:x=x[:,:1];q=q[:,:1];config.update(n_components=2)
            if case==12:config.update(tol=1.e9,max_iter=1000000000)
            if case==13:config.update(max_no_improvement=0,max_iter=1000000000)
            if case==14:config.update(batch_size=p+7)
            if case==15:x=rng.normal(size=(17,29));q=rng.normal(size=(8,29))
            k=x.shape[1] if config['n_components'] is None else config['n_components']
            ops=[(3,x),(4,q),(5,rng.normal(size=(5,k)))]
            source=MiniBatchSparsePCA(n_jobs=1,**config);lines=[f'2 {len(ops)}'];expected={}
            for i,(op,data) in enumerate(ops,1):
                c=config;prefix=f'{i}_'
                ints=[optional(c['n_components'],0),c['max_iter'],c['batch_size'],c['random_state'],optional(c['max_no_improvement'],-1)]
                lines.extend([f'{op} {data.shape[0]} {data.shape[1]}',' '.join(map(str,ints)),
                              f"{c['alpha']:.17g} {c['tol']:.17g} {c['ridge_alpha']:.17g}",f"{boolean(c['shuffle'])} {c['method']}",*rows(data)])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if op==3:result=source.fit_transform(data)
                    elif op==4:result=source.transform(data)
                    else:result=source.inverse_transform(data)
                expected[prefix+'result']=result;expected[prefix+'components']=source.components_.copy();expected[prefix+'mean']=source.mean_.copy()
                expected[prefix+'meta']=[source.n_features_in_,source.n_components_,source.n_iter_]
            compare(run(executable,lines),expected,checks,'sparse_pca',seed,case)
        print(f'SparsePCA seed {seed}: 17 configurations passed',flush=True)


def randomized_cases(executable,checks):
    for seed in [0,7,42,2026]:
        rng=np.random.RandomState(seed)
        for case,(n,p,k) in enumerate([(41,7,3),(7,41,3),(64,48,2),(48,64,4),(24,24,15),(15,6,11),(20,12,5),(8,13,5)]):
            x=rng.normal(size=(n,p))
            if case==6:x=x[:,:1]@x[:1,:]
            if case==7:x[:]=0
            state=np.random.RandomState(seed);u,s,vt=randomized_svd(x,k,random_state=state)
            expected={'1_singular':s,'1_rng':[state.uniform(),state.normal()]}
            actual=run(executable,['3 1',f'1 {n} {p}',f'{k} {seed}',*rows(x)])
            au=actual.pop('1_left').reshape((n,len(s)),order='F');av=actual.pop('1_right').reshape((len(s),p),order='F')
            asv=actual['1_singular'];nz=np.count_nonzero(s>np.finfo(float).eps*max(n,p)*s.max())
            expected['1_reconstruction']=(u*s)@vt;actual['1_reconstruction']=(au*asv)@av
            if nz:
                expected['1_left']=u[:,:nz];actual['1_left']=au[:,:nz]
                expected['1_right']=vt[:nz];actual['1_right']=av[:nz]
                expected['1_left_orthogonality']=np.eye(nz);actual['1_left_orthogonality']=au[:,:nz].T@au[:,:nz]
                expected['1_right_orthogonality']=np.eye(nz);actual['1_right_orthogonality']=av[:nz]@av[:nz].T
            compare(actual,expected,checks,'randomized_svd',seed,case)
        print(f'randomized SVD seed {seed}: 8 configurations passed',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path);args=parser.parse_args();RESULTS.mkdir(exist_ok=True)
    if args.executable:executable=args.executable.resolve()
    else:
        proc=subprocess.run(['fpm','run','--flag',FLAGS,'--target','reference_minibatch_dictionary_driver','--runner'],cwd=ROOT,
                            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/line.strip().strip('"') for line in proc.stdout.splitlines()
               if line.strip().strip('"').endswith(('reference_minibatch_dictionary_driver','reference_minibatch_dictionary_driver.exe'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,proc.stdout
        executable=paths[0]
    checks=[];randomized_cases(executable,checks);dictionary_cases(executable,checks);sparse_pca_cases(executable,checks)
    (RESULTS/'minibatch_dictionary.json').write_text(json.dumps(dict(sklearn=sklearn.__version__,checks=checks),indent=2)+'\n')
    print(f'{len(checks)} mini-batch dictionary/SparsePCA result groups passed')


if __name__=='__main__':main()
