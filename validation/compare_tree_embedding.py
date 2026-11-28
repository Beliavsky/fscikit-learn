"""RandomTreesEmbedding: seeds, complete leaf codes, paths and staged fits."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import os
import subprocess
import warnings
import tempfile
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import sklearn
from scipy.sparse import csr_matrix,issparse
from sklearn.ensemble import RandomTreesEmbedding
from sklearn_tree_runtime import tree_runtime_provenance

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile):
    opts=['--profile','release'] if profile=='release' else ['--flag',FLAGS]
    result=subprocess.run(['fpm','run',*opts,'--target','reference_tree_embedding_driver','--runner'],cwd=ROOT,
                          text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/s.strip().strip('"') for s in result.stdout.splitlines()
           if s.strip().strip('"').endswith(('reference_tree_embedding_driver','reference_tree_embedding_driver.exe'))]
    paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,result.stdout
    return paths[0]


def fixtures(seeds,extended=False):
    for seed in seeds:
        rng=np.random.RandomState(seed)
        for case in range(49 if extended else 23):
            n,p=64,4
            if case==15:n,p=1,1
            if case==16:n,p=11,2
            if case==22:p=130
            x=rng.normal(size=(n,p));q=rng.normal(size=(13,p))
            nt,depth,leaf,split,decrease,w=5,5,1,2,0.,None
            if case==1:nt,depth=1,1
            elif case==2:nt,depth=3,-1
            elif case==3:depth,leaf,split=4,4,11
            elif case==4:decrease=.02
            elif case==5:x[:,:2]=1;q[:,:2]=1
            elif case==6:x[:]=3.;q[:]=3.
            elif case==7:
                x[:,:2]=np.floor(x[:,:2])*1e-8+1.;q[:,:2]=np.floor(q[:,:2])*1e-8+1.
            elif case==8:
                w=rng.uniform(.2,3,n);w[::7]=0
            elif case==9:w=rng.randint(0,5,n).astype(float)
            elif case==10:w=np.zeros(n)
            elif case==11:w=-np.ones(n)
            elif case==12:w=np.r_[np.ones(n//2),-np.ones(n-n//2)]
            elif case==13:w=np.ones(n)*1e150
            elif case==14:w=np.ones(n)*1e-200
            elif case==16:nt=100
            elif case==17:x[::3]=x[0]
            if case in [0,5,8]:q[0]=np.nan
            stages=[dict(x=x,q=q,nt=nt,depth=depth,leaf=leaf,split=split,decrease=decrease,seed=seed,w=w,
                         warm=False,method=case%2)]
            if case==18:
                stages[0].update(nt=2,warm=True)
                stages.append(dict(stages[0],nt=5,method=1))
                stages.append(dict(stages[0],x=x[:20],q=x[:13],nt=5,w=np.ones(20)*2,method=0))
            elif case==19:
                stages[0].update(nt=2,depth=2,w=rng.uniform(.2,2,n),warm=True)
                changed=rng.normal(size=(37,p))
                stages.append(dict(stages[0],x=changed,q=changed[:13],nt=5,depth=4,seed=seed+1,w=rng.uniform(.2,2,37)))
                reset=rng.normal(size=(17,2))
                stages.append(dict(stages[0],x=reset,q=reset[:13],nt=3,depth=-1,seed=0,w=None,warm=False))
            elif case==20:
                # A no-growth fit regenerates the encoder using current rows.
                stages[0].update(nt=3,warm=True)
                stages.append(dict(stages[0],x=x[:9],q=x[:9],w=None,method=1))
            elif case==21:
                stages[0].update(nt=2,warm=True)
                stages.append(dict(stages[0],nt=4,w=np.zeros(n)))
            for data in stages:data.update(budget=0,weight_fraction=0.,leaf_fraction=None,split_fraction=None)
            if 23<=case<=27:stages[0].update(budget=[2,3,5,17,100][case-23],depth=-1,method=2)
            elif 28<=case<=32:stages[0].update(weight_fraction=[.001,.05,.2,.49,.5][case-28],w=rng.uniform(.1,3,n),method=2)
            elif 33<=case<=37:stages[0].update(leaf_fraction=[.001,.02,.15,.5,.999][case-33],method=2)
            elif 38<=case<=42:stages[0].update(split_fraction=[.001,.02,.15,.5,1.][case-38],method=2)
            elif case==43:stages[0].update(budget=3,weight_fraction=.2,leaf_fraction=.1,split_fraction=.3,w=rng.uniform(.1,3,n),method=2)
            elif case==44:stages[0].update(budget=5,weight_fraction=.1,w=np.r_[np.full(n-8,2.),np.full(8,-1.)],method=2)
            elif case==45:stages[0].update(budget=3,weight_fraction=.1,w=np.zeros(n),method=2)
            elif case==46:
                stages[0].update(budget=3,leaf_fraction=.1,warm=True,nt=2,method=2)
                stages.append(dict(stages[0],budget=7,nt=5,x=x[:37],q=q,leaf_fraction=.02))
                stages.append(dict(stages[0],warm=False,budget=0,leaf_fraction=None,method=0))
            elif case==47:
                stages[0].update(budget=3,split_fraction=.2,warm=True,nt=2,method=2)
                stages.append(dict(stages[0],budget=7,nt=5,x=x[:31],q=q,split_fraction=.01))
            elif case==48:
                stages[0].update(warm=True,nt=3,method=2)
                stages.append(dict(stages[0],x=x[:9],q=x[:9],method=3))
            yield seed,case,stages


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path)
    parser.add_argument('--profile',choices=['checked','release'],default='checked')
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,7,42,2026,314159]);parser.add_argument('--extended',action='store_true');parser.add_argument('--report',type=Path);args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True);exe=args.executable.resolve() if args.executable else resolve(args.profile)
    runtime=tree_runtime_provenance()
    assert runtime['feature_threshold']==0.,'tested native tree baseline requires runtime FEATURE_THRESHOLD=0'
    before=hashlib.sha256(exe.read_bytes()).hexdigest()
    work=tempfile.TemporaryDirectory(prefix="fsklearn-embedding-reference-")
    checks=[];source_warnings=[];controls=[];cases=0;fits=0
    for seed,case,stages in fixtures(args.seeds,args.extended):
        model=RandomTreesEmbedding(sparse_output=case%2==0);expected={};lines=[str(len(stages))];old=0
        with warnings.catch_warnings(record=True) as caught,np.errstate(all='ignore'):
            warnings.simplefilter('always')
            for stage,data in enumerate(stages,1):
                x,q,w=data['x'],data['q'],data['w'];n,p=x.shape
                model.set_params(n_estimators=data['nt'],max_depth=None if data['depth']==-1 else data['depth'],
                    min_samples_leaf=data['leaf'] if data['leaf_fraction'] is None else data['leaf_fraction'],
                    min_samples_split=data['split'] if data['split_fraction'] is None else data['split_fraction'],
                    max_leaf_nodes=data['budget'] or None,min_weight_fraction_leaf=data['weight_fraction'],min_impurity_decrease=data['decrease'],
                    random_state=data['seed'],warm_start=data['warm'])
                if data['method']==3:model.one_hot_encoder_.set_params(drop='first',handle_unknown='ignore',max_categories=2)
                encoded=(model.fit_transform(x,sample_weight=w) if data['method'] in [0,2] else
                         model.fit(x,sample_weight=w).transform(x))
                path,pointer=model.decision_path(q);prefix=f's{stage}_'
                nodes=np.array([[t.tree_.node_count,t.get_depth(),t.get_n_leaves(),t.max_features_] for t in model.estimators_])
                cats=model.one_hot_encoder_.categories_;growth=len(model.estimators_)-(old if data['warm'] else 0)
                values=dict(train=encoded.toarray() if issparse(encoded) else encoded,query=model.transform(q).toarray() if model.sparse_output else model.transform(q),leaves=model.apply(x)+1,query_leaves=model.apply(q)+1,
                    path=path.toarray(),pointer=pointer,samples=np.column_stack(model.estimators_samples_)+1,
                    seeds=[t.random_state for t in model.estimators_],
                    random_target=np.random.RandomState(data['seed']).uniform(size=n),importances=model.feature_importances_,
                    counts=[model.n_features_in_,model._n_features_out,len(model.estimators_),model._n_samples,growth,model.n_outputs_],
                    tree_stats=nodes,categories=np.concatenate(cats)+1,
                    category_offsets=np.r_[0,np.cumsum([len(c) for c in cats])],names=model.get_feature_names_out())
                for label,array in [('csr_train',csr_matrix(encoded)),('csr_query',csr_matrix(model.transform(q))),('csr_path',path.tocsr())]:
                    values.update({label+'_shape':array.shape,label+'_indptr':array.indptr+1,label+'_indices':array.indices+1,label+'_data':array.data})
                expected.update({prefix+k:v for k,v in values.items()});old=len(model.estimators_)
                lines+=[f"{n} {p} {len(q)} {data['nt']} {data['depth']} {data['leaf']} {data['split']} {data['seed']} "
                        f"{int(data['warm'])} {int(w is not None)} {data['method']} {data['decrease']}"]
                if args.extended:lines+=[f"{data['budget']} {data['weight_fraction']} {int(data['leaf_fraction'] is not None)} {data['leaf_fraction'] or 0} {int(data['split_fraction'] is not None)} {data['split_fraction'] or 0}"]
                lines+=[' '.join(map(str,row)) for row in x]
                if w is not None:lines+=[' '.join(map(str,w))]
                lines+=[' '.join(map(str,row)) for row in q]
                controls.append(dict(seed=seed,case=case,stage=stage,n=n,p=p,trees=data['nt'],depth=data['depth'],
                    leaf=data['leaf'],split=data['split'],budget=data['budget'],weight_fraction=data['weight_fraction'],
                    leaf_fraction=data['leaf_fraction'],split_fraction=data['split_fraction'],sparse_output=model.sparse_output,
                    warm=data['warm'],method=data['method'],weighted=w is not None))
                fits+=1
        source_warnings.extend(dict(seed=seed,case=case,category=w.category.__name__,message=str(w.message)) for w in caught)
        inp=Path(work.name)/'input.txt';out=Path(work.name)/'output.txt'
        inp.write_text('\n'.join(lines)+'\n')
        result=subprocess.run([str(exe),str(inp),str(out),'extended' if args.extended else 'legacy'],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert result.returncode==0,(seed,case,result.stdout)
        actual={}
        for line in out.read_text().splitlines():
            key,*values=line.split();actual[key]=np.asarray(values,str if key.endswith('_names') else float)
        assert actual.keys()==expected.keys(),(actual.keys(),expected.keys())
        for name,want in expected.items():
            want=np.asarray(want).ravel(order='F')
            if name.endswith('_importances'):
                np.testing.assert_allclose(actual[name],want,rtol=3e-10,atol=3e-12,equal_nan=True,err_msg=f'{seed} {case} {name}')
            elif name.endswith('_random_target'):
                np.testing.assert_array_equal(actual[name],want,err_msg=f'{seed} {case} {name}')
            else:np.testing.assert_array_equal(actual[name],want,err_msg=f'{seed} {case} {name}')
            checks.append(dict(seed=seed,case=case,name=name,passed=True,
                               oracle='numpy_random_protocol' if name.endswith('_random_target') else 'sklearn'))
        cases+=1
        if case==(48 if args.extended else 22):print(f'seed {seed}: tree embeddings passed',flush=True)
    assert before==hashlib.sha256(exe.read_bytes()).hexdigest()
    work.cleanup()
    root=Path(sklearn.__file__).parent
    paths=['ensemble/_forest.py','ensemble/_base.py','preprocessing/_encoders.py','tree/_classes.py',
           'tree/_splitter.pyx','tree/_criterion.pyx','tree/_tree.pyx','utils/_random.pxd']
    report=dict(sklearn=sklearn.__version__,numpy=np.__version__,profile=args.profile,extended=args.extended,executable_sha256=before,seeds=args.seeds,fixtures=cases,
                fits=fits,controls=controls,checks=checks,source_warnings=source_warnings,tree_runtime=runtime,
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})
    (args.report or RESULTS/f'tree_embedding_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} tree embedding groups across {cases} fixtures and {fits} fits passed')


if __name__=='__main__':main()
