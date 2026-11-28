"""Direct seeded BernoulliRBM training, incremental and sampling contracts."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import tempfile
import hashlib
import warnings
import copy
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import sklearn
import scipy
import sklearn.neural_network._rbm as source_rbm
import scipy.sparse._sparsetools as source_sparse
import scipy.special._ufuncs as source_ufuncs
import scipy.special._special_ufuncs as source_special
import numpy._core._multiarray_umath as source_numpy
from scipy.sparse import csr_matrix
from sklearn.neural_network import BernoulliRBM

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def sparse_data(data,layout):
    if layout=='canonical':return csr_matrix(data)
    pointers=[0];indices=[];values=[]
    for row in data:
        columns=range(len(row)-1,-1,-1) if layout=='reverse' else range(len(row))
        for j in columns:
            v=row[j]
            if layout=='zeros':entries=[v]
            elif layout=='cancellation':entries=[1.e8,v,-1.e8]
            elif v==0:continue
            elif layout=='duplicates':entries=[.25*v,.75*v]
            else:entries=[v]
            values.extend(entries);indices.extend([j]*len(entries))
        pointers.append(len(values))
    return csr_matrix((np.array(values,float),np.array(indices,dtype=np.int32),np.array(pointers,dtype=np.int32)),shape=data.shape)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path);parser.add_argument('--sparse-input',action='store_true');parser.add_argument('--report',type=Path);parser.add_argument('--layout',choices=['canonical','reverse','duplicates','zeros','cancellation'],default='canonical');parser.add_argument('--mixed-input',action='store_true');parser.add_argument('--boundary-cases',action='store_true');parser.add_argument('--diagnostic',action='store_true');args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable: executable=args.executable.resolve()
    else:
        proc=subprocess.run(['fpm','run','--flag',FLAGS,'--target','reference_rbm_driver','--runner'],cwd=ROOT,
                            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/line.strip().strip('"') for line in proc.stdout.splitlines()
               if line.strip().strip('"').endswith(('reference_rbm_driver','reference_rbm_driver.exe'))]
        paths=[p.resolve() for p in paths if p.is_file()]
        assert len(paths)==1,proc.stdout
        executable=paths[0]
    digest=hashlib.sha256(executable.read_bytes()).hexdigest()
    reference_files=[Path(m.__file__).resolve() for m in [source_rbm,source_sparse,source_ufuncs,source_special,source_numpy]]
    reference_files+=[Path(scipy.__file__).parent/'sparse/_compressed.py']
    for package in [np,scipy]:
        libraries=Path(package.__file__).parent.parent/(package.__name__+'.libs')
        if libraries.is_dir():reference_files+=sorted(libraries.glob('*.dll'))
    reference_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in reference_files}
    work=tempfile.TemporaryDirectory(prefix="fsklearn-rbm-reference-")
    checks=[];replays=[];mismatches=[];sequences=0;case_count=26 if args.boundary_cases else 20
    for seed,n,p in [(0,27,7),(7,19,4),(42,33,11),(2026,13,1)]:
        rng=np.random.RandomState(seed);binary=(rng.uniform(size=(n,p))<.4).astype(float)
        query=(rng.uniform(size=(9,p))<.6).astype(float)
        for case in range(case_count):
            x=binary.copy();q=query.copy()
            controls=dict(n_components=5,batch_size=10,n_iter=5,random_state=seed,learning_rate=.1)
            ops=[]
            def add(op,data,**changes):
                controls.update(changes);ops.append((op,np.array(data,copy=True),controls.copy()))
            if case==1:controls.update(n_iter=0,n_components=3,batch_size=7)
            if case==2:controls.update(n_iter=3,n_components=4,batch_size=1)
            if case==3:controls.update(n_iter=4,n_components=2,batch_size=n+7)
            if case==4:controls.update(n_iter=5,n_components=6,batch_size=n)
            if case==5:controls.update(n_iter=6,n_components=3,batch_size=n//2)
            if case==6:
                controls.update(n_iter=7,n_components=7,batch_size=4,learning_rate=.03)
                x=rng.uniform(size=(n,p));q=rng.uniform(size=(9,p))
            if case==7:
                controls.update(n_iter=3,n_components=4,batch_size=5)
                x=rng.uniform(-2,3,size=(n,p));q=rng.uniform(-2,3,size=(9,p))
            if case==8:controls.update(n_iter=4,n_components=2,batch_size=3,learning_rate=2.)
            if case==9:controls.update(n_iter=1,n_components=1,learning_rate=1.e-5)
            if case==10:x[:]=0;controls.update(n_iter=8,n_components=3,batch_size=4)
            if case==11:x[:]=1;controls.update(n_iter=8,n_components=5,batch_size=7)
            if case==19:controls.update(n_iter=2,n_components=31,batch_size=n-1)
            if case==20:controls.update(n_components=3,batch_size=4,n_iter=3,learning_rate=1.e300);x*=1.e100;q=(2*q-1)*1.e100
            if case==21:controls.update(n_components=3,batch_size=4,n_iter=1,learning_rate=1.e200);x*=1.e150
            if case==22:controls.update(n_components=3,batch_size=4,n_iter=2,learning_rate=1.e-300);x*=1.e300;q=(2*q-1)*1.e300
            if case==23:controls.update(n_components=31,n_iter=0);q=(2*q-1)*1.e308
            if case==24:controls.update(n_components=3,n_iter=1,learning_rate=1.e10);x*=1.e300;q[:]=0
            if case==25:controls.update(n_components=3,n_iter=1,learning_rate=1.e308);x[:]=0;q=2*q-1
            if case==12:
                controls.update(n_iter=0,n_components=4,batch_size=6)
                for a in np.array_split(x,4):add(2,a)
            elif case==13:
                controls.update(n_components=3,batch_size=2)
                add(2,x);add(2,x[::-1],learning_rate=.07,random_state=seed+11)
            elif case==14:
                controls.update(batch_size=4);add(1,x);add(2,x[:2]);add(5,q);add(2,x[-7:])
            elif case==15:
                add(1,x);add(5,q)
                x=np.column_stack([x,np.ones(n)]);q=np.column_stack([q,np.zeros(len(q))])
                add(1,x,n_components=3,random_state=seed+1);add(5,q)
            elif case==16:
                add(3,x);q=1000*(2*q-1)
            elif case==17:
                add(2,x,batch_size=5);add(5,q);add(1,x,n_components=3,batch_size=3)
                add(2,x[:1],batch_size=9);add(2,x[-4:],random_state=seed+2)
            elif case==18:
                add(1,x);add(5,q);add(6,q);add(6,q);add(2,x)
            else:add(1,x)
            for op,data in [(4,q),(6,q),(6,q),(5,q),(5,q),(2,x[:3]),(4,q),(6,q)]:add(op,data)
            source=BernoulliRBM();expected={};snapshots={};before_states={};source_inputs={};lines=[str(len(ops))]
            for i,(op,data,config) in enumerate(ops,1):
                source.set_params(**config);prefix=f'{i}_'
                lines.extend([f'{op} {data.shape[0]} {data.shape[1]}',
                              f'{config["n_components"]} {config["batch_size"]} {config["n_iter"]} {config["random_state"]} {config["learning_rate"]:.17g}'])
                storage=int(args.sparse_input and (not args.mixed_input or i%3!=0))
                source_data=sparse_data(data,args.layout) if storage else data
                if args.mixed_input:lines.append(str(storage))
                raw=storage and (args.mixed_input or args.layout!='canonical')
                if raw:
                    lines.extend([str(source_data.nnz),' '.join(str(v+1) for v in source_data.indptr),' '.join(str(v+1) for v in source_data.indices),' '.join(format(v,'.17g') for v in source_data.data)])
                else:lines.extend(' '.join(f'{v:.17g}' for v in row) for row in data)
                if op==2:
                    initial=copy.deepcopy(source)
                    with warnings.catch_warnings(),np.errstate(all='ignore'):
                        warnings.simplefilter('ignore')
                        if not hasattr(initial,'components_'):initial.set_params(n_iter=0).fit(source_data)
                        expected[prefix+'positive_before']=initial.transform(source_data)
                if args.diagnostic:before_states[i]=copy.deepcopy(source)
                result=None
                with warnings.catch_warnings(),np.errstate(all='ignore'):
                    warnings.simplefilter('ignore')
                    if op==1:source.fit(source_data)
                    elif op==2:source.partial_fit(source_data)
                    elif op==3:result=source.fit_transform(source_data)
                    elif op==4:result=source.transform(source_data)
                    elif op==5:result=source.gibbs(source_data)
                    elif op==6:result=source.score_samples(source_data)
                if args.diagnostic:
                    snapshots[i]=copy.deepcopy(source);source_inputs[i]=source_data.copy()
                if result is not None:expected[prefix+'result']=result
                for name,attr in [('components','components_'),('hidden','intercept_hidden_'),('visible','intercept_visible_'),('chain','h_samples_')]:
                    expected[prefix+name]=getattr(source,attr).copy()
                expected[prefix+'features']=[source.n_features_in_]
            inp=Path(work.name)/'input.txt';out=Path(work.name)/'output.txt';inp.write_text('\n'.join(lines)+'\n')
            proc=subprocess.run([str(executable),str(inp),str(out),'mixed' if args.mixed_input else 'raw' if args.sparse_input and args.layout!='canonical' else 'csr' if args.sparse_input else 'dense'],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            assert proc.returncode==0,(seed,case,proc.stdout)
            actual={}
            for line in out.read_text().splitlines():
                name,*values=line.split();actual[name]=np.array(values,dtype=float)
            assert actual.keys()==expected.keys()
            for name,value in expected.items():
                op=int(name.split('_')[0]);discrete=name.endswith(('chain','features')) or (name.endswith('result') and ops[op-1][0]==5)
                tol=0 if discrete else 2.e-10
                passed=True
                try:
                    np.testing.assert_allclose(actual[name],np.asarray(value).ravel(order='F'),atol=tol,rtol=tol,
                                               err_msg=f'seed={seed} case={case} {name}')
                except AssertionError:
                    passed=False
                    if not args.diagnostic:raise
                    want=np.asarray(value).ravel(order='F');got=actual[name]
                    finite=np.isfinite(want)&np.isfinite(got)
                    mismatches.append(dict(seed=seed,case=case,name=name,max_abs_error=float(np.max(abs(got[finite]-want[finite]))) if finite.any() else None,nonfinite_masks_match=bool(np.array_equal(np.isnan(want),np.isnan(got)) and np.array_equal(np.isinf(want),np.isinf(got))),discrete=bool(discrete)))
                checks.append(dict(seed=seed,case=case,name=name,passed=passed,discrete=bool(discrete)))
            if args.diagnostic:
                for i,(operation,data,config) in enumerate(ops,1):
                    if operation==2:
                        replay=before_states[i]
                        if i>1:
                            previous=f'{i-1}_'
                            replay.components_=actual[previous+'components'].reshape(snapshots[i-1].components_.shape,order='F').copy(order='F')
                            replay.intercept_hidden_=actual[previous+'hidden'].copy();replay.intercept_visible_=actual[previous+'visible'].copy()
                            replay.h_samples_=actual[previous+'chain'].reshape(snapshots[i-1].h_samples_.shape,order='F').copy()
                        prefix=f'{i}_';positive=actual[prefix+'positive_before'].reshape(len(data),snapshots[i].components_.shape[0],order='F').copy()
                        if not hasattr(replay,'components_'):
                            initial=copy.deepcopy(replay).set_params(n_iter=0)
                            with warnings.catch_warnings(),np.errstate(all='ignore'):
                                warnings.simplefilter('ignore');initial.fit(source_inputs[i])
                            positive_reference=initial.transform(source_inputs[i])
                        else:positive_reference=replay.transform(source_inputs[i])
                        np.testing.assert_allclose(positive,positive_reference,atol=2.e-10,rtol=2.e-10,equal_nan=True,err_msg=f'positive means {seed} {case} {i}')
                        original_mean=replay._mean_hiddens;mean_calls=[0]
                        def mean_for_update(values):
                            mean_calls[0]+=1
                            return positive.copy() if mean_calls[0]==1 else original_mean(values)
                        replay._mean_hiddens=mean_for_update
                        with warnings.catch_warnings(),np.errstate(all='ignore'):
                            warnings.simplefilter('ignore');replay.partial_fit(source_inputs[i])
                        replays.append(dict(seed=seed,case=case,operation=i,passed=True,oracle='Source positive means using native preceding arrays'))
                        prefix=f'{i}_'
                        for field,attribute in [('components','components_'),('hidden','intercept_hidden_'),('visible','intercept_visible_'),('chain','h_samples_')]:
                            tolerance=0 if field=='chain' else 2.e-10
                            np.testing.assert_allclose(actual[prefix+field],np.asarray(getattr(replay,attribute)).ravel(order='F'),atol=tolerance,rtol=tolerance,equal_nan=True,err_msg=f'same-state update {seed} {case} {i} {field}')
                        replays.append(dict(seed=seed,case=case,operation=i,passed=True,oracle='Source partial update using native preceding arrays and independently checked positive means, matching RNG stream'))
                    if operation not in [3,4,6]:continue
                    prefix=f'{i}_';replay=snapshots[i]
                    replay.components_=actual[prefix+'components'].reshape(replay.components_.shape,order='F').copy(order='F')
                    replay.intercept_hidden_=actual[prefix+'hidden'].copy();replay.intercept_visible_=actual[prefix+'visible'].copy()
                    with warnings.catch_warnings(),np.errstate(all='ignore'):
                        warnings.simplefilter('ignore')
                        want=replay.score_samples(source_inputs[i]) if operation==6 else replay.transform(source_inputs[i])
                    np.testing.assert_allclose(actual[prefix+'result'],np.asarray(want).ravel(order='F'),atol=2.e-10,rtol=2.e-10,equal_nan=True,err_msg=f'same-state replay {seed} {case} {i}')
                    replays.append(dict(seed=seed,case=case,operation=i,passed=True,oracle='Source forward operation using native fitted arrays'))
            sequences+=1
        print(f'seed {seed}: {case_count} configurations processed',flush=True)
    assert digest==hashlib.sha256(executable.read_bytes()).hexdigest()
    assert reference_hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in reference_files}
    work.cleanup()
    (args.report or RESULTS/('rbm_csr.json' if args.sparse_input else 'rbm.json')).write_text(json.dumps(dict(sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,reference_sha256=reference_hashes,executable_sha256=digest,sparse_input=args.sparse_input,layout=args.layout,mixed_input=args.mixed_input,boundary_cases=args.boundary_cases,sequences=sequences,diagnostic=args.diagnostic,direct_passed=all(c["passed"] for c in checks),completed=True,direct_mismatches=mismatches,same_state_replays=replays,checks=checks),indent=2)+'\n')
    print(f'{len(checks)} BernoulliRBM groups observed; {len(mismatches)} direct mismatches; {len(replays)} same-state replays passed')


if __name__=='__main__':main()
