"""KernelCenterer against C-contiguous sklearn/NumPy reference arithmetic."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.preprocessing import KernelCenterer

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'
NUMPY_PAIRWISE_SOURCE = {'version': '2.2.6', 'url': 'https://raw.githubusercontent.com/numpy/numpy/v2.2.6/numpy/_core/src/umath/loops_utils.h.src', 'sha256': 'ed85d5b99f0b457cb7a316bf8c4acb884034bbbb81a89c5cedb62be80bf7c4f8'}


def resolve(profile):
    opts=['--profile','release'] if profile=='release' else ['--flag',FLAGS]
    result=subprocess.run(['fpm','run',*opts,'--target','reference_kernel_centering_driver','--runner'],cwd=ROOT,
                          text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/s.strip().strip('"') for s in result.stdout.splitlines()
           if s.strip().strip('"').endswith(('reference_kernel_centering_driver','reference_kernel_centering_driver.exe'))]
    paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,result.stdout
    return paths[0]


def fixtures(seeds):
    for seed in seeds:
        rng=np.random.RandomState(seed)
        for n in [1,2,7,8,9,16,17,31,32,127,128,129,257]:
            for kind in range(8):
                x=rng.normal(size=(n,4));z=rng.normal(size=(7,4))
                kernel=x@x.T;query=z@x.T
                if kind==1:
                    kernel=np.exp(-.3*np.sum((x[:,None,:]-x[None,:,:])**2,axis=2))
                    query=np.exp(-.3*np.sum((z[:,None,:]-x[None,:,:])**2,axis=2))
                elif kind==2:
                    kernel=rng.normal(size=(n,n));query=rng.normal(size=(7,n))
                elif kind==3:
                    kernel+=1e12;query+=1e12
                elif kind==4:
                    kernel=rng.normal(size=(n,n));query=rng.normal(size=(7,n))
                    if n>1:
                        kernel[0]=1e16;kernel[-1]=-1e16;query[:,0]=1e16;query[:,-1]=-1e16
                elif kind==5:
                    kernel[:]=.1;query[:]=.1
                elif kind==6:
                    kernel[:]=-0.;query[:]=-0.
                elif kind==7:
                    kernel[:]=1e308;query[:]=1e308
                    if n>1:query[:,1::2]=-1e308
                yield seed,n,kind,np.array(kernel,order='C'),np.array(query,order='C')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path)
    parser.add_argument('--profile',choices=['checked','release'],default='checked')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,7,42,2026,314159]);args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True);exe=args.executable.resolve() if args.executable else resolve(args.profile)
    checks=[];source_warnings=[];cases=0
    for seed,n,kind,kernel,query in fixtures(args.seeds):
        original=kernel.copy();model=KernelCenterer()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');centered=model.fit_transform(kernel);prediction=model.transform(query)
            expected=dict(rows=model.K_fit_rows_,all=[model.K_fit_all_],train=centered,query=prediction,
                          zero_signs=((centered==0)&np.signbit(centered)).astype(int),metadata=[model.n_features_in_,1],
                          names=model.get_feature_names_out())
            saved=copy.deepcopy(model)
            model.fit(np.array([[2.,5.],[-3.,7.]]))
            expected.update(refit=model.transform(np.array([[3.,4.]])),refit_rows=model.K_fit_rows_,
                            refit_all=[model.K_fit_all_],refit_features=[model.n_features_in_],copy_query=saved.transform(query))
        assert np.array_equal(kernel,original)
        source_warnings.extend(dict(seed=seed,n=n,kind=kind,category=w.category.__name__,message=str(w.message)) for w in caught)
        lines=[f'{n} {len(query)}']+[' '.join(map(str,row)) for row in kernel]+[' '.join(map(str,row)) for row in query]
        inp=RESULTS/f'kernel_centering_{args.profile}_input.txt';out=RESULTS/f'kernel_centering_{args.profile}_output.txt'
        inp.write_text('\n'.join(lines)+'\n')
        result=subprocess.run([str(exe),str(inp),str(out)],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert result.returncode==0,(seed,n,kind,result.stdout)
        actual={}
        for line in out.read_text().splitlines():
            key,*values=line.split();actual[key]=np.asarray(values,str if key=='names' else float)
        assert actual.keys()==expected.keys(),(actual.keys(),expected.keys())
        for name,want in expected.items():
            want=np.asarray(want).ravel(order='F')
            if name in ['names','zero_signs','metadata','refit_features']:
                np.testing.assert_array_equal(actual[name],want,err_msg=f'{seed} {n} {kind} {name}')
            else:np.testing.assert_allclose(actual[name],want,rtol=3e-13,atol=3e-13,equal_nan=True,
                                           err_msg=f'{seed} {n} {kind} {name}')
            checks.append(dict(seed=seed,n=n,kind=kind,name=name,passed=True,
                               oracle='input_identity' if name=='metadata' else 'sklearn'))
        cases+=1
        if n==257 and kind==7:print(f'seed {seed}: kernel centering passed',flush=True)
    root=Path(sklearn.__file__).parent
    paths=['preprocessing/_data.py','base.py']
    report=dict(sklearn=sklearn.__version__,numpy=np.__version__,profile=args.profile,seeds=args.seeds,fixtures=cases,
                checks=checks,source_warnings=source_warnings,array_order='C',numpy_pairwise_source=NUMPY_PAIRWISE_SOURCE,
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})
    (RESULTS/f'kernel_centering_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} kernel centering groups across {cases} fixtures passed')


if __name__=='__main__':main()
