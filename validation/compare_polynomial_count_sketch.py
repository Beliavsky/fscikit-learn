"""PolynomialCountSketch hashes, features and live transform controls."""
from pathlib import Path
import argparse
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.kernel_approximation import PolynomialCountSketch
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path);args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable: executable=args.executable.resolve()
    else:
        proc=subprocess.run(['fpm','run','--flag',FLAGS,'--target','reference_polynomial_count_sketch_driver','--runner'],
                            cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/line.strip().strip('"') for line in proc.stdout.splitlines()
               if line.strip().strip('"').endswith(('reference_polynomial_count_sketch_driver','reference_polynomial_count_sketch_driver.exe'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,proc.stdout
        executable=paths[0]
    checks=[]
    for seed,n,p in [(0,13,5),(7,11,9),(42,7,21),(2026,17,3),(2147483647,5,4)]:
        rng=np.random.RandomState(seed);base=rng.normal(size=(n,p))*.3
        for case in range(28):
            expected={};lines=[];ops=[];model=None
            d=2;components=100;gamma=1.;coef0=0.;x=base.copy()
            if case<16: components=[1,2,3,5,7,8,15,16,17,31,32,63,64,127,257,1009][case]
            if case==16: d=1;components=13
            if case==17: d=3;components=24
            if case==18: d=7;components=67;x/=2
            if case==19: gamma=0;coef0=0;components=11
            if case==20: gamma=0;coef0=2;components=19;d=4
            if case==21: gamma=2.3;coef0=.7;components=27;d=3
            if case==22: coef0=-.3;components=10
            if case==23: x[:]=0;components=23
            if case==24: x=np.ones((n,1))*.5;coef0=1;components=29
            if case==25: components=129;d=5;coef0=.1
            if case==26: components=2048;d=2;gamma=3
            if case==27: components=37;d=4;gamma=.001;coef0=.0001
            def add(op,a,degree=d,ncomp=components,g=gamma,c=coef0):
                nonlocal model
                a=np.array(a,copy=True)
                if op in (1,2): model=PolynomialCountSketch(degree=degree,n_components=ncomp,random_state=seed,gamma=g,coef0=c)
                else:model.set_params(degree=degree,n_components=ncomp,gamma=g,coef0=c)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    if op==1: model.fit(a);features=model.transform(a)
                    elif op==2: features=model.fit_transform(a)
                    else:features=model.transform(a)
                k=len(ops)+1
                values=dict(hash=model.indexHash_.copy(),bits=model.bitHash_.copy(),features=features,
                            gram=features@features.T,meta=[model.n_features_in_,model._n_features_out,len(a),features.shape[1]])
                expected.update({f'{k}_{name}':np.asarray(value).ravel(order='F') for name,value in values.items()})
                ops.append((op,a,degree,ncomp,g,c))
            add(1,x);add(3,x*.7+.1);add(2,x)
            # Safe parameter changes follow source's live controls and cached output metadata.
            if case in (16,17,18,21,25):add(3,x,degree=1,ncomp=components+11,g=.5,c=coef0*2)
            if case==0:add(3,x,g=2.)
            # A full refit replaces both fitted dimensions and hash shapes.
            add(2,base[:,:min(2,p)],degree=3,ncomp=6,g=.7,c=1.)
            lines.append(str(len(ops)))
            for op,a,degree,ncomp,g,c in ops:
                lines.append(f'{op} {len(a)} {a.shape[1]} {degree} {ncomp} {seed} {g:.17g} {c:.17g}')
                lines.extend(' '.join(f'{v:.17g}' for v in row) for row in a)
            inp=RESULTS/'polynomial_count_sketch_input.txt';out=RESULTS/'polynomial_count_sketch_output.txt'
            inp.write_text('\n'.join(lines)+'\n')
            proc=subprocess.run([str(executable),str(inp),str(out)],cwd=ROOT,text=True,
                                stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            assert proc.returncode==0,(seed,case,proc.stdout)
            actual={}
            for line in out.read_text().splitlines():
                name,*values=line.split();actual[name]=np.array(values,dtype=float)
            assert actual.keys()==expected.keys(),(seed,case)
            for name,want in expected.items():
                tol=0 if name.endswith(('hash','bits','meta')) else 3.e-8
                np.testing.assert_allclose(actual[name],want,atol=tol,rtol=tol,equal_nan=True,
                                           err_msg=f'{seed} {case} {name}')
                checks.append(dict(seed=seed,case=case,name=name,passed=True,oracle='sklearn'))
        print(f'seed {seed}: 28 scenarios passed',flush=True)
    (RESULTS/'polynomial_count_sketch.json').write_text(json.dumps(dict(sklearn=sklearn.__version__,checks=checks),indent=2)+'\n')
    print(f'{len(checks)} PolynomialCountSketch result groups passed')


if __name__=='__main__':main()
