"""Direct seeded BernoulliRBM training, incremental and sampling contracts."""
from pathlib import Path
import argparse
import json
import os
import subprocess
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import sklearn
from sklearn.neural_network import BernoulliRBM

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path);args=parser.parse_args()
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
    checks=[]
    for seed,n,p in [(0,27,7),(7,19,4),(42,33,11),(2026,13,1)]:
        rng=np.random.RandomState(seed);binary=(rng.uniform(size=(n,p))<.4).astype(float)
        query=(rng.uniform(size=(9,p))<.6).astype(float)
        for case in range(20):
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
            source=BernoulliRBM();expected={};lines=[str(len(ops))]
            for i,(op,data,config) in enumerate(ops,1):
                source.set_params(**config);prefix=f'{i}_'
                lines.extend([f'{op} {data.shape[0]} {data.shape[1]}',
                              f'{config["n_components"]} {config["batch_size"]} {config["n_iter"]} {config["random_state"]} {config["learning_rate"]:.17g}'])
                lines.extend(' '.join(f'{v:.17g}' for v in row) for row in data)
                result=None
                if op==1:source.fit(data)
                elif op==2:source.partial_fit(data)
                elif op==3:result=source.fit_transform(data)
                elif op==4:result=source.transform(data)
                elif op==5:result=source.gibbs(data)
                elif op==6:result=source.score_samples(data)
                if result is not None:expected[prefix+'result']=result
                for name,attr in [('components','components_'),('hidden','intercept_hidden_'),('visible','intercept_visible_'),('chain','h_samples_')]:
                    expected[prefix+name]=getattr(source,attr).copy()
                expected[prefix+'features']=[source.n_features_in_]
            inp=RESULTS/'rbm_input.txt';out=RESULTS/'rbm_output.txt';inp.write_text('\n'.join(lines)+'\n')
            proc=subprocess.run([str(executable),str(inp),str(out)],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            assert proc.returncode==0,(seed,case,proc.stdout)
            actual={}
            for line in out.read_text().splitlines():
                name,*values=line.split();actual[name]=np.array(values,dtype=float)
            assert actual.keys()==expected.keys()
            for name,value in expected.items():
                op=int(name.split('_')[0]);discrete=name.endswith(('chain','features')) or (name.endswith('result') and ops[op-1][0]==5)
                tol=0 if discrete else 2.e-10
                np.testing.assert_allclose(actual[name],np.asarray(value).ravel(order='F'),atol=tol,rtol=tol,
                                           err_msg=f'seed={seed} case={case} {name}')
                checks.append(dict(seed=seed,case=case,name=name,passed=True))
        print(f'seed {seed}: 20 configurations passed',flush=True)
    (RESULTS/'rbm.json').write_text(json.dumps(dict(sklearn=sklearn.__version__,checks=checks),indent=2)+'\n')
    print(f'{len(checks)} BernoulliRBM result groups passed')


if __name__=='__main__':main()
