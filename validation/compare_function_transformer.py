"""FunctionTransformer: numeric callbacks, source inverse checks and feature names."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import FunctionTransformer
from transform_callbacks import callbacks,prefixed_names

ROOT=Path(__file__).resolve().parents[1]
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile,target):
    options=['--profile','release'] if profile=='release' else ['--flag',FLAGS]
    r=subprocess.run(['fpm','run',*options,'--target',target,'--runner'],cwd=ROOT,text=True,
                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/s.strip().strip('"') for s in r.stdout.splitlines()
           if s.strip().strip('"').endswith((target,target+'.exe'))]
    paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,r.stdout
    return paths[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);args=ap.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_function_transformer_driver')
    checks=[];count=0;max_error=0.;caught_warnings={}
    for seed in [0,42,2026]:
        rng=np.random.RandomState(seed)
        for mode in [0,1,2,3,4,5,6,7,8,10]:
            for case in range(10):
                n,p=[(1,1),(7,3),(17,2),(107,4),(251,2)][case%5];m=5
                x=rng.uniform(.2,4,size=(n,p));q=rng.uniform(.2,4,size=(m,p))
                valid=case%2;check=int(case%3!=0);bits=[0,1,2,3][case%4]
                if mode==0:bits=0
                if mode in [4,5,6,7,8,10]:bits=3
                if mode==7:x[0,0]=0
                if mode==10 and n>2:x[2,0]=7.7
                if mode==8:valid=0
                operation=case%3;stored=int(case%4==0);naming=case%4
                if naming==3:valid=1;operation=0
                if mode==8 and naming==3:naming=2;valid=0
                if case==8 and not valid:n=0;x=np.empty((0,p))
                f,inv=callbacks(mode);f=f if bits&1 else None;inv=inv if bits&2 else None
                names=['field_'+str(j+1) for j in range(p)]
                naming_arg=prefixed_names if naming==2 else ('one-to-one' if naming in [1,3] else None)
                model=FunctionTransformer(func=f,inverse_func=inv,validate=bool(valid),check_inverse=bool(check),
                                          feature_names_out=naming_arg)
                data=pd.DataFrame(x,columns=names) if stored else x
                with warnings.catch_warnings(record=True) as warned,np.errstate(invalid='ignore',over='ignore'):
                    warnings.simplefilter('always')
                    if operation==2:z=np.asarray(model.fit_transform(data))
                    else:
                        if operation==0:model.fit(data)
                        z=np.asarray(model.transform(x))
                    query=np.asarray(model.transform(q));back=np.asarray(model.inverse_transform(z))
                    nf=getattr(model,'n_features_in_',0)
                    inv_checked=bool(operation!=1 and check and f is not None and inv is not None)
                    consistent=True
                    if inv_checked:
                        step=max(1,n//100);selected=x[::step];trip=inv(f(selected))
                        consistent=bool(np.allclose(selected,trip,rtol=1e-7,atol=1e-9))
                    expected_names=[]
                    if naming:
                        if (stored and operation!=1) or naming==3:
                            expected_names=list(model.get_feature_names_out())
                        else:expected_names=list(model.get_feature_names_out(names))
                for w in warned:caught_warnings[str(w.message)]=caught_warnings.get(str(w.message),0)+1
                header=[n,p,m,mode,bits,valid,check,naming,operation,stored]
                payload='\n'.join([' '.join(map(str,header)),' '.join(format(v,'.17g') for v in x.ravel(order='F')),
                                   ' '.join(format(v,'.17g') for v in q.ravel(order='F'))])+'\n'
                run=subprocess.run([str(exe)],input=payload,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                c=dict(seed=seed,mode=mode,case=case,n=n,p=p,bits=bits,validate=valid,check_inverse=check,
                       naming=naming,operation=operation,stored=stored)
                assert run.returncode==0,(c,run.stdout)
                lines=iter(run.stdout.splitlines());meta=np.fromstring(next(lines),sep=' ',dtype=int)
                np.testing.assert_array_equal(meta,[nf,int(inv_checked),int(consistent)],err_msg=str(c))
                checks.append(dict(**c,name='metadata',passed=True))
                for label,want in [('forward',z),('query',query),('inverse',back)]:
                    shape=tuple(map(int,next(lines).split()));assert shape==want.shape,(c,label,shape,want.shape)
                    actual=np.fromstring(next(lines),sep=' ').reshape(shape,order='F')
                    np.testing.assert_allclose(actual,want,rtol=3e-12,atol=3e-12,equal_nan=True,err_msg=str(c))
                    finite=np.isfinite(actual)&np.isfinite(want)
                    max_error=max(max_error,float(np.max(abs(actual[finite]-want[finite]),initial=0)))
                    checks.append(dict(**c,name=label,passed=True))
                actual_names=[next(lines) for _ in range(int(next(lines)))]
                assert actual_names==expected_names,(c,actual_names,expected_names)
                if naming:checks.append(dict(**c,name='feature_names',passed=True))
                assert next(lines,None) is None
                count+=1
                if count%100==0:print(f'{count} FunctionTransformer fixtures passed',flush=True)
    root=Path(sklearn.__file__).parent
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,
                fixtures=count,checks=checks,max_absolute_error=max_error,source_warnings=caught_warnings,
                native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                               ['preprocessing/_function_transformer.py','utils/validation.py']})
    (ROOT/'validation/results').mkdir(exist_ok=True)
    (ROOT/'validation/results'/f'function_transformer_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {count} FunctionTransformer fixtures passed')


if __name__=='__main__':main()
