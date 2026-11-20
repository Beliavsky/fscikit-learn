"""MiniBatchNMF beta divergences, initialization, fit/stream state and inference."""
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
from sklearn.decomposition import MiniBatchNMF
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def beta_number(value):
    return {'frobenius':2.,'kullback-leibler':1.,'itakura-saito':0.}[value] if isinstance(value,str) else value

def optional(value,sentinel):return sentinel if value is None else value

def rows(x):return [' '.join(f'{v:.17g}' for v in row) for row in x]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path);args=parser.parse_args();RESULTS.mkdir(exist_ok=True)
    if args.executable:executable=args.executable.resolve()
    else:
        proc=subprocess.run(['fpm','run','--flag',FLAGS,'--target','reference_minibatch_nmf_driver','--runner'],cwd=ROOT,
                            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/line.strip().strip('"') for line in proc.stdout.splitlines()
               if line.strip().strip('"').endswith(('reference_minibatch_nmf_driver','reference_minibatch_nmf_driver.exe'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,proc.stdout
        executable=paths[0]
    checks=[]
    for seed,n,p in [(0,23,7),(7,31,5),(42,28,11),(2026,19,3)]:
        rng=np.random.RandomState(seed);base=.2+abs(rng.normal(size=(n,p)));query=.2+abs(rng.normal(size=(8,p)))
        for case in range(41):
            x=base.copy();q=query.copy();ops=[]
            config=dict(n_components=3,init=None,max_iter=5,batch_size=7,random_state=seed,max_no_improvement=None,
                        fresh_restarts_max_iter=30,transform_max_iter=None,beta_loss='frobenius',tol=0.,alpha_W=0.,
                        alpha_H='same',l1_ratio=0.,forget_factor=.7,fresh_restarts=False)
            def add(op,data,custom=None,**changes):
                config.update(changes);ops.append((op,np.array(data,copy=True),config.copy(),custom))
            if case in (1,2,3,4):config.update(init=['random','nndsvd','nndsvda','nndsvdar'][case-1])
            if case in (5,6,37):config.update(init='custom')
            if case==6:config.update(n_components='auto')
            if case==7:config.update(n_components=p+3)
            if case==8:config.update(beta_loss='kullback-leibler')
            if case==9:config.update(beta_loss='itakura-saito')
            if case==10:config.update(beta_loss=-.5)
            if case==11:config.update(beta_loss=.5)
            if case==12:config.update(beta_loss=1.5)
            if case==13:config.update(beta_loss=2.5)
            if case==14:config.update(beta_loss=3.)
            if case==15:config.update(alpha_W=.03,l1_ratio=1.)
            if case==16:config.update(alpha_W=.02,alpha_H=.05)
            if case==17:config.update(alpha_W=.01,alpha_H=.03,l1_ratio=.6)
            if case==18:config.update(forget_factor=0.)
            if case==19:config.update(forget_factor=1.)
            if case==20:config.update(fresh_restarts=True,fresh_restarts_max_iter=5,transform_max_iter=10)
            if case==21:config.update(batch_size=1,max_iter=2)
            if case==22:config.update(batch_size=n+10)
            if case==23:config.update(tol=1.e9,max_iter=1000000000,transform_max_iter=4)
            if case==24:config.update(max_no_improvement=1,max_iter=30,init='random');x[:]=0;q[:]=0
            if case==25:config.update(max_no_improvement=2,max_iter=40,init='random');x[:]=0;q[:]=0
            if case==29:config.update(beta_loss=1.,init='random');x[:]=1.e-9;q[:]=2.e-9
            if case==30:
                config.update(beta_loss=1.,init='random');x[rng.uniform(size=x.shape)<.4]=0;q[rng.uniform(size=q.shape)<.4]=0
            if case==31:config.update(beta_loss=0.,fresh_restarts=True,fresh_restarts_max_iter=3,transform_max_iter=7)
            if case==32:config.update(beta_loss=.5,alpha_W=.01,alpha_H=.02,l1_ratio=.5)
            if case==33:config.update(beta_loss=1.,init='nndsvd')
            if case==34:config.update(beta_loss=3.,init='nndsvdar')
            if case==35:config.update(n_components='auto',max_iter=200,batch_size=1024,tol=1.e-4,max_no_improvement=10)
            if case==36:config.update(fresh_restarts=True,forget_factor=0.,alpha_H=.01,beta_loss=1.)
            if case==37:config.update(alpha_W=.01,l1_ratio=.5,forget_factor=1.)
            if case==38:config.update(init='random')
            if case==40:config.update(n_components='auto',max_iter=40,max_no_improvement=2,tol=.001);x[:]=0;q[:]=0
            if case==39:x=.2+abs(rng.normal(size=(7,17)));q=.2+abs(rng.normal(size=(8,17)))
            if case in (5,6,37,38):
                k=4 if case==6 else 3
                w0=.1+rng.uniform(size=(len(x),k));h0=.1+rng.uniform(size=(k,x.shape[1]))
                if case==37:w0[0]=0;h0[0,0]=0
                add(3,x,custom=(w0,h0))
            elif case==26:
                add(2,x[:5]);add(2,x[5:13],n_components=2,beta_loss=3.,forget_factor=0.,batch_size=3)
                add(2,x[13:],alpha_W=.03)
            elif case==27:
                add(1,x);add(2,x[:2],alpha_W=.02,alpha_H=.01,l1_ratio=.3,transform_max_iter=1,random_state=seed+2)
            elif case==28:
                add(2,x[:7]);x=np.column_stack([x,np.ones(len(x))]);q=np.column_stack([q,np.ones(len(q))])
                add(3,x,n_components=2,random_state=seed+1)
            else:add(3,x)
            k=x.shape[1] if config['n_components']=='auto' else config['n_components']
            if case==6:k=4
            if case==26:k=3 # valid changed control is ignored by the initialized stream
            for op,data in [(4,q),(5,rng.normal(size=(5,k))),(2,x[:6]),(4,q)]:add(op,data)
            source=MiniBatchNMF();expected={};lines=[str(len(ops))]
            for i,(op,data,c,custom) in enumerate(ops,1):
                source.set_params(**c);prefix=f'{i}_'
                ints=[0 if c['n_components']=='auto' else c['n_components'],c['max_iter'],c['batch_size'],c['random_state'],
                      optional(c['max_no_improvement'],-1),c['fresh_restarts_max_iter'],optional(c['transform_max_iter'],0)]
                real=[beta_number(c['beta_loss']),c['tol'],c['alpha_W'],-1 if c['alpha_H']=='same' else c['alpha_H'],c['l1_ratio'],c['forget_factor']]
                lines.extend([f'{op} {data.shape[0]} {data.shape[1]}',' '.join(map(str,ints)),
                              ' '.join(f'{v:.17g}' for v in real),f"{optional(c['init'],'auto')} {'T' if c['fresh_restarts'] else 'F'} {0 if custom is None else custom[0].shape[1]}"])
                kw={}
                if custom is not None:
                    lines.extend(rows(custom[0]));lines.extend(rows(custom[1]));kw=dict(W=custom[0].copy(),H=custom[1].copy())
                lines.extend(rows(data));result=None
                with warnings.catch_warnings(),np.errstate(divide='ignore',invalid='ignore'):
                    warnings.simplefilter('ignore')
                    if op==1:source.fit(data,**kw)
                    elif op==2:source.partial_fit(data,**kw)
                    elif op==3:result=source.fit_transform(data,**kw)
                    elif op==4:result=source.transform(data)
                    elif op==5:result=source.inverse_transform(data)
                if result is not None:expected[prefix+'result']=result
                expected[prefix+'components']=source.components_.copy()
                expected[prefix+'numerator']=source._components_numerator.copy();expected[prefix+'denominator']=source._components_denominator.copy()
                expected[prefix+'meta']=[source.n_features_in_,source.n_components_,source._batch_size,source._transform_max_iter,
                                         source.n_steps_,getattr(source,'n_iter_',0)]
                expected[prefix+'resolved']=[source._beta_loss,source._gamma,source._rho]
                expected[prefix+'convergence']=[optional(getattr(source,'_ewa_cost',None),np.nan),
                    optional(getattr(source,'_ewa_cost_min',None),np.nan),getattr(source,'_no_improvement',0)]
                expected[prefix+'error']=[getattr(source,'reconstruction_err_',np.nan)]
            inp=RESULTS/'minibatch_nmf_input.txt';out=RESULTS/'minibatch_nmf_output.txt';inp.write_text('\n'.join(lines)+'\n')
            proc=subprocess.run([str(executable),str(inp),str(out)],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            assert proc.returncode==0,(seed,case,proc.stdout)
            actual={}
            for line in out.read_text().splitlines():
                name,*values=line.split();actual[name]=np.array(values,dtype=float)
            assert actual.keys()==expected.keys()
            for name,value in expected.items():
                tol=0 if name.endswith('meta') else 3.e-8
                np.testing.assert_allclose(actual[name],np.asarray(value).ravel(order='F'),atol=tol,rtol=tol,equal_nan=True,
                                           err_msg=f'seed={seed} case={case} {name}')
                checks.append(dict(seed=seed,case=case,name=name,passed=True))
        print(f'seed {seed}: 41 configurations passed',flush=True)
    (RESULTS/'minibatch_nmf.json').write_text(json.dumps(dict(sklearn=sklearn.__version__,checks=checks),indent=2)+'\n')
    print(f'{len(checks)} MiniBatchNMF result groups passed')


if __name__=='__main__':main()
