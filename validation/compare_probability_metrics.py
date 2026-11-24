"""Cross-entropy and D2 log-loss contracts against installed sklearn 1.6.1."""
from pathlib import Path
import argparse,hashlib,itertools,json,subprocess,warnings
import numpy as np
from scipy import sparse
import sklearn
import sklearn.metrics._classification as source
import sklearn.preprocessing._label as label_source
from sklearn.metrics import log_loss,d2_log_loss_score

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def target(c):
    y=c['y']
    if c['rep'] in [0,1]:y=y[:,0]
    if c['rep'] in [1,3]:y=y.astype(bool)
    if c['rep']==4:y=sparse.csr_matrix(y)
    if c['rep']==5:
        n,q=y.shape;ptr=np.arange(n+1)*2*q
        cols=np.tile(np.repeat(np.arange(q-1,-1,-1),2),n)
        data=np.repeat(y[:,::-1].ravel().astype(float),2)
        y=sparse.csr_matrix((data,cols,ptr),shape=y.shape)
    return y


def oracle(c):
    p=c['p'][:,0] if c['vector'] else c['p']
    options=dict(labels=c['labels'],sample_weight=c['weights'])
    if c['op']==0:return log_loss(target(c),p,normalize=c['normalize'],**options)
    return d2_log_loss_score(target(c),p,**options)


def payload(c):
    y,p=c['y'],c['p'];labels,weights=c['labels'],c['weights']
    header=[c['op'],c['rep'],*y.shape,*p.shape,int(c['vector']),
            -1 if labels is None else len(labels),-1 if weights is None else len(weights),int(c['normalize'])]
    lines=[' '.join(map(str,header)),' '.join(map(str,y.ravel(order='F'))),
           ' '.join(map(str,p.ravel(order='F')))]
    if labels is not None:lines+=[' '.join(map(str,labels))]
    if weights is not None:lines+=[' '.join(map(str,weights))]
    return '\n'.join(lines)+'\n'


def configuration(name,y,p,op=0,rep=0,vector=False,labels=None,weights=None,normalize=True):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if y.ndim==1:y=y[:,None]
    if p.ndim==1:p=p[:,None]
    return dict(name=name,y=y,p=p,op=op,rep=rep,vector=vector,labels=labels,
                weights=None if weights is None else np.asarray(weights,float),normalize=normalize)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path)
    ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);ap.add_argument('--case')
    args=ap.parse_args();exe=args.executable.resolve()
    paths=[exe,Path(__file__),Path(source.__file__),Path(label_source.__file__),
           ROOT/'src/sklearn_probability_metrics.f90',ROOT/'src/sklearn_metrics.f90',
           ROOT/'src/sklearn_label_binarization.f90',ROOT/'src/sklearn_ieee_math.f90',
           ROOT/'app/reference_probability_metrics_driver.f90']
    hashes={str(p):sha(p) for p in paths};cases=[]
    def add(name,y,p,**kw):
        if args.case is not None and name!=args.case:return
        c=configuration(name,y,p,**kw)
        with warnings.catch_warnings(),np.errstate(all='ignore'):
            warnings.simplefilter('ignore');want=float(oracle(c))
        cases.append((c,want))
    for n in range(2,5):
        for bits in itertools.product([0,1],repeat=n):
            for j,p in enumerate([np.full(n,.3),np.linspace(0,1,n),np.linspace(-2,3,n)]):
                for rep in [0,1,2,3]:
                    for vector in [False,True]:
                        for norm in [False,True]:
                            add(f'exhaustive_{n}_{bits}_{j}_{rep}_{vector}_{norm}',bits,p,rep=rep,vector=vector,
                                labels=[1,0,1] if rep in [0,1,2,3] else None,normalize=norm)
                            if norm and rep in [0,1]:
                                add(f'exhaustive_d2_{n}_{bits}_{j}_{rep}_{vector}',bits,p,rep=rep,vector=vector,
                                    labels=[0,1],op=1)
    for seed in [0,1,7,42,314159]:
        rng=np.random.RandomState(seed)
        for n,k in [(2,2),(7,3),(31,5),(129,3)]:
            labels=np.array([-11,2,99,103,900])[:k];y=rng.choice(labels,n);y[:k]=labels;rng.shuffle(y)
            base=rng.uniform(.01,1,size=(n,k));base/=base.sum(axis=1,keepdims=True)
            for scenario,p in enumerate([base,np.full((n,k),.8),rng.uniform(-2,3,size=(n,k))]):
                weights=[None,np.ones(n),np.arange(n,dtype=float),rng.uniform(.1,3,n),-np.ones(n),
                         np.resize([-1.,2.],n)]
                for iw,w in enumerate(weights):
                    for norm in [False,True]:
                        for rep in [0,2]:
                            add(f'random_{seed}_{n}_{k}_{scenario}_{iw}_{norm}_{rep}',y,p,rep=rep,weights=w,
                                labels=labels[::-1].tolist(),normalize=norm)
                    add(f'd2_{seed}_{n}_{k}_{scenario}_{iw}',y,p,op=1,weights=w,labels=labels.tolist())
            for wname,w in [('nan',np.full(n,np.nan)),('inf',np.full(n,np.inf)),('negative_inf',np.full(n,-np.inf))]:
                for norm in [False,True]:
                    add(f'nonfinite_weights_{seed}_{n}_{wname}_{norm}',y,base,weights=w,normalize=norm)
            add(f'zero_weight_sum_{seed}_{n}',y,base,weights=np.zeros(n),normalize=False)
        for n,k in [(2,2),(7,3),(31,17)]:
            mask=rng.randint(0,2,size=(n,k));p=rng.uniform(-1,2,size=(n,k))
            for kind,y in enumerate([mask,mask*2,mask*(-2),mask*0,mask+3]):
                for rep in [2,3]:
                    for norm in [False,True]:
                        add(f'indicator_{seed}_{n}_{k}_{kind}_{rep}_{norm}',y,p,rep=rep,normalize=norm,
                            weights=rng.uniform(.1,3,n))
        for rep in [0,1,2,3]:
            for vector in [False,True]:
                add(f'single_d2_{seed}_{rep}_{vector}',[[9,8]] if rep in [2,3] else [9],[.2],
                    op=1,rep=rep,vector=vector,labels=[],weights=[np.nan])
    for rep in [0,1,2,3]:
        for pval in [0.,1.,-1.e308,1.e308]:
            add(f'clipping_{rep}_{pval}',[0,1],np.full((2,1),pval),rep=rep,vector=True,labels=[0,1])
    for labels,y in [([1,2],[9,2]),([1,2],[9,2,1]),([0,1,2],[9,2,0]),([0,1],[1,1]),([0,1],[0,0])]:
        p=np.full((len(y),len(labels)),.25)
        add(f'unknown_{labels}_{y}',y,p,labels=labels)
        if len(set(y))==1:add(f'constant_d2_{labels}_{y}',y,p,labels=labels,op=1)
    for rep in [2]:
        y=np.array([[0,-2147483648],[-2147483648,0]],int)
        add(f'integer_extreme_{rep}',y,[[.2,.8],[.7,.3]],rep=rep)
    n,k=129,137;y=np.zeros((n,k),int);y[np.arange(n),np.arange(n)]=1
    p=np.full((n,k),1/k)
    for rep in [2,3]:add(f'wide_indicator_{rep}',y,p,rep=rep)
    add('d2_overflow_weight_total', [0,1], [[.8,.2],[.2,.8]], op=1, weights=[1.e308,1.e308])
    add('d2_infinite_loss_ratio', [0,1,2], np.full((3,3),1/3), op=1, weights=np.full(3,1.e308))
    assert cases and len({c['name'] for c,_ in cases})==len(cases)
    run=subprocess.run([str(exe)],input=str(len(cases))+'\n'+''.join(payload(c) for c,_ in cases),text=True,capture_output=True)
    if run.returncode:
        begins=[line for line in run.stdout.splitlines() if line.startswith('BEGIN')]
        last=int(begins[-1].split()[1])-1 if begins else -1
        raise AssertionError((cases[last][0]['name'],run.stdout[-1000:],run.stderr[-1800:]))
    tokens=iter(run.stdout.split());checks=[];configs=[];nonfinite=0
    for i,(c,want) in enumerate(cases,1):
        assert next(tokens)=='BEGIN' and int(next(tokens))==i
        got=float(next(tokens));preserved=int(next(tokens));assert preserved==1,c['name']
        np.testing.assert_allclose(got,want,rtol=3.e-12,atol=5.e-13,equal_nan=True,err_msg=c['name'])
        if want==0:assert np.signbit(got)==np.signbit(want),(c['name'],got,want,'signed zero')
        nonfinite+=int(not np.isfinite(want))
        error=0. if not np.isfinite(want) else abs(got-want)
        checks.extend([dict(fixture=c['name'],field='log_loss' if c['op']==0 else 'd2',passed=True,max_abs_error=error),
                       dict(fixture=c['name'],field='input_preserved',passed=True,max_abs_error=0.)])
        configs.append(dict(name=c['name'],operation=c['op'],representation=c['rep'],n=len(c['y']),
                            target_columns=c['y'].shape[1],probability_columns=c['p'].shape[1],
                            probability_vector=c['vector'],normalize=c['normalize']))
    assert next(tokens,None) is None
    invalid=[]
    def bad(name,y=[0,1],p=[[.8,.2],[.2,.8]],**kw):invalid.append(configuration(name,y,p,**kw))
    bad('single_class',y=[1,1]);bad('single_explicit_label',labels=[1]);bad('empty_labels',labels=[])
    bad('class_width',y=[0,1,2],p=np.ones((3,2)))
    bad('target_length',y=[0]);bad('weight_length',weights=[1]);bad('zero_average_weights',weights=[0,0])
    bad('signed_zero_average_weights',weights=[-1,1]);bad('nan_probability',p=[[np.nan,.2],[.2,.8]])
    bad('inf_probability',p=[[np.inf,.2],[.2,.8]])
    bad('empty_rows',y=[],p=np.zeros((0,2)));bad('empty_probability_columns',p=np.zeros((2,0)))
    for rep in [2,3,4,5]:
        bad(f'indicator_labels_{rep}',y=[[1,0],[0,1]],rep=rep,labels=[0,1])
        bad(f'd2_matrix_{rep}',y=[[1,0],[0,1]],rep=rep,op=1)
    for rep in [2,4,5]:bad(f'non_indicator_{rep}',y=[[0,1],[2,0]],rep=rep)
    for rep in [4,5]:
        bad(f'sparse_one_column_{rep}',y=[[0],[1]],rep=rep)
        bad(f'sparse_two_nonzero_values_{rep}',y=[[3,4],[4,3]],rep=rep)
        bad(f'sparse_single_d2_{rep}',y=[[1,0]],p=[[.8,.2]],rep=rep,op=1)
    for weights,name in [([0,0],'zero'),([np.nan,1],'nan'),([np.inf,1],'inf')]:
        bad(f'd2_weights_{name}',weights=weights,op=1)
    bad('d2_overflow_class_count',y=[0,0],labels=[0,1],weights=[1.e308,1.e308],op=1)
    bad('d2_zero_null_loss',y=[7,8,9],p=np.full((3,3),.3),labels=[0,1,2],op=1)
    bad('d2_absent_class',labels=[0,1,2],p=[[.8,.1,.1],[.1,.8,.1]],op=1)
    bad('d2_one_column_matrix',y=[[0],[1]],rep=2,op=1)
    bad('d2_empty',y=[],p=np.zeros((0,2)),op=1)
    bad('d2_single_weight_length',y=[0],p=[.2],vector=True,op=1,weights=[1,1])
    rejections=[]
    for c in invalid:
        with warnings.catch_warnings(),np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            try:oracle(c)
            except (ValueError,TypeError,ZeroDivisionError) as error:source_error=type(error).__name__
            else:raise AssertionError(('source accepts invalid fixture',c['name']))
        result=subprocess.run([str(exe)],input='1\n'+payload(c),text=True,capture_output=True)
        out=result.stdout+result.stderr
        assert result.returncode!=0 and 'fscikit-learn:' in out,(c['name'],out)
        assert not any(s in out for s in ['SIGFPE','Fortran runtime error']),(c['name'],out)
        rejections.append(dict(name=c['name'],source_error=source_error,passed=True))
    assert hashes=={str(p):sha(p) for p in paths}
    report=dict(profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,configurations=configs,
                rejections=rejections,source_nonfinite_values=nonfinite,hashes=hashes)
    path=args.report or ROOT/'validation/results'/f'probability_metrics_{args.profile}.json'
    path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {len(cases)} fixtures / {len(rejections)} diagnostics / {nonfinite} nonfinite source values: PASS')


if __name__=='__main__':main()
