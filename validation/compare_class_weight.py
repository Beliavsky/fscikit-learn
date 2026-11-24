"""Class and sample weighting against the installed sklearn 1.6.1 utilities."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import subprocess
import warnings
import numpy as np
from scipy import sparse
import sklearn
import sklearn.utils.class_weight as source
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def payload(c):
    y = np.asarray(c['y'], dtype=int)
    if y.ndim == 1: y = y[:, None]
    maps = c.get('maps'); indices = c.get('indices'); classes = c.get('classes', [])
    header = [c['op'], *y.shape, len(classes), int(c['mode']=='balanced'),
              -1 if maps is None else len(maps), -1 if indices is None else len(indices)]
    lines = [' '.join(map(str, header)), ' '.join(map(str, classes)),
             ' '.join(map(str, y.ravel(order='F')))]
    if maps is not None:
        for m in maps:
            lines += [str(len(m)), ' '.join(map(str, m.keys())), ' '.join(map(str, m.values()))]
    if indices is not None: lines += [' '.join(map(str, indices))]
    return '\n'.join(lines) + '\n'


def expected(c):
    y = np.asarray(c['y'], dtype=int)
    if c['op'] in [3, 4, 7]: y = y.astype(bool)
    maps = c.get('maps'); mode = c['mode']; indices = c.get('indices')
    option = 'balanced' if mode=='balanced' else None
    if maps is not None:
        option = maps[0] if c['op'] in [0, 1, 3, 7] or y.shape[1]==1 else maps
    if c['op'] in [0, 7]:
        classes = np.asarray(c['classes'], dtype=bool if c['op']==7 else int)
        return compute_class_weight(option, classes=classes, y=y.ravel())
    if c['op'] in [5, 6]:
        if c['op']==5: y = sparse.csr_matrix(y)
        else:
            n,q = y.shape; columns=[];values=[];ptr=[0]
            for row in y:
                for j in reversed(range(q)):
                    columns += [j,j]; values += [row[j]*.5,row[j]*.5]
                ptr.append(len(values))
            y = sparse.csr_matrix((values,columns,ptr), shape=(n,q))
    if indices is not None:
        indices = np.asarray(indices, dtype=int)
        indices = np.where(indices>0, indices-1, indices)
    return compute_sample_weight(option, y, indices=indices)


def main():
    ap = argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path)
    ap.add_argument('--profile',default='checked');ap.add_argument('--report',type=Path);args=ap.parse_args()
    exe = args.executable.resolve()
    paths = [exe, Path(__file__), Path(source.__file__), ROOT/'src/sklearn_class_weight.f90',
             ROOT/'src/sklearn.f90', ROOT/'app/reference_class_weight_driver.f90']
    before = {str(p):sha(p) for p in paths};fixtures=[]

    def add(name,op,y,mode='none',classes=None,maps=None,indices=None):
        c=dict(name=name,op=op,y=np.asarray(y,dtype=int),mode=mode,classes=[] if classes is None else classes,
               maps=maps,indices=indices)
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore');want=np.atleast_1d(expected(c)).astype(float)
        fixtures.append((c,want))

    for n in range(1,6):
        for bits in itertools.product([0,1],repeat=n):
            y=np.array(bits);classes=np.unique(y)
            for order_id,order in enumerate([classes,classes[::-1],np.repeat(classes,2)]):
                for mode in ['none','balanced']:
                    add(f'exhaustive_class_{n}_{bits}_{order_id}_{mode}',0,y,mode,classes=order.tolist())
            for op in [1,3]:
                for mode in ['none','balanced']:
                    add(f'exhaustive_sample_{n}_{bits}_{op}_{mode}',op,y,mode)
    for seed in [0,1,7,42,314159]:
        rng=np.random.RandomState(seed)
        for n in [1,2,5,17,129]:
            for q in [1,2,5]:
                y=rng.choice([-11,0,3,99],size=(n,q))
                selections=[None,list(rng.permutation(n)+1),list(rng.randint(1,n+1,size=2*n)),[-1],[-n,-1]]
                for i,indices in enumerate(selections):
                    for op in [2,5,6]:
                        add(f'random_{seed}_{n}_{q}_{i}_{op}',op,y,'balanced',indices=indices)
                    if q==1:add(f'vector_{seed}_{n}_{i}',1,y[:,0],'balanced',indices=indices)
                for kind in ['empty','partial','full','nonfinite']:
                    maps=[]
                    for j in range(q):
                        labels=np.unique(y[:,j]).tolist()
                        values=rng.uniform(-3,3,size=len(labels))
                        if kind=='nonfinite': values=np.resize([0.,np.inf,np.nan,-np.inf],len(labels))
                        m=dict(zip(labels,values));
                        if kind=='empty':m={}
                        if kind=='partial':m={labels[0]:float(values[0])}
                        maps.append(m)
                    for op in [2,5,6]:add(f'map_{seed}_{n}_{q}_{kind}_{op}',op,y,maps=maps)
                    if q==1:
                        add(f'map_vector_{seed}_{n}_{kind}',1,y[:,0],maps=maps)
                        add(f'map_class_{seed}_{n}_{kind}',0,y[:,0],classes=np.unique(y).tolist(),maps=maps)
                logical=rng.randint(0,2,size=(n,q))
                for op in [2,4,5,6]:add(f'logical_{seed}_{n}_{q}_{op}',op,logical,'balanced')
                if q==1:
                    add(f'logical_vector_{seed}_{n}',3,logical[:,0],'balanced')
                    add(f'logical_class_{seed}_{n}',7,logical[:,0],'balanced',classes=np.unique(logical).tolist())
    for n in [0,1,10]:
        add(f'empty_class_{n}',0,np.array([],int),classes=[])
        add(f'empty_sample_{n}',1,np.array([],int))
        for op in [2,5,6]:add(f'zero_outputs_{n}_{op}',op,np.zeros((n,0),int),'balanced')
    for classes,maps in [([0,1,9],[{0:2.,1:3.}]),([0,1],[{0:2.,1:3.,9:7.}]),
                         ([1,1,0],[{0:-2.,1:0.}]),([],[{9:2.}])]:
        y=np.array([],int) if not classes else np.array([0,1,1])
        add(f'class_extra_{classes}_{maps}',0,y,classes=classes,maps=maps)
    y=np.array([-2147483648,-2147483648,2147483647,0],dtype=int)
    for op in [1,2,5,6]:add(f'extreme_integer_{op}',op,y if op==1 else y[:,None],'balanced')
    large=np.arange(10000,dtype=int)
    add('many_distinct_classes',0,large,'balanced',classes=large[::-1].tolist())
    add('many_distinct_samples',1,large,'balanced')
    wide=np.zeros((3,10000),int);wide[0,::2]=1;wide[1,1::2]=1
    for op in [2,5,6]:add(f'wide_multilabel_{op}',op,wide,'balanced')

    assert len({c['name'] for c,_ in fixtures})==len(fixtures)
    data=str(len(fixtures))+'\n'+''.join(payload(c) for c,_ in fixtures)
    run=subprocess.run([str(exe)],input=data,text=True,capture_output=True)
    assert run.returncode==0,(run.stdout[-2000:],run.stderr[-2000:])
    tokens=iter(run.stdout.split());checks=[];nonfinite=0;certificates=0
    for c,want in fixtures:
        size=int(next(tokens));got=np.array([float(next(tokens)) for _ in range(size)])
        np.testing.assert_allclose(got,want,atol=3.e-12,rtol=3.e-12,equal_nan=True,err_msg=c['name'])
        zeros=(want==0)
        np.testing.assert_array_equal(np.signbit(got[zeros]),np.signbit(want[zeros]),err_msg=c['name'])
        nonfinite+=int(np.count_nonzero(~np.isfinite(want)))
        if c['mode']=='balanced' and c['op'] in [1,3]:
            y=c['y'].ravel();selection=c['indices']
            selected=np.arange(len(y)) if selection is None else np.array([i-1 if i>0 else len(y)+i for i in selection])
            labels,counts=np.unique(y[selected],return_counts=True)
            np.testing.assert_allclose(got[selected].sum(),len(selected),rtol=3.e-12,atol=3.e-12)
            for label in labels:
                np.testing.assert_allclose(got[selected][y[selected]==label].sum(),len(selected)/len(labels),rtol=3.e-12,atol=3.e-12)
            certificates+=1
        finite=np.isfinite(want)
        checks.append(dict(name=c['name'],operation=c['op'],passed=True,size=size,
                           max_abs_error=float(np.max(np.abs(got[finite]-want[finite]),initial=0))))
    assert next(tokens,None) is None

    invalid=[]
    def bad(name,op,y,mode='none',classes=None,maps=None,indices=None):
        invalid.append(dict(name=name,op=op,y=np.asarray(y,int),mode=mode,classes=[] if classes is None else classes,maps=maps,indices=indices))
    bad('missing_target_class',0,[1,2],classes=[1])
    bad('balanced_unknown_class',0,[1,2],'balanced',classes=[1,2,3])
    bad('balanced_empty_class',0,[],'balanced',classes=[])
    bad('balanced_empty_sample',1,[],'balanced')
    bad('map_unknown_with_omission',0,[1,2],classes=[1,2],maps=[{1:2.,9:1.}])
    bad('map_duplicate_class_omission',0,[1,2],classes=[1,1,2],maps=[{1:2.}])
    bad('sample_map_unknown',1,[1,2],maps=[{1:2.,9:1.}])
    for op in [1,2,5,6]:
        y=np.array([1,2,2]) if op==1 else np.array([[1],[2],[2]])
        bad(f'empty_indices_{op}',op,y,'balanced',indices=[])
        bad(f'positive_index_{op}',op,y,'balanced',indices=[4])
        bad(f'negative_index_{op}',op,y,'balanced',indices=[-4])
        bad(f'unweighted_indices_{op}',op,y,indices=[1])
    for op in [2,5,6]:
        bad(f'multioutput_none_{op}',op,[[0,1],[1,0]])
        bad(f'map_count_{op}',op,[[0,1],[1,0]],maps=[{}])
        bad(f'empty_multioutput_{op}',op,np.zeros((0,2),int),'balanced')
    rejections=[]
    for c in invalid:
        try:expected(c)
        except (ValueError,IndexError) as error:source_error=type(error).__name__
        else:raise AssertionError(('source accepts invalid case',c['name']))
        result=subprocess.run([str(exe)],input='1\n'+payload(c),text=True,capture_output=True)
        output=result.stdout+result.stderr
        assert result.returncode!=0 and 'fscikit-learn: class' in output,(c['name'],output)
        assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(c['name'],output)
        rejections.append(dict(name=c['name'],source_error=source_error,passed=True))
    assert before=={str(p):sha(p) for p in paths}
    report=dict(profile=args.profile,sklearn=sklearn.__version__,fixtures=len(fixtures),checks=checks,rejections=rejections,
                source_nonfinite_values=nonfinite,balanced_mass_certificates=certificates,hashes=before)
    path=args.report or ROOT/'validation/results'/f'class_weight_{args.profile}.json'
    path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} class/sample-weight comparisons and {len(rejections)} diagnostics: PASS')
    print(f'{certificates} balanced-mass certificates and {nonfinite} nonfinite source values: PASS')


if __name__=='__main__':main()
