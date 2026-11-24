"""Bootstrap and shuffle index/data adapters against sklearn.utils 1.6.1."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
from scipy import sparse
import sklearn
import sklearn.utils._indexing as source
import sklearn.utils.extmath as extmath
from sklearn.utils import resample,shuffle

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
INTEGER_FIELDS={'indices','iv','im','jv','jm','bv','bm','cv','cm','char_length','csr_ptr','csr_indices','data_preserved','rng_preserved'}


def stratum(c):
    kind=c['kind'];s=c['s']
    if kind==0:return None
    if kind==11:return sparse.csr_matrix(np.eye(2))
    if kind==12:return 1
    if kind==13:return np.array([[[1,2]]])
    if kind in [5,6]:s=s.astype(bool)
    if kind in [7,8]:s=np.array([str(v).ljust(64) for v in s.ravel()],dtype='U64').reshape(s.shape)
    return s[:,0] if kind in [1,3,5,7,9] else s


def payload(c):
    s=c['s'];n,q=(0,0) if s is None else s.shape
    lines=[' '.join(map(str,[c['n'],c['draw'],c['seed'],int(c['replace']),int(c['shuffle']),c['kind'],n,q,int(c['typed'])]))]
    if c['kind'] in [7,8]:lines+=list(map(str,s.ravel(order='F')))
    elif c['kind'] in [1,2,3,4,5,6,9,10]:lines+=[' '.join(map(str,s.ravel(order='F')))]
    return '\n'.join(lines)+'\n'


def expected(c):
    n=c['n'];ids=np.arange(1,n+1,dtype=np.int64);arrays=[ids]
    if c['typed']:
        iv=ids.copy();im=np.c_[ids,-ids];jv=ids+2**40;jm=np.c_[jv,-jv]
        rv=ids*.125;rm=np.c_[rv,-rv];bv=ids%2==0;bm=np.c_[bv,~bv]
        cv=np.array([str(i).ljust(24) for i in ids],dtype='U24')
        cm=np.array([[str(i).ljust(24),str(-i).ljust(24)] for i in ids],dtype='U24').reshape(n,2)
        ptr=np.arange(n+1)*4;cols=np.tile([1,0,0,2],n)
        data=np.c_[-ids.astype(float),ids*.25,ids*.75,np.full(n,-0.)].ravel()
        matrix=sparse.csr_matrix((data,cols,ptr),shape=(n,3))
        arrays += [iv,im,jv,jm,rv,rm,bv,bm,cv,cm,matrix]
    options=dict(n_samples=None if c['draw']==-1 else c['draw'],random_state=c['seed'])
    if c['shuffle']:sampled=shuffle(*arrays,**options)
    else:sampled=resample(*arrays,replace=c['replace'],stratify=stratum(c),**options)
    if not c['typed']:return dict(indices=np.asarray(sampled),rng_preserved=np.ones(1,int))
    out=dict(zip(['indices','iv','im','jv','jm','rv','rm','bv','bm','cv','cm'],sampled[:-1]))
    out['cv']=out['cv'].astype(np.int64);out['cm']=out['cm'].astype(np.int64)
    mat=sampled[-1];out.update(char_length=np.array([24,24]),csr_ptr=mat.indptr+1,csr_indices=mat.indices+1,
                             csr_data=mat.data,data_preserved=np.ones(1,int),rng_preserved=np.ones(1,int))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path)
    ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);ap.add_argument('--case')
    args=ap.parse_args();exe=args.executable.resolve()
    paths=[exe,Path(__file__),Path(source.__file__),Path(extmath.__file__),ROOT/'src/sklearn_resampling.f90',
           ROOT/'src/sklearn_legacy_rng.f90',ROOT/'src/sklearn_sparse.f90',ROOT/'app/reference_array_resampling_driver.f90']
    before={str(p):sha(p) for p in paths};cases=[]
    def config(name,n,draw=-1,seed=0,replace=True,shuffle_=False,kind=0,s=None,typed=False):
        if s is not None:
            s=np.asarray(s);s=s[:,None] if s.ndim==1 else s
        return dict(name=name,n=n,draw=draw,seed=seed,replace=replace,shuffle=shuffle_,kind=kind,s=s,typed=typed)
    def add(name,n,**kw):
        if args.case is not None and name!=args.case:return
        c=config(name,n,**kw);cases.append((c,expected(c)))
    seeds=[0,1,7,42,2147483647]
    for seed in seeds:
        for n in [0,1,2,3,17,129]:
            draws=[-1] if n==0 else sorted(set([-1,1,min(3,n),n,3*n]))
            for draw in draws:
                for mode in ['bootstrap','subsample','shuffle']:
                    if mode!='bootstrap' and draw>n:continue
                    add(f'plain_{seed}_{n}_{draw}_{mode}',n,seed=seed,draw=draw,replace=mode=='bootstrap',
                        shuffle_=mode=='shuffle',typed=n<=17 and draw in [-1,1])
        rng=np.random.RandomState(seed)
        for n in [1,2,7,31,129]:
            for kind in range(1,11):
                q=1 if kind in [1,3,5,7,9] else 2
                if kind in [1,2]:s=rng.choice([-12,-2,0,2,10,100],size=(n,q))
                elif kind in [3,4]:s=rng.normal(size=(n,q))
                elif kind in [5,6]:s=rng.randint(0,2,size=(n,q))
                elif kind in [7,8]:s=rng.choice(['a',' a','10','2'],size=(n,q))
                else:s=rng.choice(np.array([-2**63,-2**40,0,2**40,2**63-1],dtype=np.int64),size=(n,q))
                for replacement in [False,True]:
                    for draw in sorted(set([-1,1,max(1,n//2),2*n] if replacement else [-1,1,max(1,n//2)])):
                        add(f'strata_{seed}_{n}_{kind}_{replacement}_{draw}',n,seed=seed,draw=draw,
                            replace=replacement,kind=kind,s=s,typed=(n<=2 or seed==0 and n==7) and draw==-1)
    floats=np.array([0.,-0.,1e-5,1e-4,1e15,1e16,1e23,1.2e16,1e-6,1.23456789012345,
                     np.nextafter(1e-4,0),np.nextafter(1e16,0),np.finfo(float).max,-np.finfo(float).max,
                     np.nextafter(0.,1.),-np.nextafter(0.,1.)])
    for seed in seeds:
        for q in [1,2]:
            s=np.tile(floats[:,None],(1,q));s[::2,-1]*=-1
            for replacement in [False,True]:
                add(f'float_boundaries_{seed}_{q}_{replacement}',len(s),seed=seed,kind=4,s=s,replace=replacement)
        rng=np.random.RandomState(seed)
        values=rng.uniform(1,9,size=71)*np.power(10.,rng.randint(-300,300,size=71))
        values[::2]*=-1
        add(f'float_exponents_{seed}',len(values),seed=seed,kind=4,s=values,typed=False)
        n=9;s=np.c_[np.resize([2,10,-2,-20],n),np.resize([0,1,0],n)]
        add(f'lexical_integer_rows_{seed}',n,seed=seed,kind=2,s=s,typed=True)
        add(f'lexical_integer_columns_{seed}',n,seed=seed,kind=2,s=s[:,:1])
    for kind in [2,4,6,8,10]:
        s=np.array([[2],[10],[2],[10]])
        if kind==8:s=s.astype(str)
        add(f'one_column_joint_{kind}',4,kind=kind,s=s,typed=True)
    wide=np.zeros((5,1201),int);wide[:,-1]=[0,1,2,0,1]
    add('wide_joint_rows',5,kind=2,s=wide,typed=True)
    many=np.arange(10000)
    add('many_strata',10000,kind=1,s=many,draw=129)
    add('long_bootstrap',10000,draw=20001,seed=7)
    add('short_stratification',7,kind=1,s=[0,1,1],seed=7,typed=True)
    add('short_stratification_subsample',7,kind=1,s=[0,1,1],seed=7,draw=2,replace=False,typed=True)
    add('long_stratification_valid_selection',4,kind=1,s=np.zeros(7,int),draw=1,seed=3,typed=True)
    assert cases and len({c['name'] for c,_ in cases})==len(cases)
    run=subprocess.run([str(exe)],input=str(len(cases))+'\n'+''.join(payload(c) for c,_ in cases),text=True,capture_output=True)
    if run.returncode:
        begins=[line for line in run.stdout.splitlines() if line.startswith('BEGIN')]
        index=int(begins[-1].split()[1])-1 if begins else -1
        raise AssertionError((cases[index][0]['name'],run.stdout[-700:],run.stderr[-1500:]))
    checks=[];inventories=[];fields={};index=-1
    for line in run.stdout.splitlines():
        parts=line.split()
        if parts[0]=='BEGIN':index+=1;fields={};assert int(parts[1])==index+1
        elif parts[0]=='END':
            c,want=cases[index];assert fields.keys()==want.keys(),(c['name'],fields.keys()^want.keys())
            for name,value in fields.items():
                expected_values=np.asarray(want[name]).ravel(order='F')
                np.testing.assert_array_equal(value,expected_values,err_msg=str((c['name'],name)))
                if name not in INTEGER_FIELDS:
                    zeros=expected_values==0
                    np.testing.assert_array_equal(np.signbit(value[zeros]),np.signbit(expected_values[zeros]),err_msg=c['name'])
                checks.append(dict(fixture=index,field=name,passed=True))
            inventories.append(sorted(fields))
        else:
            field=parts[0];assert field not in fields
            fields[field]=np.array(parts[2:],dtype=np.int64 if field in INTEGER_FIELDS else float)
            assert len(fields[field])==int(parts[1])
    assert index+1==len(cases)
    invalid=[]
    def bad(name,n=3,**kw):invalid.append(config(name,n,**kw))
    bad('zero_draw',draw=0);bad('negative_draw',draw=-2);bad('negative_seed',seed=-1)
    bad('oversample_without_replacement',draw=4,replace=False);bad('shuffle_oversample',draw=4,shuffle_=True)
    bad('draw_from_empty',n=0,draw=1);bad('shuffle_zero_draw',draw=0,shuffle_=True)
    for replacement in [False,True]:
        bad(f'empty_stratification_{replacement}',kind=1,s=np.array([],int),replace=replacement)
        bad(f'zero_column_stratification_{replacement}',kind=2,s=np.zeros((3,0),int),replace=replacement)
        bad(f'empty_stratified_result_{replacement}',n=0,kind=1,s=[0,1],replace=replacement)
    for kind in [3,4]:
        for value,name in [(np.nan,'nan'),(np.inf,'inf'),(-np.inf,'negative_inf')]:
            bad(f'nonfinite_stratification_{kind}_{name}',kind=kind,s=[0.,value,1.])
    bad('sparse_stratification',kind=11);bad('scalar_stratification',kind=12);bad('rank3_stratification',kind=13)
    bad('short_stratification_overdraw',n=7,kind=1,s=[0,1,1],replace=False)
    bad('long_stratification_out_of_bounds',n=4,kind=1,s=np.zeros(7,int),draw=1,seed=0)
    rejections=[]
    for c in invalid:
        try:expected(c)
        except (ValueError,TypeError,IndexError) as error:source_error=type(error).__name__
        else:raise AssertionError(('source accepts invalid fixture',c['name']))
        result=subprocess.run([str(exe)],input='1\n'+payload(c),text=True,capture_output=True);output=result.stdout+result.stderr
        assert result.returncode!=0 and 'fscikit-learn:' in output,(c['name'],output)
        assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(c['name'],output)
        rejections.append(dict(name=c['name'],passed=True,source_error=source_error))
    assert before=={str(p):sha(p) for p in paths}
    configurations=[{k:v for k,v in c.items() if k!='s'} for c,_ in cases]
    report=dict(profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,inventories=inventories,
                configurations=configurations,rejections=rejections,typed_fixtures=sum(c['typed'] for c,_ in cases),hashes=before)
    path=args.report or ROOT/'validation/results'/f'array_resampling_{args.profile}.json'
    path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {len(cases)} fixtures / {len(rejections)} diagnostics / {report["typed_fixtures"]} typed-data fixtures: PASS')


if __name__=='__main__':main()
