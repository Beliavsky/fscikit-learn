"""Poisson and Friedman regression-tree structures versus installed sklearn."""
from pathlib import Path
import argparse,hashlib,itertools,json,subprocess,warnings
import numpy as np
import sklearn
import sklearn.tree._classes as source
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
CRITERIA=['squared_error','friedman_mse','poisson','unknown']
INTEGER_FIELDS={'leaves','metadata','left','right','feature','count','criterion_preserved'}


def payload(c):
    x,y,q=c['x'],c['y'],c['q'];w=c['w']
    header=[c['mode'],len(x),x.shape[1],len(q),c['criterion'],c['depth'],c['leaf'],c['split'],c['nf'],c['seed'],
            -1 if w is None else len(w),int(np.array(c['decrease'],float).view(np.int64))]
    lines=[' '.join(map(str,header))]
    for a in [x,y,q,*([] if w is None else [w])]:
        lines+=[' '.join(map(str,np.asarray(a,np.float64).ravel(order='F').view(np.int64)))]
    return '\n'.join(lines)+'\n'


def oracle(c):
    kind=DecisionTreeRegressor if c['mode']==0 else ExtraTreeRegressor
    model=kind(criterion=CRITERIA[c['criterion']],max_depth=c['depth'],min_samples_leaf=c['leaf'],
               min_samples_split=c['split'],max_features=c['nf'] or None,random_state=c['seed'],
               min_impurity_decrease=c['decrease'])
    with warnings.catch_warnings(),np.errstate(all='ignore'):
        warnings.simplefilter('ignore');model.fit(c['x'],c['y'],sample_weight=c['w']);t=model.tree_
        return dict(prediction=model.predict(c['q']),leaves=model.apply(c['q'])+1,importance=model.feature_importances_,
                    metadata=np.array([t.node_count,t.max_depth,model.get_n_leaves(),model.n_features_in_,model.max_features_]),
                    values=t.value[:,0,0],impurity=t.impurity,mass=t.weighted_n_node_samples,
                    threshold=np.where(t.children_left<0,0.,t.threshold),left=t.children_left+1,right=t.children_right+1,
                    feature=np.where(t.children_left<0,0,t.feature+1),count=t.n_node_samples,criterion_preserved=np.ones(1,int))


def config(name,x,y,mode=0,criterion=2,seed=0,depth=3,leaf=1,split=2,nf=0,w=None,decrease=0.,q=None):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim==1:x=x[:,None]
    q=x.copy() if q is None else np.asarray(q,float)
    if q.ndim==1:q=q[:,None]
    return dict(name=name,x=x,y=y,q=q,mode=mode,criterion=criterion,seed=seed,depth=depth,leaf=leaf,split=split,nf=nf,
                w=None if w is None else np.asarray(w,float),decrease=decrease)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path)
    ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);ap.add_argument('--case')
    args=ap.parse_args();exe=args.executable.resolve()
    source_dir=Path(source.__file__).parent
    paths=[exe,Path(__file__),Path(source.__file__),source_dir/'_criterion.pyx',source_dir/'_splitter.pyx',
           source_dir/'_tree.pyx',source_dir/'_utils.pyx',ROOT/'src/sklearn_tree.f90',ROOT/'src/sklearn_ieee_math.f90',
           ROOT/'src/sklearn_legacy_rng.f90',ROOT/'app/reference_tree_regression_criteria_driver.f90']
    hashes={str(p):sha(p) for p in paths};cases=[]
    def add(name,x,y,**kw):
        if args.case is not None and name!=args.case:return
        c=config(name,x,y,**kw);cases.append((c,oracle(c)))
    for seed in [0,1,7,42,314159]:
        rng=np.random.RandomState(seed)
        for n,p in [(7,1),(17,3),(41,5)]:
            x=rng.normal(size=(n,p));y=rng.poisson(3,size=n).astype(float)+.125
            query=np.r_[x,rng.normal(size=(5,p)),np.full((1,p),np.nan)]
            weights=[None,np.ones(n),rng.uniform(.1,3,n),np.where(np.arange(n)%4==0,0.,1.)]
            for criterion,mode,iw,depth in itertools.product([1,2],[0,1],range(4),[1,3]):
                for leaf,nf,decrease in [(1,0,0.),(2,max(1,p//2),.05)]:
                    add(f'random_{seed}_{n}_{p}_{criterion}_{mode}_{iw}_{depth}_{leaf}',x,y,criterion=criterion,
                        mode=mode,seed=seed,depth=depth,leaf=leaf,nf=nf,w=weights[iw],decrease=decrease,q=query)
            add(f'squared_default_{seed}_{n}_{p}',x,y,criterion=0,seed=seed,w=weights[2],q=query)
        for n in [2,3,5]:
            x=np.arange(n,dtype=float)
            targets=[np.zeros(n),np.ones(n),np.arange(n,dtype=float),np.resize([0.,1.,0.],n)]
            for j,y in enumerate(targets):
                for mode,criterion in itertools.product([0,1],[1,2]):
                    if criterion==2 and y.sum()==0:continue
                    add(f'counts_{seed}_{n}_{j}_{mode}_{criterion}',x,y,criterion=criterion,mode=mode,seed=seed)
    for criterion,mode in itertools.product([1,2],[0,1]):
        for wname,w in [('zero',[0.,0.,0.]),('negative',[-1.,-1.,-1.]),('signed',[1.,-1.,2.]),
                        ('cancel',[1.,-1.,0.]),('small',[1.e-300]*3),('huge',[1.e300]*3)]:
            add(f'weights_{criterion}_{mode}_{wname}',[0,1,2],[1,2,3],criterion=criterion,mode=mode,w=w)
        for yname,y in [('tiny',[1.e-16]*3),('large',[1.e154,2.e154,3.e154]),('overflow',[1.e308]*3)]:
            add(f'scale_{criterion}_{mode}_{yname}',[0,1,2],y,criterion=criterion,mode=mode)
    for criterion in [0,1,2]:
        for decrease in [0.,.5,1.,2.,4.,4.1]:
            add(f'improvement_{criterion}_{decrease}',[0,1,2,3],[0,0,2,2],criterion=criterion,decrease=decrease)
    for mode,criterion in itertools.product([0,1],[1,2]):
        x=np.c_[np.zeros(9),np.resize([0.,0.,1.],9),np.arange(9,dtype=float)]
        add(f'constant_features_{mode}_{criterion}',x,np.resize([0.,1.,4.],9),mode=mode,criterion=criterion,leaf=2)
        add(f'constant_x_{mode}_{criterion}',np.ones((7,3)),np.arange(1,8),mode=mode,criterion=criterion)
    assert cases and len({c['name'] for c,_ in cases})==len(cases)
    run=subprocess.run([str(exe)],input=str(len(cases))+'\n'+''.join(payload(c) for c,_ in cases),text=True,capture_output=True)
    if run.returncode:
        begins=[line for line in run.stdout.splitlines() if line.startswith('BEGIN')];i=int(begins[-1].split()[1])-1 if begins else -1
        raise AssertionError((cases[i][0]['name'],run.stdout[-600:],run.stderr[-1500:]))
    fields={};index=-1;checks=[];inventories=[];nonfinite=0
    for line in run.stdout.splitlines():
        parts=line.split()
        if parts[0]=='BEGIN':index+=1;fields={};assert int(parts[1])==index+1
        elif parts[0]=='END':
            c,want=cases[index];assert fields.keys()==want.keys(),(c['name'],fields.keys()^want.keys())
            for name,value in fields.items():
                expected=np.asarray(want[name]).ravel(order='F')
                if name in INTEGER_FIELDS:np.testing.assert_array_equal(value,expected,err_msg=str((c['name'],name)))
                else:np.testing.assert_allclose(value,expected,rtol=2.e-10,atol=2.e-12,equal_nan=True,err_msg=str((c['name'],name)))
                finite=np.isfinite(expected);nonfinite+=int(np.count_nonzero(~finite))
                checks.append(dict(fixture=index,field=name,passed=True,
                                   max_abs_error=float(np.max(np.abs(value[finite]-expected[finite]),initial=0))))
            inventories.append(sorted(fields))
        else:
            name=parts[0];assert name not in fields
            fields[name]=np.array(parts[2:],dtype=int if name in INTEGER_FIELDS else float)
            assert len(fields[name])==int(parts[1])
    assert index+1==len(cases)
    invalid=[]
    def bad(name,x=[0,1,2],y=[1,2,3],**kw):invalid.append(config(name,x,y,**kw))
    for mode in [0,1]:
        bad(f'negative_target_{mode}',y=[0,-1,2],mode=mode)
        bad(f'zero_target_sum_{mode}',y=[0,0,0],mode=mode)
        bad(f'unknown_criterion_{mode}',criterion=3,mode=mode)
        bad(f'nan_target_{mode}',y=[1,np.nan,2],mode=mode)
        bad(f'inf_target_{mode}',y=[1,np.inf,2],mode=mode)
        bad(f'weight_length_{mode}',w=[1,1],mode=mode)
        bad(f'nan_weights_{mode}',w=[1,np.nan,1],mode=mode)
        bad(f'negative_impurity_threshold_{mode}',decrease=-1,mode=mode)
    rejections=[]
    for c in invalid:
        try:oracle(c)
        except ValueError as error:source_error=type(error).__name__
        else:raise AssertionError(('source accepts invalid',c['name']))
        result=subprocess.run([str(exe)],input='1\n'+payload(c),text=True,capture_output=True);out=result.stdout+result.stderr
        assert result.returncode!=0 and 'fscikit-learn:' in out,(c['name'],out)
        assert not any(s in out for s in ['SIGFPE','Fortran runtime error']),(c['name'],out)
        rejections.append(dict(name=c['name'],passed=True,source_error=source_error))
    assert hashes=={str(p):sha(p) for p in paths}
    configurations=[{k:v for k,v in c.items() if k not in ['x','y','q','w']} for c,_ in cases]
    report=dict(profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,
                inventories=inventories,configurations=configurations,rejections=rejections,
                source_nonfinite_values=nonfinite,hashes=hashes)
    path=args.report or ROOT/'validation/results'/f'tree_regression_criteria_{args.profile}.json'
    path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {len(cases)} fixtures / {len(rejections)} diagnostics / {nonfinite} nonfinite source values: PASS')


if __name__=='__main__':main()
