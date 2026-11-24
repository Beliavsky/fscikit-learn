"""Weighted regression forests and every component tree versus installed sklearn."""
from pathlib import Path
import argparse,hashlib,itertools,json,subprocess,warnings
import numpy as np
import sklearn
import sklearn.ensemble._forest as source
import sklearn.tree._classes as tree_source
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
CRITERIA=['squared_error','friedman_mse','poisson','unknown']
INTEGER_FIELDS={'leaves','metadata','left','right','feature','count','criterion_preserved','rng_preserved','seeds','tree_metadata','forest_metadata'}


def payload(c):
    x,y,q=c['x'],c['y'],c['q'];w=c['w']
    header=[c['mode'],len(x),x.shape[1],len(q),c['criterion'],c['depth'],c['leaf'],c['split'],c['nf'],c['seed'],
            -1 if w is None else len(w),int(np.array(c['decrease'],float).view(np.int64)),c['nt'],int(c['bootstrap']),c['ns']]
    lines=[' '.join(map(str,header))]
    for a in [x,y,q,*([] if w is None else [w])]:
        lines+=[' '.join(map(str,np.asarray(a,np.float64).ravel(order='F').view(np.int64)))]
    return '\n'.join(lines)+'\n'


def oracle(c):
    kind=RandomForestRegressor if c['mode']==0 else ExtraTreesRegressor
    model=kind(criterion=CRITERIA[c['criterion']],n_estimators=c['nt'],bootstrap=c['bootstrap'],
               max_samples=c['ns'] or None,max_depth=c['depth'],min_samples_leaf=c['leaf'],
               min_samples_split=c['split'],max_features=c['nf'] or 1.,random_state=c['seed'],
               min_impurity_decrease=c['decrease'],n_jobs=1)
    with warnings.catch_warnings(),np.errstate(all='ignore'):
        warnings.simplefilter('ignore');model.fit(c['x'],c['y'],sample_weight=c['w'])
        trees=[m.tree_ for m in model.estimators_]
        fields=dict(rng_preserved=np.ones(1,int),prediction=model.predict(c['q']))
        if len(c['y'])>1 and np.isfinite(model.predict(c['x'])).all() and (c['w'] is None or c['w'].sum()!=0):
            fields['score']=np.array([model.score(c['x'],c['y'],sample_weight=c['w'])])
        fields.update(leaves=(model.apply(c['q'])+1).ravel(order='F'),importance=model.feature_importances_,
                      seeds=np.array([m.random_state for m in model.estimators_]),
                      tree_metadata=np.array([[t.node_count,t.max_depth,m.get_n_leaves(),m.n_features_in_,m.max_features_]
                                              for m,t in zip(model.estimators_,trees)]).ravel(),
                      forest_metadata=np.array([model.n_features_in_,len(model.estimators_)]))
        for name,extract in dict(values=lambda t:t.value[:,0,0],impurity=lambda t:t.impurity,
             mass=lambda t:t.weighted_n_node_samples,threshold=lambda t:np.where(t.children_left<0,0.,t.threshold),
             left=lambda t:t.children_left+1,right=lambda t:t.children_right+1,
             feature=lambda t:np.where(t.children_left<0,0,t.feature+1),count=lambda t:t.n_node_samples).items():
            fields[name]=np.concatenate([extract(t) for t in trees])
        fields['criterion_preserved']=np.ones(c['nt'],int)
        return fields


def config(name,x,y,mode=0,criterion=2,seed=0,depth=3,leaf=1,split=2,nf=0,w=None,decrease=0.,q=None,nt=3,bootstrap=True,ns=0):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim==1:x=x[:,None]
    q=x.copy() if q is None else np.asarray(q,float)
    if q.ndim==1:q=q[:,None]
    return dict(name=name,x=x,y=y,q=q,mode=mode,criterion=criterion,seed=seed,depth=depth,leaf=leaf,split=split,nf=nf,
                w=None if w is None else np.asarray(w,float),decrease=decrease,nt=nt,bootstrap=bootstrap,ns=ns)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path)
    ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);ap.add_argument('--case')
    args=ap.parse_args();exe=args.executable.resolve()
    source_dir=Path(tree_source.__file__).parent
    paths=[exe,Path(__file__),Path(source.__file__),Path(source.__file__).with_name('_base.py'),Path(tree_source.__file__),source_dir/'_criterion.pyx',source_dir/'_splitter.pyx',
           source_dir/'_tree.pyx',source_dir/'_utils.pyx',ROOT/'src/sklearn_ensemble.f90',ROOT/'src/sklearn_tree.f90',ROOT/'src/sklearn_ieee_math.f90',
           ROOT/'src/sklearn_legacy_rng.f90',ROOT/'app/reference_regression_forests_driver.f90']
    hashes={str(p):sha(p) for p in paths};cases=[]
    def add(name,x,y,**kw):
        if args.case is not None and name!=args.case:return
        c=config(name,x,y,**kw);cases.append((c,oracle(c)))
    for seed in [0,1,7,42,314159]:
        rng=np.random.RandomState(seed)
        for n,p in [(7,1),(19,3),(43,5)]:
            x=rng.normal(size=(n,p));y=rng.poisson(3,size=n).astype(float)+.125
            query=np.r_[x,rng.normal(size=(5,p)),np.full((1,p),np.nan)]
            weights=[None,np.ones(n),rng.uniform(.1,3,n),np.where(np.arange(n)%4==0,0.,1.)]
            for criterion,mode,iw,control in itertools.product([0,1,2],[0,1],range(4),range(3)):
                bootstrap=mode==0 and control!=1
                depth,leaf,nf,decrease,ns=[(1,1,0,0.,0),(3,2,max(1,p//2),.05,0),(3,1,0,0.,max(1,n//2) if bootstrap else 0)][control]
                add(f'random_{seed}_{n}_{p}_{criterion}_{mode}_{iw}_{control}',x,y,criterion=criterion,
                    mode=mode,seed=seed,depth=depth,leaf=leaf,nf=nf,w=weights[iw],decrease=decrease,q=query,
                    bootstrap=bootstrap,ns=ns,nt=1 if control==0 else 4)
    for criterion,mode in itertools.product([0,1,2],[0,1]):
        for wname,w in [('zero',[0.,0.,0.]),('negative',[-1.,-1.,-1.]),('signed',[1.,-1.,2.]),
                        ('cancel',[1.,-1.,0.]),('small',[1.e-300]*3),('huge',[1.e300]*3)]:
            add(f'weights_{criterion}_{mode}_{wname}',[0,1,2],[1,2,3],criterion=criterion,mode=mode,w=w,bootstrap=mode==0)
        for yname,y in [('tiny',[1.e-16]*3),('large',[1.e154,2.e154,3.e154]),('overflow',[1.e308]*3)]:
            add(f'scale_{criterion}_{mode}_{yname}',[0,1,2],y,criterion=criterion,mode=mode,bootstrap=mode==0)
        for decrease in [0.,.5,1.,2.,4.,4.1]:
            add(f'improvement_{criterion}_{mode}_{decrease}',[0,1,2,3],[0,0,2,2],criterion=criterion,
                mode=mode,decrease=decrease,bootstrap=False)
        add(f'singleton_{criterion}_{mode}',[0],[1],criterion=criterion,mode=mode,bootstrap=mode==0)
        add(f'constant_x_{criterion}_{mode}',np.ones((7,3)),np.arange(1,8),mode=mode,criterion=criterion,bootstrap=mode==0)
        add(f'constant_y_{criterion}_{mode}',np.arange(7),np.ones(7),mode=mode,criterion=criterion,bootstrap=mode==0)
    for criterion,mode in itertools.product([0,1,2],[0,1]):
        for label,y,w in [('zero_counts',[0,0,0],[1,1,1]),('sparse_counts',[1,0,0],[0,1,1]),
                           ('zero_weights',[0,1,0],[1,0,1])]:
            if criterion==2 and sum(y)==0:continue
            add(f'{label}_{criterion}_{mode}',[0,1,2],y,w=w,criterion=criterion,mode=mode,bootstrap=mode==0)
        add(f'max_seed_{criterion}_{mode}',np.arange(9),np.arange(1,10),criterion=criterion,mode=mode,
            bootstrap=mode==0,seed=2147483647)
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
    def bad(name,x=[0,1,2],y=[1,2,3],**kw):
        kw.setdefault('bootstrap',kw.get('mode',0)==0);invalid.append(config(name,x,y,**kw))
    for mode in [0,1]:
        bad(f'negative_target_{mode}',y=[0,-1,2],mode=mode)
        bad(f'zero_target_sum_{mode}',y=[0,0,0],mode=mode)
        bad(f'negative_zero_weight_target_{mode}',y=[-1,1,2],w=[0,1,1],mode=mode)
        bad(f'unknown_criterion_{mode}',criterion=3,mode=mode)
        bad(f'nan_target_{mode}',y=[1,np.nan,2],mode=mode)
        bad(f'inf_target_{mode}',y=[1,np.inf,2],mode=mode)
        bad(f'weight_length_{mode}',w=[1,1],mode=mode)
        bad(f'nan_weights_{mode}',w=[1,np.nan,1],mode=mode)
        bad(f'negative_impurity_threshold_{mode}',decrease=-1,mode=mode)
        bad(f'zero_estimators_{mode}',nt=0,mode=mode)
        bad(f'negative_seed_{mode}',seed=-1,mode=mode)
        bad(f'large_max_samples_{mode}',ns=4,mode=mode)
        bad(f'negative_max_samples_{mode}',ns=-1,mode=mode)
        bad(f'max_samples_without_bootstrap_{mode}',ns=1,bootstrap=False,mode=mode)
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
    path=args.report or ROOT/'validation/results'/f'regression_forests_{args.profile}.json'
    path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {len(cases)} fixtures / {len(rejections)} diagnostics / {nonfinite} nonfinite source values: PASS')


if __name__=='__main__':main()
