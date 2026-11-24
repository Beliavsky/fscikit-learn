"""Absolute-error CART/ExtraTree/forests: direct source plus empty-queue safety identities."""
from pathlib import Path
import argparse,hashlib,itertools,json,subprocess,warnings
import numpy as np
import sklearn
import compare_tree_regression_criteria as single
import compare_regression_forests as forest
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
INTEGER_FIELDS=single.INTEGER_FIELDS|forest.INTEGER_FIELDS
single.CRITERIA.append('absolute_error');forest.CRITERIA.append('absolute_error')


def empty_tree(c):
    p=c['x'].shape[1];m=len(c['q'])
    return dict(prediction=np.full(m,np.nan),leaves=np.ones(m,int),importance=np.full(p,np.nan),
                metadata=np.array([1,0,1,p,c['nf'] or p]),values=np.array([np.nan]),impurity=np.array([np.nan]),
                mass=np.array([0.]),threshold=np.array([0.]),left=np.array([0]),right=np.array([0]),
                feature=np.array([0]),count=np.array([0]),criterion_preserved=np.ones(1,int))


def expected(c,is_forest):
    if not is_forest:
        if c['w'] is not None and not np.any(c['w']!=0):return empty_tree(c),'empty_queue_safety'
        return single.oracle(c),'source'
    n=len(c['y']);w=np.ones(n) if c['w'] is None else c['w']
    seeds=np.random.RandomState(c['seed']).randint(2147483647,size=c['nt']);weights=[]
    for seed in seeds:
        counts=np.ones(n)
        if c['bootstrap']:
            counts=np.bincount(np.random.RandomState(seed).randint(n,size=c['ns'] or n),minlength=n)
        weights.append(w*counts)
    if all(np.any(wi!=0) for wi in weights):return forest.oracle(c),'source'
    # Source WeightedMedianCalculator.get_median reads [k] and [k-1] at k=0
    # for empty queues. Use source component trees only where records exist;
    # otherwise prove the explicit native quiet-NaN root identity.
    trees=[]
    for seed,wi in zip(seeds,weights):
        kw={k:c[k] for k in ['x','y','mode','depth','leaf','split','nf','decrease','q']}
        tc=single.config(c['name'],criterion=4,seed=int(seed),w=wi,**kw)
        trees.append(single.oracle(tc) if np.any(wi!=0) else empty_tree(tc))
    with np.errstate(all='ignore'):
        importance=[t['importance'] for t in trees if t['metadata'][0]>1]
        imp=np.mean(importance,axis=0) if importance else np.zeros(c['x'].shape[1])
        if importance:imp=imp/imp.sum()
        d=dict(rng_preserved=np.ones(1,int),prediction=np.sum([t['prediction'] for t in trees],axis=0)/len(trees),
               leaves=np.concatenate([t['leaves'] for t in trees]),importance=imp,seeds=seeds,
               tree_metadata=np.concatenate([t['metadata'] for t in trees]),
               forest_metadata=np.array([c['x'].shape[1],len(trees)]))
        for name in ['values','impurity','mass','threshold','left','right','feature','count']:
            d[name]=np.concatenate([t[name] for t in trees])
        d['criterion_preserved']=np.ones(len(trees),int)
    return d,'source_components_and_empty_queue_safety'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tree-executable',required=True,type=Path)
    ap.add_argument('--forest-executable',required=True,type=Path);ap.add_argument('--profile',default='development')
    ap.add_argument('--report',type=Path);ap.add_argument('--case');args=ap.parse_args()
    tree_exe=args.tree_executable.resolve();forest_exe=args.forest_executable.resolve()
    source_root=Path(sklearn.__file__).parent
    paths=[tree_exe,forest_exe,Path(__file__),Path(single.__file__),Path(forest.__file__)]
    paths+=[ROOT/'src'/s for s in ['sklearn_tree_median.f90','sklearn_tree.f90','sklearn_ensemble.f90','sklearn_ieee_math.f90','sklearn_legacy_rng.f90']]
    paths+=[ROOT/'app'/s for s in ['reference_tree_regression_criteria_driver.f90','reference_regression_forests_driver.f90']]
    paths+=[source_root/s for s in ['tree/_criterion.pyx','tree/_utils.pyx','tree/_classes.py','tree/_splitter.pyx','tree/_tree.pyx','ensemble/_forest.py','ensemble/_base.py']]
    hashes={str(p):sha(p) for p in paths};cases=[]
    def add(c,is_forest):
        if args.case and c['name']!=args.case:return
        want,oracle=expected(c,is_forest);cases.append((c,is_forest,want,oracle))
    for seed in [0,1,7,42,314159]:
        rng=np.random.RandomState(seed)
        for n,p in [(7,1),(19,3),(43,5)]:
            x=rng.normal(size=(n,p));y=rng.normal(size=n)*5
            q=np.r_[x,rng.normal(size=(5,p))]
            weights=[None,np.ones(n),rng.uniform(.1,3,n),np.where(np.arange(n)%4==0,0.,1.)]
            for mode,iw,control in itertools.product([0,1],range(4),range(3)):
                depth,leaf,nf,decrease=[(1,1,0,0.),(3,2,max(1,p//2),.05),(3,1,0,0.)][control]
                kw=dict(mode=mode,criterion=4,seed=seed,depth=depth,leaf=leaf,nf=nf,w=weights[iw],decrease=decrease,q=q)
                add(single.config(f'tree_{seed}_{n}_{p}_{mode}_{iw}_{control}',x,y,**kw),False)
                bootstrap=mode==0 and control!=1
                add(forest.config(f'forest_{seed}_{n}_{p}_{mode}_{iw}_{control}',x,y,**kw,
                    bootstrap=bootstrap,ns=max(1,n//2) if bootstrap and control==2 else 0,nt=1 if control==0 else 4),True)
    for mode in [0,1]:
        for label,w in [('zero',[0.,0.,0.]),('negative',[-1.,-1.,-1.]),('signed',[1.,-1.,2.]),
                        ('cancel',[1.,-1.,0.]),('small',[1e-300]*3),('huge',[1e300]*3)]:
            kw=dict(mode=mode,criterion=4,w=w,seed=7)
            add(single.config(f'tree_weights_{mode}_{label}',[0,1,2],[1,2,3],**kw),False)
            add(forest.config(f'forest_weights_{mode}_{label}',[0,1,2],[1,2,3],bootstrap=mode==0,**kw),True)
        for label,y in [('ties',[0,1,7,100]),('constant',[1]*4),('large',[1e308]*4),('opposite',[-1e308,1e308,-1e308,1e308])]:
            for constant in [False,True]:
                x=np.ones(4) if constant else np.arange(4.)
                add(single.config(f'tree_values_{mode}_{label}_{constant}',x,y,mode=mode,criterion=4),False)
                add(forest.config(f'forest_values_{mode}_{label}_{constant}',x,y,mode=mode,criterion=4,bootstrap=mode==0),True)
        for decrease in [0.,.5,1.,2.,4.,4.1]:
            add(single.config(f'tree_improvement_{mode}_{decrease}',[0,1,2,3],[0,0,2,2],mode=mode,criterion=4,decrease=decrease),False)
            add(forest.config(f'forest_improvement_{mode}_{decrease}',[0,1,2,3],[0,0,2,2],mode=mode,criterion=4,decrease=decrease,bootstrap=False),True)
    assert cases and len({c['name'] for c,_,_,_ in cases})==len(cases)
    checks=[];inventories=[];configurations=[];nonfinite=0;safety=[]
    for is_forest,exe,helper in [(False,tree_exe,single),(True,forest_exe,forest)]:
        selected=[item for item in cases if item[1]==is_forest]
        if not selected:continue
        run=subprocess.run([str(exe)],input=str(len(selected))+'\n'+''.join(helper.payload(c) for c,_,_,_ in selected),text=True,capture_output=True)
        if run.returncode:
            begins=[s for s in run.stdout.splitlines() if s.startswith('BEGIN')];i=int(begins[-1].split()[1])-1 if begins else -1
            raise AssertionError((selected[i][0]['name'],run.stdout[-600:],run.stderr[-1500:]))
        index=-1;fields={}
        for line in run.stdout.splitlines():
            parts=line.split()
            if parts[0]=='BEGIN':index+=1;fields={};assert int(parts[1])==index+1
            elif parts[0]=='END':
                c,_,want,oracle=selected[index];assert fields.keys()==want.keys(),(c['name'],fields.keys()^want.keys())
                fixture=len(inventories)
                for name,value in fields.items():
                    target=np.asarray(want[name]).ravel(order='F')
                    if name in INTEGER_FIELDS:np.testing.assert_array_equal(value,target,err_msg=str((c['name'],name)))
                    else:np.testing.assert_allclose(value,target,rtol=2e-10,atol=2e-12,equal_nan=True,err_msg=str((c['name'],name)))
                    finite=np.isfinite(target)
                    if oracle=='source':nonfinite+=int(np.count_nonzero(~finite))
                    checks.append(dict(fixture=fixture,field=name,oracle=oracle,passed=True,
                                       max_abs_error=float(np.max(np.abs(value[finite]-target[finite]),initial=0))))
                inventories.append(sorted(fields));configurations.append({k:v for k,v in c.items() if k not in ['x','y','q','w']}|dict(forest=is_forest,oracle=oracle))
                if oracle!='source':safety.append(c['name'])
            else:
                name=parts[0];assert name not in fields
                fields[name]=np.array(parts[2:],dtype=int if name in INTEGER_FIELDS else float);assert len(fields[name])==int(parts[1])
        assert index+1==len(selected)
    invalid=[]
    for is_forest,helper in [(False,single),(True,forest)]:
        for mode in [0,1]:
            for label,change in [('nan_query',dict(q=[[np.nan]])),('inf_query',dict(q=[[np.inf]])),
                ('nan_target',dict(y=[1,np.nan,2])),('inf_target',dict(y=[1,np.inf,2])),
                ('bad_weights',dict(w=[1,2])),('nan_weights',dict(w=[1,np.nan,1])),
                ('negative_decrease',dict(decrease=-1))]:
                kw=dict(x=[0,1,2],y=[1,2,3],mode=mode,criterion=4);kw.update(change)
                if is_forest:kw['bootstrap']=mode==0
                c=helper.config(f'{is_forest}_{mode}_{label}',**kw)
                try:(forest.oracle if is_forest else single.oracle)(c)
                except ValueError as error:source_error=type(error).__name__
                else:raise AssertionError(('source accepts invalid',c['name']))
                exe=forest_exe if is_forest else tree_exe
                r=subprocess.run([str(exe)],input='1\n'+helper.payload(c),text=True,capture_output=True);output=r.stdout+r.stderr
                assert r.returncode!=0 and 'fscikit-learn:' in output,(c['name'],output)
                assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(c['name'],output)
                invalid.append(dict(name=c['name'],passed=True,source_error=source_error))
    assert hashes=={str(p):sha(p) for p in paths}
    report=dict(profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,inventories=inventories,
                configurations=configurations,rejections=invalid,empty_queue_safety_fixtures=safety,
                source_nonfinite_values=nonfinite,hashes=hashes)
    path=args.report or ROOT/'validation/results'/f'tree_absolute_error_{args.profile}.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {len(cases)} fixtures / {len(invalid)} diagnostics / {len(safety)} empty-queue safety fixtures: PASS')


if __name__=='__main__':main()
