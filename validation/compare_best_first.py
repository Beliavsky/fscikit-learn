"""Best-first tree growth, frontier order and complete fitted structures versus sklearn 1.6.1."""
from pathlib import Path
import argparse,hashlib,itertools,json,subprocess,warnings
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier,DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,RandomForestRegressor,ExtraTreesRegressor
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
KINDS=[DecisionTreeClassifier,ExtraTreeClassifier,DecisionTreeRegressor,ExtraTreeRegressor,RandomForestClassifier,ExtraTreesClassifier,RandomForestRegressor,ExtraTreesRegressor]
CC=['gini','entropy','log_loss'];RC=['squared_error','friedman_mse','poisson','absolute_error'];CW=[None,'balanced','balanced_subsample']
INTEGER_FIELDS={'classes','tree_classes','caller_preserved','metadata','tree_metadata','seeds','leaves','left','right','feature','count','indicator_shape','indicator_indptr','indicator_indices','node_offsets'}

def config(name,x,y,kind=2,criterion=0,seed=0,depth=4,leaf=1,nf=0,w=None,alpha=0.,q=None,nt=3,bootstrap=True,ns=0,cw=0,fraction=0.,max_leaves=0,check_path=True):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim==1:x=x[:,None]
    q=x.copy() if q is None else np.asarray(q,float)
    if q.ndim==1:q=q[:,None]
    return dict(name=name,x=x,y=y,q=q,kind=kind,criterion=criterion,seed=seed,depth=depth,leaf=leaf,nf=nf,
                w=None if w is None else np.asarray(w,float),alpha=alpha,nt=nt,bootstrap=bootstrap,ns=ns,cw=cw,fraction=fraction,max_leaves=max_leaves,check_path=check_path)

def payload(c):
    x,y,q,w=c['x'],c['y'],c['q'],c['w']
    h=[c['kind'],len(x),x.shape[1],len(q),c['criterion'],c['depth'],c['leaf'],c['nf'],c['seed'],-1 if w is None else len(w),int(np.array(c['alpha'],np.float64).view(np.int64)),c['nt'],int(c['bootstrap']),c['ns'],c['cw'],int(np.array(c['fraction'],np.float64).view(np.int64)),c['max_leaves'],int(c['check_path'])]
    lines=[' '.join(map(str,h))]
    for a in [x,y,q,*([] if w is None else [w])]:lines.append(' '.join(map(str,np.asarray(a,np.float64).ravel(order='F').view(np.int64))))
    return '\n'.join(lines)+'\n'

def model_for(c):
    kind=c['kind'];classifier=kind%4<2
    max_features=c['nf'] or ('sqrt' if classifier and kind!=0 else None)
    params=dict(criterion=(CC if classifier else RC)[c['criterion']],random_state=c['seed'],max_depth=c['depth'],
                min_samples_leaf=c['leaf'],min_weight_fraction_leaf=c['fraction'],max_leaf_nodes=c['max_leaves'] or None,max_features=max_features,ccp_alpha=c['alpha'])
    if kind>=4:
        params.update(n_estimators=c['nt'],bootstrap=c['bootstrap'],max_samples=c['ns'] or None,n_jobs=1)
        if classifier:params['class_weight']=CW[c['cw']]
    return KINDS[kind](**params)

def oracle(c):
    classifier=c['kind']%4<2;model=model_for(c)
    with warnings.catch_warnings(),np.errstate(all='ignore'):
        warnings.simplefilter('ignore');model.fit(c['x'],c['y'],sample_weight=c['w'])
        if c['kind']<4:
            path=model.cost_complexity_pruning_path(c['x'],c['y'],sample_weight=c['w']) if c['check_path'] else dict(ccp_alphas=np.array([]),impurities=np.array([]));models=[model]
        else:path=dict(ccp_alphas=np.array([]),impurities=np.array([]));models=model.estimators_
        trees=[m.tree_ for m in models]
        fields=dict(prediction=model.predict(c['q']),proba=model.predict_proba(c['q']).ravel(order='F') if classifier else np.array([]),
                    classes=model.classes_ if classifier else np.array([],int),
                    tree_classes=np.concatenate([m.classes_.astype(int) for m in models]) if classifier else np.array([],int),
                    path_alphas=path['ccp_alphas'],path_impurities=path['impurities'],caller_preserved=np.ones(1,int),
                    metadata=np.array([model.n_features_in_,len(model.classes_) if classifier else 0,len(models)]),
                    tree_metadata=np.array([[t.node_count,t.max_depth,m.get_n_leaves(),m.n_features_in_,m.max_features_] for m,t in zip(models,trees)]).ravel(),
                    seeds=np.array([m.random_state for m in models]),
                    leaves=np.column_stack([m.apply(c['q'])+1 for m in models]).ravel(order='F'),importance=model.feature_importances_)
        if c['kind']<4:
            indicator=model.decision_path(c['q']);offsets=np.array([0,indicator.shape[1]])
        else:indicator,offsets=model.decision_path(c['q'])
        fields.update(indicator_shape=np.array(indicator.shape),indicator_indptr=indicator.indptr+1,
                      indicator_indices=indicator.indices+1,indicator_data=indicator.data.astype(float),node_offsets=offsets)
        for name,extract in dict(values=lambda t:t.value[:,0,:].ravel(),impurity=lambda t:t.impurity,mass=lambda t:t.weighted_n_node_samples,
            threshold=lambda t:np.where(t.children_left<0,0.,t.threshold),left=lambda t:t.children_left+1,right=lambda t:t.children_right+1,
            feature=lambda t:np.where(t.children_left<0,0,t.feature+1),count=lambda t:t.n_node_samples).items():fields[name]=np.concatenate([extract(t) for t in trees])
    return fields

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);ap.add_argument('--case');args=ap.parse_args()
    exe=args.executable.resolve();source=Path(sklearn.__file__).parent
    paths=[exe,Path(__file__)]+[ROOT/'src'/n for n in ['sklearn_tree_pruning.f90','sklearn_tree.f90','sklearn_tree_median.f90','sklearn_ensemble.f90','sklearn_ieee_math.f90']]+[ROOT/'app/reference_best_first_driver.f90',ROOT/'src/sklearn_sparse.f90']
    paths+=[source/n for n in ['tree/_tree.pyx','tree/_classes.py','tree/_criterion.pyx','tree/_splitter.pyx','ensemble/_forest.py','ensemble/_base.py']]
    hashes={str(p):sha(p) for p in paths};cases=[]
    def add(c):
        if args.case and c['name']!=args.case:return
        cases.append((c,oracle(c)))
    for seed in [0,7,42]:
        r=np.random.RandomState(seed)
        for n,p in [(9,1),(29,3)]:
            x=r.normal(size=(n,p));yc=r.choice([-7,9,42],n);yr=r.gamma(2.,1.,n)+.1
            weights=[None,r.uniform(.2,3,n),np.where(np.arange(n)%4==0,0.,1.)]
            query=np.r_[x,r.normal(size=(3,p))]
            for kind in range(8):
                for criterion,iw,max_leaves,control in itertools.product(range(3 if kind%4<2 else 4),range(3),[2,3,4,8],range(2)):
                    q=query if kind%4>=2 and criterion==3 else np.r_[query,np.full((1,p),np.nan)]
                    add(config(f'random_{seed}_{n}_{p}_{kind}_{criterion}_{iw}_{max_leaves}_{control}',x,yc if kind%4<2 else yr,
                        kind=kind,criterion=criterion,seed=seed,w=weights[iw],alpha=.05 if control else 0.,q=q,
                        bootstrap=kind in [4,6],cw=[0,7,42].index(seed),fraction=.1 if control else 0.,max_leaves=max_leaves,
                        nf=max(1,p//2) if control else 0,depth=3 if control else 8))
    for kind in range(8):
        for criterion in range(3 if kind%4<2 else 4):
            for max_leaves in [2,3,4,5,8,16]:
                y=np.resize([-7,9],16) if kind%4<2 else np.resize([0.,0.,1.,1.],16)
                if kind%4>=2 and criterion==2:y=y+.1
                add(config(f'equal_gain_{kind}_{criterion}_{max_leaves}',np.arange(16),y,kind=kind,criterion=criterion,
                    max_leaves=max_leaves,bootstrap=False,depth=10))
            for label,w in [('negative',[-1.,-1.,-1.]),('zero',[0.,0.,0.]),('small',[1e-300]*3),('large',[1e300]*3)]:
                if label=='zero' and kind%4>=2 and criterion==3:continue
                add(config(f'weights_{kind}_{criterion}_{label}',[0,1,2],[1,2,3],kind=kind,criterion=criterion,w=w,
                    max_leaves=3,bootstrap=kind in [4,6]))
    for kind in range(8):
        add(config(f'constant_features_{kind}',np.c_[np.ones(13),np.resize([0,1,1],13),np.arange(13)],
                   np.resize([-7,9,42],13) if kind%4<2 else np.arange(1,14),kind=kind,max_leaves=5,bootstrap=kind in [4,6]))
    cube=np.array(list(itertools.product([0.,1.],repeat=3)))
    parity=(cube.sum(axis=1)%2).astype(int)
    for kind in range(8):
        for criterion in range(3 if kind%4<2 else 4):
            for max_leaves in [2,3,4,5,6,7,8,12]:
                y=np.where(parity,9,-7) if kind%4<2 else parity+.1
                add(config(f'zero_gain_{kind}_{criterion}_{max_leaves}',cube,y,kind=kind,criterion=criterion,
                    max_leaves=max_leaves,bootstrap=False,depth=8))
    for kind in [2,3,6,7]:
        for criterion in range(4):
            for label,y in [('large',[1e154,2e154,3e154,4e154]),('overflow',[1e308]*4),('opposite',[-1e308,1e308,-1e308,1e308])]:
                if criterion==2 and label=='opposite':continue
                add(config(f'nonfinite_gain_{kind}_{criterion}_{label}',np.arange(4),y,kind=kind,criterion=criterion,
                    max_leaves=3,bootstrap=kind in [4,6],check_path=False))
    assert cases and len({c['name'] for c,_ in cases})==len(cases)
    run=subprocess.run([str(exe)],input=str(len(cases))+'\n'+''.join(payload(c) for c,_ in cases),text=True,capture_output=True)
    if run.returncode:
        begins=[s for s in run.stdout.splitlines() if s.startswith('BEGIN')];i=int(begins[-1].split()[1])-1 if begins else -1
        raise AssertionError((cases[i][0]['name'],run.stdout[-600:],run.stderr[-1600:]))
    checks=[];inventories=[];index=-1;fields={};nonfinite=0
    for line in run.stdout.splitlines():
        s=line.split()
        if s[0]=='BEGIN':index+=1;fields={};assert int(s[1])==index+1
        elif s[0]=='END':
            c,want=cases[index];assert fields.keys()==want.keys(),(c['name'],fields.keys()^want.keys())
            for name,value in fields.items():
                target=np.asarray(want[name]).ravel()
                if name in INTEGER_FIELDS or (name=='prediction' and c['kind']%4<2):np.testing.assert_array_equal(value,target,err_msg=str((c['name'],name)))
                else:np.testing.assert_allclose(value,target,rtol=2e-10,atol=2e-12,equal_nan=True,err_msg=str((c['name'],name)))
                finite=np.isfinite(target);nonfinite+=int(np.count_nonzero(~finite))
                checks.append(dict(fixture=index,field=name,passed=True,max_abs_error=float(np.max(np.abs(value[finite]-target[finite]),initial=0))))
            inventories.append(sorted(fields))
        else:
            name=s[0];assert name not in fields
            fields[name]=np.array(s[2:],dtype=int if name in INTEGER_FIELDS else float);assert len(fields[name])==int(s[1])
    assert index+1==len(cases)
    invalid=[]
    for kind in range(8):
        for label,budget in [('one',1),('negative',-1)]:
            c=config(f'{kind}_{label}',[0,1,2],[1,2,3],kind=kind,max_leaves=budget,bootstrap=kind in [4,6])
            try:oracle(c)
            except ValueError as error:source_error=type(error).__name__
            else:raise AssertionError(('source accepts invalid',c['name']))
            r=subprocess.run([str(exe)],input='1\n'+payload(c),text=True,capture_output=True);out=r.stdout+r.stderr
            assert r.returncode!=0 and 'fscikit-learn:' in out,(c['name'],out)
            assert not any(s in out for s in ['SIGFPE','Fortran runtime error']),(c['name'],out)
            invalid.append(dict(name=c['name'],passed=True,source_error=source_error))
    assert hashes=={str(p):sha(p) for p in paths}
    d=dict(profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,inventories=inventories,
           configurations=[{k:v for k,v in c.items() if k not in ['x','y','q','w']} for c,_ in cases],rejections=invalid,source_nonfinite_values=nonfinite,hashes=hashes)
    path=args.report or ROOT/'validation/results'/f'best_first_{args.profile}.json';path.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {len(cases)} fixtures / {len(invalid)} diagnostics: PASS')

if __name__=='__main__':main()
