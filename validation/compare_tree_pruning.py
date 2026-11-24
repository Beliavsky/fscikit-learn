"""Cost-complexity pruning paths and complete trees/forests versus sklearn 1.6.1."""
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
INTEGER_FIELDS={'classes','tree_classes','caller_preserved','metadata','tree_metadata','seeds','leaves','left','right','feature','count'}

def config(name,x,y,kind=2,criterion=0,seed=0,depth=4,leaf=1,nf=0,w=None,alpha=0.,q=None,nt=3,bootstrap=True,ns=0,cw=0):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim==1:x=x[:,None]
    q=x.copy() if q is None else np.asarray(q,float)
    if q.ndim==1:q=q[:,None]
    return dict(name=name,x=x,y=y,q=q,kind=kind,criterion=criterion,seed=seed,depth=depth,leaf=leaf,nf=nf,
                w=None if w is None else np.asarray(w,float),alpha=alpha,nt=nt,bootstrap=bootstrap,ns=ns,cw=cw)

def payload(c):
    x,y,q,w=c['x'],c['y'],c['q'],c['w']
    h=[c['kind'],len(x),x.shape[1],len(q),c['criterion'],c['depth'],c['leaf'],c['nf'],c['seed'],-1 if w is None else len(w),int(np.array(c['alpha'],np.float64).view(np.int64)),c['nt'],int(c['bootstrap']),c['ns'],c['cw']]
    lines=[' '.join(map(str,h))]
    for a in [x,y,q,*([] if w is None else [w])]:lines.append(' '.join(map(str,np.asarray(a,np.float64).ravel(order='F').view(np.int64))))
    return '\n'.join(lines)+'\n'

def model_for(c):
    kind=c['kind'];classifier=kind%4<2
    max_features=c['nf'] or ('sqrt' if classifier and kind!=0 else None)
    params=dict(criterion=(CC if classifier else RC)[c['criterion']],random_state=c['seed'],max_depth=c['depth'],
                min_samples_leaf=c['leaf'],max_features=max_features,ccp_alpha=c['alpha'])
    if kind>=4:
        params.update(n_estimators=c['nt'],bootstrap=c['bootstrap'],max_samples=c['ns'] or None,n_jobs=1)
        if classifier:params['class_weight']=CW[c['cw']]
    return KINDS[kind](**params)

def oracle(c):
    classifier=c['kind']%4<2;model=model_for(c)
    with warnings.catch_warnings(),np.errstate(all='ignore'):
        warnings.simplefilter('ignore');model.fit(c['x'],c['y'],sample_weight=c['w'])
        if c['kind']<4:
            path=model.cost_complexity_pruning_path(c['x'],c['y'],sample_weight=c['w']);models=[model]
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
        for name,extract in dict(values=lambda t:t.value[:,0,:].ravel(),impurity=lambda t:t.impurity,mass=lambda t:t.weighted_n_node_samples,
            threshold=lambda t:np.where(t.children_left<0,0.,t.threshold),left=lambda t:t.children_left+1,right=lambda t:t.children_right+1,
            feature=lambda t:np.where(t.children_left<0,0,t.feature+1),count=lambda t:t.n_node_samples).items():fields[name]=np.concatenate([extract(t) for t in trees])
    return fields

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);ap.add_argument('--case');args=ap.parse_args()
    exe=args.executable.resolve();source=Path(sklearn.__file__).parent
    paths=[exe,Path(__file__)]+[ROOT/'src'/n for n in ['sklearn_tree_pruning.f90','sklearn_tree.f90','sklearn_tree_median.f90','sklearn_ensemble.f90','sklearn_ieee_math.f90']]+[ROOT/'app/reference_tree_pruning_driver.f90']
    paths+=[source/n for n in ['tree/_tree.pyx','tree/_classes.py','tree/_criterion.pyx','tree/_splitter.pyx','ensemble/_forest.py','ensemble/_base.py']]
    hashes={str(p):sha(p) for p in paths};cases=[]
    def add(c):
        if args.case and c['name']!=args.case:return
        cases.append((c,oracle(c)))
    for seed in [0,7,42]:
        r=np.random.RandomState(seed)
        for n,p in [(7,1),(23,3)]:
            x=r.normal(size=(n,p));yc=r.choice([-7,9,42],n);yr=r.gamma(2.,1.,n)+.1;query=np.r_[x,r.normal(size=(3,p))]
            weights=[None,r.uniform(.2,3,n),np.where(np.arange(n)%4==0,0.,1.)]
            for kind in range(8):
                for criterion,iw,alpha in itertools.product(range(3 if kind%4<2 else 4),range(3),[0.,.01,.1,.5,2.]):
                    add(config(f'random_{seed}_{n}_{p}_{kind}_{criterion}_{iw}_{alpha}',x,yc if kind%4<2 else yr,
                        kind=kind,criterion=criterion,seed=seed,w=weights[iw],alpha=alpha,q=query,bootstrap=kind in [4,6],cw=[0,7,42].index(seed)))
    for kind in range(4):
        for criterion in range(3 if kind<2 else 4):
            c=config('boundary',np.arange(8),[-7,-7,9,9,-7,9,-7,9] if kind<2 else [0,0,1,1,4,4,5,5],kind=kind,criterion=criterion,w=[1,1,2,2,1,1,2,2],depth=8)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');path=model_for(c).cost_complexity_pruning_path(c['x'],c['y'],sample_weight=c['w'])
            for j,a in enumerate(path.ccp_alphas[1:]):
                for side,alpha in [('below',np.nextafter(a,-np.inf)),('at',a),('above',np.nextafter(a,np.inf))]:
                    if alpha<0:continue
                    add(c|dict(name=f'boundary_{kind}_{criterion}_{j}_{side}',alpha=float(alpha)))
    for kind in range(8):
        for alpha in [0.,.05,2.]:
            add(config(f'constant_{kind}_{alpha}',np.ones((7,2)),np.full(7,9.) if kind%4<2 else np.arange(1,8.),kind=kind,alpha=alpha,bootstrap=kind in [4,6]))
            add(config(f'zero_weights_{kind}_{alpha}',np.arange(3),[1,2,3],kind=kind,alpha=alpha,w=np.zeros(3),bootstrap=kind in [4,6]))
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
        for label,alpha in [('negative',-1.),('nan',np.nan),('inf',np.inf)]:
            c=config(f'{kind}_{label}',[0,1,2],[1,2,3],kind=kind,alpha=alpha,bootstrap=kind in [4,6])
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
    path=args.report or ROOT/'validation/results'/f'tree_pruning_{args.profile}.json';path.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {len(cases)} fixtures / {len(invalid)} diagnostics: PASS')

if __name__=='__main__':main()
