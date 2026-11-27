"""AdaBoost/tree parameter validation against actual sklearn fitting."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import sklearn
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--executable',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
    controls={
        'estimator__min_samples_leaf':[0,-1,1,99,0.,-.1,.1,1.,np.nan,np.inf],
        'estimator__min_samples_split':[0,1,2,99,0.,-.1,.1,1.,1.01,np.nan,np.inf],
        'estimator__max_features':[-1,0,1,9,.01,1.,0.,1.01,'sqrt','log2','none','invalid',np.nan,np.inf],
        'estimator__max_leaf_nodes':[-1,0,1,2,9],
        'estimator__ccp_alpha':[-.1,0.,.1,np.nan,np.inf],
        'estimator__min_weight_fraction_leaf':[-.1,0.,.1,.5,.51,np.nan,np.inf],
        'estimator__min_impurity_decrease':[-.1,0.,.1,np.nan,np.inf],
        'estimator__max_depth':[-1,0,1,3],
        'estimator__criterion':['invalid','poisson'],
        'estimator__splitter':['invalid','random'],
        'estimator__random_state':[-1,0,91],
        'n_estimators':[0,1], 'learning_rate':[0.,.1,np.nan,np.inf],
        'estimator__n_estimators':[2], 'sample_weight':[np.nan,np.inf,-1.,0.]}
    for kind in ['regression','classification']:
        x=np.column_stack([np.arange(1.,21),np.arange(1,21)%3,np.sin(np.arange(1.,21))])
        y=x[:,0]**2 if kind=='regression' else np.where(np.arange(1,21)<=10,-9,8)
        for name,values in controls.items():
            for value in values:
                params={} if name=='sample_weight' else {name:value}
                weights=None
                if name=='sample_weight':
                    weights=np.ones(len(y));weights[0]=value
                    if value==0:weights[:]=0
                # Documented Fortran zero sentinels represent Python None.
                if name in ['estimator__max_features','estimator__max_leaf_nodes'] and isinstance(value,int) and value==0:
                    params[name]=None
                if name=='estimator__max_features' and value=='none':params[name]=None
                tree=(DecisionTreeRegressor(max_depth=3) if kind=='regression' else DecisionTreeClassifier(max_depth=1))
                model=(AdaBoostRegressor if kind=='regression' else AdaBoostClassifier)(estimator=tree,n_estimators=3,random_state=0)
                error=None
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore');model.set_params(**params);model.fit(x,y,sample_weight=weights)
                    accepted=True
                except (ValueError,TypeError,ZeroDivisionError) as e:
                    accepted=False;error=dict(type=type(e).__name__,message=str(e))
                p=subprocess.run([str(a.executable),kind,name,str(value)],text=True,capture_output=True,timeout=30)
                output=p.stdout+p.stderr
                passed=(p.returncode==0 and 'FIT ACCEPTED' in output) if accepted else (
                    p.returncode!=0 and 'fscikit-learn:' in output and not any(v in output for v in [
                        'SIGFPE','SIGSEGV','Fortran runtime error']))
                records.append(dict(kind=kind,name=name,token=str(value),accepted=accepted,source_error=error,
                    native_returncode=p.returncode,native_output=output,passed=passed))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),a.executable,Path('app/reference_ada_controls_invalid_driver.f90')]+[Path('src')/s for s in [
        'sklearn_adaboost.f90','sklearn_adaboost_regression.f90','sklearn_search_adapters.f90','sklearn_tree_features.f90']]
    report=dict(configurations=len(records),passed=all(r['passed'] for r in records),records=records,
        sklearn=sklearn.__version__,sha256={str(p):sha(p) for p in files})
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(dict(configurations=len(records),passed=report['passed']));assert report['passed']


if __name__=='__main__':main()
