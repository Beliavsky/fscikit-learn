"""Warm forests validate unchanged-count fits and reject shrinking ensembles."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,RandomForestRegressor,ExtraTreesRegressor
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
a=ap.parse_args();checks=[]
controls=dict(shrink=dict(n_estimators=2),zero=dict(n_estimators=0),criterion=dict(criterion='invalid'),depth=dict(max_depth=-1),
              leaf=dict(min_samples_leaf=0),split=dict(min_samples_split=1),budget=dict(max_leaf_nodes=1),features=dict(max_features=-1),
              decrease=dict(min_impurity_decrease=-.1),alpha=dict(ccp_alpha=-.1),fraction=dict(min_weight_fraction_leaf=.6),
              max_samples_no_bootstrap=dict(bootstrap=False,max_samples=2),oob_no_bootstrap=dict(bootstrap=False,oob_score=True),
              weights_length={},weights_nan={},input_inf={},feature_count={},output_count={})
for kind,classes in [('classification',[RandomForestClassifier,ExtraTreesClassifier]),('regression',[RandomForestRegressor,ExtraTreesRegressor])]:
 for extra,cls in enumerate(classes):
  cases=controls|({'class_width_shrink':{}} if kind=='classification' else {'poisson_negative':dict(criterion='poisson'),'poisson_zero':dict(criterion='poisson')})
  for name,params in cases.items():
   x=np.column_stack([np.arange(1,9),np.arange(1,9)%3]).astype(float)
   y=np.column_stack([np.arange(1,9)%2,(np.arange(1,9)//2)%2]) if kind=='classification' else np.column_stack([np.arange(1,9),np.arange(8,0,-1)]).astype(float)
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');model=cls(warm_start=True,n_estimators=3,random_state=7).fit(x,y)
   model.set_params(**params);xx=x.copy();target=y.copy();w=None;phase='fit'
   if name=='weights_length':w=[1]
   if name=='weights_nan':w=np.r_[np.nan,np.ones(7)]
   if name=='input_inf':xx[0,0]=np.inf
   if name=='feature_count':xx=np.ones((8,3))
   if name=='output_count':target=y[:,0]
   if name in ['class_width_shrink','poisson_zero']:target[:]=0
   if name=='poisson_negative':target[0,0]=-1
   try:
    with warnings.catch_warnings():
     warnings.simplefilter('ignore');model.fit(xx,target,sample_weight=w)
     if name in ['feature_count','output_count','class_width_shrink']:phase='predict';model.predict(xx)
   except (ValueError,TypeError) as error:source_error=dict(type=type(error).__name__,message=str(error),phase=phase)
   else:raise AssertionError(('Source accepted',kind,extra,name))
   p=subprocess.run([str(a.executable),kind,str(extra),name],capture_output=True,text=True);out=p.stdout+p.stderr
   passed=p.returncode!=0 and 'fscikit-learn:' in out and 'invalid warm forest input was accepted' not in out and not any(s in out for s in ['SIGFPE','SIGSEGV','Fortran runtime error'])
   checks.append(dict(kind=kind,extra=extra,case=name,source_error=source_error,native_output=out,passed=passed))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=dict(count=len(checks),checks=checks,passed=all(c['passed'] for c in checks),sha256={str(p):sha(p) for p in [Path(__file__),a.executable,Path('src/sklearn_ensemble.f90')]})
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n');print(report['count'],'warm forest guards passed:',report['passed']);assert report['passed']
