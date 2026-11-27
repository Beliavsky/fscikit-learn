"""Actual sklearn fit/query errors for classic gradient boosting."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
a=ap.parse_args();checks=[]
common=dict(count=dict(n_estimators=0),depth=dict(max_depth=0),leaf=dict(min_samples_leaf=0),split=dict(min_samples_split=1),
 seed=dict(random_state=-1),rate_negative=dict(learning_rate=-.1),rate_nan=dict(learning_rate=np.nan),rate_inf=dict(learning_rate=np.inf),
 decrease=dict(min_impurity_decrease=-.1),criterion=dict(criterion='invalid'),loss=dict(loss='invalid'),
 feature_integer=dict(max_features=-1),feature_zero=dict(max_features=0.),feature_large=dict(max_features=1.01),
 feature_nan=dict(max_features=np.nan),feature_inf=dict(max_features=np.inf),feature_name=dict(max_features='invalid'),
 input_nan={},input_inf={},float32_overflow={},query_nan={},query_inf={},query_width={},target_size={},unfit={},
 subsample_zero=dict(subsample=0.),subsample_negative=dict(subsample=-.1),subsample_large=dict(subsample=1.01),
 subsample_nan=dict(subsample=np.nan),subsample_inf=dict(subsample=np.inf),
 warm_count=dict(warm_start=True,n_estimators=1),warm_width=dict(warm_start=True))
reg=dict(alpha_zero=dict(alpha=0.),alpha_one=dict(alpha=1.),alpha_nan=dict(alpha=np.nan),alpha_inf=dict(alpha=np.inf),
 budget=dict(max_leaf_nodes=1),ccp=dict(ccp_alpha=-.1),weight_fraction=dict(min_weight_fraction_leaf=.6),
 weights_shape={},weights_zero={},weights_nan={},weights_inf={},target_nan={},target_inf={})
for kind,cls in [('regression',GradientBoostingRegressor),('classification',GradientBoostingClassifier)]:
 cases=common|(reg if kind=='regression' else dict(one_class={},exponential_multiclass=dict(loss='exponential'),
  budget=dict(max_leaf_nodes=1),ccp=dict(ccp_alpha=-.1),weight_fraction=dict(min_weight_fraction_leaf=.6),
  weights_shape={},weights_zero={},weights_nan={},weights_inf={},weights_one_class={},weights_cancel={},warm_classes=dict(warm_start=True,n_estimators=3)))
 for name,params in cases.items():
  x=np.column_stack([np.arange(1,9),np.arange(1,9)%3,np.arange(1,9)**2]).astype(float)
  y=np.arange(1,9).astype(float) if kind=='regression' else np.arange(1,9)%2
  model=cls(n_estimators=2,random_state=0)
  if name!='unfit':model.fit(x,y)
  model.set_params(**params);xx=x.copy();target=y.copy();w=None;phase='fit'
  if name.endswith('_nan'):
   if name.startswith('input') or name.startswith('query'):xx[0,0]=np.nan
   elif name.startswith('target'):target[0]=np.nan
  if name.endswith('_inf'):
   if name.startswith('input') or name.startswith('query'):xx[0,0]=np.inf
   elif name.startswith('target'):target[0]=np.inf
  if name=='float32_overflow':xx[0,0]=2*float(np.finfo(np.float32).max)
  if name=='target_size':target=y[:7]
  if name=='query_width':xx=x[:,:2]
  if name=='warm_width':xx=x[:,:2]
  if name=='warm_classes':target=np.arange(8)%3
  if name=='one_class':target[:]=0
  if name=='exponential_multiclass':target[0]=99
  if name.startswith('weights_'):
   w=np.ones(8)
   if name=='weights_zero':w[:]=0
   if name=='weights_shape':w=np.ones(1)
   if name=='weights_nan':w[0]=np.nan
   if name=='weights_inf':w[0]=np.inf
   if name=='weights_one_class':w[y==1]=0
   if name=='weights_cancel':w[y==1]=-1
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    if name.startswith('query') or name=='unfit':phase='predict';model.predict(xx)
    else:model.fit(xx,target,sample_weight=w)
  except (ValueError,TypeError,ZeroDivisionError,IndexError) as e:error=dict(type=type(e).__name__,message=str(e),phase=phase)
  else:raise AssertionError(('Source accepted',kind,name))
  p=subprocess.run([str(a.executable),kind,name],capture_output=True,text=True);output=p.stdout+p.stderr
  passed=p.returncode!=0 and 'fscikit-learn:' in output and 'bad gradient input accepted' not in output and (name=='unfit' or 'BASE_FIT_PASSED' in output) and not any(s in output for s in ['SIGFPE','SIGSEGV','Fortran runtime error'])
  checks.append(dict(kind=kind,case=name,source_error=error,native_output=output,passed=passed))
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
report=dict(count=len(checks),passed=all(c['passed'] for c in checks),checks=checks,sha256={str(f):sha(f) for f in [Path(__file__),a.executable,Path('src/sklearn_classic_gradient_boosting.f90'),Path('src/sklearn_ensemble.f90'),Path('src/sklearn_boosting_loss.f90'),Path('src/sklearn_tree.f90')]})
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n');print('Classic boosting guards',report['count'],report['passed']);assert report['passed']
