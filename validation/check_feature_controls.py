"""Actual sklearn rejection checks for eight tree/forest max_features interfaces."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor,DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,RandomForestClassifier,ExtraTreesClassifier
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
a=ap.parse_args();checks=[]
classes=[[RandomForestRegressor,ExtraTreesRegressor],[RandomForestClassifier,ExtraTreesClassifier],
         [DecisionTreeRegressor,ExtraTreeRegressor],[DecisionTreeClassifier,ExtraTreeClassifier]]
cases=dict(zero=0.,negative=-.1,large=1.01,nan=np.nan,inf=np.inf,negative_inf=-np.inf,integer=-1,name='invalid',long_name='sqrt            invalid suffix')
for family,pair in enumerate(classes):
 for mode,cls in enumerate(pair):
  x=np.column_stack([np.arange(1,10),np.arange(1,10)%3,np.arange(1,10)%5]).astype(float)
  y=np.column_stack([np.arange(1,10),np.arange(9,0,-1)]).astype(float) if family%2==0 else np.column_stack([np.arange(1,10)%2,(np.arange(1,10)//2)%2])
  for name,value in cases.items():
   params=dict(random_state=0)
   if family<2:params.update(n_estimators=3,warm_start=True,bootstrap=True,n_jobs=1)
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');model=cls(**params).fit(x,y)
   model.set_params(max_features=value)
   try:
    with warnings.catch_warnings():warnings.simplefilter('ignore');model.fit(x,y)
   except (ValueError,TypeError) as e:error=dict(type=type(e).__name__,message=str(e))
   else:raise AssertionError(('Source accepted',family,mode,name))
   p=subprocess.run([str(a.executable),str(family),str(mode),name],text=True,capture_output=True)
   output=p.stdout+p.stderr
   passed=p.returncode!=0 and 'fscikit-learn:' in output and 'invalid feature control accepted' not in output and not any(s in output for s in ['SIGSEGV','SIGFPE','Fortran runtime error'])
   checks.append(dict(family=family,mode=mode,name=name,source_error=error,native_output=output,passed=passed))
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
report=dict(count=len(checks),passed=all(c['passed'] for c in checks),checks=checks,sha256={str(f):sha(f) for f in [Path(__file__),a.executable,Path('src/sklearn_tree_features.f90'),Path('src/sklearn_tree.f90'),Path('src/sklearn_ensemble.f90')]})
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n');print('Feature guards',report['count'],report['passed']);assert report['passed']
