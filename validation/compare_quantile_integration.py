"""QuantileRegressor clone, CV and GridSearchCV/Pipeline against sklearn."""
from pathlib import Path
import argparse,json,os,subprocess,warnings,hashlib
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 for seed in [7,42,2026]:
  for weighted in [0,1,2,3]:
   for chain in [0,1]:
    for scoring in [0,1]:
     for random in [0,1]:
      n,p=29,3;rng=np.random.default_rng(seed);x=rng.normal(size=(n,p));y=.7+x@np.array([.4,-.2,.6])+rng.normal(size=n)*.07;w=.2+rng.random(n);w[::7]=0
      if weighted==2:w[:]=1.7
      if weighted==3:w[:]=0
      text=f'{n} {p} {weighted} {chain} {scoring} {random}\n'+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in np.column_stack([x,y,w]))+'\n';inp=a.report.with_suffix('.in');out=a.report.with_suffix('.out');inp.write_text(text)
      proc=subprocess.run([str(a.executable),str(inp),str(out)],capture_output=True,text=True);native={};source={};error=None;differences=[]
      try:
       with warnings.catch_warnings():
        warnings.simplefilter('ignore');base=QuantileRegressor(alpha=.03,quantile=.5).fit(x,y);before=base.coef_.copy();copy=clone(base)
        fit_weight=float(w[0]) if weighted==2 else w
        cv=cross_val_score(base,x,y,cv=4,params={'sample_weight':fit_weight} if weighted else None)
        estimator=Pipeline([('scale',StandardScaler()),('model',base)]) if chain else base;prefix='model__' if chain else ''
        search_class=RandomizedSearchCV if random else GridSearchCV
        extra=dict(n_iter=4,random_state=42) if random else {}
        search=search_class(estimator,{prefix+'alpha':[0.,.05],prefix+'quantile':[.25,.5,.75]},cv=4,scoring='neg_mean_squared_error' if scoring else None,return_train_score=True,**extra)
        search.fit(x,y,**({prefix+'sample_weight':fit_weight} if weighted else {}));r=search.cv_results_
        source=dict(cv=cv.tolist(),mean=r['mean_test_score'].tolist(),std=r['std_test_score'].tolist(),train=r['mean_train_score'].tolist(),rank=r['rank_test_score'].tolist(),split=np.stack([r[f'split{i}_test_score'] for i in range(4)],axis=1).ravel(order='F').tolist(),predict=search.predict(x+.137).tolist(),best=[int(search.best_index_)+1,float(search.best_score_)],params=[v for row in r['params'] for v in [float(row[prefix+'alpha']),float(row[prefix+'quantile'])]],clone=[int(hasattr(copy,'coef_')),0,int(hasattr(copy,'n_features_in_')),int(hasattr(copy,'n_iter_')),0],caller=[float(max(abs(base.coef_-before)))])
      except Exception as exc:error=f'{type(exc).__name__}: {exc}'
      if proc.returncode==0 and error is None:
       for line in out.read_text().splitlines():
        key,*values=line.split();native[key]=[float(v) for v in values]
       for key,value in source.items():
        if np.shape(native[key])!=np.shape(value) or not np.allclose(native[key],value,atol=2e-7,rtol=2e-7,equal_nan=True):differences.append(key)
      elif (proc.returncode==0)!=(error is None):differences.append('acceptance')
      if proc.returncode not in [0,1] or proc.returncode==1 and 'fscikit-learn:' not in proc.stdout:differences.append('abnormal_native_exit')
      records.append(dict(seed=seed,weighted=weighted,pipeline=chain,scoring=scoring,random=random,input=text,source=source,source_error=error,native=native,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,differences=differences))
 root=Path(sklearn.__file__).parent
 report=dict(sklearn=sklearn.__version__,configurations=len(records),failures=[i for i,r in enumerate(records) if r['differences']],source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['base.py','model_selection/_search.py','model_selection/_validation.py','pipeline.py','linear_model/_quantile.py']},records=records)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('Quantile integration',len(records),'strict failures',report['failures']);raise SystemExit(bool(report['failures']))
if __name__=='__main__':main()
