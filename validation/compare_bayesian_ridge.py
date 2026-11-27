"""Weighted BayesianRidge / marginal likelihood against installed sklearn."""
from pathlib import Path
import os,argparse,json,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
import sklearn
from sklearn.linear_model import BayesianRidge

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 configs=[dict(n=n,p=p,kind=kind,mode=mode,intercept=intercept,setting=setting) for n,p,kind in [(31,3,'tall'),(8,8,'square'),(7,11,'wide'),(25,4,'rank'),(1,3,'single'),(13,1,'column')] for mode in range(5) for intercept in (False,True) for setting in range(3)]
 configs += [dict(n=31,p=3,kind=kind,mode=mode,intercept=intercept,setting=setting) for kind,mode,intercept,setting in [('zero_weights',1,False,0),('zero_weights',1,True,0),('zero_alpha',1,True,1),('zero_alpha_score',1,True,0),('zero_lambda',1,True,1),('zero_lambda_score',1,True,0)]]
 for index,c in enumerate(configs):
  n,p=c['n'],c['p'];rng=np.random.default_rng(791+n+p);x=rng.normal(size=(n,p))*.4;y=x@np.linspace(-.6,.7,p)+.4+rng.normal(size=n)*.03
  if c['kind']=='rank':x[:,-1]=2*x[:,0]
  w=.2+2*rng.random(n)
  if n>1:w[::5]=0
  if c['mode'] in (2,4):w[:]=1.7
  if c['kind']=='zero_weights':w[:]=0
  settings=[dict(max_iter=300,tol=1e-8),dict(max_iter=1,tol=1e-8,alpha_init=2.,lambda_init=.3),dict(max_iter=300,tol=1e-8,alpha_1=0.,lambda_1=0.,alpha_2=.01,lambda_2=.02)]
  options=dict(fit_intercept=c['intercept'],compute_score=c['setting']!=1,**settings[c['setting']]);model=BayesianRidge(**options)
  if c['kind'].startswith('zero_alpha'):model.alpha_init=0;options['alpha_init']=0
  if c['kind'].startswith('zero_lambda'):model.lambda_init=0;options['lambda_init']=0
  inp=a.report.with_suffix('.in');out=a.report.with_suffix('.out');header=[n,p,c['mode'],int(c['intercept']),int(model.compute_score),model.max_iter,model.tol,model.alpha_1,model.alpha_2,model.lambda_1,model.lambda_2,-1 if model.alpha_init is None else model.alpha_init,-1 if model.lambda_init is None else model.lambda_init]
  text=' '.join(str(v) for v in header)+'\n'+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in np.column_stack([x,y,w]))+'\n';inp.write_text(text)
  proc=subprocess.run([str(a.executable),str(inp),str(out)],capture_output=True,text=True);error=None;source={};native={};differences=[]
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');model.fit(x,y if c['mode']<3 else y[:,None],sample_weight=None if c['mode']==0 else float(w[0]) if c['mode'] in (2,4) else w)
    prediction,std=model.predict(x+.017,return_std=True)
   source=dict(coef=model.coef_.tolist(),parameters=[float(model.intercept_),float(model.alpha_),float(model.lambda_)],covariance=model.sigma_.ravel(order='F').tolist(),prediction=prediction.tolist(),std=std.tolist(),scores=list(model.scores_),offset=model.X_offset_.tolist(),n_iter=int(model.n_iter_))
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  if error is None and proc.returncode==0:
   lines=out.read_text().splitlines();assert len(lines)==8,(c,lines)
   for key,line in zip(['coef','parameters','covariance','prediction','std','scores','offset'],lines[:7]):
    native[key]=np.fromstring(line,sep=' ').tolist()
    if np.shape(native[key])!=np.shape(source[key]) or not np.allclose(native[key],source[key],atol=3e-8,rtol=3e-8,equal_nan=True):differences.append(key)
   meta=np.fromstring(lines[7],sep=' ',dtype=int);native.update(n_features=int(meta[0]),n_iter=int(meta[1]),converged=bool(meta[2]))
   if native['n_features']!=p or native['n_iter']!=source['n_iter']:differences.append('metadata')
  elif (error is None)!=(proc.returncode==0):differences.append('acceptance')
  if proc.returncode not in (0,1) or proc.returncode==1 and 'fscikit-learn:' not in proc.stdout:differences.append('abnormal_native_exit')
  records.append(dict(index=index,config=c,options=options,input=text,source=source,source_error=error,native=native,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,differences=differences))
 report=dict(sklearn_version=sklearn.__version__,configurations=len(records),failures=[r['index'] for r in records if r['differences']],records=records);a.report.write_text(json.dumps(report,indent=2)+'\n');print('BayesianRidge',len(records),'strict failures',report['failures']);raise SystemExit(bool(report['failures']))
if __name__=='__main__':main()
