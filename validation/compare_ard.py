"""ARDRegression covariance, pruning and histories against installed sklearn."""
from pathlib import Path
import os,argparse,json,subprocess,warnings,hashlib
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
import sklearn
from sklearn.linear_model import ARDRegression

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);args=ap.parse_args();records=[]
 configs=[dict(n=n,p=p,kind=kind,column=column,intercept=intercept,setting=setting) for n,p,kind in [(31,3,'tall'),(8,8,'square'),(7,11,'wide'),(25,4,'rank'),(2,3,'two'),(13,1,'column')] for column in (False,True) for intercept in (False,True) for setting in range(6)]
 configs += [dict(n=21,p=5,kind=kind,column=False,intercept=intercept,setting=0) for kind in ['zero_target','constant_target','constant_design','offset'] for intercept in (False,True)]
 for index,c in enumerate(configs):
  n,p=c['n'],c['p'];rng=np.random.default_rng(401+n+p);x=rng.normal(size=(n,p))*.4;y=x@np.linspace(-.6,.7,p)+.4+rng.normal(size=n)*.03
  if c['kind']=='rank':x[:,-1]=2*x[:,0]
  if c['kind']=='zero_target':y[:]=0
  if c['kind']=='constant_target':y[:]=2
  if c['kind']=='constant_design':x[:]=1
  if c['kind']=='offset':x+=100
  settings=[dict(max_iter=300,tol=1e-6),dict(max_iter=1),dict(threshold_lambda=0),dict(alpha_1=0,lambda_1=0,alpha_2=.02,lambda_2=.01),dict(threshold_lambda=1.2),dict(tol=0,max_iter=20)]
  options=dict(fit_intercept=c['intercept'],compute_score=c['setting']!=1,**settings[c['setting']]);m=ARDRegression(**options)
  header=[n,p,int(c['column']),int(c['intercept']),int(m.compute_score),m.max_iter,m.tol,m.alpha_1,m.alpha_2,m.lambda_1,m.lambda_2,m.threshold_lambda]
  text=' '.join(str(v) for v in header)+'\n'+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in np.column_stack([x,y]))+'\n'
  inp=args.report.with_suffix('.in');out=args.report.with_suffix('.out');inp.write_text(text)
  proc=subprocess.run([str(args.executable),str(inp),str(out)],capture_output=True,text=True);source={};native={};error=None;differences=[]
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');m.fit(x,y[:,None] if c['column'] else y);prediction,std=m.predict(x+.017,return_std=True)
   source=dict(coef=m.coef_.tolist(),parameters=[float(m.intercept_),float(m.alpha_)],precision=m.lambda_.tolist(),covariance=m.sigma_.ravel(order='F').tolist(),prediction=prediction.tolist(),std=std.tolist(),scores=list(m.scores_),offset=m.X_offset_.tolist(),n_iter=int(m.n_iter_),rank=int(sum(m.lambda_<m.threshold_lambda)))
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  if error is None and proc.returncode==0:
   lines=out.read_text().splitlines();assert len(lines)==9,(c,lines)
   for key,line in zip(['coef','parameters','precision','covariance','prediction','std','scores','offset'],lines[:8]):
    native[key]=np.fromstring(line,sep=' ').tolist()
    if np.shape(native[key])!=np.shape(source[key]) or not np.allclose(native[key],source[key],atol=3e-8,rtol=3e-8,equal_nan=True):differences.append(key)
   meta=np.fromstring(lines[8],sep=' ',dtype=int);native.update(n_features=int(meta[0]),n_iter=int(meta[1]),rank=int(meta[2]))
   if native['n_features']!=p or native['n_iter']!=source['n_iter'] or native['rank']!=source['rank']:differences.append('metadata')
  elif (error is None)!=(proc.returncode==0):differences.append('acceptance')
  if proc.returncode not in (0,1) or proc.returncode==1 and 'fscikit-learn:' not in proc.stdout:differences.append('abnormal_native_exit')
  records.append(dict(index=index,config=c,options=options,input=text,source=source,source_error=error,native=native,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,differences=differences))
 root=Path(sklearn.__file__).parent
 report=dict(sklearn_version=sklearn.__version__,source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['linear_model/_bayes.py','linear_model/_base.py']},configurations=len(records),failures=[r['index'] for r in records if r['differences']],records=records)
 args.report.write_text(json.dumps(report,indent=2)+'\n');print('ARDRegression',len(records),'strict failures',report['failures']);raise SystemExit(bool(report['failures']))
if __name__=='__main__':main()
