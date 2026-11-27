"""Fresh-clone regression CV and fitting-weight routing versus sklearn."""
from pathlib import Path
import argparse,json,os,subprocess,warnings,hashlib
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
import sklearn
from sklearn.linear_model import HuberRegressor,PoissonRegressor,GammaRegressor,TweedieRegressor
from sklearn.model_selection import cross_val_score

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 for family in range(4):
  for iterations in [1,4,50]:
   for warm in [0,1]:
    for prefit in [0,1]:
     for weighted in [0,1,2]:
      n,p=35,3;rng=np.random.default_rng(924+family);x=rng.normal(size=(n,p))*.4;linear=.3+x@np.array([.4,-.2,.3]);alternative=1.1+x@np.array([-.8,.7,-.2]);noise=rng.normal(size=n)*.03
      if family==0:
       y=linear+noise;y[::9]+=.4;other=alternative;m=HuberRegressor(max_iter=iterations,tol=1e-7,warm_start=bool(warm))
      else:
       y=np.exp(linear+noise);other=np.exp(alternative);cls=[None,PoissonRegressor,GammaRegressor,TweedieRegressor][family];kw=dict(alpha=.1,max_iter=iterations,tol=1e-8,warm_start=bool(warm));m=cls(**kw,**(dict(power=1.5) if family==3 else {}))
      w=.2+rng.random(n);w[::7]=0
      if weighted==2:w[:]=1.7
      text=f'{n} {p} {family} {iterations} {warm} {prefit} {weighted}\n'+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in np.column_stack([x,y,other,w]))+'\n';inp=a.report.with_suffix('.in');out=a.report.with_suffix('.out');inp.write_text(text)
      proc=subprocess.run([str(a.executable),str(inp),str(out)],capture_output=True,text=True);source={};native={};error=None;differences=[]
      try:
       with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if prefit:m.fit(x,other)
        before=m.coef_.copy() if hasattr(m,'coef_') else None;intercept=float(m.intercept_) if hasattr(m,'intercept_') else 0.
        fit_weight=float(w[0]) if weighted==2 else w
        scores=cross_val_score(m,x,y,cv=4,params={'sample_weight':fit_weight} if weighted else None)
       change=max(abs(float(m.intercept_)-intercept),max(abs(m.coef_-before))) if before is not None else 0.
       source=dict(scores=scores.tolist(),caller=[float(change),int(before is not None),int(hasattr(m,'coef_'))])
      except Exception as exc:error=f'{type(exc).__name__}: {exc}'
      if proc.returncode==0 and error is None:
       lines=out.read_text().splitlines();native=dict(scores=np.fromstring(lines[0],sep=' ').tolist(),caller=np.fromstring(lines[1],sep=' ').tolist())
       for key in ['scores','caller']:
        if np.shape(native[key])!=np.shape(source[key]) or not np.allclose(native[key],source[key],atol=2e-7,rtol=2e-7,equal_nan=True):differences.append(key)
      elif (proc.returncode==0)!=(error is None):differences.append('acceptance')
      if proc.returncode not in [0,1] or proc.returncode==1 and 'fscikit-learn:' not in proc.stdout:differences.append('abnormal_native_exit')
      records.append(dict(family=family,iterations=iterations,warm=warm,prefit=prefit,weighted=weighted,input=text,source=source,source_error=error,native=native,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,differences=differences))
 root=Path(sklearn.__file__).parent
 report=dict(sklearn=sklearn.__version__,configurations=len(records),failures=[i for i,r in enumerate(records) if r['differences']],source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['base.py','model_selection/_validation.py','linear_model/_huber.py','linear_model/_glm/glm.py']},records=records)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('Fresh clone CV',len(records),'strict failures',report['failures']);raise SystemExit(bool(report['failures']))
if __name__=='__main__':main()
