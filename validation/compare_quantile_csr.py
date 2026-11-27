"""QuantileRegressor dense/CSR, rank adapters, weights and LP certificates."""
from pathlib import Path
import argparse,os,json,subprocess,warnings,hashlib
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
import scipy,sklearn
from scipy.sparse import csr_matrix
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import r2_score

def raw_csr(a):
 n,p=a.shape;indices=[];data=[];ptr=[0]
 for row in a:
  for j in reversed(range(p)):indices.extend([j]*3);data.extend([.6*row[j],.4*row[j],0.])
  ptr.append(len(data))
 return csr_matrix((data,indices,ptr),shape=a.shape)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);args=ap.parse_args();records=[]
 configs=[dict(n=n,p=p,kind=kind,mode=mode,setting=setting) for n,p,kind in [(23,3,'tall'),(7,11,'wide'),(19,4,'rank'),(13,1,'column'),(9,3,'zero_design')] for mode in range(12) for setting in range(4)]
 for index,c in enumerate(configs):
  n,p,mode,setting=c['n'],c['p'],c['mode'],c['setting'];rng=np.random.default_rng(933+n+p);x=rng.normal(size=(n,p));y=.7+x@np.linspace(-.4,.8,p)+rng.normal(size=n)*.04
  if c['kind']=='rank':x[:,-1]=x[:,0]
  if c['kind']=='zero_design':x[:]=0
  w=.2+rng.random(n);w[::5]=0
  if mode in [2,4,7,9]:w[:]=1.7
  if setting==3:w[:]=0
  q,alpha,intercept=[(.1,0.,False),(.5,.03,True),(.9,1.,True),(.5,.1,False)][setting]
  text=f'{n} {p} {mode} {int(intercept)} {q} {alpha}\n'+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in np.column_stack([x,y,w]))+'\n'
  inp=args.report.with_suffix('.in');out=args.report.with_suffix('.out');inp.write_text(text);native={};source={};error=None;differences=[]
  proc=subprocess.run([str(args.executable),str(inp),str(out)],capture_output=True,text=True)
  train=raw_csr(x) if mode>=10 else csr_matrix(x) if mode>=5 else x
  query=raw_csr(x+.137) if mode>=10 else csr_matrix(x+.137) if mode>=5 else x+.137
  logical=train.toarray() if mode>=5 else x;query_logical=query.toarray() if mode>=5 else query
  fit_weight=None if mode in [0,5] else float(w[0]) if mode in [2,4,7,9] else w
  actual_weight=np.ones(n) if fit_weight is None else np.full(n,fit_weight) if np.ndim(fit_weight)==0 else w
  norm=actual_weight/actual_weight.sum() if actual_weight.sum()>0 else np.zeros(n);score_weight=w if w.sum()>0 else np.ones(n)
  column=mode in [3,4,8,9,11];nonunique=c['kind']=='rank' or n<p and alpha==0
  try:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');m=QuantileRegressor(quantile=q,alpha=alpha,fit_intercept=intercept).fit(train,y[:,None] if column else y,sample_weight=fit_weight)
   source=dict(coef=m.coef_.tolist(),intercept=[float(m.intercept_)],predict=m.predict(train).tolist(),query=m.predict(query).tolist(),n_iter=int(m.n_iter_))
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  checks={}
  if proc.returncode==0 and error is None:
   for line in out.read_text().splitlines():
    key,*values=line.split();native[key]=[float(v) for v in values]
   def check(name,a,b,atol=2e-7,rtol=2e-7):
    checks[name]=np.shape(a)==np.shape(b) and bool(np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True))
   def objective(pred,coef):
    residual=y-np.array(pred);return np.dot(norm,np.maximum(q*residual,(q-1)*residual))+alpha*np.abs(coef).sum()
   source['objective']=float(objective(source['predict'],source['coef']));scale=max(1.,abs(source['objective']),max(abs(y)))
   check('objective',objective(native['predict'],native['coef']),source['objective'],2e-8*scale)
   check('reported_objective',native['diagnostics'][0],objective(native['predict'],native['coef']),2e-8*scale)
   check('dual_objective',np.dot(y,native['dual']),native['diagnostics'][0],2e-8*scale)
   check('dual_gap',native['diagnostics'][1],0.,2e-8*scale);check('optimality',native['diagnostics'][2],0.,1.01e-10)
   dual=np.array(native['dual']);check('dual_upper',np.maximum(dual-q*norm,0),np.zeros(n));check('dual_lower',np.maximum((q-1)*norm-dual,0),np.zeros(n))
   check('dual_features',np.maximum(abs(logical.T@dual)-alpha,0),np.zeros(p))
   if intercept:check('dual_intercept',dual.sum(),0.)
   check('prediction_identity',native['predict'],logical@native['coef']+native['intercept'][0]);check('query_identity',native['query'],query_logical@native['coef']+native['intercept'][0])
   check('weighted_score',native['score'][0],r2_score(y,native['predict'],sample_weight=score_weight))
   check('state',native['state'][1:],[p,1],0.,0.)
   if not nonunique:
    for key in ['coef','intercept','predict','query']:check(key,native[key],source[key],2e-7*scale)
   differences=[name for name,passed in checks.items() if not passed]
  elif (proc.returncode==0)!=(error is None):differences.append('acceptance')
  if proc.returncode not in [0,1] or proc.returncode==1 and 'fscikit-learn:' not in proc.stdout:differences.append('abnormal_native_exit')
  records.append(dict(index=index,config=c,input=text,nonunique=nonunique,source=source,source_error=error,native=native,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,checks=checks,differences=differences))
 root=Path(sklearn.__file__).parent
 report=dict(sklearn=sklearn.__version__,scipy=scipy.__version__,configurations=len(records),groups=sum(len(r['checks']) for r in records),failures=[r['index'] for r in records if r['differences']],source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['linear_model/_quantile.py']},records=records)
 args.report.write_text(json.dumps(report,indent=2)+'\n');print('Quantile dense/CSR',len(records),'strict failures',report['failures']);raise SystemExit(bool(report['failures']))
if __name__=='__main__':main()
