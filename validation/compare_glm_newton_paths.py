"""Actual Source Newton failure/last-budget paths without prediction-domain filtering."""
from pathlib import Path
import os,argparse,json,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model._glm._newton_solver import NewtonCholeskySolver

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[];inputs=[]
 fixtures=[('line-search',np.ones((1,1)),np.array([1e12]),1.,'log',False,0.,1),('line-search',np.ones((1,1)),np.array([1e12]),1.,'log',False,0.,2),('singular/ill-conditioned',np.zeros((5,2)),np.linspace(1.,2.,5),0.,'identity',False,0.,1),('curvature',np.arange(1.,6.)[:,None]*.1,np.linspace(5.,6.,5),0.,'log',False,.2,100),('singular/ill-conditioned',np.c_[np.arange(5.)*.1,np.arange(5.)*1e-11],np.linspace(1.,2.,5),0.,'identity',True,0.,100)]
 original=NewtonCholeskySolver.solve
 for reason,x,y,power,link,intercept,alpha,cap in fixtures:
  for representation in (0,1,2):
   n,p=x.shape;w=np.ones(n);point=np.zeros(p+int(intercept))
   if intercept:point[-1]=np.log(y.mean()) if link=='log' else y.mean()
   data=[];columns=[];pointers=[0]
   for row in x:
    for col in reversed(range(p)):
     data.extend([.25*row[col],.75*row[col]]);columns.extend([col,col])
    data.append(0.);columns.append(0);pointers.append(len(data))
   sx=x if representation==0 else csr_matrix(x) if representation==1 else csr_matrix((data,columns,pointers),shape=x.shape)
   state=[]
   def tracked(solver,*args,**kwargs):
    result=original(solver,*args,**kwargs);state.append(dict(converged=bool(solver.converged),fallback=bool(solver.use_fallback_lbfgs_solve)));return result
   with patch.object(NewtonCholeskySolver,'solve',tracked),warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always');model=TweedieRegressor(power=power,link=link,alpha=alpha,tol=1e-10,max_iter=cap,fit_intercept=intercept,solver='newton-cholesky').fit(sx,y,sample_weight=w)
   text=f'{n} {p} {power} {link} {alpha} 1e-10 {cap} {int(intercept)} {representation}\n'+' '.join(format(v,'.17g') for v in point)+'\n'+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in np.column_stack([x,y,w]))+'\n';inputs.append(text)
   expected=np.r_[model.coef_,model.intercept_] if intercept else model.coef_
   records.append(dict(reason=reason,representation=representation,input=text,source=expected.tolist(),n_iter=int(model.n_iter_),**state[-1],warnings=[str(w.message) for w in caught]))
 inp=a.report.with_suffix('.in');out=a.report.with_suffix('.out');inp.write_text(str(len(inputs))+'\n'+''.join(inputs));p=subprocess.run([str(a.executable),str(inp),str(out)],capture_output=True,text=True);assert p.returncode==0,(p.returncode,p.stdout,p.stderr)
 lines=out.read_text().splitlines();assert len(lines)==3*len(records)
 for i,r in enumerate(records):
  point=np.fromstring(lines[3*i],sep=' ');meta=np.fromstring(lines[3*i+1],sep=' ',dtype=int);message=lines[3*i+2]
  r.update(native=point.tolist(),native_iterations=int(meta[0]),native_converged=bool(meta[1]),ieee_preserved=bool(meta[2]),native_message=message)
  r['passed']=bool(np.allclose(point,r['source'],atol=3e-8,rtol=3e-8,equal_nan=True) and meta[0]==r['n_iter'] and bool(meta[1])==r['converged'] and bool(meta[2]) and r['reason'] in message and r['fallback'])
 report=dict(records=records,failed=[i for i,r in enumerate(records) if not r['passed']]);a.report.write_text(json.dumps(report,indent=2)+'\n');print('Newton fallback paths',len(records),'failures',report['failed']);raise SystemExit(bool(report['failed']))
if __name__=='__main__':main()
