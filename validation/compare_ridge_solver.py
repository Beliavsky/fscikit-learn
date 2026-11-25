"""Compare public ridge solver coefficients and LSQR iterations with sklearn."""
from pathlib import Path
import argparse, hashlib, json, subprocess, warnings
import scipy, sklearn
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import ridge_regression


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--executable',required=True)
    parser.add_argument('--report')
    args=parser.parse_args()
    cases=[]; lines=[]
    rng=np.random.default_rng(2026)
    for n,p in [(31,7),(7,19),(12,12)]:
      for t in [1,3]:
       x=rng.normal(size=(n,p)); x[rng.random(x.shape)<.4]=0
       y=rng.normal(size=(n,t))
       for sparse in [False,True]:
        for solver in ['auto','svd','cholesky','sparse_cg','lsqr','lbfgs']:
         if sparse and solver=='svd':continue
         for weighted in [False,True]:
          for cap in [-1,2]:
           weights=rng.uniform(.1,2,n);weights[0]=0
           alpha=np.arange(t)*.7+.3
           positive=solver=='lbfgs'
           with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            coef,it=ridge_regression(csr_matrix(x) if sparse else x,y,alpha,
              solver=solver,tol=1e-8,max_iter=None if cap<0 else cap,
              sample_weight=weights if weighted else None,positive=positive,return_n_iter=True)
           cases.append((np.asarray(coef).reshape(t,p),np.full(t,-1) if it is None else it,solver))
           lines += [f'{n} {p} {t} {t} {int(sparse)} {int(weighted)}',f'{solver} 1e-8 {cap} {int(positive)}']
           lines += [' '.join(map(str,row)) for row in x]
           lines += [' '.join(map(str,row)) for row in y]
           lines += [' '.join(map(str,alpha)),' '.join(map(str,weights))]
    for variant in ['zero_design','zero_target','rank_deficient','zero_weights']:
     x=np.eye(5);y=np.arange(1.,6.)[:,None]
     if variant=='zero_design':x[:]=0
     if variant=='zero_target':y[:]=0
     if variant=='rank_deficient':x[:,1]=x[:,0]
     weights=np.zeros(5) if variant=='zero_weights' else np.ones(5)
     for sparse in [False,True]:
      for solver in ['svd','cholesky','sparse_cg','lsqr','lbfgs']:
       if sparse and (solver=='svd' or (solver=='cholesky' and variant=='rank_deficient')):continue
       for cap in [0,-1]:
        positive=solver=='lbfgs';alpha=np.array([0. if variant=='rank_deficient' else 1.])
        with warnings.catch_warnings():
         warnings.simplefilter('ignore')
         coef,it=ridge_regression(csr_matrix(x) if sparse else x,y,alpha,solver=solver,
           tol=1e-8,max_iter=None if cap<0 else cap,sample_weight=weights,
           positive=positive,return_n_iter=True)
        cases.append((np.asarray(coef).reshape(1,5),np.array([-1]) if it is None else it,solver))
        lines += [f'5 5 1 1 {int(sparse)} 1',f'{solver} 1e-8 {cap} {int(positive)}']
        lines += [' '.join(map(str,row)) for row in x]
        lines += [' '.join(map(str,row)) for row in y]
        lines += [' '.join(map(str,alpha)),' '.join(map(str,weights))]
    for sparse in [False,True]:
      x=np.array([[1.,0.,0.],[1.,0.,0.]])
      y=np.array([[1.,2.],[2.,4.]])
      with warnings.catch_warnings():
       warnings.simplefilter('ignore')
       coef,it=ridge_regression(csr_matrix(x) if sparse else x,y,0.,solver='cholesky',return_n_iter=True)
      cases.append((coef,np.full(2,-1),'cholesky'))
      lines += [f'2 3 2 1 {int(sparse)} 0','cholesky 1e-8 -1 0']
      lines += [' '.join(map(str,row)) for row in x]
      lines += [' '.join(map(str,row)) for row in y]
      lines += ['0','1 1']
    proc=subprocess.run([args.executable],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True)
    rows=iter(proc.stdout.splitlines()); worst=0
    for i,(coef,it,solver) in enumerate(cases):
      actual=np.array([np.fromstring(next(rows),sep=' ') for _ in coef])
      actual_it=np.fromstring(next(rows),sep=' ',dtype=int)
      err=np.max(np.abs(actual-coef)/(1+np.abs(coef)));worst=max(worst,float(err))
      if not np.allclose(actual,coef,rtol=3e-8,atol=3e-8):
       raise AssertionError((i,solver,err,actual,coef))
      np.testing.assert_array_equal(actual_it,it,err_msg=f'case {i} {solver}')
    report={'cases':len(cases),'groups':2*len(cases),'max_scaled_error':worst,
      'sklearn':sklearn.__version__,'numpy':np.__version__,'scipy':scipy.__version__,
      'executable_sha256':hashlib.sha256(Path(args.executable).read_bytes()).hexdigest(),
      'source_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'src/sklearn_ridge_solver.f90').read_bytes()).hexdigest(),
      'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(report,indent=2))
    if args.report:Path(args.report).write_text(json.dumps(report,indent=2)+"\n")
if __name__=='__main__':main()
