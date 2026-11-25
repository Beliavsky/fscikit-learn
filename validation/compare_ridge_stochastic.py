"""Pinned squared-loss SAG/SAGA ridge: draws, lagged updates and intercepts."""
from pathlib import Path
import argparse, hashlib, json, subprocess, warnings, os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy, sklearn
from scipy.sparse import csr_matrix
from sklearn.linear_model import ridge_regression


def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--executable',required=True);parser.add_argument('--report')
 args=parser.parse_args();lines=[];cases=[];source_warnings=0
 for n,p in [(31,7),(7,19),(1,1)]:
  for seed in [0,42,2026]:
   rng=np.random.default_rng(seed);x=rng.normal(size=(n,p));x[rng.random(x.shape)<.6]=0
   x[0,0]=.7
   y=rng.normal(size=(n,2))+.5
   for sparse in [False,True]:
    for weighting in ['none','ones','fractional','zeros','signed']:
     weights=np.ones(n)
     if weighting=='fractional':weights=rng.uniform(0,.9,n);weights[0]=0
     if weighting=='zeros':weights[:]=0
     if weighting=='signed':weights=rng.uniform(.1,.5,n);weights[0]=-.01
     for solver,intercept in [('sag',False),('saga',False),('sag',True),('auto',True)]:
      for cap in [0,1,3,1000]:
       alpha=np.array([.7,2.]);tol=1e-8
       with warnings.catch_warnings(record=True)as caught:
        warnings.simplefilter('always')
        result=ridge_regression(csr_matrix(x) if sparse else x,y,alpha,
          solver=solver,tol=tol,max_iter=cap,random_state=seed,
          sample_weight=None if weighting=='none' else weights,
          return_n_iter=True,return_intercept=intercept)
       source_warnings+=len(caught)
       coef,it=result[:2];bias=np.asarray(result[2])if intercept else np.zeros(2)
       label=dict(n=n,p=p,seed=seed,sparse=sparse,weighting=weighting,solver=solver,intercept=intercept,cap=cap)
       cases.append((coef,it,bias,label))
       lines += [f'{n} {p} 2 2 {int(sparse)} {int(weighting!="none")}',f'{solver} {tol} {cap} 0 {seed} {int(intercept)}']
       lines += [' '.join(map(str,row))for row in x]
       lines += [' '.join(map(str,row))for row in y]
       lines += [' '.join(map(str,alpha)),' '.join(map(str,weights))]
 for layout in [2,3,4]:
  rng=np.random.default_rng(123);n,p=23,8
  x=rng.normal(size=(n,p));x[rng.random(x.shape)<.7]=0
  data=[];indices=[];ptr=[0]
  for row in x:
   columns=range(p) if layout==4 else range(p-1,-1,-1)
   for j in columns:
    if layout!=4 and row[j]==0:continue
    if layout==3:data += [.4*row[j],.6*row[j]];indices += [j,j]
    else:data.append(row[j]);indices.append(j)
   ptr.append(len(data))
  source_x=csr_matrix((data,indices,ptr),shape=x.shape)
  y=rng.normal(size=(n,2));weights=rng.uniform(0,.9,n)
  for solver,intercept in [('sag',False),('saga',False),('sag',True)]:
   for cap in [1,4,1000]:
    alpha=np.array([.7,2.]);tol=1e-8;seed=2026
    with warnings.catch_warnings(record=True)as caught:
     warnings.simplefilter('always')
     result=ridge_regression(source_x,y,alpha,solver=solver,tol=tol,max_iter=cap,
       random_state=seed,sample_weight=weights,return_n_iter=True,return_intercept=intercept)
    source_warnings+=len(caught)
    coef,it=result[:2];bias=np.asarray(result[2])if intercept else np.zeros(2)
    label=dict(layout=layout,solver=solver,intercept=intercept,cap=cap)
    cases.append((coef,it,bias,label))
    lines += [f'{n} {p} 2 2 {layout} 1',f'{solver} {tol} {cap} 0 {seed} {int(intercept)}']
    lines += [' '.join(map(str,row))for row in x]
    lines += [' '.join(map(str,row))for row in y]
    lines += [' '.join(map(str,alpha)),' '.join(map(str,weights))]
 for special in ['reset','zero_design','zero_target','zero_penalty']:
  rng=np.random.default_rng(7);n,p=(257,5)if special=='reset' else (11,4)
  x=rng.normal(size=(n,p));y=rng.normal(size=(n,2));weights=np.ones(n)
  alpha=np.array([.7,2.])
  if special=='reset':alpha=np.array([1e8,1e9]);y*=1e7
  if special=='zero_design':x[:]=0
  if special=='zero_target':y[:]=0
  if special=='zero_penalty':alpha[:]=0
  for sparse in [False,True]:
   for solver,intercept in [('sag',False),('saga',False),('sag',True)]:
    if special=='zero_design' and solver=='sag' and not intercept:continue
    for cap in [1,3,1000]:
     seed=42;tol=1e-8
     with warnings.catch_warnings(record=True)as caught:
      warnings.simplefilter('always')
      result=ridge_regression(csr_matrix(x)if sparse else x,y,alpha,solver=solver,
        max_iter=cap,tol=tol,random_state=seed,return_n_iter=True,return_intercept=intercept)
     source_warnings+=len(caught)
     coef,it=result[:2];bias=np.asarray(result[2])if intercept else np.zeros(2)
     cases.append((coef,it,bias,dict(special=special,sparse=sparse,solver=solver,intercept=intercept,cap=cap)))
     lines += [f'{n} {p} 2 2 {int(sparse)} 0',f'{solver} {tol} {cap} 0 {seed} {int(intercept)}']
     lines += [' '.join(map(str,row))for row in x]
     lines += [' '.join(map(str,row))for row in y]
     lines += [' '.join(map(str,alpha)),' '.join(map(str,weights))]
 proc=subprocess.run([args.executable],input='\n'.join(lines)+'\n',capture_output=True,text=True,check=True)
 rows=iter(proc.stdout.splitlines());worst=0
 for coef,it,bias,label in cases:
  actual=np.array([np.fromstring(next(rows),sep=' ')for _ in coef])
  actual_it=np.fromstring(next(rows),sep=' ',dtype=int);actual_bias=np.fromstring(next(rows),sep=' ')
  for got,want in [(actual,coef),(actual_bias,bias)]:
   worst=max(worst,float(np.max(np.abs(got-want)/(1+np.abs(want)))))
   np.testing.assert_allclose(got,want,rtol=3e-8,atol=3e-8,err_msg=str(label))
  np.testing.assert_array_equal(actual_it,it,err_msg=str(label))
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),groups=3*len(cases),max_scaled_error=worst,source_warnings=source_warnings,
   sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,
   executable_sha256=hashlib.sha256(Path(args.executable).read_bytes()).hexdigest(),
   source_sha256=hashlib.sha256((root/'src/sklearn_ridge_solver.f90').read_bytes()).hexdigest(),
   validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   input_sha256=hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest(),
   configuration_inventory=[case[3]for case in cases],
   reference_sha256={name:hashlib.sha256((Path(sklearn.__file__).parent/name).read_bytes()).hexdigest()
     for name in ['linear_model/_ridge.py','linear_model/_sag.py','linear_model/_sag_fast.pyx.tp',
                  'utils/_seq_dataset.pyx.tp','utils/_random.pxd']})
 print(json.dumps(report,indent=2))
 if args.report:Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
