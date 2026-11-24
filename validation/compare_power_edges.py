"""Power maps: source MLE, free API and retained nonfinite scalar boundaries."""
from pathlib import Path
import argparse,hashlib,io,json,subprocess,warnings
import numpy as np
import scipy,sklearn
from sklearn.preprocessing import PowerTransformer,power_transform
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 paths=[Path(__file__),ROOT/'src/sklearn_power_transform.f90',ROOT/'app/reference_power_edges_driver.f90',a.executable,Path(sklearn.__file__).parent/'preprocessing/_data.py',Path(scipy.__file__).parent/'stats/_morestats.py',Path(scipy.__file__).parent/'optimize/_optimize.py']
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashes={str(p):sha(p) for p in paths}
 checks=[];rejects=[];configs=[];nonfinite=0
 def run(x,q,method,std,op,lam=1):
  nonlocal nonfinite
  cfg=dict(id=len(configs),n=len(x),p=x.shape[1],m=len(q),method=method,standardize=std,op=op,lambda_=str(lam),input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),query_sha256=hashlib.sha256(q.tobytes()).hexdigest());configs.append(cfg)
  stream=io.StringIO();stream.write(f'{len(x)} {x.shape[1]} {len(q)} {op} {int(std)} {method} {lam}\n');np.savetxt(stream,x,fmt='%.17g');np.savetxt(stream,q,fmt='%.17g')
  proc=subprocess.run([str(a.executable)],input=stream.getvalue(),text=True,capture_output=True)
  with warnings.catch_warnings(),np.errstate(all='ignore'):
   warnings.simplefilter('ignore')
   try:
    model=PowerTransformer(method=method,standardize=std)
    if op==0: z=model.fit_transform(x.copy());want={'lambda':model.lambdas_,'train':z,'query':model.transform(q.copy())}
    elif op==1:want={'free':power_transform(x.copy(),method=method,standardize=std)}
    else:
     model.fit(x.copy());model.lambdas_[:]=lam
     want={'scalar':(model.transform if op==2 else model.inverse_transform)(q.copy())}
   except (ValueError,RuntimeError) as e:
    assert proc.returncode!=0,(cfg,'accepted source rejection',str(e))
    assert 'ERROR STOP' in proc.stderr and not any(t in proc.stderr for t in ['SIGFPE','Segmentation fault','Fortran runtime error']), (cfg,proc.stderr)
    diagnostic=('rejects infinity' if np.isinf(x).any() else 'strictly positive' if method=='box-cox' and np.any(x<=0) else 'all-missing feature' if np.isnan(x).all() else 'nonconstant data' if method=='box-cox' else 'could not bracket likelihood optimum')
    assert diagnostic in proc.stdout+proc.stderr,(cfg,diagnostic,proc.stdout,proc.stderr)
    rejects.append(dict(configuration=cfg,source_type=type(e).__name__,source_message=str(e),native_diagnostic=diagnostic,passed=True));return
  assert proc.returncode==0,(cfg,proc.stderr)
  actual={k:np.array(v,float) for k,*v in (line.split() for line in proc.stdout.splitlines())}
  assert actual.keys()==want.keys(),(cfg,actual.keys(),want.keys())
  for k,v in want.items():
   v=np.asarray(v).ravel(order='F');got=actual[k]
   assert got.shape==v.shape,(cfg,k,got.shape,v.shape)
   assert np.array_equal(np.isnan(got),np.isnan(v)) and np.array_equal(np.isposinf(got),np.isposinf(v)) and np.array_equal(np.isneginf(got),np.isneginf(v)),(cfg,k,got,v)
   finite=np.isfinite(v);nonfinite+=int((~finite).sum());err=float(np.max(np.abs(got[finite]-v[finite]),initial=0))
   np.testing.assert_allclose(got[finite],v[finite],rtol=3e-6 if op<2 else 3e-12,atol=3e-6 if op<2 else 3e-12,err_msg=str((cfg,k)))
   checks.append(dict(configuration=cfg['id'],field=k,max_abs_error=err,passed=True))
 rng=np.random.default_rng(4701)
 for method in ['yeo-johnson','box-cox']:
  for n,p in [(7,1),(33,3),(129,2),(260,1)]:
   for dist in range(3):
    x=[rng.normal(size=(n,p)),rng.lognormal(size=(n,p)),rng.uniform(-3,5,size=(n,p))][dist]
    if method=='box-cox':x=np.exp(x*.3)
    for missing in [False,True]:
     train=x.copy()
     if missing:train[1,0]=np.nan
     q=x[:5].copy()
     for std in [False,True]:
      run(train,q,method,std,0);run(train,q,method,std,1)
 # Scalar maps are queried independently: inverse domain results can be nonfinite.
 train=np.array([.1,.5,1,2,4.]).reshape(-1,1)
 for method in ['yeo-johnson','box-cox']:
  for lam in [-20,-3,-1,-.5,-1e-20,0,1e-20,np.finfo(float).eps,.5,1,1.5,2,3,20,np.nan,np.inf,-np.inf]:
   for op in [2,3]:
    q=np.array([-1e200,-10,-2,-1,-.5,-1e-20,0,1e-20,.5,1,2,10,1e200,np.nan]).reshape(-1,1)
    if method=='box-cox' and op==2:q=np.abs(q)+1e-100
    run(train,q,method,False,op,lam)
 for method in ['yeo-johnson','box-cox']:
  for x in [np.ones((5,1)),np.full((5,1),1e-20),np.full((5,1),np.nan),np.array([[1.],[1.],[np.nan]]),np.array([[0.],[1.]]),np.array([[np.inf],[1.]])]:
   for std in [False,True]:run(x,np.ones((2,1)),method,std,0)
 # Retain constrained and unconstrained likelihoods at extreme finite scales.
 for values in [[1,1.47,1.48,1.49,1.5],[1,1.01,1.02,1.03,1.5]]:
  x=np.array(values).reshape(-1,1)*1e300
  run(x,x.copy(),'box-cox',False,0);run(x,x.copy(),'box-cox',False,1)
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,sklearn_version=sklearn.__version__,scipy_version=scipy.__version__,numpy_version=np.__version__,hashes=hashes,configurations=configs,checks=checks,rejections=rejects,source_nonfinite_values=nonfinite),indent=2,allow_nan=False)+'\n')
 print(f'{len(configs)} configurations; {len(checks)} groups; {len(rejects)} source rejections; {nonfinite} nonfinite values PASS')
if __name__=='__main__':main()
