"""Weighted/multi-target/CSR KernelRidge and complete kernel dispatch."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings,os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,sklearn,scipy
from scipy.sparse import csr_matrix,issparse
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from scipy.linalg import lstsq,svdvals
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
import sklearn.kernel_ridge as source_model
import sklearn.linear_model._ridge as source_solve
import sklearn.metrics.pairwise as source_kernels
import scipy.linalg._flapack as source_lapack


def callback(a,b):
 if issparse(a):a=a.toarray().ravel()
 if issparse(b):b=b.toarray().ravel()
 return np.dot(a,b)+.5


def asymmetric_callback(a,b):
 if issparse(a):a=a.toarray().ravel()
 if issparse(b):b=b.toarray().ravel()
 return np.dot(a,b)+.05*np.sum(a)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True)
 parser.add_argument('--reference-cache',type=Path);parser.add_argument('--report',type=Path)
 parser.add_argument('--allow-singular-certificates',action='store_true')
 args=parser.parse_args();root=Path(__file__).resolve().parents[1]
 names=['linear','rbf','poly','sigmoid','cosine','laplacian','additive_chi2','chi2','precomputed','callback','gp','asymmetric']
 provenance=dict(sklearn=sklearn.__version__,scipy=scipy.__version__,numpy=np.__version__,
   files={m.__file__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
          for m in [source_model,source_solve,source_kernels,source_lapack]})
 cache={};cases=[];lines=[];source_calls=cache_hits=source_warnings=0
 if args.reference_cache and args.reference_cache.exists():
  saved=json.loads(args.reference_cache.read_text());assert saved['provenance']==provenance
  cache=saved['cases']

 def add(x,q,y,truth,w,qw,penalty,storage,kind,weight_mode,target_rank,alpha_mode,gamma,coef,label):
  nonlocal source_calls,cache_hits,source_warnings
  n,p=x.shape;m=len(q);t=y.shape[1]
  block=[f'{n} {p} {m} {t} {storage} {kind+1} {weight_mode} {target_rank} {alpha_mode} {gamma:.17g} {coef:.17g}']
  for array in [x,q,y,truth]:block+=[' '.join(map(str,row))for row in array]
  for array in [w,qw,penalty]:block+=[' '.join(map(str,array))]
  key=hashlib.sha256(('\n'.join(block)+'\n').encode()).hexdigest()
  if key not in cache:
   xx=csr_matrix(x)if storage in [1,3]else x
   qq=csr_matrix(q)if storage in [1,2]else q
   kernel=callback if kind==9 else asymmetric_callback if kind==11 else RBF(1.2)+WhiteKernel(.3)if kind==10 else names[kind]
   model=KernelRidge(alpha=penalty[0]if alpha_mode==0 else penalty,kernel=kernel,
                    gamma=None if gamma==-1 else gamma,degree=3,coef0=coef)
   target=y[:,0]if target_rank==0 else y
   weights=None if weight_mode==0 else w if weight_mode==1 else float(w[0])
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    try:
     model.fit(xx,target,sample_weight=weights)
     pred=model.predict(qq);train=model.predict(model.X_fit_)
     score=model.score(qq,truth[:,0]if target_rank==0 else truth,sample_weight=qw)
     failure=None
    except np.linalg.LinAlgError as error:
     failure=dict(type=type(error).__name__,message=str(error))
   if failure is not None:
    cache[key]=dict(failure=failure,warnings=len(caught))
   else:cache[key]=dict(dual=np.asarray(model.dual_coef_).reshape(n,t).ravel(order='F').tolist(),
     pred=np.asarray(pred).reshape(m,t).ravel(order='F').tolist(),
     train=np.asarray(train).reshape(n,t).ravel(order='F').tolist(),score=float(score),
     fallback=any('least-squares solution'in str(w.message)for w in caught),warnings=len(caught))
   source_calls+=1
  else:cache_hits+=1
  label=dict(label,n=n,p=p,m=m,t=t,storage=storage,kernel=names[kind],weight_mode=weight_mode,
    target_rank=target_rank,alpha_mode=alpha_mode,gamma=gamma,coef0=coef,penalty=penalty.tolist())
  cases.append((cache[key],label,block));source_warnings+=cache[key]['warnings']
  if 'failure'not in cache[key]:lines.extend(block)

 def certificate(expected,label,block,actual,used_lstsq):
  assert all(a==0 for a in label['penalty']),label
  n,p,m,t=[label[k]for k in ['n','p','m','t']];cursor=1
  def matrix(rows,cols):
   nonlocal cursor
   out=np.array([np.fromstring(row,sep=' ')for row in block[cursor:cursor+rows]]).reshape(rows,cols)
   cursor+=rows;return out
  x=matrix(n,p);q=matrix(m,p);y=matrix(n,t);truth=matrix(m,t)
  w=np.fromstring(block[cursor],sep=' ');qw=np.fromstring(block[cursor+1],sep=' ')
  sw=np.ones(n)if label['weight_mode']==0 else np.sqrt(w)
  xx=csr_matrix(x)if label['storage']in [1,3]else x
  qq=csr_matrix(q)if label['storage']in [1,2]else q
  metric=callback if label['kernel']=='callback'else asymmetric_callback if label['kernel']=='asymmetric'else RBF(1.2)+WhiteKernel(.3)if label['kernel']=='gp'else label['kernel']
  probe=KernelRidge(kernel=metric,gamma=None if label['gamma']==-1 else label['gamma'],degree=3,coef0=label['coef0'])
  gram=probe._get_kernel(xx);cross=probe._get_kernel(qq,xx)
  a=gram*np.outer(sw,sw);rhs=y*sw[:,None];singular=svdvals(a)
  eps=np.finfo(float).eps
  assert singular[-1]<=16*eps*singular[0],(label,singular)
  reference_ls=lstsq(a,rhs,cond=eps)[0]
  def solve_check(values,fallback):
   dual=np.asarray(values).reshape(n,t,order='F');c=np.zeros_like(dual)
   np.divide(dual,sw[:,None],out=c,where=sw[:,None]!=0)
   if fallback:
    np.testing.assert_allclose(c,reference_ls,rtol=3e-8,atol=3e-8,err_msg=str(label))
    return dict(kind='minimum_norm_lstsq',max_scaled=float(np.max(abs(c-reference_ls)/(1+abs(reference_ls)))))
   # Cholesky's assumption consumes the upper triangle, even if rounded
   # kernel construction has minute asymmetric entries.
   upper=np.triu(a);system=upper+np.triu(a,1).T
   residual=system@c-rhs
   eta=np.max(abs(residual),axis=0)/(np.linalg.norm(system,np.inf)*np.max(abs(c),axis=0)+np.max(abs(rhs),axis=0))
   eta=np.nan_to_num(eta,nan=0.)
   assert np.max(eta)<=512*eps*n,(label,eta)
   return dict(kind='cholesky_backward_error',maximum=float(np.max(eta)))
  source=solve_check(expected['dual'],expected['fallback']);native=solve_check(actual['dual'],used_lstsq)
  dual=actual['dual'].reshape(n,t,order='F')
  for key,kernel in [('pred',cross),('train',gram)]:
   have=actual[key].reshape(kernel.shape[0],t,order='F');want=kernel@dual
   bound=64*eps*n*(abs(kernel)@abs(dual)+1)+3e-8
   assert np.all(abs(have-want)<=bound),(label,key,np.max(abs(have-want)),np.max(bound))
  score=r2_score(truth,actual['pred'].reshape(m,t,order='F'),sample_weight=qw)
  np.testing.assert_allclose(actual['score'],score,rtol=3e-8,atol=3e-8,err_msg=str(label))
  differences={key:float(np.max(abs(actual[key]-np.atleast_1d(expected[key]))))for key in ['dual','pred','train','score']}
  return dict(configuration=label,singular_values=singular.tolist(),source_solve=source,native_solve=native,
    source_outputs={key:expected[key]for key in ['dual','pred','train','score']},
    native_outputs={key:actual[key].tolist()for key in ['dual','pred','train','score']},maximum_absolute_differences=differences)

 for seed in [7,42,2026]:
  rng=np.random.default_rng(seed)
  for n,p in [(13,3),(23,5),(2,1),(1,2)]:
   for data in ['normal','duplicates','zeros']:
    base=rng.normal(size=(n,p));query=rng.normal(size=(3,p))
    if data=='duplicates'and n>1:base[1]=base[0]
    if data=='zeros':base[:]=0
    for kind in range(len(names)):
     x=base.copy();q=query.copy()
     if kind in [6,7]:x=abs(x);q=abs(q)
     if kind==8:
      x=base@base.T+.2*np.eye(n);q=query@base.T
      # Positive upper triangle is consumed by the solve, while prediction
      # consumes the entire supplied matrix.
      if data=='normal':x[np.tril_indices(n,-1)]+=.137
     for storage in ([0]if kind in [6,7,8,10]else range(4)):
      for target_rank,t in [(0,1),(1,1),(1,3)]:
       y=rng.normal(size=(n,t));truth=rng.normal(size=(3,t));qw=np.array([.5,2.,1.])
       for weight_mode in [0,1,2]:
        w=np.ones(n)
        if weight_mode==1:
         w=rng.uniform(.2,2,size=n)
         if n>2:w[1]=0
        if weight_mode==2:w[:]=1.75
        index=len(cases);penalty=np.full(t,[1.,.03,0.][(index//3)%3])
        alpha_mode=0 if(index%2)==0 else 1
        if t>1 and index%5==0:penalty=np.linspace(.2,1.2,t);alpha_mode=1
        gamma=.3 if kind==7 else [-1.,.2,0.][(index//9)%3]
        add(x,q,y,truth,w,qw,penalty,storage,kind,weight_mode,target_rank,alpha_mode,gamma,.7,
            dict(seed=seed,data=data))
 if args.reference_cache:
  args.reference_cache.parent.mkdir(exist_ok=True,parents=True)
  args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 payload='\n'.join(lines)+'\n';proc=subprocess.run([args.executable],input=payload,text=True,capture_output=True)
 if proc.returncode:
  valid=[c for c in cases if 'failure'not in c[0]]
  index=len(proc.stdout.splitlines())//5
  raise RuntimeError(str(valid[min(index,len(valid)-1)][1])+'\n'+proc.stderr+'\nlast output:\n'+proc.stdout[-1400:])
 rows=iter(proc.stdout.splitlines());worst=0;fallbacks=0;rejections=0;successes=0;certificates=[];direct_groups=0;raw_worst=0
 for expected,label,block in cases:
  if 'failure'in expected:
   rejected=subprocess.run([args.executable],input='\n'.join(block)+'\n',text=True,capture_output=True)
   assert rejected.returncode!=0 and 'ERROR STOP'in rejected.stderr and 'SIGFPE'not in rejected.stderr,(label,rejected.stderr)
   rejections+=1;continue
  successes+=1
  meta=next(rows).split()
  assert list(map(int,meta[:3]))==[label[k]for k in ['n','p','t']],label
  assert(meta[3]=='T')==(label['storage']in [1,3]),label
  fallbacks+=meta[4]=='T'
  actual={key:np.fromstring(next(rows),sep=' ')for key in ['dual','pred','train','score']}
  matches={key:np.allclose(actual[key],np.atleast_1d(expected[key]),rtol=3e-8,atol=3e-8,equal_nan=True)for key in actual}
  raw_worst=max(raw_worst,max(float(np.max(abs(actual[key]-np.atleast_1d(expected[key]))/(1+abs(np.atleast_1d(expected[key])))))for key in actual))
  direct_groups+=1 # Fitted metadata remains a direct source comparison.
  if not all(matches.values()):
   if not args.allow_singular_certificates:
    for key in actual:np.testing.assert_allclose(actual[key],np.atleast_1d(expected[key]),rtol=3e-8,atol=3e-8,err_msg=str((label,key)))
   certificates.append(certificate(expected,label,block,actual,meta[4]=='T'))
  else:
   direct_groups+=4
   worst=max(worst,max(float(np.max(abs(actual[key]-np.atleast_1d(expected[key]))/(1+abs(np.atleast_1d(expected[key])))))for key in actual))
 assert next(rows,None)is None
 report=dict(configurations=len(cases),groups=5*successes,successful_configurations=successes,source_rejections=rejections,max_scaled_error=worst,raw_source_max_scaled_difference=raw_worst,direct_groups=direct_groups,certificate_groups=4*len(certificates),certificates=certificates,
   source_warnings=source_warnings,native_fallbacks=fallbacks,source_fallbacks=sum(c[0].get('fallback',False)for c in cases),
   configuration_inventory=[dict(c[1],source_failure=c[0].get('failure'))for c in cases],reference_provenance=provenance,
   source_calls=source_calls,reference_cache_hits=cache_hits,
   reference_cache_sha256=hashlib.sha256(args.reference_cache.read_bytes()).hexdigest()if args.reference_cache else None,
   source_sha256=hashlib.sha256((root/'src/sklearn_kernel_ridge.f90').read_bytes()).hexdigest(),
   validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   executable_sha256=hashlib.sha256(Path(args.executable).read_bytes()).hexdigest(),
   input_sha256=hashlib.sha256(payload.encode()).hexdigest())
 print(json.dumps({k:v for k,v in report.items()if k not in ['configuration_inventory','certificates']},indent=2))
 if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
