"""Barnes-Hut spatial trees, sparse affinities and complete sklearn t-SNE fits."""
import argparse
import hashlib
import json
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='2')
from pathlib import Path
import subprocess
import warnings
import ctypes
import numpy as np
import scipy
import sklearn
from scipy.sparse import csr_matrix
from sklearn.manifold import TSNE
from sklearn.manifold._t_sne import _joint_probabilities_nn,_kl_divergence_bh,_gradient_descent
from sklearn.metrics import pairwise_distances
from sklearn.neighbors._quad_tree import _QuadTree
from sklearn.manifold import _barnes_hut_tsne
from compare_image_graph import fields

ROOT=Path(__file__).resolve().parents[1]
BACKEND_REJECTIONS={
 'angle_nan':'Barnes-Hut angle must be finite','angle_inf':'Barnes-Hut angle must be finite',
 'angle_negative':'Barnes-Hut angle must be in [0,1]','angle_large':'Barnes-Hut angle must be in [0,1]',
 'degrees':'Barnes-Hut degrees must be positive','skip_negative':'invalid Barnes-Hut skipped point count',
 'skip_large':'invalid Barnes-Hut skipped point count','probability_negative':'Barnes-Hut probabilities must be nonnegative',
 'probability_nan':'Barnes-Hut probabilities must be finite','probability_range':'Barnes-Hut probabilities exceed float32 range',
 'probability_shape':'Barnes-Hut probability shape mismatch','position_range':'Barnes-Hut positions exceed working range',
 'dimension':'Barnes-Hut t-SNE requires one to three components','perplexity_zero':'t-SNE requires 0 < perplexity < n_samples',
 'perplexity_large':'t-SNE requires 0 < perplexity < n_samples','distance_negative':'Barnes-Hut distances must be nonnegative',
 'distance_inf':'Barnes-Hut distances must be finite','distance_range':'Barnes-Hut distances exceed float32 range',
 'distance_shape':'Barnes-Hut distances must be square with at least two samples'}
FIT_REJECTIONS={
 'dimensions':'Barnes-Hut t-SNE requires one to three components','angle_nan':'t-SNE angle must be finite',
 'angle_negative':'t-SNE angle must be in [0,1]','angle_large':'t-SNE angle must be in [0,1]',
 'csr_exact':'CSR t-SNE requires method=barnes_hut','csr_metric':'CSR t-SNE requires metric=precomputed',
 'csr_shape':'manifold precomputed distances must be square','csr_neighbors':'not enough stored neighbors for Barnes-Hut t-SNE',
 'csr_pca':'t-SNE PCA initialization requires feature input','csr_negative':'manifold precomputed distances must be nonnegative'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def vector(a):return ' '.join(format(v,'.17g') for v in np.asarray(a).ravel(order='F'))+'\n'
def csr_payload(p):return vector(p.indptr+1)+vector(p.indices+1)+vector(p.data)

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--executable',type=Path,required=True)
 parser.add_argument('--fit-executable',type=Path,required=True)
 parser.add_argument('--profile',default='checked')
 parser.add_argument('--report',type=Path,default=ROOT/'validation/results/tsne_bh.json')
 args=parser.parse_args();checks=[];fixtures=0
 def check(actual,expected,label,field,exact=False,rtol=3e-6,atol=2e-7):
  actual=np.asarray(actual);expected=np.asarray(expected)
  assert actual.shape==expected.shape,(label,field,actual.shape,expected.shape)
  if exact:np.testing.assert_array_equal(actual,expected,err_msg=f'{label} {field}')
  else:np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol,equal_nan=True,err_msg=f'{label} {field}')
  mask=np.isfinite(expected)
  checks.append(dict(fixture=label,field=field,exact=exact,rtol=0 if exact else rtol,atol=0 if exact else atol,
    max_abs_error=float(np.max(np.abs(actual[mask]-expected[mask]),initial=0)),passed=True))
 def run(mode,p,x=None,dof=1,angle=.5,skip=0,compute=True,perplexity=2,optimizer=None):
  n=p.shape[0];d=2 if x is None else x.shape[1]
  payload=vector([mode,n,d,dof,angle,skip,int(compute),perplexity,p.nnz])+csr_payload(p)
  if x is not None:payload+=vector(x)
  if optimizer is not None:payload+=vector(optimizer)
  result=subprocess.run([str(args.executable.resolve())],input=payload,text=True,capture_output=True)
  assert result.returncode==0,result.stderr
  return fields(result.stdout)
 for seed in [0,42,123]:
  rng=np.random.RandomState(seed)
  for n in [2,7,31,100]:
   for k in sorted(set([1,min(4,n-1),n-1,n])):
    idx=np.array([rng.choice(n,k,replace=False) if k==n else rng.choice([j for j in range(n) if j!=i],k,replace=False) for i in range(n)])
    for scale in [0.,1.,1e4]:
     dist=csr_matrix((rng.uniform(0,20,size=n*k)*scale,idx.ravel(),np.arange(0,n*k+1,k)),shape=(n,n))
     perplexity=min(2,n-.5);native=run(0,dist,perplexity=perplexity)
     source=_joint_probabilities_nn(dist.copy(),perplexity,0);label=f'prob/{seed}/{n}/{k}/{scale}'
     check(native[0].ravel(),source.indptr+1,label,'indptr',True)
     check(native[1].ravel(),source.indices+1,label,'indices',True)
     check(native[2].ravel(),source.data,label,'probabilities',rtol=2e-14,atol=2e-16);fixtures+=1
  for d in [1,2,3]:
   for n in [2,7,31,100]:
    mat=rng.uniform(size=(n,n));mat+=mat.T;np.fill_diagonal(mat,0);mat/=mat.sum();p=csr_matrix(mat)
    for pattern in ['tiny','normal','large','duplicate','collapsed']:
     scale={'tiny':1e-4,'normal':1.,'large':20.,'duplicate':1.,'collapsed':0.}[pattern]
     x=(rng.normal(size=(n,d))*scale).astype(np.float32)
     if pattern=='duplicate':x[-1]=x[0]
     for angle in [0.,.2,.5,.8,1.]:
      # Source's summary buffer has n-skip slots. One duplicate pair makes
      # skip=1 safe even at angle zero (at most n-1 occupied leaves).
      skip=0 if pattern!='duplicate' else 1;compute=pattern!='tiny';dof=max(d-1,1)
      native=run(1,p,x,dof,angle,skip,compute)
      error,grad=_kl_divergence_bh(x.ravel(),p,dof,n,d,angle=angle,skip_num_points=skip,compute_error=compute,num_threads=1)
      if skip:
       # sklearn's Python wrapper forgets to forward skip_num_points.
       # Exercise the compiled source backend directly for this helper option.
       grad=np.zeros_like(x)
       error=_barnes_hut_tsne.gradient(p.data.astype(np.float32),x,p.indices.astype(np.int64),
         p.indptr.astype(np.int64),grad,angle,d,0,dof=dof,skip_num_points=skip,compute_error=compute,num_threads=1)
       grad=grad.ravel();grad*=2.*(dof+1.)/dof
      label=f'force/{seed}/{d}/{n}/{pattern}/{angle}'
      check(native[0][0,0],error,label,'objective')
      check(native[1],grad.reshape(n,d),label,'gradient')
      tree=_QuadTree(d,0);tree.build_tree(x);cells=tree.__getstate__()['cells']
      expected=np.column_stack([cells['parent']+1,cells['children']+1,cells['point_index']+1,cells['depth'],cells['cumulative_size'],cells['is_leaf']])
      expected[:,1+2**d:9]=0;check(native[2],expected,label,'tree_indices',True)
      expected=np.column_stack([cells['min_bounds'][:,:d],cells['max_bounds'][:,:d],cells['center'][:,:d],cells['barycenter'][:,:d],cells['squared_max_width']])
      check(native[3],expected,label,'tree_bounds_centers_masses',True);fixtures+=1
  print(f'seed {seed}: sparse affinities and spatial trees passed',flush=True)
 # Short independent source optimizer comparisons include both NumPy scalar-rate paths.
 rng=np.random.RandomState(73)
 for d in [1,2,3]:
  n=11;mat=rng.uniform(size=(n,n));mat+=mat.T;np.fill_diagonal(mat,0);mat/=mat.sum();p=csr_matrix(mat)
  for single in [False,True]:
   for auto in [False,True]:
    x=rng.normal(scale=.02,size=(n,d)).astype(np.float32 if single else np.float64)
    rate=np.float64(.5) if auto else .5;steps=12;dof=max(d-1,1)
    native=run(2,p,x,dof,optimizer=[steps,0,300,rate,.5,0,int(single),int(auto)])
    source,error,iteration=_gradient_descent(_kl_divergence_bh,x.ravel(),0,steps,n_iter_check=50,
      n_iter_without_progress=300,momentum=.5,learning_rate=rate,min_grad_norm=0,
      args=[p,dof,n,d],kwargs=dict(angle=.5,num_threads=1))
    label=f'optimizer/{d}/{single}/{auto}'
    check(native[0],source.reshape(n,d),label,'embedding',rtol=3e-6,atol=2e-8)
    check(native[1][0,0],error,label,'objective');check(native[2][0,0],iteration,label,'iteration',True)
    check(native[3][-1,:n*d],native[0].ravel(),label,'trace_final_position',True);fixtures+=1
 # Full trajectories are deliberately strict: no alternative-oracle acceptance.
 for seed in [0,42,123]:
  rng=np.random.RandomState(seed)
  for mode in [0,1,2]:
   for d in [1,2,3]:
    for controlled in [True,False]:
     n=17;x=rng.normal(size=(n,5));initial=rng.normal(scale=.02,size=(n,d))
     perp=4.;early=1. if controlled else 12.;rate=.5 if controlled else -1.
     full_fit(x,initial,mode,seed,0,0,perp,early,rate,.5,300,0.,run=None,args=args,check=check)
     fixtures+=1
  # Extra metrics, truncated sparse neighborhoods, stopping and iteration boundaries.
  for metric,kind,angle,steps,minnorm in [(1,0,0.,300,0.),(2,0,1.,300,0.),(3,0,.5,300,0.),
       (3,1,.2,300,0.),(3,1,.8,250,0.),(0,0,.5,350,1e10)]:
   n=31;x=rng.normal(size=(n,5));initial=rng.normal(scale=.02,size=(n,2))
   if metric==3:
    x=pairwise_distances(x)
    if kind:x=csr_matrix((x.ravel(),np.tile(np.arange(n),n),np.arange(0,n*n+1,n)),shape=(n,n))
   full_fit(x,initial,1,seed,metric,kind,2.,1.,.5,angle,steps,minnorm,run=None,args=args,check=check);fixtures+=1
  print(f'seed {seed}: complete Barnes-Hut fits passed',flush=True)
 rejections=[]
 for executable,names in [(args.executable,BACKEND_REJECTIONS),(args.fit_executable,FIT_REJECTIONS)]:
  for name,message in names.items():
   result=subprocess.run([str(executable.resolve()),name],text=True,capture_output=True)
   output=result.stdout+result.stderr
   assert result.returncode!=0 and 'unknown Barnes-Hut' not in output and 'input accepted' not in output,(name,output)
   assert 'fscikit-learn: '+message in output and 'SIGFPE' not in output and 'Fortran runtime error' not in output,(name,output)
   rejections.append(dict(driver=executable.name,case=name,passed=True))
 from sklearn.neighbors import _quad_tree
 capsule=_quad_tree.__pyx_capi__['EPSILON'];pointer=ctypes.pythonapi.PyCapsule_GetPointer
 pointer.restype=ctypes.c_void_p;pointer.argtypes=[ctypes.py_object,ctypes.c_char_p]
 epsilon=ctypes.c_float.from_address(pointer(capsule,b'float')).value
 assert epsilon==0.,epsilon
 source_root=Path(sklearn.__file__).parent
 source_paths=['manifold/_t_sne.py','manifold/_utils.pyx','manifold/_barnes_hut_tsne.pyx','neighbors/_quad_tree.pyx','neighbors/_quad_tree.pxd']
 report=dict(sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,profile=args.profile,
   sklearn_source_hashes={p:digest(source_root/p) for p in source_paths},duplicate_epsilon=epsilon,
   reference_binary_hashes={m.__file__:digest(m.__file__) for m in [_quad_tree,_barnes_hut_tsne]},
   validator_sha256=digest(__file__),
   fixtures=fixtures,checks=checks,rejections=rejections,executables={str(p):digest(p) for p in [args.executable,args.fit_executable]},
   source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'src/sklearn_tsne.f90',ROOT/'src/sklearn_tsne_core.f90',ROOT/'src/sklearn_tsne_bh.f90']})
 args.report.parent.mkdir(exist_ok=True,parents=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
 print(f'{len(checks)} result groups across {fixtures} fixtures and {len(rejections)} rejections passed',flush=True)

def full_fit(x,initial,mode,seed,metric,kind,perp,early,rate,angle,steps,minnorm,*,run,args,check):
 n,p=x.shape;d=initial.shape[1]
 header=[kind,n,p,d,mode,steps,300,seed,metric,perp,early,rate,minnorm,angle,1]
 payload=vector(header)+(vector([x.nnz])+csr_payload(x) if kind else vector(x))+vector(initial)
 proc=subprocess.run([str(args.fit_executable.resolve())],input=payload,text=True,capture_output=True)
 assert proc.returncode==0,proc.stderr
 native=fields(proc.stdout)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  source=TSNE(n_components=d,init=initial if mode==0 else ['','random','pca'][mode],perplexity=perp,
    early_exaggeration=early,learning_rate=rate if rate>0 else 'auto',max_iter=steps,min_grad_norm=minnorm,
    random_state=seed,method='barnes_hut',angle=angle,metric=['euclidean','manhattan','cosine','precomputed'][metric],n_jobs=1).fit(x.copy())
 label=f'fit/{seed}/{mode}/{d}/{early}/{metric}/{kind}/{angle}/{steps}/{minnorm}'
 check(native[0],source.embedding_,label,'embedding',rtol=3e-8,atol=3e-8)
 check(native[1].ravel(),[source.n_features_in_,source.n_iter_],label,'metadata',True)
 check(native[2].ravel(),[source.kl_divergence_,source.learning_rate_],label,'diagnostics',rtol=3e-8,atol=3e-8)
 check(native[3][0,0],1,label,'input_preservation_and_refit',True)

if __name__=='__main__':main()
