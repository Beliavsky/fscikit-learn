"""Isomap radius/distance graph fitting, shortest paths and projection vs sklearn."""
import argparse,hashlib,json,os,subprocess,warnings
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='2')
import numpy as np
import scipy,sklearn
from scipy.sparse import csr_matrix,issparse
from scipy.sparse.csgraph import connected_components,minimum_spanning_tree
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from compare_image_graph import fields

REJECTIONS={
 'neighbors_negative':'Isomap n_neighbors must be nonnegative','neighbors_large':'Isomap requires 0 < n_neighbors < n_samples',
 'both':'either n_neighbors or radius','neither':'radius must be nonnegative','radius_negative':'invalid Isomap radius',
 'radius_nan':'Isomap radius is NaN','components':'invalid Isomap n_components','path':'invalid Isomap path_method',
 'p_inf':'Isomap p must be finite','radius_inf':'Isomap radius must be finite',
 'p_small':'Isomap p must be at least one','p_nan':'Isomap p is NaN','pre_shape':'precomputed fit distances must be square',
 'pre_negative':'Isomap distances must be nonnegative','pre_nan':'X contains NaN or infinity',
 'csr_metric':'CSR Isomap requires metric=precomputed','csr_shape':'precomputed fit distances must be square',
 'csr_neighbors':'not enough stored neighbors for Isomap','csr_disconnected':'disconnected precomputed CSR Isomap graph',
 'unfitted_names':'Isomap is not fitted','names_length':'Isomap input feature name count mismatch',
 'unfitted':'Isomap is not fitted','unfitted_csr':'Isomap is not fitted','unfitted_error':'Isomap is not fitted',
 'query_width':'Isomap query feature count mismatch','query_negative':'Isomap distances must be nonnegative',
 'query_empty_radius':'Isomap query has no neighbors within radius','query_csr_metric':'CSR Isomap query requires metric=precomputed',
 'query_csr_width':'Isomap precomputed distance shape mismatch','query_csr_neighbors':'not enough stored neighbors for Isomap',
 'query_inf':'X contains NaN or infinity',
}
METRICS=['euclidean','manhattan','minkowski','chebyshev','precomputed']
METHODS=['auto','FW','D']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sparse_values(a,mask):
 rows,cols=np.where(mask)
 return csr_matrix((a[rows,cols],(rows,cols)),shape=a.shape)

def fixtures():
 for seed in [0,42,2026]:
  rng=np.random.RandomState(seed);x=rng.normal(size=(24,4));query=x[:7]+rng.normal(scale=.01,size=(7,4))
  for metric,power in [('euclidean',2.),('manhattan',2.),('minkowski',1.5)]:
   d=pairwise_distances(x,metric=metric,**({'p':power} if metric=='minkowski' else {}))
   q=pairwise_distances(query,x,metric=metric,**({'p':power} if metric=='minkowski' else {}))
   radius=float(minimum_spanning_tree(d).data.max()*1.1)
   for k,r in [(5,-1.),(0,radius)]:
    for method in METHODS:
     configs=[(0,0,x,query,metric)]
     for kind in [1,2]:
      if kind==2:
       mask=np.zeros_like(d,dtype=bool)
       if k:
        np.put_along_axis(mask,np.argsort(d,axis=1)[:,:12],True,axis=1)
       else:mask=d<=r
       train=sparse_values(d,mask)
      else:train=d
      for query_kind in [0,1]:
       target=q if query_kind==0 else sparse_values(q,np.ones_like(q,dtype=bool) if k else q<=r)
       configs.append((kind,query_kind,train,target,'precomputed'))
     for kind,qkind,train,target,effective in configs:
      yield dict(seed=seed,case='ordinary',kind=kind,query_kind=qkind,k=k,radius=r,method=method,
                 metric=effective,p=power,c=2,train=train,query=target)
 for kind in [0,1,2]:
  train=np.array([[2.]]) if kind==0 else np.array([[0.]])
  if kind==2:train=sparse_values(train,np.ones((1,1),dtype=bool))
  yield dict(seed=0,case='singleton',kind=kind,query_kind=0,k=0,radius=0.,method='D',metric='euclidean' if kind==0 else 'precomputed',p=2.,c=1,train=train,query=np.array([[2.]]) if kind==0 else np.array([[0.]]))
 # Disconnected components, asymmetric/nonzero diagonal distances, ties and zero edges.
 for case in ['disconnected','duplicates','asymmetric','diagonal','zero']:
  rng=np.random.RandomState(73);x=rng.normal(size=(12,3))
  if case=='disconnected':x[:6]-=10;x[6:]+=10
  if case=='duplicates':x[::2]=x[1::2]
  if case=='zero':x[:]=0
  d=pairwise_distances(x);q=d[:4].copy()
  if case=='asymmetric':d*=rng.uniform(.7,1.3,size=d.shape);q=d[:4].copy()
  if case=='diagonal':np.fill_diagonal(d,100);q=d[:4].copy()
  for method in METHODS:
   for kind in [0,1,2]:
    if case in ['asymmetric','diagonal'] and kind==0:continue
    for radius_mode in [False,True]:
     k=0 if radius_mode else (1 if case=='disconnected' else 11)
     r=0. if radius_mode else -1.
     train=x if kind==0 else (d if kind==1 else sparse_values(d,np.ones_like(d,dtype=bool)))
     target=x[:4] if kind==0 else q
     yield dict(seed=73,case=case,kind=kind,query_kind=0,k=k,radius=r,method=method,
                metric='euclidean' if kind==0 else 'precomputed',p=2.,c=2,train=train,query=target)
 # Preserve unsorted duplicate CSR columns as separate stored neighbor candidates.
 rng=np.random.RandomState(96);x=rng.normal(size=(12,3));d=pairwise_distances(x)
 pointers=[0];indices=[];values=[]
 for row in range(12):
  order=rng.permutation(12)
  for col in order:
   indices.append(col);values.append(d[row,col])
   if col==(row+1)%12:indices.append(col);values.append(d[row,col]*.75)
  pointers.append(len(indices))
 train=csr_matrix((values,indices,pointers),shape=d.shape)
 for method in METHODS:
  for k,r in [(3,-1.),(0,float(d.max()))]:
   yield dict(seed=96,case='duplicate_csr',kind=2,query_kind=1,k=k,radius=r,method=method,
              metric='precomputed',p=2.,c=2,train=train,query=train[:3].copy())
 # Invalid infinite power, oversized components and the sparse auto-method branch.
 for power in [1.,np.inf]:
  x=np.random.RandomState(8).normal(size=(15,3))
  yield dict(seed=8,case='power',kind=0,query_kind=0,k=4,radius=-1.,method='D',metric='minkowski',p=power,
             c=2,train=x,query=x[:3])
 x=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
 yield dict(seed=0,case='oversized',kind=0,query_kind=0,k=3,radius=-1.,method='FW',metric='euclidean',p=2.,
            c=8,train=x,query=x)
 for method in ['auto','D']:
  x=np.arange(100,dtype=float)[:,None]**1.01
  yield dict(seed=0,case='auto_sparse',kind=2,query_kind=0,k=3,radius=-1.,method=method,metric='precomputed',p=2.,
             c=1,train=sparse_values(pairwise_distances(x),np.abs(np.arange(100)[:,None]-np.arange(100))<=4),
             query=pairwise_distances(x[:3],x))

def encode(a):
 if issparse(a):
  return '\n'.join([' '.join(map(str,[*a.shape,a.nnz])), ' '.join(map(str,a.indptr+1)),
                    ' '.join(map(str,a.indices+1)), ' '.join(format(v,'.17g') for v in a.data)])
 return ' '.join(format(v,'.17g') for v in np.asarray(a).ravel(order='F'))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path,required=True)
 parser.add_argument('--profile',choices=['development','checked','release'],required=True)
 parser.add_argument('--report',type=Path,required=True);args=parser.parse_args();exe=args.executable.resolve()
 checks=[];inventories=[];source_errors=[];roundoff=[];warns={};cases=list(fixtures())
 for index,cfg in enumerate(cases):
  x=cfg['train'];query=cfg['query'];n,p=x.shape;m,width=query.shape;mutate=index%11==0
  options=dict(n_neighbors=cfg['k'] or None,radius=None if cfg['radius']<0 else cfg['radius'],
               n_components=cfg['c'],path_method=cfg['method'],metric=cfg['metric'],p=cfg['p'],
               eigen_solver='dense',neighbors_algorithm='brute')
  stage='fit';error=None
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always')
   try:
    model=Isomap(**options).fit(x)
    neighbor_graph=(model.nbrs_.kneighbors_graph(mode='distance') if cfg['k'] else model.nbrs_.radius_neighbors_graph(mode='distance'))
    components=connected_components(neighbor_graph)[0]
    if mutate:
     if model.n_neighbors is not None:model.n_neighbors=1
     model.radius=0;model.metric='cosine';model.p=99
    stage='query';projected=model.transform(query);again=model.transform(x)
    reconstruction=model.reconstruction_error()
   except (ValueError,RuntimeError) as e:error=e
   for w in caught:warns[str(w.message)]=warns.get(str(w.message),0)+1
  header=[cfg['kind'],cfg['query_kind'],n,p,m,width,cfg['k'],cfg['c'],cfg['radius'],METHODS.index(cfg['method']),
          METRICS.index(cfg['metric']),cfg['p'],int(mutate)]
  payload=' '.join(map(str,header))+'\n'+encode(x)+'\n'+encode(query)+'\n'
  proc=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
  cfg_info={k:v for k,v in cfg.items() if k not in ['train','query']};cfg_info['p']=str(cfg_info['p'])
  if error:
   message='Isomap p must be finite' if "'p' parameter of Isomap" in str(error) else (
      'disconnected precomputed CSR Isomap graph' if isinstance(error,RuntimeError) else (
      'Isomap query has no neighbors within radius' if stage=='query' else 'kernel PCA selected significant negative eigenvalues'))
   assert proc.returncode!=0 and message in proc.stdout+proc.stderr,(index,cfg_info,str(error),proc.stderr)
   assert 'SIGFPE' not in proc.stderr and 'Fortran runtime error' not in proc.stderr,(index,proc.stderr)
   source_errors.append(dict(fixture=index,stage=stage,error=str(error)));names=['source_'+stage+'_rejection']
   checks.append(dict(fixture=index,field=names[0],max_abs_error=0.))
  else:
   assert proc.returncode==0,(index,cfg_info,proc.stderr)
   actual=fields(proc.stdout);assert len(actual)==11,(index,len(actual))
   metadata,d,z,q,t,e,v,means,err,preserved,name_codes=actual;e=e.ravel();names=[]
   def check(name,a,b,atol=3e-8,rtol=3e-9):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(index,name,a.shape,b.shape)
    np.testing.assert_allclose(a,b,atol=atol,rtol=rtol,err_msg=f'{index} {cfg_info} {name}')
    checks.append(dict(fixture=index,field=name,max_abs_error=float(np.max(np.abs(a-b),initial=0))));names.append(name)
   check('metadata',metadata,np.array([p,n,model.embedding_.shape[1],components])[:,None],0,0)
   check('distances',d,model.dist_matrix_,2e-9,2e-11)
   check('embedding_gram',z@z.T,model.embedding_@model.embedding_.T)
   check('query_gram',q@q.T,projected@projected.T)
   check('query_cross',q@z.T,projected@model.embedding_.T)
   check('training_projection_gram',t@t.T,again@again.T)
   check('training_projection_cross',t@z.T,again@model.embedding_.T)
   check('eigenvalues',e,model.kernel_pca_.eigenvalues_)
   positive=e>0;ref_positive=model.kernel_pca_.eigenvalues_>0
   check('positive_eigenspace',v[:,positive]@v[:,positive].T,
         model.kernel_pca_.eigenvectors_[:,ref_positive]@model.kernel_pca_.eigenvectors_[:,ref_positive].T)
   check('kernel_center_rows',means,model.kernel_pca_._centerer.K_fit_rows_[:,None])
   g=-.5*model.dist_matrix_**2;g=g-g.mean(0)[None,:]-g.mean(1)[:,None]+g.mean()
   mass=np.sum(g*g);spectral=np.sum(model.kernel_pca_.eigenvalues_**2)
   residual=max(0.,mass-spectral)/n**2;bound=256*np.finfo(float).eps*max(1.,mass,spectral)/n**2
   assert np.isfinite(err[0,0]) and err[0,0]>=0
   check('squared_reconstruction_error',err**2,np.array([[residual]]),bound,2e-9)
   if not np.isfinite(reconstruction):roundoff.append(dict(fixture=index,source_residual=float(mass-spectral),bound=bound))
   check('input_preservation',preserved,np.ones((1,1),dtype=int),0,0)
   check('feature_names',name_codes,np.array([[ord(c) for c in s.ljust(48)] for s in model.get_feature_names_out()]),0,0)
  inventories.append(dict(fixture=index,**cfg_info,mutate=mutate,fields=names))
  if index%50==0:print(index,'/',len(cases),'fixtures passed',flush=True)
 for case,message in REJECTIONS.items():
  proc=subprocess.run([str(exe),case],text=True,capture_output=True)
  assert proc.returncode!=0 and message in proc.stdout+proc.stderr,(case,proc.stdout,proc.stderr)
  assert 'SIGFPE' not in proc.stderr and 'Fortran runtime error' not in proc.stderr,(case,proc.stderr)
 source=Path(sklearn.__file__).parent
 report=dict(profile=args.profile,fixtures=len(cases),checks=checks,inventories=inventories,rejections=list(REJECTIONS),
  source_errors=source_errors,source_roundoff=roundoff,source_warnings=warns,sklearn_version=sklearn.__version__,
  numpy_version=np.__version__,scipy_version=scipy.__version__,executable=str(exe),executable_sha256=sha(exe),
  validator_sha256=sha(__file__),parser_sha256=sha(Path(__file__).with_name('compare_image_graph.py')),
  source_sha256={s:sha(source/s) for s in ['manifold/_isomap.py','utils/graph.py','neighbors/_base.py','decomposition/_kernel_pca.py','base.py']})
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print('PASS:',len(cases),'fixtures,',len(checks),'groups,',len(REJECTIONS),'diagnostics')
if __name__=='__main__':main()
