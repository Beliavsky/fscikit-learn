"""KernelDensity kernels, metric parameters and seeded mixture draws vs sklearn."""
import argparse,hashlib,json,os,subprocess,warnings
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='2')
import numpy as np
import scipy,sklearn
from sklearn.neighbors import KernelDensity
import sklearn.neighbors._kd_tree as source_kd
import sklearn.neighbors._ball_tree as source_ball
import scipy.special._ufuncs as source_special
from compare_image_graph import fields
KERNELS=['gaussian','tophat','epanechnikov','exponential','linear','cosine']
METRICS=['euclidean','l2','minkowski','p','manhattan','cityblock','l1','chebyshev','infinity']
REJECTIONS={
 'algorithm':'invalid density algorithm','leaf_size':'density leaf_size must be positive',
 'atol_negative':'density tolerances must be nonnegative','rtol_negative':'density tolerances must be nonnegative',
 'atol_nan':'density tolerances must be finite','rtol_inf':'density tolerances must be finite',
 'bandwidth_zero':'density bandwidth must be positive','bandwidth_negative':'density bandwidth must be positive',
 'bandwidth_nan':'density bandwidth must be positive','bandwidth_inf':'density bandwidth must be positive',
 'rule':'unknown density bandwidth rule','kernel':'unknown density kernel','metric':'unsupported density metric',
 'power_zero':'density Minkowski p must be positive','power_negative':'density Minkowski p must be positive',
 'power_nan':'density Minkowski p is NaN','weighted_power_inf':'weighted density Minkowski p must be finite',
 'metric_weights_type':'density metric weights require Minkowski','metric_weights_length':'density metric weight count mismatch',
 'metric_weights_negative':'negative density metric weights','metric_weights_nan':'density metric weights must be finite',
 'weights_negative':'invalid density sample weights','weights_nan':'nonfinite density sample weights',
 'weights_inf':'nonfinite density sample weights','weights_length':'density sample weight size mismatch',
 'unfitted_tree':'KernelDensity is not fitted','unfitted_scores':'KernelDensity is not fitted','unfitted_sample':'KernelDensity is not fitted',
 'query_width':'density query feature count mismatch','query_nan':'X contains NaN or infinity',
 'sample_negative':'density sample count must be nonnegative','sample_seed':'density sample seed must be nonnegative',
 'sample_kernel':'density sampling supports gaussian/tophat only','rng_unseeded':'legacy RNG is not seeded',
}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fixtures():
 # Main metric families, rules, weighted fits and representative dimensions.
 configs=[('euclidean',2,False),('minkowski',.5,False),('minkowski',3.25,False),('minkowski',2,True),
          ('minkowski',.75,True),('minkowski',np.inf,False),('manhattan',2,False),('chebyshev',2,False)]
 for seed in [0,42,2026]:
  rng=np.random.RandomState(seed)
  for n,p in [(11,1),(24,3),(100,8)]:
   x=rng.normal(size=(n,p));query=np.vstack([x[:3],rng.normal(size=(4,p)),np.full((1,p),30.)])
   w=rng.uniform(.05,3,size=n);w[::5]=0;fw=rng.uniform(.1,2,size=p);fw[0]=0
   for kernel in KERNELS:
    for metric,power,weighted_features in configs:
     for weighted in [False,True]:
      yield dict(seed=seed,case='ordinary',x=x,q=query,w=w,fw=fw,weighted=weighted,feature_weighted=weighted_features,
        kernel=kernel,metric=metric,power=power,bandwidth=.65 if weighted else 1.7,rule=(seed+p+int(weighted))%3,
        draws=[0,1,7,19][(seed+p)%4],mutate=int((seed+p)%7==0))
 # Exact metric aliases, small fractional orders and negative infinite alias behavior.
 rng=np.random.RandomState(71);x=rng.normal(size=(9,2));q=x[:4]+.01;w=np.arange(9,dtype=float);fw=np.ones(2)
 for metric,power in [('l2',2),('p',1.5),('cityblock',2),('l1',2),('infinity',2),('minkowski',-np.inf)]:
  for kernel in KERNELS:
   yield dict(seed=71,case='aliases',x=x,q=q,w=w,fw=fw,weighted=True,feature_weighted=False,
      kernel=kernel,metric=metric,power=power,bandwidth=1.,rule=0,draws=-2,mutate=0)
 # IEEE/boundary states and cosine's source normalization, including dimensions > 8.
 for case in ['zero_weights','tiny_weights','overflow_weights','single_weight','tiny_bandwidth','huge_bandwidth',
              'constant_tiny','boundary','cosine16','cosine32','cosine64','gamma128','gamma1024','changed_kernel']:
  p=16 if case=='cosine16' else (32 if case=='cosine32' else (64 if case=='cosine64' else (
      128 if case=='gamma128' else (1024 if case=='gamma1024' else 2))))
  n=1 if case in ['constant_tiny','boundary'] else 5
  rng=np.random.RandomState(95);x=rng.normal(scale=.1,size=(n,p));q=x[:3].copy();w=np.ones(n);bw=1.
  if case=='zero_weights':w[:]=0
  if case=='tiny_weights':w[:]=1.e-320;w[0]=0
  if case=='overflow_weights':w[:]=np.finfo(float).max;w[0]=0
  if case=='single_weight':w[:]=0;w[-1]=1
  if case in ['tiny_bandwidth','constant_tiny']:bw=1.e-200
  if case=='huge_bandwidth':bw=1.e200
  if case=='constant_tiny':x[:]=0;q[:]=0
  if case=='boundary':
   x[:]=0;q=np.zeros((5,p));q[:,0]=[0.,1.,np.nextafter(1.,0.),np.nextafter(1.,np.inf),-1.]
  for kernel in KERNELS if not case.startswith('gamma') else ['gaussian','tophat']:
   yield dict(seed=95,case=case,x=x,q=q,w=w,fw=np.ones(p),weighted=case.endswith('weights') or case=='single_weight',
      feature_weighted=False,kernel=kernel,metric='euclidean',power=2,bandwidth=bw,rule=0,draws=5,
      mutate=2 if case=='changed_kernel' else 0)

def backend_fixtures():
 rng=np.random.RandomState(101);x=rng.normal(size=(129,3));q=np.vstack([x[:3],rng.normal(size=(5,3))]);w=rng.uniform(.1,2,size=129)
 w[::7]=0;fw=np.array([.25,2.,0.])
 for algorithm in ['kd_tree','ball_tree']:
  for breadth in [False,True]:
   for leaf in [1,7,40]:
    for kernel in KERNELS:
     for feature_weighted in [False,True]:
      yield dict(seed=101,case='tree',x=x,q=q,w=w,fw=fw,weighted=True,feature_weighted=feature_weighted,
        kernel=kernel,metric='minkowski' if feature_weighted else 'euclidean',power=2.,bandwidth=.8,rule=0,draws=7,mutate=0,
        algorithm=algorithm,leaf_size=leaf,breadth=breadth,atol=0.,rtol=0.)
 # Approximate queries use the same public tolerance contract and tree bounds.
 for algorithm in ['kd_tree','ball_tree']:
  for breadth in [False,True]:
   for kernel in KERNELS:
    yield dict(seed=101,case='approximate',x=x,q=q,w=w,fw=fw,weighted=True,feature_weighted=False,
      kernel=kernel,metric='euclidean',power=2.,bandwidth=.8,rule=0,draws=0,mutate=0,
      algorithm=algorithm,leaf_size=7,breadth=breadth,atol=1.e-5,rtol=1.e-3)

def compare(actual,expected,index,field):
 expected=np.asarray(expected);assert actual.shape==expected.shape,(index,field,actual.shape,expected.shape)
 for classify in [np.isnan,np.isposinf,np.isneginf]:
  np.testing.assert_array_equal(classify(actual),classify(expected),err_msg=f'{index} {field}')
 finite=np.isfinite(expected)
 np.testing.assert_allclose(actual[finite],expected[finite],rtol=3e-11,atol=3e-12,err_msg=f'{index} {field}')
 return float(np.max(np.abs(actual[finite]-expected[finite]),initial=0))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path,required=True)
 parser.add_argument('--profile',choices=['development','checked','release'],required=True)
 parser.add_argument('--report',type=Path,required=True);parser.add_argument('--start-fixture',type=int,default=0)
 args=parser.parse_args();exe=args.executable.resolve();cases=list(fixtures())+list(backend_fixtures());checks=[];inventory=[];nonfinite=[];warns={};certificates=[]
 for index,cfg in enumerate(cases):
  if index<args.start_fixture:continue
  for k,v in dict(algorithm='auto',leaf_size=40,breadth=True,atol=0.,rtol=0.).items():cfg.setdefault(k,v)
  x=cfg['x'];q=cfg['q'];w=cfg['w'];fw=cfg['fw'];n,p=x.shape
  params=None
  if cfg['metric'] in ['minkowski','p']:
   params=dict(p=cfg['power'])
   if cfg['feature_weighted']:params['w']=fw
  bandwidth=cfg['bandwidth'] if cfg['rule']==0 else ['fixed','scott','silverman'][cfg['rule']]
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always')
   options=dict(kernel=cfg['kernel'],metric=cfg['metric'],metric_params=params,bandwidth=bandwidth,
     algorithm=cfg['algorithm'],leaf_size=cfg['leaf_size'],breadth_first=cfg['breadth'],atol=cfg['atol'],rtol=cfg['rtol'])
   model=KernelDensity(**options).fit(
      x,sample_weight=w if cfg['weighted'] else None)
   raw=w if cfg['weighted'] else np.ones(n);normalized=raw.copy()
   if raw.max()>0:normalized=raw/raw.max();normalized/=normalized.sum()
   if cfg['mutate']:
    model.metric='invalid';model.metric_params={'p':-1,'w':-fw};model.bandwidth=-1
   if cfg['mutate']==2:model.kernel='cosine'
   rng=np.random.RandomState(cfg['seed']);rng.normal()
   if model.kernel in ['gaussian','tophat']:
    kwargs={} if cfg['draws']<0 else dict(n_samples=cfg['draws'])
    a=model.sample(random_state=cfg['seed'],**kwargs);b=model.sample(random_state=cfg['seed'],**kwargs)
    c=model.sample(3,random_state=rng);d=model.sample(4,random_state=rng)
   else:a=b=c=d=np.empty((0,p))
   after=np.array([rng.uniform(),rng.normal()])[:,None]
   expected=[('metadata',np.array([p,n,len(model.tree_.get_arrays()[2]),1 if type(model.tree_).__name__.startswith('KDTree') else 2])[:,None]),('bandwidth_mass',np.array([model.bandwidth_,model.tree_.sum_weight])[:,None]),
    ('legacy_weight_view',normalized[:,None]),('raw_weights',raw[:,None]),('training_scores',model.score_samples(x)[:,None]),
    ('query_scores',model.score_samples(q)[:,None]),('total_scores',np.array([model.score(x),model.score(q)])[:,None]),
    ('seeded_samples',a),('repeated_seed_samples',b),('shared_rng_first',c),('shared_rng_second',d),
    ('shared_rng_next',after),('input_preservation',np.ones((1,1),dtype=int))]
   tree_data,tree_order,tree_nodes,tree_bounds=model.tree_.get_arrays()
   spans=np.column_stack([tree_nodes['idx_start']+1,tree_nodes['idx_end'],tree_nodes['is_leaf']])
   bounds=np.zeros((2,len(tree_nodes),p));bounds[:tree_bounds.shape[0]]=tree_bounds
   masses=np.array([sum(raw[tree_order[a['idx_start']:a['idx_end']]]) for a in tree_nodes])[:,None]
   expected.extend([('tree_partitions',(tree_order+1)[:,None]),('tree_spans',spans),
    ('tree_bounds',bounds.reshape((2*len(tree_nodes),p),order='F')),('tree_radii',tree_nodes['radius'][:,None]),('tree_masses',masses)])
   for item in caught:warns[str(item.message)]=warns.get(str(item.message),0)+1
  header=[n,p,len(q),KERNELS.index(cfg['kernel']),METRICS.index(cfg['metric']),cfg['power'],cfg['bandwidth'],
    cfg['rule'],int(cfg['weighted']),int(cfg['feature_weighted']),cfg['draws'],cfg['seed'],cfg['mutate'],
    ['auto','kd_tree','ball_tree'].index(cfg['algorithm']),cfg['leaf_size'],int(cfg['breadth']),cfg['atol'],cfg['rtol']]
  payload=' '.join(map(str,header))+'\n'+'\n'.join(' '.join(format(v,'.17g') for v in arr.ravel(order='F'))
                                               for arr in [x,q,w,fw])+'\n'
  proc=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
  assert proc.returncode==0,(index,cfg['case'],proc.stderr)
  actual=fields(proc.stdout);assert len(actual)==len(expected),(index,len(actual),len(expected))
  np.testing.assert_array_equal(actual[7],actual[8])
  assert actual[11][0,0]==expected[11][1][0,0],(index,'RNG uniform state')
  names=[]
  certified_rows={}
  for native,(field,reference) in zip(actual,expected):
   try:
    if field=='tree_partitions':
     np.testing.assert_array_equal(np.sort(native.ravel()),np.arange(1,n+1))
     for start,end,is_leaf in spans:
      np.testing.assert_array_equal(np.sort(native[int(start)-1:int(end),0]),np.sort(reference[int(start)-1:int(end),0]))
     error=0.
    else:error=compare(native,reference,index,field)
   except AssertionError:
    if field not in ['training_scores','query_scores','total_scores']:raise
    if field=='total_scores':
     if not certified_rows:raise
     canonical=np.array([np.sum(actual[4]),np.sum(actual[5])])[:,None]
     error=compare(native,canonical,index,'sum_of_native_scores')
     certificates.append(dict(fixture=index,field=field,kind='sum_identity',source=[float(z) if np.isfinite(z) else str(z) for z in reference.ravel()],native=[float(z) if np.isfinite(z) else str(z) for z in native.ravel()]))
    else:
     # Fitted partitions, bounds, radii and masses are checked separately.
     # Bound cross-runtime arithmetic differences in density space. A public
     # single-leaf fit additionally diagnoses bound-subtraction residues and
     # source zero-weight-node state effects; retain all three outputs.
     single_options={**options,'leaf_size':n,'atol':0.,'rtol':0.}
     if cfg['mutate']==2:single_options['kernel']='cosine'
     with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      single=KernelDensity(**single_options).fit(x,sample_weight=w if cfg['weighted'] else None)
      stable=single.score_samples(x if field=='training_scores' else q)[:,None]
     from sklearn.neighbors._kd_tree import kernel_norm
     norm=kernel_norm(model.bandwidth_,p,model.kernel,return_log=True)
     if not np.isfinite(norm) or norm>600:raise
     for a in [native,reference,stable]:
      if np.any(np.isnan(a)) or np.any(np.isposinf(a)):raise
     dn,dr,ds=np.exp(native),np.exp(reference),np.exp(stable)
     bound=256*np.finfo(float).eps*len(model.tree_.get_arrays()[2])*np.exp(norm)
     assert np.max(np.abs(dn-dr),initial=0)<=bound,(index,field,'tree density discrepancy exceeds rounding budget')
     error=float(np.max(np.abs(dn-dr),initial=0));certified_rows[field]=True
     serial=lambda a:[float(z) if np.isfinite(z) else str(z) for z in a.ravel()]
     certificates.append(dict(fixture=index,field=field,kind='tree_density_roundoff',bound=bound,max_density_error=error,
                              source=serial(reference),native=serial(native),single_leaf=serial(stable),
       source_to_single_leaf=float(np.max(np.abs(dr-ds),initial=0)),native_to_single_leaf=float(np.max(np.abs(dn-ds),initial=0))))
   checks.append(dict(fixture=index,field=field,max_abs_error=error));names.append(field)
   if np.any(~np.isfinite(reference)):
    nonfinite.append(dict(fixture=index,field=field,nan=int(np.isnan(reference).sum()),positive_inf=int(np.isposinf(reference).sum()),
                          negative_inf=int(np.isneginf(reference).sum())))
  inventory.append(dict(fixture=index,**{k:v for k,v in cfg.items() if k not in ['x','q','w','fw','power']},
                        power=str(cfg['power']),n=n,p=p,fields=names))
  if index%100==0:print(index,'/',len(cases),'fixtures passed',flush=True)
 for case,message in REJECTIONS.items():
  proc=subprocess.run([str(exe),case],text=True,capture_output=True)
  assert proc.returncode!=0 and message in proc.stdout+proc.stderr,(case,proc.stdout,proc.stderr)
  assert 'SIGFPE' not in proc.stderr and 'Fortran runtime error' not in proc.stderr,(case,proc.stderr)
 source=Path(sklearn.__file__).parent
 report=dict(profile=args.profile,fixtures=len(cases),start_fixture=args.start_fixture,checks=checks,inventories=inventory,
  nonfinite_reference_results=nonfinite,source_warnings=warns,roundoff_certificates=certificates,rejections=list(REJECTIONS),sklearn_version=sklearn.__version__,
  numpy_version=np.__version__,scipy_version=scipy.__version__,executable=str(exe),executable_sha256=sha(exe),validator_sha256=sha(__file__),
  parser_sha256=sha(Path(__file__).with_name('compare_image_graph.py')),
  reference_binary_sha256={str(Path(m.__file__)):sha(m.__file__) for m in [source_kd,source_ball,source_special,np.random.mtrand]},
  source_sha256={s:sha(source/s) for s in ['neighbors/_kde.py','neighbors/_binary_tree.pxi.tp','neighbors/_kd_tree.pyx.tp','neighbors/_ball_tree.pyx.tp','metrics/_dist_metrics.pyx.tp']})
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print('PASS:',len(cases),'fixtures,',len(checks),'groups,',len(REJECTIONS),'diagnostics')
if __name__=='__main__':main()
