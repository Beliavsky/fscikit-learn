"""Public streamed OPTICS graphs against sklearn's compute_optics_graph."""
from pathlib import Path
import argparse, hashlib, json, subprocess, warnings, os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.cluster import compute_optics_graph
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster._optics import _set_reach_dist


def callback(a,b):return np.sum(np.abs(a-b))+np.sum(np.abs(a-b)**2)/8


def replay_certificate(x,k,metric,power,radius,core,order,actual_reach,pred):
 """Replay the source state along the native permutation within roundoff bounds.

 Core distances remain direct source comparisons. Every selection, reachable
 value, predecessor edge and strict-improvement boundary is checked against
 sklearn's own radius queries and _set_reach_dist, not a relaxed permutation
 comparison. Precomputed data never use this exception.
 """
 n=len(x);order=order-1;pred=np.where(pred<0,-1,pred-1)
 np.testing.assert_array_equal(np.sort(order),np.arange(n))
 processed=np.zeros(n,bool);reach=np.full(n,np.inf);previous=np.full(n,-1)
 bound_factor=8*np.finfo(float).eps*(x.shape[1]+1)
 def bound(value):return 2e-15+bound_factor*max(1.,abs(value))
 def close(a,b):
  if np.isinf(a)or np.isinf(b):return a==b
  return abs(a-b)<=bound(max(abs(a),abs(b)))
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  nbrs=NearestNeighbors(n_neighbors=k,algorithm='brute',metric=metric,p=power,n_jobs=1).fit(x)
 selections=[]
 for step,point in enumerate(order):
  candidates=np.flatnonzero(~processed);best=candidates[np.argmin(reach[candidates])]
  assert close(actual_reach[point],reach[point]),('selected reach mismatch',step,point,actual_reach[point],reach[point])
  assert close(reach[point],reach[best]),('nonminimal choice',step,point,best,reach[point],reach[best])
  if point!=best:
   assert np.isfinite(reach[point]),('unreachable tie must use lowest index',step,point,best)
   selections.append(dict(step=step,point=int(point+1),source_choice=int(best+1),
     chosen_source_reach=float(reach[point]),minimum_source_reach=float(reach[best]),bound=bound(reach[point])))
  if pred[point]>=0:
   parent=pred[point];assert parent<n and processed[parent],('invalid predecessor',point,parent)
   assert point in nbrs.radius_neighbors(x[parent:parent+1],radius=radius,return_distance=False)[0],('edge outside radius',parent,point)
   parameters={'p':power}if metric=='minkowski'else {}
   distance=pairwise_distances(x[parent:parent+1],x[point:point+1],metric=metric,**parameters)[0,0]
   candidate=np.around(max(core[parent],distance),15)
   assert close(candidate,actual_reach[point]),('predecessor edge cost',parent,point,candidate,actual_reach[point])
  else:
   assert np.isinf(actual_reach[point]),('missing predecessor',point)
  processed[point]=True
  if np.isfinite(core[point]):
   _set_reach_dist(core_distances_=core,reachability_=reach,predecessor_=previous,point_index=point,
     processed=processed,X=x,nbrs=nbrs,metric=metric,metric_params=None,p=power,max_eps=radius)
 for i in range(n):assert close(actual_reach[i],reach[i]),('final replay reach',i,actual_reach[i],reach[i])
 return dict(native_order=(order+1).tolist(),selections=selections,roundoff_factor=bound_factor,
   rounded_absolute_bound=2e-15,predecessor_edges_checked=int(np.sum(pred>=0))),reach


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True);parser.add_argument('--report');parser.add_argument('--allow-rounding-certificates',action='store_true')
 args=parser.parse_args();cases=[];lines=[];warning_count=0
 def add(x,k,metric,p=2.,radius=np.inf):
  nonlocal warning_count
  mode=2 if callable(metric) else int(isinstance(k,float))
  metric_name='callback'if callable(metric)else metric
  with warnings.catch_warnings(record=True)as caught:
   warnings.simplefilter('always')
   expected=compute_optics_graph(x,min_samples=k,max_eps=radius,metric=metric,p=p,
     metric_params=None,algorithm='brute',leaf_size=30,n_jobs=1)
  warning_count+=len(caught)
  label=dict(n=len(x),p=x.shape[1],k=k,metric=metric_name,power=str(p),radius=str(radius))
  cases.append((expected,label,max(2,int(k*len(x)))if isinstance(k,float)else k,x,metric,p,radius))
  lines.extend([f'{len(x)} {x.shape[1]} {k if isinstance(k,int)else 2} {mode}',
    f'{metric_name} {k if isinstance(k,float)else .2} {radius} {p}'])
  lines.extend(' '.join(map(str,row))for row in x)
 for seed in [7,42,2026]:
  rng=np.random.default_rng(seed)
  for n,p in [(31,2),(23,5),(7,1)]:
   x=rng.normal(size=(n,p));x[1]=x[0]
   for metric,power in [('minkowski',2.),('minkowski',1.5),('minkowski',.5),
      ('euclidean',2.),('manhattan',2.),('chebyshev',2.),('sqeuclidean',2.),('cosine',2.),(callback,2.)]:
    for k in [2,5,.25,1.]:
     if callable(metric)and isinstance(k,float):continue
     for radius in [np.inf,.7,0.]:add(x,k,metric,power,radius)
   d=pairwise_distances(x,metric='euclidean');np.fill_diagonal(d,0)
   for variant in ['symmetric','asymmetric','diagonal']:
    a=d.copy()
    if variant=='asymmetric':a[0,2]+=1
    if variant=='diagonal':np.fill_diagonal(a,np.linspace(.1,1,n))
    for k in [2,5,.25,1.]:
     for radius in [np.inf,.7,0.]:add(a,k,'precomputed',radius=radius)
 for n in [2,4,13]:
  for kind in ['zeros','plateau','huge']:
   d=np.ones((n,n))if kind=='plateau'else np.zeros((n,n))
   if kind=='huge':d[:]=1e300
   np.fill_diagonal(d,0)
   for k in [2,1.]:add(d,k,'precomputed')
 x=np.arange(24,dtype=float).reshape(8,3)/8
 for metric,power in [('l1',2.),('cityblock',2.),('l2',2.),('minkowski',np.inf)]:
  for k in [2,.5]:
   for radius in [np.inf,.7]:add(x,k,metric,power,radius)
 for scale in [0.,1e-20]:
  for k in [2,.5]:add(x*scale,k,'cosine')
 proc=subprocess.run([args.executable],input='\n'.join(lines)+'\n',capture_output=True,text=True,check=True)
 rows=iter(proc.stdout.splitlines());worst=0;certificates=[]
 for (order,core,reach,pred),label,k,x,metric,power,radius in cases:
  actual_order=np.fromstring(next(rows),sep=' ',dtype=int)
  actual_core=np.fromstring(next(rows),sep=' ');actual_reach=np.fromstring(next(rows),sep=' ')
  actual_pred=np.fromstring(next(rows),sep=' ',dtype=int);count,unreachable=next(rows).split()
  np.testing.assert_allclose(actual_core,core,rtol=3e-14,atol=3e-14,err_msg=str(label))
  discrete=np.array_equal(actual_order,order+1)and np.array_equal(actual_pred,np.where(pred<0,-1,pred+1))
  direct_reach=np.allclose(actual_reach,reach,rtol=3e-14,atol=3e-14)
  compared_reach=reach
  if not discrete or not direct_reach:
   if not args.allow_rounding_certificates or metric=='precomputed':
    np.testing.assert_array_equal(actual_order,order+1,err_msg=str(label))
    np.testing.assert_array_equal(actual_pred,np.where(pred<0,-1,pred+1),err_msg=str(label))
    np.testing.assert_allclose(actual_reach,reach,rtol=3e-14,atol=3e-14,err_msg=str(label))
   certificate,compared_reach=replay_certificate(x,k,metric,power,radius,core,actual_order,actual_reach,actual_pred)
   certificate['configuration']=label;certificate['source_order']=(order+1).tolist()
   certificate['direct_reach_match']=direct_reach
   finite=np.isfinite(reach)&np.isfinite(actual_reach)
   certificate['global_reachability_max_scaled_error']=float(np.max(np.abs(actual_reach[finite]-reach[finite])/(1+np.abs(reach[finite])),initial=0))
   certificate['global_infinite_pattern_match']=bool(np.array_equal(np.isinf(reach),np.isinf(actual_reach)))
   certificate['source_predecessor']=np.where(pred<0,-1,pred+1).tolist()
   certificate['native_predecessor']=actual_pred.tolist()
   certificates.append(certificate)
  for got,want in [(actual_core,core),(actual_reach,compared_reach)]:
   finite=np.isfinite(want);worst=max(worst,float(np.max(np.abs(got[finite]-want[finite])/(1+np.abs(want[finite])),initial=0)))
  assert int(count)==k and (unreachable=='T')==bool(np.all(np.isinf(actual_reach)))==bool(np.all(np.isinf(reach))),label
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),groups=len(cases)*5,max_scaled_error=worst,source_warnings=warning_count,
   sklearn=sklearn.__version__,numpy=np.__version__,certificates=certificates,
   exact_configurations=len(cases)-len(certificates),
   executable_sha256=hashlib.sha256(Path(args.executable).read_bytes()).hexdigest(),
   source_sha256=hashlib.sha256((root/'src/sklearn_optics.f90').read_bytes()).hexdigest(),
   validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   input_sha256=hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest(),
   native_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()for name in
     ['src/sklearn_optics.f90','src/sklearn_cluster.f90','src/sklearn_kernel_dispatch.f90']},
   configuration_inventory=[case[1]for case in cases],
   reference_sha256=hashlib.sha256((Path(sklearn.__file__).parent/'cluster/_optics.py').read_bytes()).hexdigest())
 print(json.dumps(report,indent=2))
 if args.report:Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
