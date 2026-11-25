"""Actual sklearn chunk boundaries, distance blocks, callable order and typed reductions."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,warnings
import numpy as np,scipy,sklearn
from scipy.sparse import csr_matrix,issparse
from sklearn.metrics import pairwise_distances_chunked
import sklearn.metrics.pairwise as source
import scipy.spatial.distance as distance
NAMES=['callback','euclidean','sqeuclidean','l2','manhattan','chebyshev','minkowski','cosine','nan_euclidean','haversine','braycurtis','canberra','correlation','hamming','dice','jaccard','rogerstanimoto','russellrao','sokalmichener','sokalsneath','yule','cityblock','seuclidean','mahalanobis','precomputed']

def configurations():
 for seed in [0,7,42]:
  rng=np.random.default_rng(seed)
  for n in [1,7]:
   for p in [2,5]:
    x=rng.normal(size=(n,p));y=rng.normal(size=(4,p));x[0]=0;y[0]=0
    for code in range(24):
     if code==9 and p!=2:continue
     for identity in [0,1,2]:
      for memory in [0.,8*(n if identity else 4)*2/2**20,1024.]:
       representations=[(0,0)]
       if code in [0,1,3,4,7,9,21]:representations += [(1,1)]if identity else[(0,1),(1,0),(1,1)]
       for spx,spy in representations:
        yield dict(code=code,x=x,y=y,identity=identity,spx=spx,spy=spy,memory=memory,seed=seed)
 for sparse in [0,1]:
  for identity in [0,1,2]:
   n=7;m=7 if identity else 4;rng=np.random.default_rng(10);x=rng.random((n,m));y=np.zeros((m,m))
   for memory in [0.,1024.]:yield dict(code=24,x=x,y=y,identity=identity,spx=sparse,spy=sparse,memory=memory,seed=-1)
 # Float floor-division boundaries and copy-versus-identity behavior.
 for boundary in [8*7/2**20,8*7*3/2**20]:
  for memory in [np.nextafter(boundary,0),boundary,np.nextafter(boundary,np.inf)]:
   yield dict(code=1,x=np.arange(21.).reshape(7,3),y=np.zeros((7,3)),identity=1,spx=0,spy=0,memory=memory,seed=-2)
 for power in [-1.,0.,.5,1.,3.,float('inf'),-float('inf')]:
  yield dict(code=6,x=np.array([[0.,1.],[1.,3.],[2.,2.]]),y=np.array([[1.,2.],[2.,3.]]),identity=0,spx=0,spy=0,memory=0.,seed=-3,power=power)

 for code in [1,2,3,6,7,8,10,11,12]:
  for identity in [0,1]:
   yield dict(code=code,x=np.array([[1e155,1e155,0.],[1e308,-1e308,1.],[0.,0.,0.]]),y=np.array([[1e155,-1e155,0.],[1e308,1e308,0.]]),identity=identity,spx=0,spy=0,memory=0.,seed=-4)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--report',required=True,type=Path);ap.add_argument('--reference-cache',type=Path);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 provenance=dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,pairwise_sha256=sha(source.__file__),distance_sha256=sha(distance.__file__))
 cache={}
 if args.reference_cache and args.reference_cache.exists():
  old=json.loads(args.reference_cache.read_text());assert old['provenance']==provenance;cache=old['cases']
 cases=[];calls=hits=0
 def encode(a):
  if issparse(a):return[a.indptr.tolist(),(a.indices+1).tolist(),a.data.tolist()]
  return[np.asarray(a).ravel(order='F').tolist()]
 for index,c in enumerate(configurations()):
  if args.smoke and index%37:continue
  code=c['code'];x=c['x'].copy();y=c['y'].copy();identity=c['identity'];n,p=x.shape;m=n if identity else len(y)
  if code==8:x[::2,0]=np.nan;y[::2,-1]=np.nan
  if identity:y=x.copy()
  mode=index%7 if c['seed']>=0 else 0
  parameters=int(code in [6,22,23]and(index%2==0 or identity==0));power=c.get('power',1.5)
  squared=int(code in [1,3,8]and index%2==1)
  lines=[f'{code} {n} {m} {p} {c["spx"]} {c["spy"]} {identity} {c["memory"]:.17g} {power} {parameters} {squared} {mode}']
  lines+=[' '.join(map(str,row))for row in x];lines+=[' '.join(map(str,row))for row in y]
  request='\n'.join(lines)+'\n';key=hashlib.sha256(request.encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   a=csr_matrix(x)if c['spx']else x;b=csr_matrix(y)if c['spy']else y
   if identity==1:b=None
   elif identity==2:b=a
   counters=[0,0,0.];starts=[];sizes=[]
   def metric(a,b):
    counters[0]+=1
    a=a.toarray().ravel()if issparse(a)else a;b=b.toarray().ravel()if issparse(b)else b
    return np.sum((a-b)**2)+.01*a[0]+.03*b[0]+counters[0]*1e-5
   def reducer(d,start):
    counters[1]+=1;data=d.toarray()if issparse(d)else d;counters[2]+=float(np.sum(data));starts.append(start);sizes.append(len(data))
    if mode==1:return np.sum(data,axis=1),np.argmin(data,axis=1)
    if mode==3:return csr_matrix(data<1.)
    if mode==4:return ()
    if mode==5:return data.copy()
    if mode==6:return (data<1.).astype(int)
   kwargs=dict(metric=metric if code==0 else NAMES[code],working_memory=c['memory'])
   if code==6:kwargs['p']=power
   if parameters:
    if code==6:kwargs['w']=np.arange(1,p+1)/p+.5
    if code==22:kwargs['V']=np.arange(1,p+1)/p+.5
    if code==23:kwargs['VI']=np.full((p,p),.1)+np.eye(p)*1.9
   if squared:kwargs['squared']=True
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always');outputs=list(pairwise_distances_chunked(a,b,reduce_func=reducer if mode else None,**kwargs))
   blocks=[];start=0
   for i,out in enumerate(outputs):
    if mode==0:
     count=out.shape[0];values=encode(out);sparse=int(issparse(out))
    else:
     start=starts[i];count=sizes[i];sparse=0
     if mode==1:values=[out[0].tolist(),out[1].tolist()]
     elif mode in [3,5,6]:values=encode(out)
     else:values=[]
    blocks.append(dict(start=start,rows=count,values=values,sparse=sparse));start+=count
   exceeded=int(code!=24 and c['memory']*2**20<8*m)
   expected=dict(rows=blocks[0]['rows'],exceeded=exceeded,blocks=blocks,counters=counters,warnings=len(caught));cache[key]=expected;calls+=1
  cases.append((request,expected,dict(index=index,code=code,n=n,m=m,p=p,identity=identity,spx=c['spx'],spy=c['spy'],mode=mode,memory=c['memory'])))
 if args.reference_cache:args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 request=''.join(c[0]for c in cases);r=subprocess.run([args.executable],input=request,text=True,capture_output=True)
 if r.returncode:raise RuntimeError(r.stderr+'\n'+r.stdout[-2000:])
 lines=iter(r.stdout.splitlines());groups=0;worst=0
 def compare(actual,want,label,exact=False):
  nonlocal groups,worst
  a=np.asarray(actual);b=np.asarray(want);groups+=1
  if exact:np.testing.assert_array_equal(a,b,err_msg=str(label))
  else:
   np.testing.assert_allclose(a,b,rtol=3e-10,atol=3e-10,equal_nan=True,err_msg=str(label))
   finite=np.isfinite(a)&np.isfinite(b)
   if finite.any():worst=max(worst,float(np.max(abs(a[finite]-b[finite])/(1+abs(b[finite])))))
 for _,expected,label in cases:
  meta=np.fromstring(next(lines),sep=' ');compare(meta,[expected['rows'],expected['exceeded']],label,True)
  for block in expected['blocks']:
   parts=next(lines).split();assert parts[0]=='BLOCK';compare(list(map(int,parts[1:])),[block['start'],block['rows']],label,True)
   if label['mode']==0:assert int(next(lines))==block['sparse']
   arrays=block['values']
   if not arrays:assert next(lines)=='NONE'
   else:
    for j,want in enumerate(arrays):
     if (label['mode']==0 and block['sparse']or label['mode']==3)and j==0:want=(np.asarray(want)+1).tolist()
     compare(np.fromstring(next(lines),sep=' '),want,(label,block['start'],j),j<2 if label['mode']==3 or block['sparse']else label['mode']==1 and j==1)
  end=next(lines).split();assert end[0]=='END';compare(list(map(int,end[1:3])),expected['counters'][:2],label,True);compare([float(end[3])],[expected['counters'][2]],label)
 assert next(lines,None)is None
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),groups=groups,max_scaled_error=worst,source_calls=calls,reference_cache_hits=hits,source_warnings=sum(c[1]['warnings']for c in cases),provenance=provenance,input_sha256=hashlib.sha256(request.encode()).hexdigest(),executable_sha256=sha(args.executable),validator_sha256=sha(__file__),reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,source_sha256={n:sha(root/'src'/n)for n in ['sklearn_pairwise_chunks.f90','sklearn_pairwise_reductions.f90','sklearn_metrics.f90']},inventory=[c[2]for c in cases])
 args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='inventory'},indent=2))
if __name__=='__main__':main()
