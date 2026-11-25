"""Direct sklearn classification reports: numeric rows and exact text."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings,sys
import numpy as np,scipy,sklearn
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report,precision_recall_fscore_support
import sklearn.metrics._classification as source

def fixtures():
 for seed in [0,7,42]:
  rng=np.random.default_rng(seed)
  for n in [1,7,33]:
   for c in [2,5]:
    y=rng.integers(c,size=n);p=rng.integers(c,size=n)
    for selection in [None,list(range(c-1,-1,-1)),[0],[c+1,0,0],[]]:
     for wm in range(6):
      w=None if wm==0 else np.ones(n)if wm==1 else np.arange(1,n+1)/n if wm==2 else np.zeros(n)if wm==3 else np.where(np.arange(n)%2,-.5,1.)if wm==4 else np.full(n,1e-7)
      for z in [0.,1.,float('nan')]:
       yield dict(y=y,p=p,labels=selection,w=w,z=z,kind=0 if c==5 else 1,seed=seed)
 for seed in [7,42]:
  rng=np.random.default_rng(seed)
  for n in [1,7,33,129]:
   for q in [2,5,17]:
    y=rng.integers(2,size=(n,q));p=rng.integers(2,size=(n,q))
    for selection in [None,list(range(q-1,-1,-1)),[0],[q-1,0,0],[]]:
     for wm in range(5):
      w=None if wm==0 else np.ones(n)if wm==1 else np.arange(1,n+1)/n if wm==2 else np.zeros(n)if wm==3 else np.where(np.arange(n)%2,-.5,1.)
      for z in [0.,1.,float('nan')]:
       yield dict(y=y,p=p,labels=selection,w=w,z=z,kind=2,seed=seed)
 # Negative numeric labels, single-column adapters, empty vectors and name quirks.
 for kind in [0,2]:
  yield dict(y=np.array([-9,2,2,9]),p=np.array([2,2,9,9]),labels=[9,-9,2,7],w=None,z=0.,kind=kind,seed=-1)
 for y,p in [(np.array([],int),np.array([],int)),(np.array([0,1]),np.array([1,1]))]:
  for names in [None,[],['a'],['a','b','unused long name'],['same','same']]:
   yield dict(y=y,p=p,labels=[0,1],names=names,w=None,z=float('nan'),kind=0,seed=-2,dict=True)

 for w in [None,np.zeros(2),np.array([.5,1.])]:
  yield dict(y=np.array([0,1]),p=np.array([1,1]),labels=[],names=['unused'],w=w,z=0.,kind=0,seed=-3,force_text=True)

 for kind in [0,1]:
  yield dict(y=np.array([],int),p=np.array([],int),labels=None,w=None,z=float('nan'),kind=kind,seed=-4,dict=True)
 for w in [np.array([float('nan'),1.]),np.array([float('inf'),1.]),np.array([-float('inf'),1.])]:
  yield dict(y=np.array([0,1]),p=np.array([0,0]),labels=None,w=w,z=0.,kind=0,seed=-5)

 for numeric_only in [False,True]:
  yield dict(y=np.array([0,1]),p=np.array([0,0]),labels=[0,0,1],w=None,z=0.,kind=0,seed=-6,dict=numeric_only)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--report',required=True,type=Path);ap.add_argument('--reference-cache',type=Path);args=ap.parse_args()
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 provenance=dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,classification_sha256=sha(source.__file__))
 cache={}
 if args.reference_cache and args.reference_cache.exists():
  saved=json.loads(args.reference_cache.read_text());assert saved['provenance']==provenance;cache=saved['cases']
 cases=[];calls=hits=0
 for index,c in enumerate(fixtures()):
  y,p=c['y'],c['p'];multi=y.ndim==2;kind=c['kind']
  if multi:kind=2+index%7
  elif c['seed']>=0:
   if kind==1:kind=[0,1,2,3][index%4]
   else:kind=0 if index%2 else 2
  if not multi and kind in [2,3]:y=y[:,None];p=p[:,None]
  q=y.shape[1]if y.ndim==2 else 1;n=len(y);multi=q>1
  labels=c['labels'];chosen=np.unique(np.concatenate([y.ravel(),p.ravel()])).tolist()if not multi else list(range(q))
  observed=chosen.copy()
  if labels is not None:chosen=labels
  names=c.get('names',None)
  if 'names'not in c and index%7==0:names=['class '+str(j)for j in chosen]
  if 'names'not in c and labels is not None and index%37==0:names=['only']
  digits=[0,2,4,9][index%4];dict_=c.get('dict',False)or len(chosen)==0 or names==[]
  if c.get('force_text'):dict_=False
  w=c['w'];z=c['z']
  sy=csr_matrix(y)if kind in [4,5,7]else y.astype(bool)if kind in [1,3,8]else y
  sp=csr_matrix(p)if kind in [4,6,8]else p.astype(bool)if kind in [1,3,7]else p
  lines=[f'{kind} {n} {q} {-1 if labels is None else len(labels)} {-1 if names is None else len(names)} {int(w is not None)} {digits} {int(dict_)} {z}']
  if kind<=1:lines+=[' '.join(map(str,y.ravel())),' '.join(map(str,p.ravel()))]
  else:
   for a,b in zip(y,p):lines+=[' '.join(map(str,a)),' '.join(map(str,b))]
  if labels:lines+=[' '.join(map(str,labels))]
  if names is not None:lines+=names
  if w is not None:lines+=[' '.join(map(str,w))]
  request='\n'.join(lines)+'\n';key=hashlib.sha256(request.encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    kw=dict(labels=labels,target_names=names,sample_weight=w,zero_division=z,digits=digits)
    source_dict=classification_report(sy,sp,output_dict=True,**kw)
    text=''if dict_ else classification_report(sy,sp,**kw)
    pr,re,fs,su=precision_recall_fscore_support(sy,sp,labels=chosen,sample_weight=w,zero_division=z)
    rownames=list(map(str,chosen))if names is None else names[:len(chosen)]
    if names is None and labels is None and not multi and kind in [1,3]:rownames=list(map(str,np.unique(np.concatenate([sy.ravel(),sp.ravel()]))))
    values=[[float(a),float(b),float(f),float(s)]for a,b,f,s in zip(pr,re,fs,su)][:len(rownames)]
    micro_accuracy=not multi and set(chosen)>=set(observed)
    for avg in ['micro','macro','weighted']+(['samples']if multi else []):
     a,b,f,_=precision_recall_fscore_support(sy,sp,labels=chosen,sample_weight=w,zero_division=z,average=avg)
     name='accuracy'if avg=='micro'and micro_accuracy else avg+' avg'
     rownames.append(name);values.append([float(a),float(b),float(f),float(np.sum(su))])
     # Cross-check the actual classification_report dictionary, including its scalar accuracy.
     target=source_dict[name]
     if name=='accuracy':np.testing.assert_allclose(target,a,equal_nan=True,rtol=0,atol=0)
     else:np.testing.assert_allclose(list(target.values()),values[-1],equal_nan=True,rtol=0,atol=0)
    expected=dict(meta=[len(rownames),len(values)-3-int(multi),int(micro_accuracy),int(names is not None and len(names)!=len(chosen))],names=rownames,values=values,text=text,warnings=len(caught));cache[key]=expected;calls+=1
  cases.append((request,expected,dict(index=index,kind=kind,n=n,q=q,labels=labels,digits=digits,dict=dict_)))
 if args.reference_cache:args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 request=''.join(c[0]for c in cases);r=subprocess.run([args.executable],input=request,text=True,capture_output=True)
 if r.returncode:raise RuntimeError(r.stderr+'\n'+r.stdout[-2000:])
 lines=iter(r.stdout.splitlines());worst=0;groups=0
 for _,expected,label in cases:
  meta=np.fromstring(next(lines),sep=' ',dtype=int);np.testing.assert_array_equal(meta,expected['meta'],err_msg=str(label));groups+=1
  names=[];values=[]
  for _ in range(meta[0]):names.append(next(lines));values.append(np.fromstring(next(lines),sep=' '))
  assert names==expected['names'],(label,names,expected['names']);groups+=1
  values=np.asarray(values);want=np.asarray(expected['values'])
  for j in range(4):
   np.testing.assert_allclose(values[:,j],want[:,j],rtol=3e-12,atol=3e-12,equal_nan=True,err_msg=str((label,j)));groups+=1
  finite=np.isfinite(values)&np.isfinite(want)
  if finite.any():worst=max(worst,float(np.max(abs(values[finite]-want[finite])/(1+abs(want[finite])))))
  text=''.join(chr(int(i))for i in next(lines).split());assert text==expected['text'],(label,repr(text),repr(expected['text']));groups+=1
 assert next(lines,None)is None
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),groups=groups,max_scaled_error=worst,source_calls=calls,reference_cache_hits=hits,source_warnings=sum(c[1]['warnings']for c in cases),provenance=provenance,input_sha256=hashlib.sha256(request.encode()).hexdigest(),executable_sha256=sha(args.executable),validator_sha256=sha(__file__),reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,source_sha256={n:sha(root/'src'/n)for n in ['sklearn_classification_report.f90','sklearn_classification_metrics.f90','sklearn_metrics.f90','sklearn_resampling.f90']},inventory=[c[2]for c in cases])
 args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='inventory'},indent=2))
if __name__=='__main__':main()
