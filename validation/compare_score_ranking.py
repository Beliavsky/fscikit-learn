"""Margins, top-k class ranking and discounted relevance against sklearn 1.6.1."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.metrics import hinge_loss,top_k_accuracy_score,dcg_score,ndcg_score
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def weights(n,mode):
 if mode==0:return None
 w=np.arange(1,n+1,dtype=float)
 if mode==2 and n>1:w[0]=0
 if mode==3:w[0]=-1
 return w
def classification(y,scores,labels,k=0,w=None,kind=0,mode='classification'):
 return dict(kind=kind,mode=mode,y=np.array(y,dtype=int),scores=np.asarray(scores,dtype=float),labels=None if labels is None else np.array(labels),k=k,w=w,ignore=False,base=2.)
def relevance(y,scores,k=0,w=None,ignore=False,base=2.,mode='relevance'):
 return dict(kind=2,mode=mode,y=np.asarray(y,dtype=float),scores=np.asarray(scores,dtype=float),labels=None,k=k,w=w,ignore=ignore,base=base)
def fixtures():
 for p in [3,4]:
  for target,score,k in itertools.product(range(p),itertools.product([0.,1.,2.],repeat=p),range(1,p+2)):
   yield classification([target],[score],np.arange(p),k,mode='exhaustive_top')
 for y,score,k in itertools.product(itertools.product([0,1],repeat=2),itertools.product([-2.,-.5,0.,.5,1.,2.],repeat=2),[0,1,3]):
  yield classification(y,np.array(score)[:,None],[0,1],k,mode='exhaustive_binary')
 for seed,n,p,pattern,weight in itertools.product([0,7,42,2026,2147483647],[1,2,17],[1,3,7],range(3),range(4)):
  rng=np.random.RandomState(seed);classes=np.arange(2 if p==1 else p)*12-7;y=classes[rng.randint(len(classes),size=n)];s=rng.normal(size=(n,p))
  if pattern==1:s=np.round(s)
  if pattern==2:s=rng.uniform(size=(n,p))
  for k in sorted({1,2,p+1}):yield classification(y,s,classes,k,w=weights(n,weight))
 for seed,p,weight in itertools.product([0,7,42,2026,2147483647],[1,3,7,31],range(4)):
  rng=np.random.RandomState(seed);classes=np.arange(2 if p==1 else p)*12-7;y=np.tile(classes,2);rng.shuffle(y);s=rng.normal(size=(len(y),p))
  for k in sorted({1,2,p+1}):
   yield classification(y,s,None,k,w=weights(len(y),weight),mode='implicit_classes')
   yield classification(y,s,classes,k,w=weights(len(y),weight),mode='paired_classes')
 for p in [2,3]:
  for target,score,k in itertools.product(itertools.product([0.,1.],repeat=p),itertools.product([0.,1.,2.],repeat=p),[0,1,p+1]):
   yield relevance([target],[score],k,mode='exhaustive_dcg')
 for seed,(n,p),pattern,truth,weight,cut in itertools.product([0,7,42,2026,2147483647],[(1,2),(3,7),(17,31)],range(3),range(4),range(4),range(3)):
  rng=np.random.RandomState(seed);s=rng.normal(size=(n,p));r=np.exp(rng.normal(size=(n,p)))
  if pattern==1:s=np.round(s)
  if pattern==2:s[:]=0
  if truth==1:r[:]=0
  if truth==2:r=rng.randint(0,6,(n,p)).astype(float)
  if truth==3:r=rng.randint(-3,4,(n,p)).astype(float)
  base=[.5,1.,2.,10.,1e300][(seed+pattern+truth+weight+cut)%5]
  yield relevance(r,s,[0,1,p+1][cut],weights(n,weight),ignore=pattern==0 and truth%2==0,base=base)
 for seed,p in itertools.product([0,7,42,2026,2147483647],[129,100000]):
  rng=np.random.RandomState(seed);s=rng.randint(-2,3,(1,p)).astype(float);r=rng.randint(0,6,(1,p)).astype(float)
  yield relevance(r,s,k=17,mode='wide')
  yield classification([p//2],s,np.arange(p),k=17,mode='wide')
 for labels in [[30,10,20,20],[20,30,10],[10,20],[10]]:
  p=3 if len(np.unique(labels))>2 else 1
  yield classification([10,20,30],np.arange(3*p).reshape(3,p),labels,kind=3,mode='hinge_labels')
 for y in [[0,1],[0,0],[1,1]]:yield classification(y,[[-1.],[1.]],[0,1],1,w=np.array([-1.,1.]),kind=4,mode='raw_zero_sum')
 for n,p in [(2,1),(2,3)]:
  classes=np.arange(2 if p==1 else p);s=np.full((n,p),1e308);s[0,0]=-1e308
  yield classification([0,1],s,classes,1,mode='extreme')
 for y,s in [(np.array([[1e308,1e308,1e308]]),np.array([[3.,2.,1.]])),(np.array([[1e308,-1e308,1e308]]),np.array([[1.,1.,1.]])),(np.array([[1.,2.,3.]]),np.array([[-1e308,0.,1e308]]))]:
  yield relevance(y,s,mode='extreme')
def expected(c):
 y,s,w=c['y'],c['scores'],c['w'];k=None if c['k']==0 else c['k'];out={}
 if c['kind']!=2:
  labels=c['labels'];score=s.ravel() if s.shape[1]==1 else s
  if c['kind']!=4:
   out['hinge']=np.array([hinge_loss(y,score,labels=labels,sample_weight=w)])
   if s.shape[1]==1:out['vector_hinge']=out['hinge']
  if c['kind']!=3:
   options=dict(k=2 if k is None else k,sample_weight=w,labels=labels)
   if c['kind']!=4:out['top_k']=np.array([top_k_accuracy_score(y,score,**options)])
   out['top_k_count']=np.array([top_k_accuracy_score(y,score,normalize=False,**options)])
   if s.shape[1]==1:
    if c['kind']!=4:out['vector_top_k']=out['top_k']
    out['vector_top_k_count']=out['top_k_count']
  return out
 opts=dict(k=k,log_base=c['base'],ignore_ties=c['ignore']);normopts=dict(k=k,ignore_ties=c['ignore'])
 out['dcg']=np.array([dcg_score(y,s,sample_weight=w,**opts)])
 out['dcg_rows']=np.array([dcg_score(y[i:i+1],s[i:i+1],**opts) for i in range(len(y))])
 if np.all(y>=0):
  out['ndcg']=np.array([ndcg_score(y,s,sample_weight=w,**normopts)])
  out['ndcg_rows']=np.array([ndcg_score(y[i:i+1],s[i:i+1],**normopts) for i in range(len(y))])
 out['logical_dcg']=np.array([dcg_score(y!=0,s,sample_weight=w,**opts)])
 out['logical_ndcg']=np.array([ndcg_score(y!=0,s,sample_weight=w,**normopts)])
 if np.all(np.abs(y)<=np.iinfo(np.int32).max) and np.all(y==np.trunc(y)):
  out['integer_dcg']=np.array([dcg_score(y.astype(int),s,sample_weight=w,**opts)])
  if np.all(y>=0):out['integer_ndcg']=np.array([ndcg_score(y.astype(int),s,sample_weight=w,**normopts)])
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--report',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--case',type=int);args=ap.parse_args();cases=list(fixtures())
 if args.case is not None:cases=[cases[args.case]]
 inputs=[str(len(cases))];refs=[];configs=[]
 for i,c in enumerate(cases):
  y,s,labels,w=c['y'],c['scores'],c['labels'],c['w'];n,p=s.shape
  with warnings.catch_warnings(),np.errstate(all='ignore'):
   warnings.simplefilter('ignore');refs.append(expected(c))
  integral=int(np.all(np.abs(y)<=np.iinfo(np.int32).max) and np.all(y==np.trunc(y))) if c['kind']==2 else 0
  inputs.append(' '.join(map(str,[c['kind'],n,p,0 if labels is None else len(labels),c['k'],int(w is not None),int(c['ignore']),int(np.all(y>=0)),integral,c['base']])))
  if labels is not None:inputs.append(' '.join(map(str,labels)))
  if c['kind']!=2:
   inputs.append(' '.join(map(str,y)))
   inputs.extend(' '.join(map(repr,map(float,row))) for row in s)
  else:
   for a,b in zip(y,s):inputs.extend([' '.join(map(repr,map(float,a))),' '.join(map(repr,map(float,b)))])
  if w is not None:inputs.append(' '.join(map(repr,map(float,w))))
  configs.append(dict(kind=c['kind'],mode=c['mode'],n=n,p=p,weighted=w is not None,explicit_labels=labels is not None,integral=integral,nonnegative=bool(np.all(y>=0)),k=c['k'],base=c['base'],ignore=c['ignore']))
  if (i+1)%500==0:print('Prepared',i+1,'source fixtures',flush=True)
 exe=args.executable.resolve();r=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True);assert r.returncode==0,(r.stdout[-1800:],r.stderr[-1800:])
 checks=[];inventories=[];index=-1;nonfinite=0
 for line in r.stdout.splitlines():
  parts=line.split()
  if parts[0]=='BEGIN':index+=1;fields={}
  elif parts[0]=='END':
   assert fields.keys()==refs[index].keys(),(index,fields.keys(),refs[index].keys());inventories.append(sorted(fields))
   for name,value in fields.items():
    want=refs[index][name];np.testing.assert_allclose(value,want,rtol=3e-12,atol=3e-12,equal_nan=True,err_msg=str((index,configs[index],name)))
    finite=np.isfinite(value)&np.isfinite(want);nonfinite+=int(np.count_nonzero(~np.isfinite(want)))
    checks.append(dict(fixture=index,field=name,passed=True,max_abs_error=float(np.max(np.abs(value[finite]-want[finite]),initial=0)),max_scaled_error=float(np.max(np.abs(value[finite]-want[finite])/(1+np.abs(want[finite])),initial=0))))
  else:
   assert parts[0] not in fields;fields[parts[0]]=np.array(list(map(float,parts[2:])));assert len(fields[parts[0]])==int(parts[1])
 assert index+1==len(cases)
 invalid=dict(empty='nonempty',samples='sample counts differ',hinge_vector='require a matrix',hinge_columns='match sorted classes',hinge_binary='one score column',hinge_labels='absent from labels',top_k='positive',top_labels='unique',top_order='ordered',top_missing='absent from labels',top_columns='one score column',top_binary_classes='match all classes',score_nan='finite',score_inf='finite',weight_length='length mismatch',weight_nan='finite',weight_inf='finite',weight_zero='nonzero total',weight_cancel='nonzero total',documents='at least two',dcg_shape='shapes differ',relevance_nan='finite',relevance_inf='finite',negative_relevance='nonnegative',dcg_k='positive',base_zero='positive',base_negative='positive',base_nan='finite',base_inf='finite');rejections=[]
 for mode,message in invalid.items():
  r=subprocess.run([str(exe),mode],text=True,capture_output=True);output=r.stdout+r.stderr;primary=next((s for s in output.splitlines() if s.startswith('fscikit-learn:')),'')
  assert r.returncode!=0 and message in primary,(mode,output);assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(mode,output)
  rejections.append(dict(case=mode,passed=True))
 import sklearn.metrics._ranking as ranking_source
 import sklearn.metrics._classification as class_source
 report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=configs,rejections=rejections,source_nonfinite_values=nonfinite,executable_sha256=sha(exe),validator_sha256=sha(__file__),ranking_source_sha256=sha(ranking_source.__file__),classification_source_sha256=sha(class_source.__file__),native_source_sha256={n:sha(ROOT/'src'/n) for n in ['sklearn_score_ranking.f90','sklearn_metrics.f90','sklearn_ieee_math.f90']})
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(len(checks),'groups /',len(cases),'fixtures /',len(rejections),'diagnostics PASS;',nonfinite,'nonfinite source values',flush=True)
if __name__=='__main__':main()
