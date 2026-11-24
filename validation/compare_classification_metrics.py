"""Weighted classification reductions, indicators and sparse confusion summaries."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from scipy.sparse import csr_matrix
from sklearn.metrics import precision_score,recall_score,f1_score,fbeta_score,jaccard_score,precision_recall_fscore_support,multilabel_confusion_matrix
from sklearn.metrics._classification import _prf_divide
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
AVERAGES=[None,'binary','micro','macro','weighted','samples']
def weight_vector(n,mode):
 if mode==0:return None
 w=np.arange(1,n+1,dtype=float)
 if mode==2 and n>1:w[0]=0
 if mode==3 and n>0:w[0]=-1
 if mode==4:w[:]=0
 if mode==5 and n>0:w[:]=0;w[0]=-1;w[-1]+=1
 return w
def fixture(kind,y,p,average=None,beta=1.,zero=0.,w=None,labels=None,pos=1,mode='ordinary'):
 return dict(kind=kind,y=np.asarray(y,dtype=int),p=np.asarray(p,dtype=int),average=average,beta=beta,zero=zero,w=w,labels=None if labels is None else np.array(labels,dtype=int),pos=int(pos),mode=mode)
def fixtures():
 # Exhaustive binary targets/predictions with both positive-label orientations.
 for n in [0,1,2,3]:
  for y,p,avg,pos in itertools.product(itertools.product([0,1],repeat=n),itertools.product([0,1],repeat=n),[None,'binary','micro','macro','weighted'],[0,1]):
   yield fixture(0,y,p,average=avg,pos=pos,mode='exhaustive_binary')
 for seed,n,classes,avg,zero,weight in itertools.product([0,7,42,2026,2147483647],[1,2,17],[[0,1],[-7,5,19]],[None,'micro','macro','weighted'],[0.,1.,np.nan],range(6)):
  rng=np.random.RandomState(seed);y=rng.choice(classes,n);p=rng.choice(classes,n)
  beta=[0.,.5,1.,2.,np.inf][(seed+weight)%5]
  label_options=[None,[classes[-1],classes[0],101],[classes[-1],classes[0],classes[-1]],[]]
  labels=label_options[(weight+seed)%4]
  yield fixture(0,y,p,average=avg,beta=beta,zero=zero,w=weight_vector(n,weight),labels=labels,pos=classes[-1],mode='weighted_classes')
 for seed,n,q,avg,zero,weight in itertools.product([0,7,42,2026,2147483647],[1,2,7],[1,3,17],[None,'micro','macro','weighted','samples'],[0.,1.,np.nan],range(6)):
  if q==1 and avg=='samples':continue
  rng=np.random.RandomState(seed);y=rng.randint(0,2,(n,q));p=rng.randint(0,2,(n,q));beta=[0.,.5,1.,2.,np.inf][(seed+weight)%5]
  labels=[None,[q-1,0],[q-1,0,q-1],[]][(seed+weight)%4]
  if q==1:labels=None
  yield fixture(1,y,p,average=avg,beta=beta,zero=zero,w=weight_vector(n,weight),labels=labels,mode='multilabel')
 # All-zero/full rows, two-valued nonstandard indicator codes and nonfinite weights.
 for scenario,avg,zero in itertools.product(range(6),[None,'micro','macro','weighted','samples'],[0.,1.,np.nan]):
  y=np.array([[1,0,1],[0,1,0],[0,0,0]],int);p=np.array([[1,1,0],[0,1,1],[0,0,0]],int)
  if scenario==0:y[:]=0;p[:]=0
  if scenario==1:y[:]=1;p[:]=1
  if scenario==2:y*=2;p*=-1
  w=[None,None,None,np.array([np.nan,1.,2.]),np.array([np.inf,1.,2.]),np.array([np.inf,-np.inf,1.])][scenario]
  yield fixture(1,y,p,average=avg,zero=zero,w=w,mode='special')
 for seed,q in itertools.product([0,7,42,2026,2147483647],[129,100000]):
  y=np.zeros((2,q),int);p=y.copy();y[0,[0,q-1]]=1;y[1,q//2]=1;p[0,0]=1;p[1,[q//2,q-2]]=1
  for avg in [None,'micro','weighted','samples']:
   yield fixture(1,y,p,average=avg,beta=.5,labels=[q-1,0,q//2],w=np.array([.25,2.]),mode='wide')
 # Full class vectors and fractional cancellation weights exercise pairwise reductions.
 for seed,avg in itertools.product([0,7,42,2026,2147483647],[None,'micro','macro','weighted']):
  rng=np.random.RandomState(seed);y=rng.randint(-2,3,129);p=rng.randint(-2,3,129);w=rng.normal(size=129)
  yield fixture(0,y,p,average=avg,beta=.3,zero=np.nan,w=w,mode='long_signed')
 yield fixture(1,np.zeros((1,10000),int),np.zeros((1,10000),int),average='micro',mode='wide_full')
 for w in [np.array([np.nan,1.,2.]),np.array([np.inf,1.,2.]),np.array([np.inf,-np.inf,1.])]:
  for avg,z in itertools.product([None,'binary','micro','macro','weighted'],[0.,1.,np.nan]):
   yield fixture(0,[0,1,1],[0,1,0],average=avg,zero=z,w=w,mode='special_vector')
 for avg,z in itertools.product([None,'micro','macro','weighted'],[0.,1.,np.nan]):
  yield fixture(1,np.array([-7,5,19])[:,None],np.array([5,5,-7])[:,None],average=avg,beta=2.,zero=z,w=np.array([.25,0.,2.]),labels=[-7,19,-7,101],mode='one_column_classes')
def representations(c):
 y,p=c['y'],c['p'];kind=c['kind'];out={'int':(y,p)}
 boolean=np.all((y==0)|(y==1)) and np.all((p==0)|(p==1))
 if kind==0:
  if boolean:out['logical']=(y!=0,p!=0)
 else:
  if y.shape[1]>1 or boolean:out['logical']=(y!=0,p!=0)
  if y.shape[1]>1:
   sy,sp=csr_matrix(y!=0,dtype=float),csr_matrix(p!=0,dtype=float)
   out.update(csr=(sy,sp),csr_int=(sy,p),int_csr=(y,sp),csr_logical=(sy,p!=0),logical_csr=(y!=0,sp))
 return out
def raw_statistics(y,p,c):
 avg=c['average'];labels=[c['pos']] if avg=='binary' else c['labels'];w=c['w'];zero=c['zero'];beta=c['beta']
 cm=multilabel_confusion_matrix(y,p,sample_weight=w,labels=labels,samplewise=avg=='samples').astype(float)
 tp=cm[:,1,1];pred=tp+cm[:,0,1];true=tp+cm[:,1,0];union=tp+cm[:,0,1]+cm[:,1,0]
 if avg=='micro':tp=np.array([tp.sum()]);pred=np.array([pred.sum()]);true=np.array([true.sum()]);union=np.array([union.sum()])
 div=lambda a,b:_prf_divide(a,b,'precision','predicted',avg,(),zero)
 pr=div(tp,pred);re=div(tp,true)
 f1=div(2*tp,true+pred)
 fb=re if np.isposinf(beta) else pr if beta==0 else div((1+beta**2)*tp,beta**2*true+pred)
 return dict(precision=pr,recall=re,f1=f1,fbeta=fb,jaccard=div(tp,union))
def expected(c):
 y,p,w,z=c['y'],c['p'],c['w'],c['zero'];avg=c['average'];scalaravg='macro' if avg is None else avg
 opts=dict(labels=c['labels'],pos_label=c['pos'],average=avg,sample_weight=w,zero_division=z)
 pr,re,fb,support=precision_recall_fscore_support(y,p,beta=c['beta'],**opts)
 base=dict(summary_precision=np.atleast_1d(pr),summary_recall=np.atleast_1d(re),summary_fbeta=np.atleast_1d(fb),support_present=np.array([float(support is not None)]),support=np.array([]) if support is None else np.asarray(support))
 cm=multilabel_confusion_matrix(y,p,sample_weight=w,labels=c['labels']);base['confusion']=cm.ravel()
 if c['kind']==1 and y.shape[1]>1:base['sample_confusion']=multilabel_confusion_matrix(y,p,sample_weight=w,labels=c['labels'],samplewise=True).ravel()
 raw=raw_statistics(y,p,c);scalaropts={**opts,'average':scalaravg};jok=False;jrejected=False
 for name,f in [('precision',precision_score),('recall',recall_score),('f1',f1_score),('fbeta',fbeta_score)]:
  extra=dict(beta=c['beta']) if name=='fbeta' else {}
  base[name]=np.array([f(y,p,**scalaropts,**extra)]);base[name+'_raw']=raw[name]
 if not np.isnan(z):
  try:base['jaccard']=np.array([jaccard_score(y,p,**scalaropts)])
  except ZeroDivisionError:jrejected=True
  else:base['jaccard_raw']=raw['jaccard'];jok=True
 out={};repchecks=[]
 for prefix,(a,b) in representations(c).items():
  # Public confusion matrices certify representation-equivalent sufficient statistics.
  actual=multilabel_confusion_matrix(a,b,sample_weight=w,labels=c['labels'])
  np.testing.assert_allclose(actual.ravel(),base['confusion'],rtol=0,atol=0,equal_nan=True)
  if 'sample_confusion' in base:
   actual=multilabel_confusion_matrix(a,b,sample_weight=w,labels=c['labels'],samplewise=True)
   np.testing.assert_allclose(actual.ravel(),base['sample_confusion'],rtol=0,atol=0,equal_nan=True)
  repchecks.append(prefix)
  for name,value in base.items():out[prefix+'_'+name]=np.asarray(value,dtype=float).ravel()
 return out,jok,jrejected,repchecks
def payload(c,jok):
 y,p,w,labels=c['y'],c['p'],c['w'],c['labels'];n=len(y);q=1 if c['kind']==0 else y.shape[1]
 lines=[' '.join(map(str,[c['kind'],n,q,AVERAGES.index(c['average']),c['pos'],int(w is not None),int(labels is not None),0 if labels is None else len(labels),int(jok),c['beta'],c['zero']]))]
 if c['kind']==0:lines+=[' '.join(map(str,y)),' '.join(map(str,p))]
 else:
  for a,b in zip(y,p):lines+=[' '.join(map(str,a)),' '.join(map(str,b))]
 if w is not None:lines.append(' '.join(map(repr,map(float,w))))
 if labels is not None and len(labels):lines.append(' '.join(map(str,labels)))
 return '\n'.join(lines)+'\n'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--report',required=True,type=Path);ap.add_argument('--profile',default='development');selection=ap.add_mutually_exclusive_group();selection.add_argument('--case',type=int);selection.add_argument('--cases',type=int,nargs='+');args=ap.parse_args();cases=list(fixtures())
 if args.case is not None:cases=[cases[args.case]]
 elif args.cases is not None:cases=[cases[i] for i in args.cases]
 refs=[];inputs=[str(len(cases))];configs=[];source_rejected=[]
 for i,c in enumerate(cases):
  with warnings.catch_warnings(),np.errstate(all='ignore'):
   warnings.simplefilter('ignore');ref,jok,jrejected,reps=expected(c)
  refs.append(ref);inputs.append(payload(c,jok))
  configs.append(dict(kind=c['kind'],mode=c['mode'],n=len(c['y']),q=1 if c['kind']==0 else c['y'].shape[1],average=c['average'],beta='inf' if np.isinf(c['beta']) else c['beta'],zero='nan' if np.isnan(c['zero']) else c['zero'],weighted=c['w'] is not None,labels=None if c['labels'] is None else c['labels'].tolist(),jaccard_valid=jok,representations=reps))
  if jrejected:source_rejected.append(i)
  if (i+1)%500==0:print('Prepared',i+1,'source fixtures',flush=True)
 exe=args.executable.resolve();r=subprocess.run([str(exe)],input=inputs[0]+'\n'+''.join(inputs[1:]),text=True,capture_output=True);assert r.returncode==0,(r.stdout[-2500:],r.stderr[-2500:])
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
 probes={};source_rejections=[];jobs=[]
 for i in source_rejected:
  data='1\n'+payload(cases[i],False)
  if data not in probes:
   probes[data]=len(probes);jobs.append((i,data))
  source_rejections.append(dict(fixture=i,probe=probes[data],passed=True))
 def rejection_probe(job):
  i,data=job;r=subprocess.run([str(exe),'jaccard_stdin'],input=data,text=True,capture_output=True);output=r.stdout+r.stderr
  assert r.returncode!=0 and 'fscikit-learn: Jaccard averaging weights have zero total' in output,(i,configs[i],output)
  assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(i,output)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(rejection_probe,jobs))
 invalid=dict(length='lengths differ',shape='shapes differ',matrix_empty='nonempty',average='unknown',multiclass_binary='binary targets',multilabel_binary='multilabel targets',pos_label='valid observed label',samples_vector='multilabel targets',labels_high='[0,n_labels)',labels_low='[0,n_labels)',indicator='at most two',weight_length='length mismatch',zero_division='zero, one or NaN',jaccard_nan='zero or one',beta_negative='nonnegative',beta_nan='nonnegative',beta_overflow='square overflows',jaccard_zero_weights='zero total',csr_single='at least two',csr_value='store ones',csr_nan='finite',csr_index='out of bounds',csr_duplicate='sorted and unique',csr_unsorted='sorted and unique');rejections=[]
 def invalid_probe(item):
  mode,message=item;r=subprocess.run([str(exe),mode],text=True,capture_output=True);output=r.stdout+r.stderr;primary=next((s for s in output.splitlines() if s.startswith('fscikit-learn:')),'')
  assert r.returncode!=0 and message in primary,(mode,output);assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(mode,output)
  return dict(case=mode,passed=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rejections=list(pool.map(invalid_probe,invalid.items()))
 import sklearn.metrics._classification as source
 import sklearn.utils.extmath as extmath
 report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=configs,rejections=rejections,source_rejections=source_rejections,unique_source_rejection_probes=len(probes),source_nonfinite_values=nonfinite,executable_sha256=sha(exe),validator_sha256=sha(__file__),source_sha256=sha(source.__file__),extmath_source_sha256=sha(extmath.__file__),native_source_sha256={n:sha(ROOT/'src'/n) for n in ['sklearn_classification_metrics.f90','sklearn_metrics.f90','sklearn_ieee_math.f90','sklearn_sparse.f90']})
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(len(checks),'groups /',len(cases),'fixtures /',len(rejections),'diagnostics PASS;',len(source_rejections),'source rejections /',len(probes),'probes;',nonfinite,'nonfinite source values',flush=True)
if __name__=='__main__':main()
