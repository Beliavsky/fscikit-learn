"""Exhaustive ties and weighted multilabel ranking against installed sklearn."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.metrics import coverage_error,label_ranking_average_precision_score,label_ranking_loss
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fixtures():
 for p in [2,3,4]:
  for truth,score in itertools.product(itertools.product([0,1],repeat=p),itertools.product([0.,1.,2.],repeat=p)):
   yield dict(mode='exhaustive',y=np.array([truth]),scores=np.array([score]),weights=None)
 for seed,(n,p),pattern,style,weight in itertools.product([0,7,42,2026,2147483647],[(1,1),(3,1),(1,7),(5,13),(17,64)],range(4),range(4),range(4)):
  rng=np.random.RandomState(seed);y=rng.randint(0,2,(n,p));scores=rng.normal(size=(n,p))
  if pattern==1:scores=np.round(scores)
  if pattern==2:scores[:]=0
  if pattern==3:scores=np.array([-np.finfo(float).max,-0.,np.finfo(float).max])[rng.randint(0,3,(n,p))]
  if style==1:y=y*2
  if style==2:y=-y
  if style==3:y=y+1
  weights=None
  if weight:
   weights=np.arange(1,n+1,dtype=float)
   if weight==1 and n>1:weights[0]=0
   if weight==2:weights[0]=-1
   if weight==3:weights*=1e200
  yield dict(mode='random',seed=seed,pattern=pattern,style=style,weight=weight,y=y,scores=scores,weights=weights)
 for seed,p in itertools.product([0,7,42,2026,2147483647],[129,1025,100000]):
  rng=np.random.RandomState(seed);yield dict(mode='wide',seed=seed,y=rng.randint(0,2,(1,p)),scores=rng.randint(-2,3,(1,p)).astype(float),weights=None)
def expected(c):
 y,s,w=c['y'],c['scores'],c['weights'];out={};p=y.shape[1]
 if p>1:
  out.update(coverage=np.array([coverage_error(y,s,sample_weight=w)]),loss=np.array([label_ranking_loss(y,s,sample_weight=w)]),logical_coverage=np.array([coverage_error(y!=0,s,sample_weight=w)]),logical_loss=np.array([label_ranking_loss(y!=0,s,sample_weight=w)]),row_coverage=np.array([coverage_error(y[i:i+1],s[i:i+1]) for i in range(len(y))]),row_loss=np.array([label_ranking_loss(y[i:i+1],s[i:i+1]) for i in range(len(y))]))
 out.update(precision=np.array([label_ranking_average_precision_score(y,s,sample_weight=w)]),logical_precision=np.array([label_ranking_average_precision_score(y!=0,s,sample_weight=w)]),row_precision=np.array([label_ranking_average_precision_score(y[i:i+1],s[i:i+1]) for i in range(len(y))]))
 if np.all((y==0)|(y==1)):
  from scipy.sparse import csr_matrix
  csr=csr_matrix(y);out['csr_precision']=np.array([label_ranking_average_precision_score(csr,s,sample_weight=w)])
  if p>1:out['csr_loss']=np.array([label_ranking_loss(csr,s,sample_weight=w)])
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--report',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--case',type=int);args=ap.parse_args();cases=list(fixtures())
 if args.case is not None:cases=[cases[args.case]]
 refs=[];inputs=[str(len(cases))];configs=[]
 for i,c in enumerate(cases):
  y,s,w=c['y'],c['scores'],c['weights']
  with np.errstate(over='ignore',invalid='ignore'):refs.append(expected(c))
  inputs.append(f'{len(y)} {y.shape[1]} {int(w is not None)} {int(np.all((y==0)|(y==1)))}')
  for row,score in zip(y,s):inputs.extend([' '.join(map(str,row)),' '.join(map(repr,map(float,score)))])
  if w is not None:inputs.append(' '.join(map(repr,map(float,w))))
  config={k:v for k,v in c.items() if k not in ['y','scores','weights']};config.update(n=len(y),p=y.shape[1],weighted=w is not None,csr=np.all((y==0)|(y==1)).item());configs.append(config)
  if (i+1)%500==0:print('Prepared',i+1,'source fixtures',flush=True)
 exe=args.executable.resolve();r=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True);assert r.returncode==0,(r.stdout[-1800:],r.stderr[-1800:])
 checks=[];inventories=[];index=-1
 for line in r.stdout.splitlines():
  parts=line.split()
  if parts[0]=='BEGIN':index+=1;fields={}
  elif parts[0]=='END':
   assert fields.keys()==refs[index].keys(),(index,fields.keys(),refs[index].keys());inventories.append(sorted(fields))
   for name,value in fields.items():
    want=refs[index][name];np.testing.assert_allclose(value,want,rtol=3e-13,atol=3e-13,err_msg=str((index,configs[index],name)))
    checks.append(dict(fixture=index,field=name,passed=True,max_abs_error=float(np.max(np.abs(value-want),initial=0))))
  else:
   assert parts[0] not in fields;fields[parts[0]]=np.array(list(map(float,parts[2:])));assert len(fields[parts[0]])==int(parts[1])
 assert index+1==len(cases)
 invalid=dict(empty='nonempty',no_labels='nonempty',single_coverage='at least two',single_loss='at least two',shape='shapes differ',target='at most two',nan='finite',inf='finite',weight_length='length mismatch',weight_nan='finite',weight_inf='finite',weight_zero='nonzero total',weight_cancel='nonzero total',csr_value='store ones',csr_duplicate='sorted and unique',csr_unsorted='sorted and unique',csr_shape='shapes differ',csr_invalid='out of bounds');rejections=[]
 for mode,message in invalid.items():
  r=subprocess.run([str(exe),mode],text=True,capture_output=True);output=r.stdout+r.stderr;primary=next((s for s in output.splitlines() if s.startswith('fscikit-learn:')),'')
  assert r.returncode!=0 and message in primary,(mode,output);assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(mode,output)
  rejections.append(dict(case=mode,passed=True))
 import sklearn.metrics._ranking as source
 report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=configs,rejections=rejections,executable_sha256=sha(exe),validator_sha256=sha(__file__),source_sha256=sha(source.__file__),native_source_sha256={n:sha(ROOT/'src'/n) for n in ['sklearn_multilabel_ranking.f90','sklearn_metrics.f90','sklearn_sparse.f90','sklearn_ieee_math.f90']})
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(len(checks),'groups /',len(cases),'fixtures /',len(rejections),'diagnostics PASS',flush=True)
if __name__=='__main__':main()
