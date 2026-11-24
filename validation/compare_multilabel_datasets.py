"""Multilabel dataset arrays, CSR storage, sets and probabilities against sklearn."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy
from scipy.sparse import csr_matrix
import sklearn
from sklearn.datasets import make_multilabel_classification
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SEEDS=[0,7,42,2026,2147483647]
def fixtures():
 for seed,(n,p),nc,nl,length,allow in itertools.product(SEEDS,[(1,1),(7,5),(19,13),(31,31)],[1,5,9,33],[0,1,2,9,10],[1,9,10,50,250],[False,True]):
  if nl==0 and not allow:continue
  yield dict(mode=0,n=n,p=p,nc=nc,nl=nl,length=length,allow=allow,seed=seed)
 for seed in SEEDS:
  for p,nc,nl,length,allow in [(1,129,2,10,True),(7,129,10,250,False),(131,5,2,10,True),(131,1,1,250,False)]:
   yield dict(mode=0,n=7,p=p,nc=nc,nl=nl,length=length,allow=allow,seed=seed)
  for nl,allow in [(0,True),(2,True),(2,False)]:yield dict(mode=1,n=3,p=100000,nc=2,nl=nl,length=3,allow=allow,seed=seed)
def expected(c):
 x,y,pc,pwc=make_multilabel_classification(n_samples=c['n'],n_features=c['p'],n_classes=c['nc'],n_labels=c['nl'],length=c['length'],allow_unlabeled=c['allow'],sparse=True,return_indicator='sparse',return_distributions=True,random_state=c['seed'])
 x.sort_indices();y.sort_indices()
 result=dict(x_data=x.data,x_indices=x.indices+1,x_indptr=x.indptr+1,y_data=y.data,y_indices=y.indices+1,y_indptr=y.indptr+1,metadata=[c['n'],c['p'],c['nc']],p_c=pc)
 if c['mode']==0:
  offsets=[1];labels=[]
  for i in range(c['n']):labels.extend(y.indices[y.indptr[i]:y.indptr[i+1]]);offsets.append(len(labels)+1)
  result.update(x=x.toarray(),y=y.toarray(),consistency=np.ones(5),p_w_c=pwc,set_offsets=offsets,set_labels=labels)
 else:result['word_probability_sums']=pwc.sum(axis=0)
 return {name:np.asarray(values,dtype=float).ravel() for name,values in result.items()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--report',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--case',type=int)
 args=ap.parse_args();cases=list(fixtures())
 if args.case is not None:cases=[cases[args.case]]
 inputs=[str(len(cases))];references=[]
 for i,c in enumerate(cases):
  inputs.append(' '.join(map(str,[c['mode'],c['n'],c['p'],c['nc'],c['nl'],c['length'],int(c['allow']),c['seed']])));references.append(expected(c))
  if (i+1)%500==0:print('Prepared',i+1,'source configurations',flush=True)
 exe=args.executable.resolve();run=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,cwd=ROOT)
 assert run.returncode==0,(run.stdout[-2500:],run.stderr[-2500:])
 checks=[];inventories=[];i=-1;actual={}
 for line in run.stdout.splitlines():
  parts=line.split()
  if parts[0]=='BEGIN':i+=1;actual={}
  elif parts[0]=='END':
   want=references[i];assert actual.keys()==want.keys(),(i,actual.keys()^want.keys());inventories.append(sorted(actual))
   for field,value in actual.items():
    tolerance=3e-12 if field in ['p_c','p_w_c','word_probability_sums'] else 0
    np.testing.assert_allclose(value,want[field],atol=tolerance,rtol=tolerance,err_msg=str((i,cases[i],field)))
    checks.append(dict(fixture=i,field=field,passed=True,max_abs_error=float(np.max(np.abs(value-want[field]),initial=0))))
   assert actual['x_data'].min(initial=1)>0 and np.all(actual['x_data']==np.floor(actual['x_data']))
   assert np.all(actual['y_data']==1)
  else:
   field=parts[0];assert field not in actual;actual[field]=np.array(list(map(float,parts[2:])));assert int(parts[1])==len(actual[field])
 assert i+1==len(cases)
 invalid=dict(samples='n_samples',rows_limit='row pointers',features='n_features',classes='n_classes',labels='n_labels',length='length',seed='nonnegative',impossible='zero n_labels')
 rejections=[]
 for mode,message in invalid.items():
  result=subprocess.run([str(exe),mode],text=True,capture_output=True,cwd=ROOT);output=result.stdout+result.stderr
  diagnostic=next((s for s in output.splitlines() if s.startswith('fscikit-learn:')), '')
  assert result.returncode!=0 and message.lower() in diagnostic.lower(),(mode,output)
  assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(mode,output)
  rejections.append(dict(case=mode,passed=True))
 import sklearn.datasets._samples_generator as source
 report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,rejections=rejections,source_sha256=sha(source.__file__),validator_sha256=sha(__file__),executable_sha256=sha(exe),native_source_sha256={name:sha(ROOT/'src'/name) for name in ['sklearn_multilabel_datasets.f90','sklearn_datasets.f90','sklearn_legacy_rng.f90','sklearn_sparse.f90']},numpy_distribution_source_sha256=sha(ROOT/'validation/results/tools/numpy-2.2.6-distributions.c'))
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(len(checks),'groups /',len(cases),'fixtures /',len(rejections),'diagnostics passed',flush=True)
if __name__=='__main__':main()
