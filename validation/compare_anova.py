"""Public sklearn f_oneway: ragged dense, CSR and mixed groups."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings,sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np,scipy,sklearn
from scipy.sparse import csr_matrix
from sklearn.feature_selection import f_oneway
import sklearn.feature_selection._univariate_selection as source

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True)
 ap.add_argument('--reference-cache',type=Path);ap.add_argument('--report',type=Path);args=ap.parse_args()
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 provenance=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,source_sha256=sha(source.__file__))
 cache={};cases=[];calls=hits=warns=0
 if args.reference_cache and args.reference_cache.exists():
  old=json.loads(args.reference_cache.read_text());assert old['provenance']==provenance;cache=old['cases']
 def add(groups,label,route=0):
  nonlocal calls,hits,warns
  lines=[f'{len(groups)} {groups[0].shape[1]} {route}']
  for g in groups:
   if isinstance(g,csr_matrix):
    lines += [f'1 {g.shape[0]}',str(g.nnz),' '.join(map(str,g.indptr+1)),' '.join(map(str,g.indices+1)),' '.join(map(str,g.data))]
   else:lines += [f'0 {g.shape[0]}']+[' '.join(map(str,row))for row in g]
  payload='\n'.join(lines)+'\n';key=hashlib.sha256(payload.encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    try:
     f,p=f_oneway(*groups);expected=dict(statistic=f.tolist(),pvalue=p.tolist())
    except ValueError as error:expected=dict(error=str(error))
   expected['warnings']=len(caught);cache[key]=expected;calls+=1
  warns+=expected['warnings'];cases.append((label,payload,expected))
 for seed in [7,42,2026]:
  rng=np.random.default_rng(seed)
  for k in [1,2,3,5]:
   for counts in [[1]*k,[2+c*3 for c in range(k)],[129+c for c in range(k)]]:
    for p in [1,3,9]:
     for kind in ['normal','zero','common_constant','separate_constant','sparse','tiny','overflow']:
      dense=[]
      for c,n in enumerate(counts):
       x=rng.normal(size=(n,p))+c*.3
       if kind=='zero':x[:]=0
       elif kind=='common_constant':x[:]=2
       elif kind=='separate_constant':x[:]=c+1
       elif kind=='sparse':x[rng.random(x.shape)<.7]=0
       elif kind=='tiny':x*=1e-160
       elif kind=='overflow':x*=1e200
       dense.append(x)
      for storage in ['dense','csr','mixed','mixed_sparse_first','duplicates']:
       groups=[]
       for c,x in enumerate(dense):
        if storage=='dense'or(storage=='mixed'and c%2==0)or(storage=='mixed_sparse_first'and c%2==1):groups.append(x);continue
        s=csr_matrix(x)
        if storage=='duplicates':
         # Two raw entries per logical value, plus explicit zeros; no canonicalization.
         data=[];indices=[];indptr=[0]
         for row in x:
          for j,v in enumerate(row):
           if v!=0:data.extend([v*.25,v*.75]);indices.extend([j,j])
          data.append(0.);indices.append(p-1);indptr.append(len(data))
         s=csr_matrix((data,indices,indptr),shape=x.shape)
        groups.append(s)
       label=dict(seed=seed,k=k,counts=counts,p=p,kind=kind,storage=storage)
       add(groups,label)
       if k==2 and storage in ['dense','csr']:add(groups,dict(label,route='two'),1 if storage=='dense'else 2)
 valid=[c for c in cases if 'error'not in c[2]];rejected=[c for c in cases if 'error'in c[2]]
 payload=''.join(c[1]for c in valid)
 if args.reference_cache:
  args.reference_cache.parent.mkdir(parents=True,exist_ok=True)
  args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 run=subprocess.run([args.executable],input=payload,text=True,capture_output=True)
 if run.returncode:raise RuntimeError(run.stderr+'\n'+run.stdout[-1000:])
 rows=iter(run.stdout.splitlines());worst=0;groups=0
 for label,_,expected in valid:
  for field in ['statistic','pvalue']:
   actual=np.fromstring(next(rows),sep=' ');want=np.array(expected[field]);groups+=1
   np.testing.assert_allclose(actual,want,rtol=3e-8,atol=3e-8,equal_nan=True,err_msg=str((label,field)))
   if field=='pvalue':np.testing.assert_allclose(actual,want,rtol=3e-10,atol=1e-300,equal_nan=True,err_msg=str((label,'tail_relative')))
   finite=np.isfinite(actual)&np.isfinite(want)
   if finite.any():worst=max(worst,float(np.max(abs(actual[finite]-want[finite])/(1+abs(want[finite])))))
 assert next(rows,None)is None
 def check_rejection(case):
  label,request,expected=case
  run=subprocess.run([args.executable],input=request,text=True,capture_output=True)
  assert 'non-broadcastable output operand'in expected['error'] and label['storage']=='mixed'
  assert run.returncode and 'ANOVA dense-first mixed sparse samples cannot broadcast'in(run.stdout+run.stderr) and 'SIGFPE'not in run.stderr,(label,expected,run.stdout,run.stderr)
 with ThreadPoolExecutor(max_workers=8)as pool:list(pool.map(check_rejection,rejected))
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),successful_configurations=len(valid),source_rejections=len(rejected),groups=groups,max_scaled_error=worst,source_warnings=warns,source_calls=calls,reference_cache_hits=hits,
  reference_provenance=provenance,reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,
  source_sha256=sha(root/'src/sklearn_univariate_selection.f90'),validator_sha256=sha(__file__),executable_sha256=sha(args.executable),
  input_sha256=hashlib.sha256(payload.encode()).hexdigest(),rejection_input_sha256=hashlib.sha256(''.join(c[1]for c in rejected).encode()).hexdigest(),configuration_inventory=[c[0]for c in cases])
 print(json.dumps({k:v for k,v in report.items()if k!='configuration_inventory'},indent=2))
 if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
