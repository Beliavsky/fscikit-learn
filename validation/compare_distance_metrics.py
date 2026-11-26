from pathlib import Path
import argparse,os,json,hashlib,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np,sklearn
from scipy.sparse import csr_matrix
from sklearn.metrics import DistanceMetric
ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();r=a.report.parent;r.mkdir(parents=True,exist_ok=True);exe=a.executable;unsafe_by_input={}
names=['euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity', 'seuclidean', 'mahalanobis', 'hamming', 'canberra', 'braycurtis', 'matching', 'jaccard', 'dice', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'haversine', 'pyfunc']
rng=np.random.RandomState(131);cases=[]
def callback(u,v):return 1000+np.sum(u)-2*np.sum(v)

def source_undefined(x,y):
 # Independently trace Source cursor bounds; never read the exhausted value.
 def row(a,i):
  if hasattr(a,'indptr'):return int(a.indptr[i]),int(a.indptr[i+1]),a.indices,len(a.data)
  return 0,a.shape[1],np.arange(a.shape[1]),(len(a)-i)*a.shape[1]
 same=x is y;result=np.zeros((x.shape[0],y.shape[0]),dtype=bool)
 for i in range(x.shape[0]):
  for j in range(i if same else 0,y.shape[0]):
   ia,ea,ca,na=row(x,i);ib,eb,cb,nb=row(y,j)
   while ia<ea and ib<eb:
    if ca[ia]==cb[ib]:ia+=1;ib+=1
    elif ca[ia]<cb[ib]:ia+=1
    else:ib+=1
   bad=(ia==ea and ib<eb and ia>=na)or(ia<ea and ib>=nb)
   result[i,j]=bad
   if same:result[j,i]=bad
 return result

for n,p in [(3,2),(7,3)]:
 x=np.array([[1.,0.],[0.,1.],[0.,0.]])if p==2 else rng.normal(size=(n,p))
 if p==3:x[rng.random_sample(x.shape)<.5]=0;x[-1]=0
 y=np.array([[0.,0.],[-1.,2.]])if p==2 else np.round(rng.normal(size=(4,p)))
 specifications=[(name,2.,0,None)for name in names if name not in ['seuclidean','mahalanobis']]
 specifications += [('seuclidean',2.,2,np.arange(1,p+1,dtype=float)),('mahalanobis',2.,3,np.eye(p)+.2),('mahalanobis',2.,4,np.eye(p)+.2)]
 for power in [-np.inf,-1.,0.,.5,1.,2.,3.,np.inf,np.nan]:
  specifications += [('minkowski',power,0,None),('p',power,1,np.arange(1,p+1,dtype=float))]
 for name,power,params,parameter in specifications:
  for storage in range(4):
   for own in [False,True]:
    label=dict(n=n,p=p,metric=name,power=str(power),params=params,storage=storage,own=own)
    xx=csr_matrix(x)if storage&1 else x.copy();yy=csr_matrix(y)if storage&2 else y.copy()
    with warnings.catch_warnings(),np.errstate(all='ignore'):
     warnings.simplefilter('ignore')
     try:
      kwargs={}
      if name in ['minkowski','p']:kwargs['p']=power
      if params==1:kwargs['w']=parameter
      if params in [2,4]:kwargs['V']=parameter
      if params==3:kwargs['VI']=parameter
      metric=DistanceMetric.get_metric(callback if name=='pyfunc'else name,**kwargs)
      values=metric.pairwise(xx,None if own else yy);again=metric.pairwise(xx,None if own else yy)
      conversion=np.array([-np.inf,-1.,0.,.5,1.,np.inf,np.nan])
      expected=[type(metric).__name__,values.ravel(),again.ravel(),np.asarray(metric.dist_to_rdist(conversion)),np.asarray(metric.rdist_to_dist(conversion))]
     except Exception as e:expected=dict(error_type=type(e).__name__,error=str(e))
    text=f'{n} {p} {len(y)} {storage} {int(own)} {name} {power} {params}\n'+'\n'.join(' '.join(map(str,row))for row in x)+'\n'+'\n'.join(' '.join(map(str,row))for row in y)+'\n'
    if parameter is not None:text+='\n'.join(' '.join(map(str,np.atleast_1d(row)))for row in parameter)+'\n'
    if isinstance(expected,list):
     unsafe_by_input[text]=source_undefined(xx,xx if own else yy)if name=='braycurtis'and(hasattr(xx,'indptr')or(not own and hasattr(yy,'indptr')))else np.zeros((len(x),len(x)if own else len(y)),dtype=bool)
    cases.append((text,expected,label))
differences=[];defined_failures=[];unsafe_records=[];groups=0;maximum=0.;accepted=0;rejected=0
for offset in range(0,len(cases),1):
 text,expected,label=cases[offset];proc=subprocess.run([str(exe)],input=text,text=True,capture_output=True);lines=proc.stdout.splitlines()
 if isinstance(expected,dict):
  rejected+=1
  if proc.returncode==0:differences.append(dict(case=label,field='Source rejection accepted',source=expected))
  else:assert 'fscikit-learn:'in proc.stdout+proc.stderr and not any(t in proc.stderr for t in ['SIGFPE','SIGSEGV','Fortran runtime error']),(label,proc.stderr)
  continue
 accepted+=1
 if proc.returncode or len(lines)!=7:
  for suffix,data in [('in',text),('stdout',proc.stdout),('stderr',proc.stderr)]: (r/f'{a.report.stem}_failure_{offset}.{suffix}').write_text(data)
  differences.append(dict(case=label,field='native failure',status=proc.returncode,stderr=proc.stderr));continue
 masks=[np.fromstring(lines[k],sep=' ',dtype=int).astype(bool)for k in [2,4]]
 expected_mask=unsafe_by_input[text].ravel()
 assert all(np.array_equal(mask,expected_mask)for mask in masks),(label,'undefined mask')
 if expected_mask.any():unsafe_records.append(dict(case=label,shape=unsafe_by_input[text].shape,mask=expected_mask.tolist()))
 lines=[lines[k]for k in [0,1,3,5,6]]
 for name,line,want in zip(['type','pairwise','repeat_pairwise','dist_to_rdist','rdist_to_dist'],lines,expected):
  groups+=1
  if name=='type':
   if line!=want:differences.append(dict(case=label,field=name,actual=line,source=want))
   continue
  value=np.fromstring(line,sep=' ');want=np.asarray(want)
  if name in ['pairwise','repeat_pairwise']:
   try:np.testing.assert_allclose(value[~expected_mask],want[~expected_mask],rtol=2e-12,atol=2e-12,equal_nan=True)
   except AssertionError:defined_failures.append(dict(case=label,field=name,actual=value.tolist(),source=want.tolist(),unsafe=expected_mask.tolist()))

  try:np.testing.assert_allclose(value,want,rtol=2e-12,atol=2e-12,equal_nan=True)
  except AssertionError:differences.append(dict(case=label,field=name,actual=value.tolist(),source=want.tolist()))
  if value.shape==want.shape:
   mask=np.isfinite(value)&np.isfinite(want)
   if mask.any():maximum=max(maximum,float(np.max(abs(value[mask]-want[mask])/(1+abs(want[mask])))))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=dict(undefined_records=unsafe_records,defined_failures=defined_failures,configurations=len(cases),source_successful_calls=accepted,source_rejections=rejected,groups=groups,max_scaled_error=maximum,failed_comparison_groups=len(differences),differences=differences,inventory=[c[2]for c in cases],input_sha256=hashlib.sha256(''.join(c[0]for c in cases).encode()).hexdigest(),executable_sha256=sha(exe),sha256={str(p):sha(p)for p in [Path(__file__),Path('src/sklearn_distance_metrics.f90'),Path('src/sklearn_metrics.f90'),Path('app/reference_distance_metric_driver.f90')]},sklearn=sklearn.__version__,numpy=np.__version__)
a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items()if k not in ['inventory','differences','undefined_records','defined_failures']})
print('Differences',[(c['case'],c['field'])for c in differences[:30]])

print('Defined failures',len(defined_failures),'undefined configurations',len(unsafe_records))

assert not defined_failures,defined_failures
for failure in differences:
 assert failure['case']['metric']=='braycurtis'and failure['field']in ['pairwise','repeat_pairwise'],failure
 mask=next(c['mask']for c in unsafe_records if c['case']==failure['case'])
 bad=~np.isclose(np.asarray(failure['actual']),np.asarray(failure['source']),rtol=2e-12,atol=2e-12,equal_nan=True)
 assert not np.any(bad&~np.asarray(mask,dtype=bool)),failure
report.update(defined_contract_passed=True,strict_parity=not differences and not unsafe_records,
 rtol=2e-12,atol=2e-12,native_undefined_policy='quiet_nan_with_mask')
a.report.write_text(json.dumps(report,indent=2)+'\n')
