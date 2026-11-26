from pathlib import Path
import argparse,os,subprocess,json,hashlib,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import DistanceMetric
ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();r=a.report.parent;r.mkdir(parents=True,exist_ok=True);exe=a.executable
x=np.array([[1.,0.,2.],[0.,-1.,0.],[2.,1.,0.],[0.,0.,0.]])
y=np.array([[0.,2.,0.],[1.,0.,-1.],[0.,0.,0.]])
def callback(u,v):return 1000+np.sum(u)-2*np.sum(v)
def sparse(a,kind):
 data=[];indices=[];ptr=[0]
 for row in a:
  for j,value in enumerate(row):
   if kind=='explicit_zero'or value!=0:
    if kind=='duplicates':data.extend([.25*value,.75*value]);indices.extend([j,j])
    else:data.append(value);indices.append(j)
  ptr.append(len(data))
 a=csr_matrix((data,indices,ptr),shape=a.shape)
 if kind=='unsorted':
  for i in range(a.shape[0]):
   sl=slice(a.indptr[i],a.indptr[i+1]);a.data[sl]=a.data[sl][::-1];a.indices[sl]=a.indices[sl][::-1]
 return a
def wire_csr(a):
 return str(len(a.data))+'\n'+' '.join(map(str,a.indptr+1))+'\n'+(' '.join(map(str,a.indices+1))+'\n'+' '.join(map(str,a.data))+'\n'if len(a.data)else'')
def undefined(x,y):
 def row(a,i):
  if hasattr(a,'indptr'):return int(a.indptr[i]),int(a.indptr[i+1]),a.indices,len(a.data)
  return 0,a.shape[1],np.arange(a.shape[1]),(len(a)-i)*a.shape[1]
 same=x is y;mask=np.zeros((x.shape[0],y.shape[0]),dtype=bool)
 for i in range(x.shape[0]):
  for j in range(i if same else 0,y.shape[0]):
   ia,ea,ca,na=row(x,i);ib,eb,cb,nb=row(y,j)
   while ia<ea and ib<eb:
    if ca[ia]==cb[ib]:ia+=1;ib+=1
    elif ca[ia]<cb[ib]:ia+=1
    else:ib+=1
   mask[i,j]=(ia==ea and ib<eb and ia>=na)or(ia<ea and ib>=nb)
   if same:mask[j,i]=mask[i,j]
 return mask
names=['euclidean','manhattan','chebyshev','minkowski','seuclidean','mahalanobis','hamming','canberra','braycurtis','matching','jaccard','dice','kulsinski','rogerstanimoto','russellrao','sokalmichener','sokalsneath','haversine','pyfunc']
cases=[]
for kind in ['canonical','explicit_zero','unsorted','duplicates']:
 sx=sparse(x,kind);sy=sparse(y,kind)
 for name in names:
  params=0;parameter=None;kwargs={};power=1.5
  if name=='minkowski':params=1;parameter=np.array([1.,2.,3.]);kwargs=dict(p=power,w=parameter)
  if name=='seuclidean':params=2;parameter=np.array([1.,2.,3.]);kwargs=dict(V=parameter)
  if name=='mahalanobis':params=3;parameter=np.eye(3)+.2;kwargs=dict(VI=parameter)
  for storage in [1,2,3]:
   for own in [False,True]:
    xx=sx if storage&1 else x.copy();yy=sy if storage&2 else y.copy()
    label=dict(kind=kind,metric=name,storage=storage,own=own)
    with warnings.catch_warnings(),np.errstate(all='ignore'):
     warnings.simplefilter('ignore')
     try:
      metric=DistanceMetric.get_metric(callback if name=='pyfunc'else name,**kwargs)
      first=metric.pairwise(xx,None if own else yy);second=metric.pairwise(xx,None if own else yy)
      expected=[type(metric).__name__,first.ravel(),second.ravel()]
     except Exception as error:expected=dict(error_type=type(error).__name__,error=str(error))
    text=f'4 3 3 {storage} {int(own)} {name} {power} {params}\n'+'\n'.join(' '.join(map(str,row))for row in x)+'\n'+'\n'.join(' '.join(map(str,row))for row in y)+'\n'
    if parameter is not None:text+='\n'.join(' '.join(map(str,np.atleast_1d(row)))for row in parameter)+'\n'
    if storage&1:text+=wire_csr(sx)
    if storage&2:text+=wire_csr(sy)
    mask=undefined(xx,xx if own else yy)if name=='braycurtis'else np.zeros((len(x),len(x)if own else len(y)),dtype=bool)
    cases.append((text,expected,label,mask.ravel()))
differences=[];groups=0;source_rejections=0;undefined_records=[]
for offset,(text,expected,label,unknown)in enumerate(cases):
 p=subprocess.run([str(exe)],input=text,text=True,capture_output=True);lines=p.stdout.splitlines()
 if isinstance(expected,dict):
  source_rejections+=1
  assert p.returncode and 'fscikit-learn:'in p.stdout+p.stderr and not any(s in p.stderr for s in ['SIGFPE','SIGSEGV','Fortran runtime error']),(label,p.stderr)
  continue
 if p.returncode or len(lines)!=7:
  for suffix,data in [('in',text),('stdout',p.stdout),('stderr',p.stderr)]: (r/f'{a.report.stem}_failure_{offset}.{suffix}').write_text(data)
  differences.append(dict(case=label,field='native failure',stderr=p.stderr));continue
 assert all(np.array_equal(np.fromstring(lines[k],sep=' ',dtype=int).astype(bool),unknown)for k in [2,4]),label
 if unknown.any():undefined_records.append(dict(case=label,mask=unknown.tolist()))
 for field,line,want in zip(['type','first','repeat'],[lines[k]for k in [0,1,3]],expected):
  groups+=1
  if field=='type':assert line==want,(label,line,want);continue
  actual=np.fromstring(line,sep=' ');want=np.asarray(want)
  bad=~np.isclose(actual,want,rtol=2e-12,atol=2e-12,equal_nan=True)
  if bad.any():differences.append(dict(case=label,field=field,actual=actual.tolist(),source=want.tolist(),bad=bad.tolist(),undefined=unknown.tolist(),defined_differences=int(np.count_nonzero(bad&~unknown))))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=dict(configurations=len(cases),source_rejections=source_rejections,groups=groups,differences=differences,failed_groups=len(differences),undefined_records=undefined_records,input_sha256=hashlib.sha256(''.join(c[0]for c in cases).encode()).hexdigest(),executable_sha256=sha(exe),sha256={str(p):sha(p)for p in [Path(__file__),Path('src/sklearn_distance_metrics.f90'),Path('src/sklearn_metrics.f90'),Path('app/reference_distance_metric_csr_driver.f90')]})
a.report.write_text(json.dumps(report,indent=2)+'\n');print('configs',len(cases),'failed groups',len(differences),'defined differences',sum(c.get('defined_differences',1)for c in differences))

for failure in differences:
 assert failure['case']['metric']=='braycurtis'and failure['field']in ['first','repeat'],failure
 assert failure['defined_differences']==0,failure
report.update(defined_contract_passed=True,strict_parity=not differences and not undefined_records,
 rtol=2e-12,atol=2e-12,native_undefined_policy='quiet_nan_with_mask')
a.report.write_text(json.dumps(report,indent=2)+'\n')
