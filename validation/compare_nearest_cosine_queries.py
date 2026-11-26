"""Dense cosine KNN identity, rounding and physical-block boundaries vs sklearn."""
from pathlib import Path
import argparse,os,hashlib,json,subprocess
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np,sklearn
from sklearn.neighbors import NearestNeighbors
import sklearn.neighbors._base as source
ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--record-differences',action='store_true');a=ap.parse_args()
a.report.parent.mkdir(parents=True,exist_ok=True);rng=np.random.RandomState(441);cases=[]
for n,p in [(n,p)for n in [9,65]for p in [2,16,31,32]]+[(n,16)for n in [1023,1024,1025]]:
 original=rng.normal(size=(n,p));original[rng.random_sample((n,p))<.2]=0
 for kind in (['regular','constant','integer']if n<1000 else['constant','integer']):
  x=original.copy()
  if kind=='constant':x[:]=0
  if kind=='integer':x=np.round(x)
  q=np.vstack([x[:1],np.zeros((1,p)),rng.normal(size=(1,p))])
  for algorithm in ['auto','brute']:
   model=NearestNeighbors(metric='cosine',algorithm=algorithm,n_neighbors=1,n_jobs=1).fit(np.ascontiguousarray(x))
   for implicit,exclude in [(0,0),(1,0),(1,1)]:
    query=None if implicit and exclude else(model._fit_X if implicit else q.copy())
    for k in [1,3]:
     d,ids=model.kneighbors(query,n_neighbors=k)
     label=dict(n=n,p=p,kind=kind,algorithm=algorithm,implicit=implicit,exclude=exclude,k=k)
     m=0 if implicit else len(q)
     text=f'{n} {p} {m} {k} 1 30 0 {implicit} {exclude} cosine 2 {algorithm} 1\n'+'\n'.join(' '.join(map(str,row))for row in x)+'\n'
     if m:text+='\n'.join(' '.join(map(str,row))for row in q)+'\n'
     cases.append((text,[model._fit_method,d.ravel(),(ids+1).ravel()],label))
differences=[];groups=0;maximum=0.
for start in range(0,len(cases),8):
 batch=cases[start:start+8];text=''.join(c[0]for c in batch)
 proc=subprocess.run([str(a.executable)],input=text,text=True,capture_output=True);lines=proc.stdout.splitlines()
 if proc.returncode or len(lines)!=3*len(batch):
  for suffix,value in [('in',text),('stdout',proc.stdout),('stderr',proc.stderr)]:a.report.with_name(f'{a.report.stem}_batch_{start}.{suffix}').write_text(value)
 assert proc.returncode==0 and len(lines)==3*len(batch),(start,proc.stderr)
 for j,(_,expected,label)in enumerate(batch):
  for field,line,want in zip(['method','distances','indices'],lines[3*j:3*j+3],expected):
   groups+=1
   if field=='method':
    if line!=want:differences.append(dict(case=label,field=field,actual=line,source=want))
    continue
   value=np.fromstring(line,sep=' ');want=np.asarray(want)
   try:np.testing.assert_array_equal(value,want)
   except AssertionError:differences.append(dict(case=label,field=field,actual=value.tolist(),source=want.tolist()))
   if value.shape==want.shape:maximum=max(maximum,float(np.max(abs(value-want)/(1+abs(want)))))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=[Path(__file__),Path(source.__file__),Path('src/sklearn_neighbor_query.f90'),Path('src/sklearn_neighbors.f90'),Path('src/sklearn_pairwise_chunks.f90'),Path('src/sklearn_kernel_dispatch.f90'),Path('app/reference_nearest_neighbors_driver.f90')]
report=dict(strict_parity=not differences,configurations=len(cases),groups=groups,inventory=[c[2]for c in cases],input_sha256=hashlib.sha256(''.join(c[0]for c in cases).encode()).hexdigest(),failed_comparison_groups=len(differences),max_scaled_error=maximum,differences=differences,sha256={str(p):sha(p)for p in files},executable_sha256=sha(a.executable),sklearn=sklearn.__version__,numpy=np.__version__)
a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items()if k not in ['differences','inventory']})
if differences and not a.record_differences:raise SystemExit(1)
