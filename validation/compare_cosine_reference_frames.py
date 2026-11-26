"""Cosine numerical chunks with independent physical buffers versus sklearn."""
from pathlib import Path
import os,subprocess,json,hashlib,warnings,argparse
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn.metrics.pairwise as source
ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();r=a.report.parent;r.mkdir(parents=True,exist_ok=True);exe=a.executable
rng=np.random.RandomState(912);reports=[];input_hash=hashlib.sha256()
for n,p in [(1,32),(5,1),(65,16),(66,16),(65,32),(66,32)]:
 x=np.round(rng.normal(size=(n,p)))
 for own in [False,True]:
  y=x.copy()
  for reference in [1.,.0027]:
   for physical in [.0013,1.]:
    with warnings.catch_warnings():
     warnings.simplefilter('ignore')
     expected=np.vstack(list(source.pairwise_distances_chunked(x,None if own else y,metric='cosine',working_memory=reference)))
    text=f'{n} {n} {p} {int(own)} {physical} {reference}\n'+'\n'.join(' '.join(map(str,row))for row in x)+'\n'
    if not own:text+='\n'.join(' '.join(map(str,row))for row in y)+'\n'
    input_hash.update(text.encode());proc=subprocess.run([str(exe)],input=text,text=True,capture_output=True)
    assert proc.returncode==0,proc.stderr
    lines=proc.stdout.splitlines();actual=np.empty_like(expected);index=0;seen=[]
    while lines[index]!='END':
     first,count=map(int,lines[index].split());index+=1
     actual[first:first+count]=np.fromstring(lines[index],sep=' ').reshape(count,n);index+=1;seen.extend(range(first,first+count))
    assert seen==list(range(n))and index==len(lines)-1
    different=np.argwhere(actual!=expected)
    row=dict(n=n,p=p,own=own,physical_memory=physical,reference_memory=reference,fields=n*n,failed=len(different),differences=[dict(i=int(i),j=int(j),actual=float(actual[i,j]),source=float(expected[i,j]))for i,j in different])
    reports.append(row);print({k:v for k,v in row.items()if k!='differences'},flush=True)
    if len(different):
     name=f'{a.report.stem}_{len(reports)}'
     for suffix,data in [('in',text),('stdout',proc.stdout),('stderr',proc.stderr)]: (r/f'{name}.{suffix}').write_text(data)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=[Path(__file__),Path(source.__file__),Path('app/reference_cosine_reference_frames_driver.f90'),Path('src/sklearn_pairwise_chunks.f90'),Path('src/sklearn_kernel_dispatch.f90'),exe]
a.report.write_text(json.dumps(dict(reports=reports,input_sha256=input_hash.hexdigest(),sha256={str(p):sha(p)for p in files}),indent=2)+'\n')
assert not any(c['failed']for c in reports)
