"""Sparse precomputed DBSCAN diagonal insertion and cancellation regressions."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import scipy,sklearn
from scipy.sparse import csr_matrix,issparse
from sklearn.cluster import DBSCAN

def fixtures():
 for n,unsorted,missing_row,duplicate,weighting in itertools.product(
   [7,999,1000,1999,2000,2001,3001],[False,True],[0,-1],[False,True],range(3)):
  values=[];indices=[];indptr=[0]
  for i in range(n):
   cols=[]
   if i != (0 if missing_row==0 else n-1):cols.append(i)
   if i==0:cols.extend([1,2])
   if unsorted and i==1:cols=[2,1]
   if duplicate and i==2:cols.append(i)
   indices.extend(cols);values.extend([0.]*len(cols));indptr.append(len(values))
  x=csr_matrix((values,indices,indptr),shape=(n,n))
  weights=np.zeros(n)
  weights[:3]=[[1e16,-1e16,1.],[1.,1.,1.],[2.,-1.,1.]][weighting]
  yield dict(n=n,unsorted=unsorted,missing_row=missing_row,duplicate=duplicate,weighting=weighting),x,weights

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--executable',type=Path,required=True)
 ap.add_argument('--report',type=Path,required=True)
 a=ap.parse_args();records=[];payloads=[]
 for case,x,weights in fixtures():
  before=(x.indptr.copy(),x.indices.copy(),x.data.copy())
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');model=DBSCAN(metric='precomputed',eps=.1,min_samples=1).fit(x,sample_weight=weights)
  assert all(np.array_equal(before[i],v)for i,v in enumerate([x.indptr,x.indices,x.data]))
  c=model.components_
  expected=dict(labels=model.labels_.tolist(),core=(model.core_sample_indices_+1).tolist(),
   data=c.data.tolist()if issparse(c)else[],indices=(c.indices+1).tolist()if issparse(c)else[],
   indptr=(c.indptr+1).tolist()if issparse(c)else[])
  records.append(dict(case=case,source=expected,differences=[]))
  text=f'{x.shape[0]} {x.shape[1]} {x.nnz} 1 .1 1\n'
  for v in [x.indptr+1,x.indices+1,x.data,weights]:text+=' '.join(map(str,v))+'\n'
  payloads.append(text)
 a.report.parent.mkdir(parents=True,exist_ok=True)
 payload=''.join(payloads);a.report.with_suffix('.in').write_text(payload)
 native=subprocess.run([str(a.executable)],input=payload,text=True,capture_output=True)
 a.report.with_suffix('.stdout').write_text(native.stdout);a.report.with_suffix('.stderr').write_text(native.stderr)
 lines=native.stdout.splitlines()
 for i,record in enumerate(records):
  chunk=lines[i*6:i*6+6];record['native']={}
  if len(chunk)!=6:record['differences'].append('missing native output');continue
  if list(map(int,chunk[0].split()))!=[len(record['source']['core']),record['case']['n']]:record['differences'].append('metadata')
  for name,line in zip(['labels','core','data','indices','indptr'],chunk[1:]):
   values=np.fromstring(line,sep=' ');record['native'][name]=values.tolist()
   try:np.testing.assert_array_equal(values,record['source'][name])
   except AssertionError:record['differences'].append(name)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 report=dict(configurations=len(records),records=records,failed_configurations=sum(bool(r['differences'])for r in records),
  native_returncode=native.returncode,input_sha256=hashlib.sha256(payload.encode()).hexdigest(),
  sklearn=sklearn.__version__,scipy=scipy.__version__,numpy=np.__version__,
  sha256={str(p):sha(p)for p in [Path(__file__),a.executable,Path('src/sklearn_dbscan.f90')]})
 a.report.write_text(json.dumps(report,indent=2)+'\n')
 print({k:v for k,v in report.items()if k not in ['records','sha256']})
 assert not native.returncode and not report['failed_configurations']

if __name__=='__main__':main()
