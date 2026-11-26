"""Compare DBSCAN constructor and input rejections with actual sklearn calls."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--executable',required=True,type=Path)
ap.add_argument('--report',required=True,type=Path)
a=ap.parse_args()
x=np.ones((3,2))
cases={name:(kwargs,x,None)for name,kwargs in [
 ('eps_zero',dict(eps=0)),('eps_negative',dict(eps=-1)),('eps_nan',dict(eps=np.nan)),
 ('min_zero',dict(min_samples=0)),('leaf_zero',dict(leaf_size=0)),
 ('p_negative',dict(p=-1)),('p_nan',dict(p=np.nan)),
 ('algorithm',dict(algorithm='unknown')),('metric',dict(metric='unknown'))]}
cases.update(weights_nan=({},x,[np.nan,1,1]),weights_inf=({},x,[np.inf,1,1]),
 weights_count=({},x,[1]),input_nan=({},np.array([[np.nan,1],[1,1]]),None),
 input_inf=({},np.array([[np.inf,1],[1,1]]),None),input_empty=({},x[:0],None),
 precomputed_shape=(dict(metric='precomputed'),x,None),
 precomputed_negative=(dict(metric='precomputed'),np.array([[-1.]]),None),
 sparse_negative=(dict(metric='precomputed'),csr_matrix([[-1.]]),None),
 sparse_empty=({},csr_matrix(x[:0]),None))
checks=[]
for name,(kwargs,xx,weights)in cases.items():
 try:DBSCAN(**kwargs).fit(xx,sample_weight=weights)
 except (ValueError,TypeError)as error:source_error=dict(type=type(error).__name__,message=str(error))
 else:raise AssertionError(('Source accepted',name))
 result=subprocess.run([str(a.executable),name],capture_output=True,text=True)
 text=result.stdout+result.stderr
 passed=result.returncode!=0 and 'fscikit-learn:'in text and 'invalid DBSCAN input was accepted'not in text and not any(t in text for t in ['SIGFPE','SIGSEGV','Fortran runtime error'])
 checks.append(dict(case=name,source_error=source_error,native_output=text,passed=passed))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=dict(count=len(checks),checks=checks,passed=all(c['passed']for c in checks),sha256={str(p):sha(p)for p in [Path(__file__),a.executable]})
a.report.parent.mkdir(parents=True,exist_ok=True)
a.report.write_text(json.dumps(report,indent=2)+'\n')
print('DBSCAN invalid checks',len(checks),'passed',report['passed'])
assert report['passed']
