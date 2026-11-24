"""Conformance of internal scalar index ordering to the installed NumPy AVX512 backend."""
from pathlib import Path
import argparse,hashlib,io,json,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_numpy_ordering.f90',ROOT/'app/reference_numpy_ordering_driver.f90',a.executable];hashes={str(p):sha(p) for p in paths}
 rng=np.random.default_rng(1234);fixtures=[]
 for n in list(range(1,270))+[300,511,512,513,1023,1024,1025,4096,10000]:
  for mode in range(8):
   if mode==0:x=rng.normal(size=n)
   elif mode==1:x=rng.integers(-3,4,size=n).astype(float)
   elif mode==2:x=np.ones(n)
   elif mode==3:x=np.arange(n,dtype=float)
   elif mode==4:x=np.arange(n,0,-1,dtype=float)
   elif mode==5:x=np.minimum(np.arange(n),np.arange(n,0,-1)).astype(float)
   elif mode==6:
    x=rng.integers(-1,2,n).astype(float);x[::2]*=-0.
   else:
    x=rng.integers(-2,3,n).astype(float);x[::7]=np.inf;x[::11]=-np.inf
   for k in sorted({0,n//2,n-1,max(0,n-2),max(0,n-3)}):fixtures.append((n,mode,k,x.copy()))
 s=io.StringIO();s.write(str(len(fixtures))+'\n')
 for n,mode,k,x in fixtures:s.write(f'{n} {k+1}\n');np.savetxt(s,x.reshape(1,-1),fmt='%.17g')
 p=subprocess.run([str(a.executable)],input=s.getvalue(),capture_output=True,text=True,check=True)
 lines=iter(p.stdout.splitlines());checks=[];configs=[]
 for i,(n,mode,k,x) in enumerate(fixtures):
  configs.append(dict(id=i,n=n,mode=mode,kth=k,data_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
  for field,want in [('argsort',np.argsort(x)),('argpartition',np.argpartition(x,k))]:
   got=np.fromstring(next(lines),sep=' ',dtype=np.int64)-1
   np.testing.assert_array_equal(got,want,err_msg=str((n,mode,k,field)))
   checks.append(dict(configuration=i,field=field,passed=True))
 assert next(lines,None) is None
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,numpy=np.__version__,cpu_features=np._core._multiarray_umath.__cpu_features__,baseline='NumPy 2.2.6 AVX512 float64 index ordering; scalar Fortran adaptation',hashes=hashes,configurations=configs,checks=checks),indent=2)+'\n')
 print(len(configs),'configurations',len(checks),'exact index comparisons PASS')
if __name__=='__main__':main()
