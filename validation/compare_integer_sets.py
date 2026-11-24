"""Internal nonnegative-index insertion order versus CPython's set.add protocol."""
from pathlib import Path
import argparse,hashlib,io,json,platform,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def fixtures():
 rng=np.random.default_rng(518331)
 for n in list(range(80))+[1228,1229,4914,4915,19660,19661,50000,78642,78643,100000]:
  for pattern in range(6):
   keys=np.arange(n,dtype=np.int64)
   if pattern==1:keys=keys[::-1]
   elif pattern==2:keys=rng.permutation(keys)
   elif pattern==3:keys=(keys*65536)%2147483647
   elif pattern==4:keys=rng.choice([0,8,16,24,32,1048576,2147483647],n)
   elif pattern==5:keys=rng.integers(0,2147483648,n,dtype=np.int64)
   yield dict(length=n,pattern=pattern),keys
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_integer_sets.f90',ROOT/'app/reference_integer_sets_driver.f90',ROOT/'LICENSES/Python-3.13.3-LICENSE',a.executable];hashes={str(p):sha(p) for p in paths}
 stream=io.StringIO();items=list(fixtures());stream.write(str(len(items))+'\n');expected=[];configs=[]
 for ident,(cfg,keys) in enumerate(items):
  cfg=dict(id=ident,**cfg,keys_sha256=hashlib.sha256(keys.tobytes()).hexdigest());configs.append(cfg)
  stream.write(str(len(keys))+'\n'+' '.join(map(str,keys))+'\n');value=set()
  for key in keys:value.add(int(key))
  expected.append(list(value))
 r=subprocess.run([str(a.executable)],input=stream.getvalue(),capture_output=True,text=True);assert r.returncode==0,(r.stdout[-2000:],r.stderr)
 lines=r.stdout.splitlines();assert len(lines)==len(items);checks=[]
 for ident,(line,want) in enumerate(zip(lines,expected)):
  index,*got=map(int,line.split());assert index==ident+1;assert got==want,(configs[ident],got[:30],want[:30]);checks.append(dict(configuration=ident,field='iteration_order',passed=True))
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,python=sys.version,implementation=platform.python_implementation(),hashes=hashes,configurations=configs,checks=checks),indent=2)+'\n');print(len(checks),'CPython integer-set protocol groups PASS')
if __name__=='__main__':main()
