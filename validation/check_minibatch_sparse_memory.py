"""Peak memory for 384-by-200000 MiniBatch CSR fit, stream update and transform; no dense design."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,subprocess,sys

def peak_memory(process):
 if os.name=='nt':
  from ctypes import wintypes
  class Counters(ctypes.Structure):
   _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
  query=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo;query.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];query.restype=wintypes.BOOL
  counters=Counters();counters.cb=ctypes.sizeof(counters);assert query(int(process._handle),ctypes.byref(counters),counters.cb),ctypes.get_last_error();return counters.PeakWorkingSetSize
 import resource
 return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*(1 if sys.platform=='darwin' else 1024)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();binary=hashlib.sha256(a.executable.read_bytes()).hexdigest();records=[]
 for solver in ['frobenius','kl']:
  p=subprocess.Popen([str(a.executable),'memory' if solver=='frobenius' else 'memory_kl'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);stdout,stderr=p.communicate();peak=peak_memory(p);limit=128*1024**2
  passed=p.returncode==0 and 0<peak<limit and 'Large sparse MiniBatch memory case: PASS' in stdout
  records.append(dict(solver=solver,passed=passed,peak_working_set_bytes=peak,limit_bytes=limit,n_samples=384,n_features=200000,n_components=2,dense_design_bytes=384*200000*8,returncode=p.returncode,stdout=stdout,stderr=stderr))
 assert binary==hashlib.sha256(a.executable.read_bytes()).hexdigest()
 a.report.write_text(json.dumps(dict(passed=all(r['passed'] for r in records),records=records,executable_sha256=binary),indent=2)+'\n');print('MiniBatch CSR memory',[(r['solver'],r['peak_working_set_bytes'],r['passed']) for r in records]);raise SystemExit(not all(r['passed'] for r in records))
if __name__=='__main__':main()
