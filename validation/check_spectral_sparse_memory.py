"""Measure the native 32-by-1000000 Spectral CSR test's process high water."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,subprocess,sys

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 p=subprocess.Popen([str(a.executable),'--memory'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);stdout,stderr=p.communicate()
 if os.name=='nt':
  from ctypes import wintypes
  class Counters(ctypes.Structure):
   _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
  query=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo;query.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];query.restype=wintypes.BOOL
  c=Counters();c.cb=ctypes.sizeof(c);assert query(int(p._handle),ctypes.byref(c),c.cb),ctypes.get_last_error();peak=c.PeakWorkingSetSize
 else:
  import resource
  peak=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
 limit=128*1024**2;passed=p.returncode==0 and 0<peak<limit and 'Spectral CSR 32 by 1000000 memory: PASS' in stdout
 a.report.write_text(json.dumps(dict(passed=passed,peak_working_set_bytes=peak,limit_bytes=limit,n_features=1000000,n_samples=32,dense_design_bytes=32*1000000*8,returncode=p.returncode,stdout=stdout,stderr=stderr,executable_sha256=hashlib.sha256(a.executable.read_bytes()).hexdigest()),indent=2)+'\n')
 print('Spectral CSR memory',peak,'bytes; passed',passed);raise SystemExit(not passed)
if __name__=='__main__':main()
