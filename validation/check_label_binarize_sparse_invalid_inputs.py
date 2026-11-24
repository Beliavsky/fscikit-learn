"""Malformed CSR target guards, independently of unsafe SciPy malformed arrays."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
CASES={'unallocated':'CSR arrays are unallocated','negative_shape':'invalid CSR shape','bad_pointer':'CSR pointer endpoints are invalid','bad_index':'CSR column index is out of bounds'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_label_binarize.f90',ROOT/'app/reference_label_binarize_sparse_driver.f90',a.executable];hashes={str(p):sha(p) for p in paths};checks=[]
 for case,message in CASES.items():
  r=subprocess.run([str(a.executable),case],text=True,capture_output=True);text=r.stdout+r.stderr
  assert r.returncode and message in text and 'ERROR STOP' in text and not any(v in text for v in ['SIGFPE','Segmentation fault','Fortran runtime error','invalid CSR accepted']),(case,text)
  checks.append(dict(case=case,passed=True))
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,hashes=hashes,checks=checks),indent=2)+'\n');print(len(checks),'controlled CSR format diagnostics PASS')
if __name__=='__main__':main()
