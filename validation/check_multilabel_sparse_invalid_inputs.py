"""Sparse MultiLabelBinarizer entry-point and storage guards."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
CASES={'transform_unfitted':'MultiLabelBinarizer is not fitted','inverse_unfitted':'MultiLabelBinarizer is not fitted','duplicate_classes':'classes contain duplicates','inverse_width':'inverse class count mismatch','inverse_nonbinary':'indicators must be zero or one','inverse_nan':'indicators must be finite','inverse_inf':'indicators must be finite','csr_unallocated':'CSR arrays are unallocated','csr_bad_index':'CSR column index is out of bounds','csr_bad_pointer':'CSR pointer endpoints are invalid','csr_negative_shape':'invalid CSR shape'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_label_binarization.f90',ROOT/'app/multilabel_sparse_invalid_driver.f90',a.executable];hashes={str(p):sha(p) for p in paths};checks=[]
 for case,message in CASES.items():
  r=subprocess.run([str(a.executable),case],text=True,capture_output=True);text=r.stdout+r.stderr
  assert r.returncode and message in text and 'ERROR STOP' in text and not any(v in text for v in ['SIGFPE','Segmentation fault','Fortran runtime error','invalid multilabel sparse case accepted']),(case,text)
  checks.append(dict(case=case,passed=True))
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,hashes=hashes,checks=checks),indent=2)+'\n');print(len(checks),'controlled sparse multilabel diagnostics PASS')
if __name__=='__main__':main()
