"""Hamming categorical/multilabel target, weight and raw CSR guards."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
CASES={'vector_size':'invalid label sizes','weight_length':'sample weight length mismatch','zero_weight_normalized':'weights sum to zero','empty_vector_weight':'weights sum to zero','empty_unweighted_multi':'multilabel targets require samples','int64_range':'integral classification targets','matrix_weight_length':'sample weight length mismatch','continuous':'integral classification targets','target_nan':'targets must be finite','target_inf':'targets must be finite','dense_multioutput':'reject multioutput targets','dense_zero_width':'reject zero-width targets','empty_column':'reject an empty column target','mixed_format':'reject mixed vector and multilabel targets','matrix_shape':'indicator shapes differ','csr_column':'require CSR multilabel indicators','csr_nonindicator':'require CSR indicator values','csr_fractional':'integral classification targets','csr_nan':'targets must be finite','csr_inf':'targets must be finite','csr_unallocated':'CSR arrays are unallocated','csr_bad_index':'CSR column index is out of bounds','csr_bad_pointer':'CSR pointer endpoints are invalid','csr_negative_shape':'invalid CSR shape'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_subset_metrics.f90',ROOT/'app/hamming_loss_invalid_driver.f90',a.executable];hashes={str(p):sha(p) for p in paths};checks=[]
 for case,message in CASES.items():
  r=subprocess.run([str(a.executable),case],text=True,capture_output=True);text=r.stdout+r.stderr
  assert r.returncode and message in text and 'ERROR STOP' in text and not any(v in text for v in ['SIGFPE','Segmentation fault','Fortran runtime error','invalid hamming loss case accepted']),(case,text)
  checks.append(dict(case=case,passed=True))
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,hashes=hashes,checks=checks),indent=2)+'\n');print(len(checks),'controlled Hamming loss diagnostics PASS')
if __name__=='__main__':main()
