"""Controlled diagnostics for weighted multioutput variance scores."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
CASES={
 'empty_rows':'nonempty targets','empty_outputs':'nonempty targets',
 'row_shape':'target shapes differ','column_shape':'target shapes differ','mixed_shape':'target shapes differ',
 'target_nan':'targets must be finite','prediction_inf':'targets must be finite',
 'weight_length':'sample weight length','empty_weights':'sample weight length',
 'zero_weight':'weights sum to zero','singleton_ev_zero_weight':'weights sum to zero',
 'output_length':'output weight length','empty_output_weights':'output weight length',
 'single_output_custom':'require multiple outputs','output_nan':'output weights must be finite',
 'output_inf':'output weights must be finite','zero_output_weight':'output weights sum to zero',
 'bad_mode':'invalid variance multioutput mode','raw_scalar':'use *_raw',
 'singleton_bad_output_length':'output weight length'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 paths=[a.executable,Path(__file__),ROOT/'src/sklearn_variance_scores.f90',ROOT/'app/variance_scores_invalid_driver.f90'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before={str(p):sha(p) for p in paths};checks=[]
 for case,message in CASES.items():
  r=subprocess.run([str(a.executable),case],text=True,capture_output=True);text=r.stdout+r.stderr
  assert r.returncode and 'ERROR STOP' in text and message in text and not any(v in text for v in ['SIGFPE','Segmentation fault','Fortran runtime error','invalid variance score case accepted']),(case,text)
  checks.append(dict(case=case,passed=True))
 assert before=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,hashes=before,checks=checks),indent=2)+'\n');print(len(checks),'variance score diagnostics PASS')
if __name__=='__main__':main()
