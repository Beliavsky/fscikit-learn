"""Controlled categorical target, weight, kappa label/mode and capacity guards."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
CASES={**{m+'_length':'target lengths differ' for m in ['balanced','mcc','kappa']},**{m+'_weights':'weight length mismatch' for m in ['balanced','mcc','kappa']},'continuous':'integral classification targets','target_nan':'targets must be finite','target_inf':'targets must be finite','int64_range':'targets exceed int64 range','matrix_width':'single-column targets','empty_column':'single-column targets','labels_empty':'labels must not be empty','labels_missing':'must occur in y','label_kind':'unsupported categorical score label kind','kappa_weight_mode':'weights must be none, linear or quadratic','adjusted_no_recalls':'no non-NaN class recalls','adjusted_empty':'no non-NaN class recalls',**{m+'_empty_weights':'weight length mismatch' for m in ['balanced','mcc','kappa']},'matrix_capacity':'confusion matrix exceeds native indexing capacity'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_categorical_scores.f90',ROOT/'app/categorical_scores_invalid_driver.f90',a.executable];hashes={str(p):sha(p) for p in paths};checks=[]
 for case,message in CASES.items():
  r=subprocess.run([str(a.executable),case],text=True,capture_output=True);text=r.stdout+r.stderr
  assert r.returncode and message in text and 'ERROR STOP' in text and not any(v in text for v in ['SIGFPE','Segmentation fault','Fortran runtime error','invalid categorical score case accepted']),(case,text)
  checks.append(dict(case=case,passed=True))
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,hashes=hashes,checks=checks),indent=2)+'\n');print(len(checks),'controlled categorical score diagnostics PASS')
if __name__=='__main__':main()
