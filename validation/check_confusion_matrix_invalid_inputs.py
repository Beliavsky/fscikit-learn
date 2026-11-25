"""Controlled confusion target, label, normalization, weight and capacity guards."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
CASES={**{m+'_length':'target lengths differ' for m in ['counts','weighted','normalized']},**{m:'weight length mismatch' for m in ['weights_length','normalized_weights','empty_auto_weights','empty_auto_normalized_weights']},'labels_empty':'labels must not be empty','labels_missing':'must occur in y','label_kind':'unsupported categorical score label kind','continuous':'integral classification targets','target_nan':'targets must be finite','target_inf':'targets must be finite','int64_range':'targets exceed int64 range',**{m:'single-column targets' for m in ['matrix_width','empty_column','zero_width']},**{m:'normalize must be none, true, pred or all' for m in ['normalize_mode','weighted_normalize_mode']},'matrix_capacity':'confusion matrix exceeds native indexing capacity'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_confusion_matrix.f90',ROOT/'src/sklearn_categorical_scores.f90',ROOT/'app/confusion_matrix_invalid_driver.f90',a.executable];hashes={str(p):sha(p) for p in paths};checks=[]
 for case,message in CASES.items():
  r=subprocess.run([str(a.executable),case],text=True,capture_output=True);text=r.stdout+r.stderr
  assert r.returncode and message in text and 'ERROR STOP' in text and not any(v in text for v in ['SIGFPE','Segmentation fault','Fortran runtime error','invalid confusion matrix case accepted']),(case,text)
  checks.append(dict(case=case,passed=True))
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,hashes=hashes,checks=checks),indent=2)+'\n');print(len(checks),'controlled confusion matrix diagnostics PASS')
if __name__=='__main__':main()
