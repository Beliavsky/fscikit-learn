"""Controlled binary likelihood-ratio target, label and weight diagnostics."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
CASES={'target_length':'target lengths differ','continuous':'integral classification targets','target_nan':'targets must be finite','target_inf':'targets must be finite','int64_range':'targets exceed int64 range','multiclass':'only support binary classification','matrix_width':'single-column target','empty_column':'single-column target','empty_target':'require explicit labels','labels_empty':'labels must not be empty','labels_missing':'must occur in y','labels_width':'one or two selected labels','label_kind':'unsupported likelihood ratio label kind','weight_length':'weight length mismatch'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_likelihood_ratios.f90',ROOT/'app/likelihood_ratios_invalid_driver.f90',a.executable];hashes={str(p):sha(p) for p in paths};checks=[]
 for case,message in CASES.items():
  r=subprocess.run([str(a.executable),case],text=True,capture_output=True);text=r.stdout+r.stderr
  assert r.returncode and message in text and 'ERROR STOP' in text and not any(v in text for v in ['SIGFPE','Segmentation fault','Fortran runtime error','invalid likelihood ratio case accepted']),(case,text)
  checks.append(dict(case=case,passed=True))
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,hashes=hashes,checks=checks),indent=2)+'\n');print(len(checks),'controlled likelihood ratio diagnostics PASS')
if __name__=='__main__':main()
