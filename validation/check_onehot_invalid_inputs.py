"""Controlled native errors for OneHotEncoder parameters, input and typed object results."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
CASES={
 'empty_rows':'nonempty matrix','empty_features':'nonempty matrix','fit_inf':'rejects infinity',
 'unknown_mode':'invalid OneHotEncoder handle_unknown','drop_mode':'invalid OneHotEncoder drop mode',
 'count_negative':'minimum count must be nonnegative','fraction_nan':'frequency fraction must be finite',
 'fraction_negative':'frequency fraction must be in [0,1)','fraction_one':'frequency fraction must be in [0,1)',
 'both_frequencies':'count and fraction are mutually exclusive','maximum_negative':'max_categories must be nonnegative',
 'category_width':'category feature count mismatch','category_unallocated':'category values are unallocated',
 'category_empty':'categories must be nonempty','category_duplicate':'categories contain duplicates',
 'category_unsorted':'numeric categories must be sorted with NaN last','category_nan_order':'numeric categories must be sorted with NaN last',
 'unknown_fit':'unknown category during fit','drop_width':'drop value count mismatch','drop_unknown':'drop category was not found',
 'drop_infrequent':'cannot explicitly drop an infrequent category','fit_name_count':'feature name count mismatch',
 'fit_name_unallocated':'feature name is unallocated','transform_unfitted':'not fitted','inverse_unfitted':'not fitted',
 'names_unfitted':'not fitted','result_unallocated':'inverse result is unallocated','query_width':'feature count mismatch',
 'query_empty':'nonempty matrix','query_inf':'rejects infinity','query_unknown':'unknown category during OneHotEncoder transform',
 'query_name_count':'input name count mismatch','query_name_unallocated':'feature name is unallocated',
 'query_name_difference':'input feature names differ','inverse_width':'inverse encoded width mismatch',
 'inverse_empty':'inverse requires nonempty encoded input','inverse_nan':'inverse rejects NaN or infinity',
 'inverse_inf':'inverse rejects NaN or infinity','inverse_zero_sum':'inverse row has zero sum without a drop',
 'inverse_broadcast':'inverse zero-sum category broadcast mismatch','inverse_zero_width':'inverse requires nonempty encoded input',
 'result_none':'inverse result contains None or infrequent category','result_infrequent':'inverse result contains None or infrequent category',
 'bad_callback':'name callback returned unallocated text','csr_bad_index':'CSR column index is out of bounds'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_one_hot_encoding.f90',ROOT/'app/onehot_invalid_driver.f90',a.executable];hashes={str(p):sha(p) for p in paths};checks=[]
 for name,message in CASES.items():
  p=subprocess.run([str(a.executable),name],text=True,capture_output=True);text=p.stdout+p.stderr
  assert p.returncode!=0 and message in text,(name,message,text)
  assert not any(s in text for s in ['SIGFPE','Segmentation fault','Fortran runtime error','invalid case was accepted']),(name,text)
  checks.append(dict(case=name,passed=True))
 assert hashes=={str(p):sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,hashes=hashes,checks=checks),indent=2)+'\n');print(len(checks),'controlled OneHotEncoder diagnostics PASS')
if __name__=='__main__':main()
