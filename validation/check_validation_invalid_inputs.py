"""Isolated general cross-validation and out-of-fold rejection checks."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from compare_function_transformer import ROOT,resolve

CASES={
 'empty_x':'X must be nonempty','nonfinite_x':'X contains NaN or infinity',
 'target_count':'X and y sample counts differ','nonfinite_y':'CV targets must be finite',
 'unknown_task':'CV task must be','error_score':'CV supports only error_score=raise',
 'cv_one':'invalid KFold n_splits','cv_large':'invalid KFold n_splits',
 'no_splits':'CV requires at least one split','unallocated_indices':'CV split indices are unallocated',
 'empty_train':'CV splits must be nonempty','empty_test':'CV splits must be nonempty',
 'low_train':'CV train index out of bounds','high_train':'CV train index out of bounds',
 'low_test':'CV test index out of bounds','high_test':'CV test index out of bounds',
 'weights_count':'CV fit weight count mismatch','weights_nan':'CV fit weights must be finite',
 'weights_negative':'weights must be nonnegative','scorers_empty':'CV scorers must be nonempty',
 'scorer_name':'CV scorer names must be nonempty','scorer_duplicate':'CV scorer names must be unique',
 'scorer_conflict':'CV choose scorers or scorer_func','unknown_scorer':'unsupported native regression search scorer',
 'scalar_multiple':'cross_val_score requires a single scalar scorer',
 'unsupervised_scorer':'CV no-target scoring requires a callback',
 'unsupervised_weights':'CV unsupervised default fit does not accept weights',
 'classifier_no_target':'CV classifier requires targets',
 'class_continuous':'CV classification targets must be integral',
 'class_large':'CV classification target exceeds integer range','class_cv':'CV fold count exceeds every class count',
 'unknown_method':'CV unsupported response method','regressor_response':'CV classifier response requires a classifier',
 'partition_missing':'requires test indices to form a partition','partition_repeat':'requires test indices to form a partition',
 'partition_duplicate':'requires test indices to form a partition','response_samples':'CV response sample count mismatch',
 'response_widths':'CV fold response widths differ','classes_empty':'CV fold classes must be nonempty',
 'classes_range':'CV encoded class index out of bounds','classes_duplicate':'CV fold classes must be unique',
 'probability_width':'CV probability class width mismatch','decision_binary':'CV irregular missing-class decision output',
 'decision_two_column':'CV missing-class decision function requires more than two fold classes',
 'log_neighbors':'CV estimator has no native predict_log_proba','decision_bayes':'native base has no decision_function',
 'typed_class_regressor':'cross_val_predict_class requires a classifier',
 'typed_class_method':'cross_val_predict_class requires method=predict',
 'typed_class_columns':'cross_val_predict_class requires one response column',
 'typed_class_nonintegral':'CV classification targets must be integral',
}


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
 parser.add_argument('--executable',type=Path);args=parser.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'validation_invalid_driver')
 checks=[]
 for name,message in CASES.items():
  result=subprocess.run([str(exe),name],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  assert result.returncode!=0 and 'fscikit-learn:' in result.stdout and message in result.stdout,(name,result.stdout)
  checks.append(dict(name=name,expected=message,passed=True))
 report=dict(profile=args.profile,checks=checks,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
             validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (ROOT/'validation/results').mkdir(exist_ok=True)
 (ROOT/'validation/results'/f'validation_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2)+'\n')
 print(f'{len(checks)} isolated cross-validation invalid-input checks passed')


if __name__=='__main__':main()
