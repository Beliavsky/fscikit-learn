"""Isolated parameter-space and parameter-search rejection checks."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from compare_function_transformer import ROOT, resolve

CASES={
    'missing_estimator':'search estimator is unallocated','missing_grid':'search param_grid is unallocated',
    'empty_grids':'search requires at least one candidate','axis_name':'axis name is unallocated',
    'axis_values':'axis values are unallocated','axis_empty':'axis values must be nonempty',
    'axis_duplicate':'axis names must be unique','value_kind':'parameter value has invalid kind',
    'value_text':'parameter text is unallocated','grid_distribution':'ParameterGrid does not accept distributions',
    'axis_both':'axis must choose values or a distribution','grid_index_zero':'grid index out of bounds',
    'grid_index_large':'grid index out of bounds','sample_zero':'n_iter must be positive',
    'sample_empty':'ParameterSampler requires at least one grid','sample_seed':'seed must be nonnegative',
    'parameter_missing':'parameter name was not found','wrong_numeric':'parameter value must be numeric',
    'wrong_integer':'parameter value must be integer','wrong_logical':'parameter value must be logical',
    'wrong_text':'parameter value must be text','empty_x':'X must be nonempty',
    'target_count':'X and y sample counts differ','nonfinite_x':'X contains NaN or infinity',
    'nonfinite_y':'search targets must be finite','invalid_task':'search task must be',
    'cv_one':'invalid KFold n_splits','cv_large':'invalid KFold n_splits','cv_empty':'at least one CV split',
    'cv_unallocated':'CV indices are unallocated','cv_empty_train':'CV splits must be nonempty',
    'cv_empty_test':'CV splits must be nonempty','cv_train_zero':'train index out of bounds',
    'cv_test_large':'test index out of bounds','unsupported_scorer':'unsupported native regression search scorer',
    'error_score':'supports only error_score=raise','unknown_parameter':'unsupported native search parameter',
    'parameter_type':'parameter value must be numeric','nested_parameter':'no supported nested parameter',
    'bad_nested_name':'invalid nested parameter name','unfitted_predict':'search is not fitted',
    'no_refit_predict':'search prediction requires refit','wrong_predict_role':'predict_class requires a classifier',
    'wrong_proba_role':'predict_proba requires a classifier','wrong_decision_role':'decision_function requires a classifier',
    'multimetric_missing':'multimetric refit requires a valid refit_metric','scorers_empty':'scorers must be nonempty',
    'scorers_duplicate':'scorer names must be unique','scorer_name':'scorer name must be nonempty',
    'scorer_conflict':'choose scorer_func or scorers','refit_index_zero':'refit callback index out of bounds',
    'refit_index_large':'refit callback index out of bounds','refit_conflict':'refit callback requires refit=true',
    'weight_count':'fit weight count mismatch','weight_nan':'fit weights must be finite',
    'weight_negative':'weights must be nonnegative','classifier_continuous':'classifier targets must be integral',
    'classifier_large':'classifier label exceeds integer range','classifier_cv':'cv exceeds every class count',
    'log_loss_classes':'log loss class count mismatch','unknown_pipeline_step':'unknown Pipeline step',
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
    parser.add_argument('--executable',type=Path);args=parser.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile,'search_invalid_driver')
    checks=[]
    for name,message in CASES.items():
        run=subprocess.run([str(exe),name],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and message in run.stdout,(name,run.stdout)
        checks.append(dict(name=name,expected=message,passed=True))
    report=dict(profile=args.profile,checks=checks,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'validation/results').mkdir(exist_ok=True)
    (ROOT/'validation/results'/f'search_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} isolated search/parameter-space invalid-input checks passed')


if __name__=='__main__':main()
