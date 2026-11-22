"""Isolated rejection checks for sequential pipelines."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from compare_function_transformer import ROOT,FLAGS

CASES={
    'unallocated_steps':'steps are unallocated',
    'empty_steps':'requires at least one step',
    'unallocated_name':'step name is unallocated',
    'duplicate_name':'step names must be unique',
    'name_delimiter':'step name contains __',
    'reserved_name':'step name conflicts with a control',
    'action_drop':'invalid step action',
    'model_unallocated':'step model is unallocated',
    'intermediate_regressor':'step must support transform',
    'unsupported_intermediate':'step must support transform',
    'unsupported_final':'unsupported final model',
    'predict_unfitted':'Pipeline is not fitted',
    'proba_unfitted':'Pipeline is not fitted',
    'names_unfitted':'Pipeline is not fitted',
    'inverse_unfitted':'Pipeline is not fitted',
    'transform_estimator':'final estimator has no transform',
    'fit_transform_estimator':'step must support transform',
    'predict_class_regressor':'predict_class requires a classifier',
    'predict_classifier':'vector predict requires a regressor',
    'matrix_predict_classifier':'matrix predict requires a regressor',
    'probability_regressor':'final estimator has no predict_proba',
    'decision_regressor':'final estimator has no decision_function',
    'names_estimator':'final estimator has no output feature names',
    'inverse_estimator':'step has no inverse_transform',
    'missing_regression_target':'regressor requires a target',
    'missing_classification_target':'classifier requires a vector target',
    'nonfinite_target':'regression targets must be finite',
    'classifier_nonfinite':'classifier targets must be finite',
    'classifier_continuous':'classifier requires integral targets',
    'classifier_large':'classifier target exceeds native integer range',
    'target_count':'X and y sample counts differ',
    'weight_count':'fit weight length mismatch',
    'weight_nan':'fit weights must be finite',
    'weight_inf':'fit weights must be finite',
    'weight_negative':'weights must be nonnegative',
    'zero_weight':'weights must have a positive sum',
    'matrix_target_unsupported':'matrix targets require a native matrix-target final estimator',
    'matrix_score_unsupported':'matrix score requires a native matrix-target final estimator',
    'missing_supervised_target':'supervised transformer requires a vector target',
    'supervised_matrix_target':'supervised transformer requires a vector target',
    'feature_name_count':'feature name count mismatch',
    'feature_name_unallocated':'feature name is unallocated',
    'output_name_count':'feature name count mismatch',
    'output_name_mismatch':'input feature names differ',
    'output_name_blanks':'input feature names differ',
    'slice_zero':'slice indices out of bounds',
    'slice_large':'slice indices out of bounds',
    'slice_reverse':'slice indices out of bounds',
    'score_target_count':'score target length mismatch',
    'score_weight_count':'score weight length mismatch',
    'score_weight_nan':'score weights must be finite',
    'score_weight_zero':'score weights must have a nonzero sum',
    'transformer_score':'final transformer has no score',
    'passthrough_predict':'final passthrough has no predict',
    'passthrough_score':'final passthrough has no score',
    'unsupported_inverse':'step has no inverse_transform',
    'transformer_fit_weights':'final transformer has no native fit weights',
    'graph_classifier':'step must support transform',
}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);args=ap.parse_args()
    if args.executable:exe=args.executable.resolve()
    else:
        options=['--profile','release'] if args.profile=='release' else ['--flag',FLAGS]
        r=subprocess.run(['fpm','run',*options,'--target','pipeline_invalid_driver','--runner'],cwd=ROOT,
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/s.strip().strip('"') for s in r.stdout.splitlines()
               if s.strip().strip('"').endswith(('pipeline_invalid_driver','pipeline_invalid_driver.exe'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,r.stdout;exe=paths[0]
    checks=[]
    for case,message in CASES.items():
        r=subprocess.run([str(exe),case],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert r.returncode!=0 and 'fscikit-learn:' in r.stdout and message in r.stdout,(case,r.stdout)
        checks.append(dict(name=case,expected=message,passed=True))
    report=dict(profile=args.profile,checks=checks,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (ROOT/'validation/results').mkdir(exist_ok=True)
    (ROOT/'validation/results'/f'pipeline_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} isolated pipeline invalid-input checks passed')


if __name__=='__main__':main()
