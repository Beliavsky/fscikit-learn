"""Isolated SelectFromModel invalid-input diagnostics."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from compare_function_transformer import ROOT, resolve

CASES = {
    'missing_estimator': 'estimator is unallocated',
    'not_fitted': 'SelectFromModel is not fitted',
    'prefit_unfitted': 'prefit estimator is not fitted',
    'missing_target': 'requires targets unless prefit=true',
    'norm_zero': 'norm_order must be nonzero',
    'norm_fraction': 'norm_order must be integral or infinite',
    'norm_nan': 'norm_order is NaN',
    'negative_cap': 'max_features must be nonnegative or -1',
    'large_cap': 'max_features exceeds feature count',
    'negative_cap_callback': 'max_features exceeds feature count',
    'large_cap_callback': 'max_features exceeds feature count',
    'direct_cap_callback': 'callable max_features requires fit',
    'unknown_getter': 'invalid importance_getter',
    'missing_callback': 'importance callback is unassociated',
    'unchosen_callback': 'set importance_getter=callback',
    'empty_importance': 'importances must be nonempty',
    'nonfinite_importance': 'importances must be finite',
    'wrong_importance_width': 'importance feature count mismatch',
    'vector_importance_rows': 'vector importances must have one row',
    'missing_feature_importance': 'estimator has no feature_importances_',
    'nonfinite_coef': 'importances must be finite',
    'bad_threshold': 'threshold must be auto, value, mean, median',
    'bad_scale': 'invalid threshold scale',
    'missing_scale': 'malformed scaled threshold',
    'missing_reference': 'malformed scaled threshold',
    'bad_reference': 'threshold must be auto, value, mean, median',
    'infinite_scale': 'threshold scale must be finite',
    'nan_threshold': 'numeric threshold is NaN',
    'transform_width': 'transform feature count mismatch',
    'inverse_width': 'inverse feature count mismatch',
    'inverse_empty': 'X must be nonempty',
    'empty_x': 'X must be nonempty',
    'nonfinite_x': 'X contains NaN or infinity',
    'direct_names': 'feature names require fit',
    'name_count': 'feature name count mismatch',
    'name_unallocated': 'feature name is unallocated',
    'name_mismatch': 'input feature names differ',
    'target_count': 'X and y sample counts differ',
    'nonfinite_target': 'target must be finite',
    'weight_count': 'weight count mismatch',
    'weight_nan': 'weights must be finite',
    'weight_negative': 'weights must be nonnegative',
    'matrix_base': 'matrix targets require a native multi-task estimator',
    'partial_base': 'partial_fit requires a native SGD estimator',
    'classifier_continuous': 'classifier targets must be integral',
    'classifier_large': 'classifier label exceeds integer range',
    'no_auto_importance': 'unsupported importance estimator; use a callback',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['checked', 'release'], default='checked')
    parser.add_argument('--executable', type=Path)
    args = parser.parse_args()
    exe = args.executable.resolve() if args.executable else resolve(args.profile, 'model_selector_invalid_driver')
    checks = []
    for name, message in CASES.items():
        run = subprocess.run([str(exe), name], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert run.returncode != 0 and 'fscikit-learn:' in run.stdout and message in run.stdout, (name, run.stdout)
        checks.append(dict(name=name, expected=message, passed=True))
    report = dict(profile=args.profile, checks=checks, native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'validation/results').mkdir(exist_ok=True)
    (ROOT/'validation/results'/f'model_selector_invalid_{args.profile}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'{len(checks)} isolated SelectFromModel invalid-input checks passed')


if __name__ == '__main__':
    main()
