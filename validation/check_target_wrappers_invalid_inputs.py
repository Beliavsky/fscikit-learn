"""Isolated rejection checks for function and target wrappers."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from compare_function_transformer import ROOT,FLAGS

CASES={
    'ft_control':'invalid feature_names_out',
    'ft_fit_empty':'X must be nonempty',
    'ft_fit_nan':'X contains NaN or infinity',
    'ft_fit_inf':'X contains NaN or infinity',
    'ft_fit_names':'fit name count mismatch',
    'ft_unallocated_name':'feature name is unallocated',
    'ft_transform_width':'transform feature width mismatch',
    'ft_inverse_nan':'X contains NaN or infinity',
    'ft_inverse_empty':'X must be nonempty',
    'ft_names_none':'output feature names are not configured',
    'ft_names_unknown':'names require fit validation or input names',
    'ft_names_width':'input name count mismatch',
    'ft_names_differ':'input feature names differ',
    'ft_names_length':'input feature names differ',
    'target_unfitted':'is not fitted',
    'target_missing_inverse':'requires both func and inverse_func',
    'target_missing_forward':'requires both func and inverse_func',
    'target_conflict':'cannot combine transformer and functions',
    'target_transformer':'unsupported TransformedTargetRegressor transformer',
    'target_regressor':'unsupported TransformedTargetRegressor regressor',
    'target_y_nan':'X contains NaN or infinity',
    'target_y_inf':'X contains NaN or infinity',
    'target_sample_count':'X and y sample counts differ',
    'target_weight_count':'weight length mismatch',
    'target_weight_nan':'weights must be finite',
    'target_weight_inf':'weights must be finite',
    'target_weight_negative':'weights must be nonnegative',
    'target_latent_nan':'X contains NaN or infinity',
    'target_latent_rows':'transformed sample count mismatch',
    'target_joint_tree':'base has no joint multi-target fit',
    'target_query_width':'query feature width mismatch',
    'target_vector_multi':'vector prediction requires one output',
    'target_score_shape':'score output shape mismatch',
}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);args=ap.parse_args()
    if args.executable:exe=args.executable.resolve()
    else:
        options=['--profile','release'] if args.profile=='release' else ['--flag',FLAGS]
        r=subprocess.run(['fpm','run',*options,'--target','target_wrappers_invalid_driver','--runner'],cwd=ROOT,
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/s.strip().strip('"') for s in r.stdout.splitlines()
               if s.strip().strip('"').endswith(('target_wrappers_invalid_driver','target_wrappers_invalid_driver.exe'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,r.stdout;exe=paths[0]
    checks=[]
    for case,message in CASES.items():
        r=subprocess.run([str(exe),case],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert r.returncode!=0 and 'fscikit-learn:' in r.stdout and message in r.stdout,(case,r.stdout)
        checks.append(dict(name=case,expected=message,passed=True))
    report=dict(profile=args.profile,checks=checks,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (ROOT/'validation/results').mkdir(exist_ok=True)
    (ROOT/'validation/results'/f'target_wrappers_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} isolated target-wrapper invalid-input checks passed')


if __name__=='__main__':main()
