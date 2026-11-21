"""One-process-per-case validation of IterativeImputer diagnostics."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from compare_iterative_impute import ROOT,FLAGS

CASES={
    'unfitted':'is not fitted','names_unfitted':'is not fitted',
    'negative_iter':'max_iter must be nonnegative','huge_iter':'sequence exceeds integer capacity',
    'negative_tol':'tol must be nonnegative','nan_tol':'tol must be finite','inf_tol':'tol must be finite',
    'negative_nearest':'n_nearest_features must be nonnegative','negative_seed':'random_state must be nonnegative',
    'strategy':'unknown initial strategy','order':'unknown imputation order',
    'nan_fill':'fill_value must be finite','inf_fill':'fill_value must be finite',
    'infinite_fit':'X contains infinity','negative_infinite_fit':'X contains infinity',
    'finite_sentinel_nan':'finite-sentinel input contains NaN or infinity',
    'nan_sentinel':'missing sentinel must be finite','inf_sentinel':'missing sentinel must be finite',
    'min_nan':'bounds contain NaN','max_nan':'bounds contain NaN',
    'equal_bounds':'min_value must be below max_value','reversed_bounds':'min_value must be below max_value',
    'min_vector':'min_values width mismatch','max_vector':'max_values width mismatch',
    'nan_vector':'bounds contain NaN','reversed_vector':'min_value must be below max_value',
    'empty_fit':'X must be nonempty','zero_width_fit':'X must be nonempty',
    'query_width':'feature count mismatch','empty_query':'X must be nonempty','query_inf':'X contains infinity',
    'name_width':'feature-name width mismatch',
    'estimator_iterations':'invalid BayesianRidge iteration parameters','estimator_prior':'hyperpriors must be positive',
}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path)
    ap.add_argument('--profile',choices=['checked','release'],default='checked');args=ap.parse_args()
    if args.executable:exe=args.executable.resolve()
    else:
        flags=['--profile','release'] if args.profile=='release' else ['--flag',FLAGS]
        run=subprocess.run(['fpm','run',*flags,'--target','iterative_impute_invalid_driver','--runner'],cwd=ROOT,
                           text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/s.strip().strip('"') for s in run.stdout.splitlines()
               if s.strip().strip('"').endswith(('iterative_impute_invalid_driver.exe','iterative_impute_invalid_driver'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,run.stdout;exe=paths[0]
    checks=[]
    for case,expected in CASES.items():
        run=subprocess.run([str(exe),case],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and expected in run.stdout,(case,run.stdout)
        checks.append(dict(name=case,expected=expected,passed=True))
    report=dict(profile=args.profile,checks=checks,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (ROOT/'validation/results'/f'iterative_impute_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} isolated iterative-imputer invalid-input checks passed')

if __name__=='__main__':main()
