"""Isolated input rejections for neighborhood components analysis."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from compare_nca import ROOT,FLAGS

CASES={
    'unfitted':'NCA is not fitted','empty_rows':'X must be nonempty','empty_columns':'X must be nonempty',
    'one_row':'requires at least two samples','wrong_labels':'sample counts differ',
    'nan_x':'X contains NaN or infinity','inf_x':'X contains NaN or infinity',
    'negative_components':'invalid n_components','large_components':'invalid n_components',
    'zero_iterations':'max_iter must be positive','negative_iterations':'max_iter must be positive',
    'negative_tol':'tol must be nonnegative','nan_tol':'tol must be finite','inf_tol':'tol must be finite',
    'large_tol':'tol exceeds working range','negative_seed':'random_state must be nonnegative',
    'unknown_init':'unknown init','empty_init':'X must be nonempty','init_width':'init_matrix shape mismatch',
    'init_height':'init_matrix shape mismatch','init_components':'init_matrix and n_components disagree',
    'nan_init':'X contains NaN or infinity','pca_samples':'PCA components exceed',
    'lda_all_classes':'LDA needs more samples','lda_one_class':'LDA needs more samples',
    'lda_within_rank':'LDA within-class rank is zero','lda_between_rank':'LDA between-class rank is zero',
    'warm_width':'warm_start feature width mismatch','query_width':'transform feature width mismatch',
    'query_nan':'X contains NaN or infinity','query_inf':'X contains NaN or infinity',
    'query_overflow':'transform exceeds working range','large_input':'input exceeds SVD working range',
    'gradient_shape':'component shape mismatch','gradient_output':'gradient shape mismatch',
    'gradient_one_row':'requires at least two samples','gradient_nan':'components must be finite',
    'gradient_embedding':'embedded products exceed working range',
    'gradient_distances':'squared distances exceed working range',
    'gradient_products':'gradient products exceed working range',
}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);args=ap.parse_args()
    if args.executable:exe=args.executable.resolve()
    else:
        options=['--profile','release'] if args.profile=='release' else ['--flag',FLAGS]
        r=subprocess.run(['fpm','run',*options,'--target','nca_invalid_driver','--runner'],cwd=ROOT,
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/s.strip().strip('"') for s in r.stdout.splitlines()
               if s.strip().strip('"').endswith(('nca_invalid_driver','nca_invalid_driver.exe'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,r.stdout;exe=paths[0]
    checks=[]
    for case,message in CASES.items():
        r=subprocess.run([str(exe),case],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert r.returncode!=0 and 'fscikit-learn:' in r.stdout and message in r.stdout,(case,r.stdout)
        checks.append(dict(name=case,expected=message,passed=True))
    report=dict(profile=args.profile,checks=checks,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (ROOT/'validation/results'/f'nca_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} isolated NCA invalid-input checks passed')


if __name__=='__main__':main()
