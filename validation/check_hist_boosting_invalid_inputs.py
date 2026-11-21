"""Isolated numeric histogram boosting input-contract checks."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from compare_hist_boosting import ROOT,FLAGS

CASES={
    'unfitted_reg':'is not fitted','unfitted_staged':'is not fitted','unfitted_clf':'is not fitted',
    'uint32_negative':'seed must fit uint32','uint32_overflow':'seed must fit uint32',
    'max_iter_zero':'max_iter must be positive','max_iter_negative':'max_iter must be positive',
    'predictor_capacity':'predictor count exceeds integer capacity',
    'max_leaves_one':'invalid max_leaf_nodes','max_leaves_negative':'invalid max_leaf_nodes',
    'depth_negative':'max_depth must be nonnegative',
    'leaf_zero':'invalid min_samples_leaf','leaf_negative':'invalid min_samples_leaf',
    'bins_small':'max_bins must be in [2,255]','bins_large':'max_bins must be in [2,255]',
    'rate_zero':'learning_rate must be positive','rate_negative':'learning_rate must be positive',
    'rate_nan':'learning_rate must be finite','rate_inf':'learning_rate must be finite',
    'l2_negative':'regularization must be nonnegative','l2_nan':'regularization must be finite','l2_inf':'regularization must be finite',
    'tol_negative':'tol must be nonnegative','tol_nan':'tol must be finite','tol_inf':'tol must be finite',
    'patience_zero':'n_iter_no_change must be positive','seed_negative':'random_state must be nonnegative',
    'monotonic_width':'monotonic constraint width mismatch','monotonic_value':'constraints must be -1,0,1',
    'monotonic_minint':'constraints must be -1,0,1','loss_reg':'unknown loss',
    'quantile_low':'quantile must be in [0,1]','quantile_high':'quantile must be in [0,1]',
    'quantile_nan':'quantile must be finite','quantile_inf':'quantile must be finite',
    'poisson_negative':'Poisson y domain','poisson_zero':'Poisson y domain','gamma_zero':'Gamma y domain',
    'gamma_negative':'Gamma y domain','y_nan':'y must be finite','y_inf':'y must be finite',
    'y_count':'target count mismatch','weights_count':'weight count mismatch',
    'weights_zero':'requires positive total weight','weights_negative':'weights must be nonnegative',
    'weights_nan':'weights must be finite','weights_inf':'weights must be finite',
    'weights_overflow':'weight sum overflow','baseline_overflow':'baseline exceeds finite working range',
    'gradient_range':'gradients exceed float32 range','empty_fit':'X must be nonempty','zero_width':'X must be nonempty',
    'query_width':'feature width mismatch','empty_query':'X must be nonempty','loss_clf':'requires log_loss',
    'class_mode':'invalid histogram class_weight','class_weights_width':'class weight width mismatch',
    'class_weights_negative':'class weights must be nonnegative','class_weights_nan':'class weights must be finite',
    'class_weights_inf':'class weights must be finite','class_weights_zero':'class weights require positive total weight',
    'class_weights_overflow':'class weight overflow','class_count':'target count mismatch',
    'multiclass_monotonic':'multiclass monotonicity is unsupported',
    'warm_decrease':'iteration count decreased','warm_width':'warm-start width changed',
    'warm_loss':'warm-start loss/classes changed','warm_class_width':'warm-start classes changed',
    'warm_class_labels':'warm-start classes changed',
}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path)
    ap.add_argument('--profile',choices=['checked','release'],default='checked');args=ap.parse_args()
    if args.executable:exe=args.executable.resolve()
    else:
        flags=['--profile','release'] if args.profile=='release' else ['--flag',FLAGS]
        run=subprocess.run(['fpm','run',*flags,'--target','hist_boosting_invalid_driver','--runner'],cwd=ROOT,
                           text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/s.strip().strip('"') for s in run.stdout.splitlines()
               if s.strip().strip('"').endswith(('hist_boosting_invalid_driver.exe','hist_boosting_invalid_driver'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,run.stdout;exe=paths[0]
    checks=[]
    for case,expected in CASES.items():
        run=subprocess.run([str(exe),case],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and expected in run.stdout,(case,run.stdout)
        checks.append(dict(name=case,expected=expected,passed=True))
    report=dict(profile=args.profile,checks=checks,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (ROOT/'validation/results'/f'hist_boosting_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} isolated histogram boosting invalid-input checks passed')

if __name__=='__main__':main()
