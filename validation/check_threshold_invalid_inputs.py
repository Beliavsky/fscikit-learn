"""Isolated diagnostic checks for the native binary threshold wrappers."""
from pathlib import Path
import argparse
import ast
import json
import subprocess
ROOT=Path(__file__).resolve().parents[1]
CASES = {'th_unfitted': 'threshold classifier is not fitted',
 'th_tuned_unfitted': 'threshold classifier is not fitted',
 'th_multiclass': 'threshold fit requires two target classes',
 'th_oneclass': 'threshold fit requires two target classes',
 'th_method': 'unknown threshold response method',
 'th_no_proba': 'threshold base has no predict_proba',
 'th_no_decision': 'threshold base has no decision_function',
 'th_bad_base': 'threshold unsupported classifier base',
 'th_prefit_unfitted': 'threshold prefit base is not fitted',
 'th_prefit_single': 'threshold requires a binary fitted base',
 'th_cutoff_nan': 'threshold cutoff must not be NaN',
 'th_positive': 'threshold positive label is not a fitted class',
 'th_query_width': 'threshold query width mismatch',
 'th_query_nan': 'X contains NaN or infinity',
 'th_weight_length': 'threshold score weight length mismatch',
 'th_weight_nan': 'threshold score weights must be finite',
 'th_weight_negative': 'threshold weights must be nonnegative',
 'th_weight_zero': 'threshold score weights must have a nonzero sum',
 'th_weighted_knn': 'threshold base has no weighted fit',
 'th_score_length': 'X and y sample counts differ',
 'th_score_weight_nan': 'threshold score weights must be finite',
 'th_score_weight_zero': 'threshold score weights must have a nonzero sum',
 'th_cv_one': 'invalid threshold CV count',
 'th_cv_many': 'invalid threshold CV count',
 'th_cv_class_count': 'threshold CV exceeds every class count',
 'th_grid_empty': 'threshold grid must be nonempty',
 'th_grid_nan': 'threshold grid must be finite',
 'th_grid_infinity': 'threshold grid must be finite',
 'th_grid_order': 'threshold grid must be nondecreasing',
 'th_grid_close': 'threshold curve endpoints are too close for optimization',
 'th_grid_one': 'threshold curve endpoints are too close for optimization',
 'th_grid_count': 'threshold grid count must be positive',
 'th_grid_range': 'threshold grid exceeds working range',
 'th_scoring': 'unknown threshold scoring mode',
 'th_prefit_refit': 'threshold prefit requires refit=false',
 'th_no_refit_multiple': 'threshold refit=false requires a single split',
 'th_fraction_zero': 'invalid threshold validation fraction',
 'th_fraction_nan': 'threshold validation fraction must be finite',
 'th_fraction_large': 'invalid threshold validation fraction',
 'th_fraction_negative': 'invalid threshold validation fraction',
 'th_splits_empty': 'threshold CV needs at least one split',
 'th_split_unallocated': 'threshold split indices are unallocated',
 'th_split_train_empty': 'threshold split train indices must be nonempty',
 'th_split_test_empty': 'threshold split test indices must be nonempty',
 'th_split_bounds': 'threshold split index out of bounds',
 'th_test_weights_zero': 'threshold test weights require positive total',
 'th_train_weights_zero': 'threshold score weights must have a nonzero sum',
 'th_prefit_labels': 'threshold target classes do not match fitted base',
 'th_f1_default_label': 'threshold positive label is not a fitted class',
 'th_constant': 'threshold curve endpoints are too close for optimization',
 'th_objective_nan': 'threshold objective scores must be finite',
 'th_validation_weight_length': 'threshold score weight length mismatch',
 'th_validation_weight_nan': 'threshold score weights must be finite',
 'th_validation_weight_negative': 'threshold weights must be nonnegative',
 'th_validation_weight_zero': 'threshold score weights must have a nonzero sum',
 'th_forward_proba': 'threshold base has no predict_proba',
 'th_forward_decision': 'threshold base has no decision_function',
 'th_tuned_width': 'threshold query width mismatch'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path);args=parser.parse_args()
    tree=ast.parse(Path(__file__).read_text())
    value=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CASES' for t in n.targets))
    keys=[ast.literal_eval(k) for k in value.keys];assert len(keys)==len(set(keys)) and max(map(len,keys))<=32
    if args.executable:
        exe=args.executable.resolve()
    else:
        result=subprocess.run(['fpm','run','--flag','-fcheck=all -fbacktrace','--target','threshold_invalid_driver','--runner'],
                               cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/s.strip().strip('"') for s in result.stdout.splitlines()
               if s.strip().strip('"').endswith(('threshold_invalid_driver','threshold_invalid_driver.exe'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,result.stdout;exe=paths[0]
    results=[]
    for name,message in CASES.items():
        proc=subprocess.run([str(exe),name],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert proc.returncode!=0 and message in proc.stdout,(name,message,proc.stdout)
        assert "Index '" not in proc.stdout and 'Segmentation fault' not in proc.stdout,(name,proc.stdout)
        results.append(dict(case=name,passed=True));print('PASS '+name,flush=True)
    path=ROOT/'validation/results/threshold_invalid_inputs.json'
    path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(results,indent=2)+'\n')
    print(f'{len(results)} threshold invalid-input cases passed')


if __name__=='__main__':main()
