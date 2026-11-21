"""Isolated input-error checks for tree embeddings and kernel centering."""
from pathlib import Path
import argparse
import ast
import json
import subprocess
ROOT=Path(__file__).resolve().parents[1]
CASES = {'rte_unfitted': 'tree embedding is not fitted',
 'rte_apply_unfitted': 'tree embedding is not fitted',
 'rte_path_unfitted': 'tree embedding is not fitted',
 'rte_samples_unfitted': 'tree embedding is not fitted',
 'rte_names_unfitted': 'tree embedding is not fitted',
 'rte_estimators_zero': 'invalid tree embedding estimator count',
 'rte_estimators_negative': 'invalid tree embedding estimator count',
 'rte_estimators_overflow': 'invalid tree embedding estimator count',
 'rte_seed': 'tree embedding seed must be nonnegative',
 'rte_depth': 'invalid tree embedding depth',
 'rte_leaf': 'invalid tree embedding sample controls',
 'rte_split': 'invalid tree embedding sample controls',
 'rte_impurity_negative': 'tree embedding impurity decrease must be nonnegative',
 'rte_impurity_nan': 'tree embedding impurity decrease must be finite',
 'rte_impurity_inf': 'tree embedding impurity decrease must be finite',
 'rte_weight_length': 'tree embedding weight length mismatch',
 'rte_weight_nan': 'tree embedding weights must be finite',
 'rte_weight_inf': 'tree embedding weights must be finite',
 'rte_empty_rows': 'X must be nonempty',
 'rte_empty_features': 'X must be nonempty',
 'rte_fit_nan': 'X contains NaN or infinity',
 'rte_fit_inf': 'X contains NaN or infinity',
 'rte_fit_range': 'tree training features exceed float32 range',
 'rte_query_width': 'tree embedding query shape mismatch',
 'rte_query_empty': 'tree embedding query shape mismatch',
 'rte_query_inf': 'tree queries contain infinity',
 'rte_query_range': 'tree query features exceed float32 range',
 'rte_names_width': 'tree embedding input feature name count mismatch',
 'rte_warm_shrink': 'warm tree embedding cannot remove estimators',
 'rte_warm_width': 'warm tree embedding feature width changed',
 'rte_warm_unknown': 'unknown category',
 'rte_encoder_reset': 'unknown category',
 'kc_unfitted': 'kernel centerer is not fitted',
 'kc_names_unfitted': 'kernel centerer is not fitted',
 'kc_not_square': 'kernel centerer fit requires a square matrix',
 'kc_empty_fit': 'X must be nonempty',
 'kc_empty_features': 'X must be nonempty',
 'kc_fit_nan': 'X contains NaN or infinity',
 'kc_fit_inf': 'X contains NaN or infinity',
 'kc_query_width': 'kernel centerer query width mismatch',
 'kc_query_nan': 'X contains NaN or infinity',
 'kc_query_inf': 'X contains NaN or infinity',
 'kc_query_empty': 'X must be nonempty',
 'kc_names_width': 'kernel centerer input feature name count mismatch'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path);args=parser.parse_args()
    tree=ast.parse(Path(__file__).read_text())
    value=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CASES' for t in n.targets))
    keys=[ast.literal_eval(k) for k in value.keys];assert len(keys)==len(set(keys)) and max(map(len,keys))<=32
    if args.executable:
        exe=args.executable.resolve()
    else:
        result=subprocess.run(['fpm','run','--flag','-fcheck=all -fbacktrace','--target','feature_maps_invalid_driver','--runner'],
                               cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/s.strip().strip('"') for s in result.stdout.splitlines()
               if s.strip().strip('"').endswith(('feature_maps_invalid_driver','feature_maps_invalid_driver.exe'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,result.stdout;exe=paths[0]
    results=[]
    for name,message in CASES.items():
        proc=subprocess.run([str(exe),name],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert proc.returncode!=0 and message in proc.stdout,(name,message,proc.stdout)
        assert "Index '" not in proc.stdout and 'Segmentation fault' not in proc.stdout,(name,proc.stdout)
        results.append(dict(case=name,passed=True));print('PASS '+name,flush=True)
    path=ROOT/'validation/results/feature_maps_invalid_inputs.json'
    path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(results,indent=2)+'\n')
    print(f'{len(results)} feature map invalid-input cases passed')


if __name__=='__main__':main()
