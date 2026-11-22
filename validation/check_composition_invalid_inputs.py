"""Isolated rejection checks for dense feature composition."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from compare_function_transformer import ROOT,FLAGS

CASES={
    'column_unfitted':'ColumnTransformer is not fitted',
    'column_names_unfitted':'ColumnTransformer is not fitted',
    'column_index_unfitted':'ColumnTransformer is not fitted',
    'union_unfitted':'FeatureUnion is not fitted',
    'union_names_unfitted':'FeatureUnion is not fitted',
    'name_unallocated':'branch name is unallocated',
    'name_dupe':'branch names must be unique',
    'name_delimiter':'branch name contains __',
    'name_reserved':'name conflicts with a control',
    'union_name_reserved':'name conflicts with a control',
    'action_invalid':'invalid branch action',
    'model_unallocated':'branch transformer is unallocated',
    'model_unsupported':'unsupported transformer',
    'empty_unsupported':'unsupported transformer',
    'remainder_action':'invalid branch action',
    'remainder_model':'branch transformer is unallocated',
    'selector_none':'exactly one column selector',
    'selector_multiple':'exactly one column selector',
    'selector_mask_width':'mask width mismatch',
    'index_zero':'column index out of bounds',
    'index_negative':'column index out of bounds',
    'index_large':'column index out of bounds',
    'named_no_names':'named selection requires feature names',
    'named_unallocated':'selected name is unallocated',
    'named_missing':'selected feature name is missing',
    'named_dupe':'selected feature name is not unique',
    'callback_unallocated':'selector returned unallocated indices',
    'callback_bounds':'column index out of bounds',
    'fit_names_count':'feature name count mismatch',
    'fit_name_unallocated':'feature name is unallocated',
    'query_width':'query feature width mismatch',
    'query_name_count':'feature name count mismatch',
    'query_name_missing':'selected feature name is missing',
    'query_name_duplicate':'selected feature name is not unique',
    'get_names_count':'feature name count mismatch',
    'get_names_differ':'input feature names differ',
    'get_names_length':'input feature names differ',
    'output_unknown':'unknown branch name',
    'output_unknown_blanks':'unknown branch name',
    'output_duplicates':'output feature names are not unique',
    'output_unconfigured':'output feature names are not configured',
    'branch_row_mismatch':'branch sample counts differ',
    'union_row_mismatch':'branch sample counts differ',
}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);args=ap.parse_args()
    if args.executable:exe=args.executable.resolve()
    else:
        options=['--profile','release'] if args.profile=='release' else ['--flag',FLAGS]
        r=subprocess.run(['fpm','run',*options,'--target','composition_invalid_driver','--runner'],cwd=ROOT,
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/s.strip().strip('"') for s in r.stdout.splitlines()
               if s.strip().strip('"').endswith(('composition_invalid_driver','composition_invalid_driver.exe'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,r.stdout;exe=paths[0]
    checks=[]
    for case,message in CASES.items():
        r=subprocess.run([str(exe),case],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert r.returncode!=0 and 'fscikit-learn:' in r.stdout and message in r.stdout,(case,r.stdout)
        checks.append(dict(name=case,expected=message,passed=True))
    report=dict(profile=args.profile,checks=checks,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (ROOT/'validation/results').mkdir(exist_ok=True)
    (ROOT/'validation/results'/f'composition_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} isolated composition invalid-input checks passed')


if __name__=='__main__':main()
