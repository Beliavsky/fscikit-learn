"""Isolated learning/validation curve and permutation-test diagnostics."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from compare_function_transformer import ROOT,resolve

CASES={
 'fractions_empty':'training fractions must be nonempty','fractions_nan':'training fractions must be finite',
 'fractions_inf':'training fractions must be finite','fractions_zero':'training fractions must be in (0,1]',
 'fractions_negative':'training fractions must be in (0,1]','fractions_large':'training fractions must be in (0,1]',
 'counts_empty':'training counts must be nonempty','counts_zero':'training counts out of range',
 'counts_negative':'training counts out of range','counts_large':'training counts out of range',
 'sizes_both':'choose fractional sizes or absolute counts','learning_seed':'random_state must be nonnegative',
 'learning_multiple':'requires a single scalar scorer','incremental_unsupported':'estimator has no native partial_fit',
 'incremental_empty':'X must be nonempty','incremental_weights':'SGD sample weight count mismatch',
 'incremental_no_target':'incremental regression requires real targets',
 'incremental_wrong_target':'incremental classification requires integer targets',
 'validation_empty':'param_range must be nonempty','validation_unknown':'unsupported native search parameter',
 'validation_type':'parameter value must be numeric','validation_nested':'no supported nested parameter',
 'validation_multiple':'requires a single scalar scorer','permutation_zero':'n_permutations must be positive',
 'permutation_negative':'n_permutations must be positive','permutation_seed':'seed must be nonnegative',
 'permutation_groups':'group count mismatch','permutation_multiple':'requires a single scalar scorer',
 'permutation_target_count':'X and y sample counts differ','curve_weights':'CV fit weight count mismatch',
 'curve_nonfinite_target':'CV targets must be finite','validation_error_score':'CV supports only error_score=raise',
 'learning_error_score':'CV supports only error_score=raise','permutation_error_score':'CV supports only error_score=raise',
}


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
 parser.add_argument('--executable',type=Path);args=parser.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'curve_invalid_driver')
 checks=[]
 for name,message in CASES.items():
  result=subprocess.run([str(exe),name],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  assert result.returncode!=0 and 'fscikit-learn:' in result.stdout and message in result.stdout,(name,result.stdout)
  checks.append(dict(name=name,expected=message,passed=True))
 report=dict(profile=args.profile,checks=checks,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
             validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (ROOT/'validation/results').mkdir(exist_ok=True)
 (ROOT/'validation/results'/f'curve_invalid_{args.profile}.json').write_text(json.dumps(report,indent=2)+'\n')
 print(f'{len(checks)} isolated curve/permutation invalid-input checks passed')


if __name__=='__main__':main()
