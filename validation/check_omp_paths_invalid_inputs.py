"""OMP diagnostics, including explicitly identified native finite/shape guards."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    'empty_x': 'X must be nonempty', 'target_length': 'sample counts differ',
    'empty_targets': 'X must be nonempty', 'x_nan': 'NaN or infinity', 'y_inf': 'NaN or infinity',
    'atoms_zero': 'number of atoms must be positive', 'atoms_negative': 'number of atoms must be positive',
    'atoms_many': 'number of atoms exceeds features', 'atoms_zero_tolerance': 'number of atoms must be positive',
    'tolerance_negative': 'tolerance must be nonnegative', 'tolerance_nan': 'tolerance must be finite',
    'tolerance_inf': 'tolerance must be finite', 'precompute_mode': 'unknown OMP precompute mode',
    'precompute_matrix': 'unknown OMP precompute mode', 'gram_empty': 'X must be nonempty',
    'gram_shape': 'OMP Gram must be square', 'gram_features': 'correlations feature counts differ',
    'gram_default_zero': 'number of atoms must be positive', 'gram_atoms_many': 'number of atoms exceeds features',
    'gram_missing_norm': 'requires squared target norms', 'gram_norm_length': 'target norm count mismatch',
    'gram_norm_nan': 'target norms must be finite', 'gram_norm_inf': 'target norms must be finite',
    'gram_xy_nan': 'NaN or infinity', 'gram_nan': 'NaN or infinity',
    'gram_zero_pivot': 'nonpositive diagonal', 'gram_negative_pivot': 'nonpositive diagonal',
}
NATIVE_GUARDS = ['gram_shape', 'gram_features', 'gram_norm_length', 'gram_norm_nan', 'gram_norm_inf',
                 'gram_xy_nan', 'gram_zero_pivot', 'gram_negative_pivot']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--profile', default='checked')
    args = parser.parse_args()
    checks = []
    for name, message in CASES.items():
        proc = subprocess.run([str(args.executable.resolve()), name], capture_output=True, text=True)
        output = proc.stdout+proc.stderr
        assert proc.returncode != 0 and message in output, (name, output)
        assert 'Fortran runtime error' not in output and 'SIGFPE' not in output, (name, output)
        checks.append(dict(case=name, passed=True, native_guard=name in NATIVE_GUARDS))
    report = dict(profile=args.profile, checks=checks, executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest())
    (ROOT/f'validation/results/omp_paths_invalid_{args.profile}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'{len(checks)} OMP guards passed; {len(NATIVE_GUARDS)} native guard contracts identified')


if __name__ == '__main__':
    main()
