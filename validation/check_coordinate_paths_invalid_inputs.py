"""Dense coordinate-path controls and identified native guard contracts."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    'empty_x': 'X must be nonempty', 'target_length': 'sample counts differ',
    'empty_targets': 'X must be nonempty', 'x_nan': 'NaN or infinity', 'y_inf': 'NaN or infinity',
    'mix_negative': 'l1_ratio must be in [0,1]', 'mix_many': 'l1_ratio must be in [0,1]',
    'mix_nan': 'nonfinite path l1_ratio', 'mix_inf': 'nonfinite path l1_ratio',
    'mix_zero_auto': 'automatic path grid requires positive l1_ratio',
    'eps_zero': 'invalid path grid controls', 'eps_negative': 'invalid path grid controls',
    'eps_nan': 'nonfinite path grid eps', 'eps_inf': 'nonfinite path grid eps',
    'grid_zero': 'invalid path grid controls', 'grid_negative': 'invalid path grid controls',
    'tol_negative': 'invalid path iteration controls', 'tol_nan': 'nonfinite path tolerance',
    'tol_inf': 'nonfinite path tolerance', 'iteration_negative': 'invalid path iteration controls',
    'seed_negative': 'random_state must be nonnegative', 'selection': 'unknown path coordinate selection',
    'precompute': 'unknown path precompute mode', 'positive_column': 'positive path requires vector targets',
    'positive_multi': 'positive path requires vector targets', 'alpha_negative': 'alphas must be nonnegative',
    'alpha_nan': 'alphas must be finite', 'alpha_inf': 'alphas must be finite',
    'initial_shape': 'initial coefficient shape mismatch', 'initial_matrix': 'initial coefficient shape mismatch',
    'initial_nan': 'initial coefficients must be finite', 'gram_shape': 'path Gram shape mismatch',
    'gram_nan': 'path Gram must be finite', 'gram_sampled': 'Gram failed sampled validation',
    'xy_empty': 'X must be nonempty', 'xy_shape': 'path correlation shape mismatch',
    'xy_matrix_shape': 'path correlation shape mismatch', 'xy_nan': 'NaN or infinity',
    'gram_denominator': 'nonpositive path coordinate denominator',
}
NATIVE_GUARDS = {'tol_negative', 'tol_nan', 'tol_inf', 'alpha_negative', 'alpha_nan', 'alpha_inf',
                 'initial_shape', 'initial_matrix', 'initial_nan', 'gram_shape', 'xy_shape',
                 'xy_matrix_shape', 'gram_denominator'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--profile', default='checked')
    args = parser.parse_args()
    checks = []
    for name, diagnostic in CASES.items():
        proc = subprocess.run([str(args.executable.resolve()), name], capture_output=True, text=True)
        output = proc.stdout+proc.stderr
        assert proc.returncode != 0 and diagnostic in output, (name, output)
        assert not any(s in output for s in ['SIGFPE', 'Fortran runtime error']), (name, output)
        checks.append(dict(case=name, passed=True, native_guard=name in NATIVE_GUARDS))
    report = dict(profile=args.profile, checks=checks, executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest())
    (ROOT/f'validation/results/coordinate_paths_invalid_{args.profile}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'{len(checks)} coordinate-path guards passed; {len(NATIVE_GUARDS)} native contracts identified')


if __name__ == '__main__':
    main()
