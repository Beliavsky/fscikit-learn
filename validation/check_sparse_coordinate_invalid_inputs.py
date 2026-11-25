"""CSR path diagnostics and explicit native structure/weight/mean guards."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
ROOT = Path(__file__).resolve().parents[1]
CASES = {
    'unallocated': 'CSR arrays are unallocated', 'pointers': 'CSR pointer endpoints are invalid',
    'indices': 'CSR column index is out of bounds', 'data_count': 'CSR data/index lengths differ',
    'empty_rows': 'nonempty bounded dimensions', 'empty_features': 'nonempty bounded dimensions',
    'feature_capacity': 'nonempty bounded dimensions', 'target_length': 'target length mismatch',
    'data_nan': 'sparse path data must be finite', 'target_inf': 'sparse path data must be finite',
    'weight_length': 'weight length mismatch', 'weight_nan': 'weights must be finite', 'weight_inf': 'weights must be finite',
    'offset_needs_scale': 'offsets require scales', 'offset_length': 'offset/scale length mismatch',
    'scale_length': 'offset/scale length mismatch', 'offset_nan': 'offsets/scales must be finite',
    'scale_inf': 'offsets/scales must be finite', 'scale_zero': 'scales must be nonzero',
    'mean_overflow': 'means exceed working range', 'initial_length': 'initial coefficient shape mismatch',
    'initial_nan': 'initial coefficients must be finite', 'xy_empty': 'X must be nonempty',
    'xy_nan': 'NaN or infinity', 'alpha_negative': 'alphas must be nonnegative', 'alpha_nan': 'alphas must be finite',
    'zero_mix_grid': 'automatic path grid requires positive l1_ratio', 'negative_tolerance': 'invalid path iteration controls',
    'negative_iterations': 'invalid path iteration controls', 'negative_seed': 'random_state must be nonnegative',
    'selection': 'unknown path coordinate selection', 'precompute': 'unknown path precompute mode',
    'zero_denominator': 'zero sparse path coordinate denominator',
}
NATIVE_GUARDS = {'unallocated', 'pointers', 'indices', 'data_count', 'feature_capacity', 'weight_length',
                 'weight_nan', 'weight_inf', 'offset_needs_scale', 'offset_length', 'scale_length',
                 'offset_nan', 'scale_inf', 'scale_zero', 'mean_overflow', 'initial_length', 'initial_nan',
                 'alpha_negative', 'alpha_nan', 'negative_tolerance', 'zero_denominator'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--profile', default='checked')
    args = parser.parse_args(); checks = []
    for name, message in CASES.items():
        proc = subprocess.run([str(args.executable.resolve()), name], capture_output=True, text=True)
        output = proc.stdout+proc.stderr
        assert proc.returncode != 0 and message in output, (name, output)
        assert not any(s in output for s in ['SIGFPE', 'Fortran runtime error']), (name, output)
        checks.append(dict(case=name, passed=True, native_guard=name in NATIVE_GUARDS))
    report = dict(profile=args.profile, checks=checks, executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest())
    (ROOT/f'validation/results/sparse_coordinate_invalid_{args.profile}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'{len(checks)} sparse coordinate guards passed; {len(NATIVE_GUARDS)} native contracts identified')


if __name__ == '__main__':
    main()
