"""Native Gram-LARS guards; additional finite-input constraints are intentional."""
from pathlib import Path
import argparse
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--profile', default='checked')
    args = parser.parse_args()
    outdir = ROOT / 'validation/results'
    cases = [
        ('samples_zero', '0 2 500 0 2.22e-16 0 lar', '1 1', '1 0\n0 1', 'n_samples'),
        ('samples_negative', '-1 2 500 0 2.22e-16 0 lar', '1 1', '1 0\n0 1', 'n_samples'),
        ('iterations_negative', '10 2 -1 0 2.22e-16 0 lar', '1 1', '1 0\n0 1', 'max_iter'),
        ('alpha_negative', '10 2 500 -1 2.22e-16 0 lar', '1 1', '1 0\n0 1', 'nonnegative'),
        ('eps_negative', '10 2 500 0 -1 0 lar', '1 1', '1 0\n0 1', 'nonnegative'),
        ('alpha_nan', '10 2 500 NaN 2.22e-16 0 lar', '1 1', '1 0\n0 1', 'nonfinite LARS controls'),
        ('eps_inf', '10 2 500 0 Inf 0 lar', '1 1', '1 0\n0 1', 'nonfinite LARS controls'),
        ('unknown_method', '10 2 500 0 2.22e-16 0 invalid', '1 1', '1 0\n0 1', 'unknown LARS method'),
        ('positive_lar', '10 2 500 0 2.22e-16 1 lar', '1 1', '1 0\n0 1', 'positive constraint'),
        ('xy_nan', '10 2 500 0 2.22e-16 0 lar', 'NaN 1', '1 0\n0 1', 'inputs must be finite'),
        ('gram_inf', '10 2 500 0 2.22e-16 0 lar', '1 1', '1 Inf\nInf 1', 'inputs must be finite'),
    ]
    checks = []
    for name, controls, xy, gram, diagnostic in cases:
        inp = outdir / 'lars_gram_invalid_input.txt'; out = outdir / 'lars_gram_invalid_output.txt'
        inp.write_text(f'{controls}\n{xy}\n{gram}\n')
        proc = subprocess.run([str(args.executable.resolve()), str(inp), str(out)], capture_output=True, text=True)
        assert proc.returncode != 0 and diagnostic in proc.stdout+proc.stderr, (name, proc.stdout, proc.stderr)
        checks.append(dict(case=name, diagnostic=diagnostic, passed=True))
    (outdir / f'lars_gram_invalid_{args.profile}.json').write_text(json.dumps(dict(checks=checks), indent=2)+'\n')
    print(f'{len(checks)} Gram LARS guards passed')


if __name__ == '__main__':
    main()
