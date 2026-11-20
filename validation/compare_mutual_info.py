"""Seeded mutual information, Pearson correlation and categorical MI checks."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
import numpy as np
import sklearn
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, r_regression
from sklearn.metrics import mutual_info_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable:
        executable = args.executable.resolve()
    else:
        proc = subprocess.run(['fpm', 'run', '--flag', FLAGS, '--target', 'reference_mutual_info_driver', '--runner'],
                              cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        paths = [ROOT / line.strip().strip('"') for line in proc.stdout.splitlines()
                 if line.strip().strip('"').endswith(('reference_mutual_info_driver', 'reference_mutual_info_driver.exe'))]
        paths = [p.resolve() for p in paths if p.is_file()]
        assert len(paths) == 1, proc.stdout
        executable = paths[0]
    checks = []
    constant_rounding = []
    rejection_cases = [('empty_metric', lambda: mutual_info_score([], [])),
        ('singleton_classes', lambda: mutual_info_classif(np.arange(5.)[:, None], np.arange(5), random_state=0)),
        ('unique_discrete_features', lambda: mutual_info_regression(np.arange(5.)[:, None], np.arange(5.),
                                                                   discrete_features=True, random_state=0)),
        ('continuous_neighbor_count', lambda: mutual_info_regression(np.arange(5.)[:, None], np.arange(5.),
                                                                    n_neighbors=5, random_state=0))]
    source_rejections = []
    for name, operation in rejection_cases:
        try: operation()
        except ValueError: source_rejections.append(name)
        else: raise AssertionError(('source rejection changed', name))

    def run(mode, seed, case, x, y, mask=None, k=3, representation=1, center=True, force=True, predicted=None):
        n, p = x.shape
        if mask is None: mask = np.zeros(p, dtype=bool)
        lines = [f'{mode} {n} {p} {k} {seed} {representation} {int(center)} {int(force)}']
        if mode == 4:
            if n:
                lines += [' '.join(map(str, y)), ' '.join(map(str, predicted))]
            expected = np.array([mutual_info_score(y, predicted)])
        else:
            lines.append(' '.join(map(str, mask.astype(int))))
            lines.extend(' '.join(f'{v:.17g}' for v in row) for row in x)
            lines.append(' '.join(f'{v:.17g}' for v in y))
            if representation == 0: discrete = 'auto'
            elif representation == 2: discrete = bool(mask.all())
            elif representation == 3:
                positions = np.flatnonzero(mask)
                discrete = np.r_[positions[::-1], positions[:1]]
            else: discrete = mask
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if mode == 1:
                    expected = mutual_info_classif(x, y, discrete_features=discrete, n_neighbors=k, random_state=seed)
                elif mode == 2:
                    expected = mutual_info_regression(x, y, discrete_features=discrete, n_neighbors=k, random_state=seed)
                else:
                    expected = r_regression(x, y, center=center, force_finite=force)
        inp = RESULTS / 'mutual_info_input.txt'
        out = RESULTS / 'mutual_info_output.txt'
        inp.write_text('\n'.join(lines) + '\n')
        proc = subprocess.run([str(executable), str(inp), str(out)], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.returncode == 0, (mode, seed, case, proc.stdout)
        actual = {}
        for line in out.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        oracle = 'sklearn'
        try:
            np.testing.assert_allclose(actual['scores'], expected, atol=3.e-8, rtol=3.e-8, equal_nan=True)
        except AssertionError:
            if mode != 3 or not center:
                raise AssertionError(f'mode={mode} seed={seed} case={case} representation={representation}')
            # Pearson correlation is mathematically undefined for exact constants.
            # BLAS reduction order can turn 0/0 into signed inf or tiny finite
            # artifacts; source force_finite only replaces NaNs, not infinities.
            constant = np.all(x == x[0], axis=0)
            np.testing.assert_allclose(actual['scores'][~constant], expected[~constant],
                                       atol=3.e-8, rtol=3.e-8, equal_nan=True)
            limit = 32 * np.sqrt(n * np.finfo(float).eps)
            for values in [actual['scores'][constant], expected[constant]]:
                good = np.isinf(values) | (np.isfinite(values) & (np.abs(values) <= limit))
                if not force: good |= np.isnan(values)
                assert good.all(), (seed, case, 'nontrivial correlation on exact constants', values)
            oracle = 'sklearn_with_exact_constant_pearson_certificate'
            def serial(values):
                return [float(v) if np.isfinite(v) else str(v) for v in values]
            constant_rounding.append(dict(seed=seed, case=case, columns=np.flatnonzero(constant).tolist(),
                                          source=serial(expected[constant]), native=serial(actual['scores'][constant]),
                                          finite_bound=float(limit)))
        checks.append(dict(mode=mode, seed=seed, case=case, representation=representation,
                           name='scores', passed=True, oracle=oracle))
        if mode != 4:
            np.testing.assert_array_equal(actual['x'], x.ravel(order='F'))
            checks.append(dict(mode=mode, seed=seed, case=case, representation=representation,
                               name='preserved_x', passed=True, oracle='input_identity'))

    with threadpool_limits(limits=1):
        for seed, n in [(0, 7), (7, 16), (42, 31), (2026, 64), (2147483647, 127), (11, 257)]:
            rng = np.random.RandomState(seed)
            for mode in [1, 2]:
                for case in range(24):
                    p = 1 if case == 22 else 13 if case == 23 else 6
                    x = rng.normal(size=(n, p))
                    mask = (np.zeros(p, bool) if case % 3 == 0 else np.ones(p, bool) if case % 3 == 1
                            else np.arange(p) % 2 == 0)
                    x[:, mask] = rng.randint(0, 3, size=(n, int(mask.sum())))
                    y = (np.arange(n) % 3) * 11 - 7 if mode == 1 else np.sin(x[:, 0]) + rng.normal(size=n) * .1
                    if case in (3, 4, 5):
                        x[:, 0] = 0
                        if p > 1: x[:, 1] = 1
                    if case in (6, 7, 8): x[:, ~mask] = np.round(x[:, ~mask])
                    if case in (9, 10, 11):
                        x[:, ~mask] *= 1.e-18
                        if mode == 2: y *= 1.e-18
                    if case in (12, 13, 14): x[:, ~mask] = x[:, ~mask] * 2.7 + 50
                    if case in (15, 16, 17): y[:] = 0
                    if case in (18, 19, 20) and mode == 1:
                        y = np.arange(n)
                        y[:3] = -1
                        y[3:5] = -2
                    k = [1, 2, 3, n-1][case % 4]
                    if mode == 1 and case == 21: k = n * 2
                    run(mode, seed, case, x.copy(), y.copy(), mask=mask, k=k)
                    if mask.all() or not mask.any():
                        run(mode, seed, case, x.copy(), y.copy(), mask=mask, k=k, representation=2)
                    if mask.any():
                        run(mode, seed, case, x.copy(), y.copy(), mask=mask, k=k, representation=3)
                    else:
                        run(mode, seed, case, x.copy(), y.copy(), mask=mask, k=k, representation=0)
            for count in [1, 2, 3, 4]:
                x = np.column_stack([np.arange(count) % 2, np.zeros(count)]).astype(float)
                run(1, seed, 100 + count, x, np.zeros(count, dtype=int), mask=np.ones(2, bool),
                    representation=2, k=9)
                if count > 1:
                    x = rng.normal(size=(count, 2))
                    run(1, seed, 110 + count, x, np.zeros(count, dtype=int), k=9)
                    run(2, seed, 120 + count, x, rng.normal(size=count), k=count-1)
                    run(2, seed, 130 + count, np.zeros((count, 2)), rng.normal(size=count),
                        mask=np.ones(2, bool), representation=2, k=9)
            for case in range(8):
                x = rng.normal(size=(n, 7))
                y = rng.normal(size=n)
                x[:, 0] = y
                x[:, 1] = -y
                x[:, 2] = 0
                x[:, 3] = 1
                if case == 1: y[:] = 0
                if case == 2: y[:] = 2
                if case == 3: x *= 1.e-10
                if case == 4: x[:, 4] = x[:, 5]
                if case == 5: y *= 1.e-200
                if case == 6: x = np.ones((n, 1))
                if case == 7: x = y[:, None].copy()
                for center in [True, False]:
                    for force in [True, False]:
                        run(3, seed, case * 4 + 2 * int(center) + int(force), x, y, center=center, force=force)
            for case in range(20):
                count = 1 if case == 1 else n
                y = rng.randint(-3, 4, count)
                predicted = y.copy() if case % 3 == 0 else rng.randint(-7, 6, count)
                if case % 5 == 0: y[:] = 7
                run(4, seed, case, np.empty((count, 0)), y, predicted=predicted)
            print(f'seed {seed}: mutual information, correlation and cluster metric passed', flush=True)
    (RESULTS / 'mutual_info.json').write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks,
        constant_pearson_rounding=constant_rounding, source_rejections=source_rejections), indent=2) + '\n')
    print(f'{len(checks)} mutual information/correlation result groups passed')


if __name__ == '__main__':
    main()
