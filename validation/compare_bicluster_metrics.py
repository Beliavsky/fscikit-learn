"""Bicluster consensus versus installed sklearn, with exhaustive small assignments."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import subprocess
import warnings
import numpy as np
import sklearn
import sklearn.metrics.cluster._bicluster as source
from sklearn.datasets import make_biclusters, make_checkerboard
from sklearn.metrics import consensus_score

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def payload(arrays, representation=1, table=None):
    ar, ac, br, bc = arrays
    shape = [len(ar), len(ac), len(br), len(bc), ar.shape[1], ac.shape[1], br.shape[1], bc.shape[1]]
    lines = [' '.join(map(str, shape + [int(table is not None), representation]))]
    for x in [*arrays, *([] if table is None else [table])]:
        lines.append(' '.join(map(str, np.asarray(x, dtype=float).ravel(order='F'))))
    return '\n'.join(lines) + '\n'


def callback_score(arrays, table):
    ar, ac, br, bc = [x.copy() for x in arrays]
    ar[:, 0] = np.arange(len(ar)); br[:, 0] = np.arange(len(br))
    return consensus_score((ar, ac), (br, bc),
                           similarity=lambda r, c, s, d: table[int(r[0]), int(s[0])])


def brute_score(table):
    t = table if table.shape[0] <= table.shape[1] else table.T
    return max(sum(t[i, j] for i, j in enumerate(p))
               for p in itertools.permutations(range(t.shape[1]), t.shape[0])) / max(table.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--profile', default='checked')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    tracked = [executable, Path(__file__), Path(source.__file__),
               ROOT / 'src/sklearn_bicluster_metrics.f90', ROOT / 'src/sklearn_metrics.f90',
               ROOT / 'app/reference_bicluster_metrics_driver.f90']
    before = {str(p): sha(p) for p in tracked}
    fixtures = []

    def add(name, arrays, representation=1, table=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            want = callback_score(arrays, table) if table is not None else consensus_score(
                (arrays[0], arrays[1]), (arrays[2], arrays[3]))
        certificate = None
        if table is not None and max(table.shape) <= 7:
            certificate = brute_score(table)
            np.testing.assert_allclose(want, certificate, atol=2.e-13, rtol=2.e-13)
        fixtures.append((name, payload(arrays, representation, table), float(want), certificate))

    for seed in [0, 1, 7, 42, 314159]:
        rng = np.random.RandomState(seed)
        for na, nb in [(1, 1), (1, 4), (4, 1), (2, 3), (3, 2), (4, 4), (7, 11), (11, 7), (20, 21)]:
            for nr, nc in [(3, 4), (17, 11), (129, 137)]:
                arrays = [rng.randint(0, 2, size=(n, w)) for n, w in
                          [(na, nr), (na, nc), (nb, nr), (nb, nc)]]
                for x in arrays:
                    x[np.arange(len(x)), rng.randint(x.shape[1], size=len(x))] = 1
                for rep in [1, 2, 3, 4]:
                    values = [x.astype(bool if rep == 3 or (rep == 4 and i == 2) else float if rep == 1 else int)
                              for i, x in enumerate(arrays)]
                    add(f'random_{seed}_{na}_{nb}_{nr}_{nc}_rep{rep}', values, rep)
                add(f'swap_{seed}_{na}_{nb}_{nr}_{nc}', arrays[2:] + arrays[:2])
                add(f'self_{seed}_{na}_{nb}_{nr}_{nc}', arrays[:2] + arrays[:2])
                add(f'permuted_{seed}_{na}_{nb}_{nr}_{nc}',
                    arrays[:2] + [arrays[0][::-1], arrays[1][::-1]])
            for kind in ['normal', 'ties', 'negative', 'near_one', 'forbidden']:
                table = rng.normal(size=(na, nb))
                if kind == 'ties': table = rng.randint(-2, 3, size=(na, nb)).astype(float)
                if kind == 'negative': table = -np.abs(table)
                if kind == 'near_one': table = 1 + table * 2.e-16
                if kind == 'forbidden':
                    table[:] = -np.inf
                    for i in range(min(na, nb)): table[i, i] = rng.normal()
                arrays = [np.ones((na, 2)), np.ones((na, 3)), np.ones((nb, 4)), np.ones((nb, 5))]
                add(f'callback_{seed}_{na}_{nb}_{kind}', arrays, table=table)

    for seed in [0, 7, 42]:
        for generator in [make_biclusters, make_checkerboard]:
            _, rows, cols = generator((23, 19), 3, random_state=seed)
            add(f'{generator.__name__}_{seed}_identity', [rows, cols, rows, cols], 3)
            add(f'{generator.__name__}_{seed}_subset', [rows, cols, rows[::2], cols[::2]], 3)
            add(f'{generator.__name__}_{seed}_reordered', [rows, cols, rows[::-1], cols[::-1]], 3)
    for na, nb in [(1, 3), (3, 1), (4, 4)]:
        for ra, rb in [(1, 5), (5, 1)]:
            for ca, cb in [(1, 7), (7, 1)]:
                add(f'broadcast_{na}_{nb}_{ra}_{rb}_{ca}_{cb}',
                    [np.ones((na, ra)), np.ones((na, ca)), np.ones((nb, rb)), np.ones((nb, cb))])
        rng = np.random.RandomState(na * 10 + nb)
        add(f'fractional_{na}_{nb}', [rng.uniform(.1, .5, (n, w)) for n, w in
                                    [(na, 9), (na, 13), (nb, 9), (nb, 13)]])
    add('one_empty_bicluster', [np.zeros((1, 2)), np.ones((1, 3)), np.ones((2, 2)), np.ones((2, 3))])
    add('greedy_counterexample', [np.ones((2, 1)) for _ in range(4)],
        table=np.array([[.9, .8], [.85, .1]]))

    proc = subprocess.run([str(executable)], input=str(len(fixtures)) + '\n' + ''.join(f[1] for f in fixtures),
                          text=True, capture_output=True, check=True)
    actual = np.array([float(x) for x in proc.stdout.split()])
    assert len(actual) == len(fixtures), (len(actual), len(fixtures), proc.stderr)
    checks = []
    for got, (name, _, want, certificate) in zip(actual, fixtures):
        np.testing.assert_allclose(got, want, rtol=5.e-13, atol=5.e-13, err_msg=name)
        checks.append(dict(name=name, passed=True, expected=want, actual=float(got),
                           brute_certificate=certificate, abs_error=float(abs(got-want))))

    base = [np.ones((2, 3)), np.ones((2, 4)), np.ones((2, 3)), np.ones((2, 4))]
    invalid = []
    for i in range(4):
        for axis in [0, 1]:
            arrays = [x.copy() for x in base]
            shape = list(arrays[i].shape); shape[axis] = 0; arrays[i] = np.ones(shape)
            invalid.append((f'empty_{i}_{axis}', arrays, None))
    for i in [1, 3]:
        arrays = [x.copy() for x in base]; arrays[i] = arrays[i][:1]
        invalid.append((f'count_mismatch_{i}', arrays, None))
    for i in [2, 3]:
        arrays = [x.copy() for x in base]; arrays[i] = np.ones((2, 7))
        invalid.append((f'width_mismatch_{i}', arrays, None))
    invalid.append(('zero_union', [np.zeros_like(x) for x in base], None))
    for number, name in [(np.nan, 'nan'), (np.inf, 'inf'), (-np.inf, 'negative_inf')]:
        for i in range(4):
            arrays = [x.copy() for x in base]; arrays[i][0, 0] = number
            invalid.append((f'input_{name}_{i}', arrays, None))
        table = np.ones((2, 2)); table[0, :] = number
        invalid.append((f'callback_{name}', base, table))
    table = np.full((3, 3), -np.inf);table[:, 0] = 1;table[2, 2] = 1
    invalid.append(('callback_hall_infeasible', [np.ones((3, 1)) for _ in range(4)], table))
    rejections = []
    for name, arrays, table in invalid:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                if table is None: consensus_score((arrays[0], arrays[1]), (arrays[2], arrays[3]))
                else: callback_score(arrays, table)
            except (ValueError, IndexError) as error:
                source_error = type(error).__name__
            else: raise AssertionError(('source unexpectedly accepts', name))
        result = subprocess.run([str(executable)], input='1\n' + payload(arrays, table=table),
                                text=True, capture_output=True)
        assert result.returncode != 0 and 'fscikit-learn: consensus' in result.stdout + result.stderr, (name, result)
        assert not any(s in result.stdout + result.stderr for s in ['SIGFPE', 'Fortran runtime error']), (name, result)
        rejections.append(dict(name=name, source_error=source_error, passed=True))
    assert before == {str(p): sha(p) for p in tracked}
    report = dict(sklearn=sklearn.__version__, profile=args.profile, checks=checks, rejections=rejections,
                  fixtures=len(fixtures), brute_certificates=sum(f[3] is not None for f in fixtures), hashes=before)
    path = args.report or ROOT / 'validation/results' / f'bicluster_metrics_{args.profile}.json'
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'{len(checks)} consensus comparisons and {len(rejections)} source-matched diagnostics: PASS')
    print(f'{report["brute_certificates"]} exhaustive assignment certificates: PASS')


if __name__ == '__main__':
    main()
