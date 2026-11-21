"""Dense clustering information and ordered-pair metrics versus sklearn 1.6.1."""
from pathlib import Path
import argparse
import json
import math
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.metrics import (adjusted_mutual_info_score, normalized_mutual_info_score, fowlkes_mallows_score,
                             homogeneity_score, completeness_score, v_measure_score)
from sklearn.metrics.cluster import (contingency_matrix, pair_confusion_matrix, entropy,
                                    expected_mutual_information, homogeneity_completeness_v_measure)
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'
METHODS = ['min', 'geometric', 'arithmetic', 'max']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable:
        executable = args.executable.resolve()
    else:
        proc = subprocess.run(['fpm', 'run', '--flag', FLAGS, '--target',
                               'reference_cluster_information_driver', '--runner'], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        paths = [ROOT / line.strip().strip('"') for line in proc.stdout.splitlines()
                 if line.strip().strip('"').endswith(('reference_cluster_information_driver',
                                                      'reference_cluster_information_driver.exe'))]
        paths = [p.resolve() for p in paths if p.is_file()]
        assert len(paths) == 1, proc.stdout
        executable = paths[0]
    checks = []
    zero_chance_gap = []
    rejections = []
    for name, operation, error in [
        ('zero_beta_zero_completeness', lambda: homogeneity_completeness_v_measure([0, 0], [0, 1], beta=0),
         ZeroDivisionError),
        ('empty_expected_table', lambda: expected_mutual_information(np.zeros((0, 0), dtype=np.int64), 0),
         ValueError)]:
        try: operation()
        except error: rejections.append(name)
        else: raise AssertionError(('source rejection changed', name))

    def compare(seed, case, lines, expected, labels=None):
        inp = RESULTS / 'cluster_information_input.txt'
        out = RESULTS / 'cluster_information_output.txt'
        inp.write_text('\n'.join(lines) + '\n')
        proc = subprocess.run([str(executable), str(inp), str(out)], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.returncode == 0, (seed, case, proc.stdout)
        actual = {}
        integers = {'table', 'pairs', 'shape', 'preserved'}
        for line in out.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=np.int64 if name in integers else float)
        assert actual.keys() == expected.keys(), (seed, case, actual.keys(), expected.keys())
        for name, want in expected.items():
            want = np.asarray(want).ravel(order='F')
            oracle = 'input_identity' if name == 'preserved' else 'sklearn'
            if name in integers:
                np.testing.assert_array_equal(actual[name], want, err_msg=f'{seed} {case} {name}')
            else:
                tol = 5.e-11 if name == 'expected' else 3.e-8
                try:
                    np.testing.assert_allclose(actual[name], want, atol=tol, rtol=tol, equal_nan=True,
                                               err_msg=f'{seed} {case} {name}')
                except AssertionError:
                    if labels is None or name not in ['adjusted', 'defaults']: raise
                    a, b = labels
                    n = len(a)
                    single_a = len(np.unique(a)) == n
                    single_b = len(np.unique(b)) == n
                    assert n > 1 and (single_a or single_b), (seed, case, name)
                    mismatch = np.flatnonzero(~np.isclose(actual[name], want, atol=tol, rtol=tol, equal_nan=True))
                    allowed = list(range(4)) if single_a and single_b else [0]
                    if name == 'defaults': allowed = [1] if single_a and single_b else []
                    assert set(mismatch) <= set(allowed), (seed, case, name, mismatch)
                    # An all-singleton partition fixes MI=EMI=the other entropy
                    # under the permutation null model. The adjusted denominator
                    # is mathematically zero for min normalization (all averages
                    # if both partitions are singletons). Certify the underlying
                    # quantities and the guarded ratio instead of amplifying
                    # libm/SciPy log-gamma rounding into a source-equality claim.
                    witness = []
                    for side in [actual, expected]:
                        ha, hb = np.asarray(side['entropy'], float)
                        emi = float(np.asarray(side['expected'])[0])
                        mi = float(np.asarray(side['normal'])[2]) * ((ha + hb) / 2)
                        coarse = min(ha, hb)
                        bound = 64 * n * np.finfo(float).eps * max(1., coarse)
                        assert coarse > 0 and abs(emi-coarse) <= bound and abs(mi-coarse) <= bound
                        uncertainty = 4 * np.finfo(float).eps * max(1., abs(mi))
                        intervals = []
                        for index in mismatch:
                            method = METHODS[index] if name == 'adjusted' else 'arithmetic'
                            normalizer = dict(min=min(ha, hb), geometric=np.sqrt(ha*hb),
                                              arithmetic=(ha+hb)/2, max=max(ha, hb))[method]
                            denominator = normalizer - emi
                            denominator = min(denominator, -np.finfo(float).eps) if denominator < 0 else max(
                                denominator, np.finfo(float).eps)
                            limits = sorted([(mi-uncertainty-emi)/denominator, (mi+uncertainty-emi)/denominator])
                            value = float(np.asarray(side[name])[index])
                            assert np.isfinite(value) and limits[0]-tol <= value <= limits[1]+tol
                            intervals.append(limits)
                        witness.append(dict(mi=mi, emi=emi, entropy=float(coarse), bound=float(bound), intervals=intervals))
                    oracle = 'sklearn_with_zero_chance_gap_certificate'
                    zero_chance_gap.append(dict(seed=seed, case=case, name=name, n_samples=n,
                        true_singletons=single_a, pred_singletons=single_b, indices=mismatch.tolist(),
                        source=want[mismatch].tolist(), native=actual[name][mismatch].tolist(), witnesses=witness))
            checks.append(dict(seed=seed, case=case, name=name, passed=True,
                               oracle=oracle))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for seed in [0, 7, 42, 2026, 314159]:
            rng = np.random.RandomState(seed)
            for case in range(40):
                n = [0, 1, 2, 3, 4, 5, 7, 16, 31, 64, 127, 257, 513][min(case, 12)]
                a = rng.randint(-3, 4, n)
                b = rng.randint(-5, 6, n)
                if case < 13:
                    a = np.arange(n)
                    b = rng.permutation(a)
                elif case % 9 == 0: a[:] = 9
                elif case % 9 == 1: b[:] = -11
                elif case % 9 == 2: b = a.copy()
                elif case % 9 == 3: b = a * 17 - 31
                elif case % 9 == 4: a = np.arange(n)
                elif case % 9 == 5: b = np.arange(n)
                elif case % 9 == 6:
                    a = np.arange(n) // 2
                    b = np.arange(n) // 3
                elif case % 9 == 7:
                    a = np.zeros(n, dtype=int); a[-3:] = [1, 2, 3]
                    b = np.zeros(n, dtype=int); b[-4:] = [1, 2, 3, 4]
                beta = [0., .1, .5, 1., 2., 1.e300][case % 6]
                if n and len(np.unique(a)) == 1 and len(np.unique(b)) > 1 and beta == 0: beta = 1.
                eps = [0., 1.e-12, .25, 10.][case % 4]
                table = contingency_matrix(a, b)
                expected = dict(table=table, shape=table.shape, smoothed=contingency_matrix(a, b, eps=eps),
                    pairs=pair_confusion_matrix(a, b), entropy=[entropy(a), entropy(b)],
                    fowlkes=[fowlkes_mallows_score(a, b), fowlkes_mallows_score(b, a)],
                    normal=[normalized_mutual_info_score(a, b, average_method=m) for m in METHODS],
                    adjusted=[adjusted_mutual_info_score(a, b, average_method=m) for m in METHODS],
                    defaults=[normalized_mutual_info_score(a, b), adjusted_mutual_info_score(a, b)],
                    hcv=homogeneity_completeness_v_measure(a, b, beta=beta), preserved=np.r_[a, b],
                    wrappers=[homogeneity_score(a, b), completeness_score(a, b), v_measure_score(a, b, beta=beta)])
                if n: expected['expected'] = [expected_mutual_information(table, n)]
                lines = [f'1 {n} {beta:.17g} {eps:.17g}']
                if n: lines += [' '.join(map(str, a)), ' '.join(map(str, b))]
                compare(seed, case, lines, expected, labels=(a, b))
            for case in range(25):
                table = rng.randint(0, 8, size=(2 + case % 5, 2 + case % 7))
                if case == 0: table[:] = 0
                if case == 1: table[:, 0] = 0
                if case == 2: table[0, :] = 0
                if case == 3: table = np.array([[2, 3, 4]])
                if case == 4: table = np.array([[2, 0], [0, 2]])
                if case == 5: table = np.array([[300, 200, 100], [100, 200, 300]])
                n = int(table.sum())
                emi = expected_mutual_information(table, n)
                lines = [f'2 {n} 1 0', f'{table.shape[0]} {table.shape[1]}']
                lines.extend(' '.join(map(str, row)) for row in table)
                compare(seed, 100 + case, lines, dict(expected=[emi, emi]))
            print(f'seed {seed}: cluster information metrics passed', flush=True)
    (RESULTS / 'cluster_information.json').write_text(json.dumps(
        dict(sklearn=sklearn.__version__, checks=checks, source_rejections=rejections,
             zero_chance_gap=zero_chance_gap), indent=2) + '\n')
    print(f'{len(checks)} cluster information result groups passed')


if __name__ == '__main__':
    main()
