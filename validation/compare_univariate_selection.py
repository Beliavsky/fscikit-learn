"""Five univariate selectors, statistical tails and degenerate/custom scores."""
from pathlib import Path
import argparse
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.feature_selection import (SelectPercentile, SelectFpr, SelectFdr, SelectFwe,
    GenericUnivariateSelect, f_classif, f_regression, chi2)
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def compare(actual, expected, identities, regression_df, checks, differences, seed, kind):
    assert actual.keys() == expected.keys(), (seed, kind, actual.keys() ^ expected.keys())
    for name, value in expected.items():
        want = np.asarray(value).ravel(order='F'); got = actual[name]
        assert got.shape == want.shape, (seed, kind, name, got.shape, want.shape)
        tol = 0 if name.endswith(('mask', 'indices', 'meta', 'inverse', 'transform')) else 3.e-8
        oracle = 'zero-column inverse identity' if name in identities else 'sklearn'
        if name in regression_df and not np.allclose(got, want, atol=tol, rtol=tol, equal_nan=True):
            sensitive = np.isfinite(got) & np.isfinite(want) & (got > 1.e10) & (want > 1.e10)
            assert sensitive.any(), (seed, kind, name, got, want)
            np.testing.assert_allclose(got[~sensitive], want[~sensitive], atol=tol, rtol=tol, equal_nan=True)
            df = regression_df[name]
            native_r2 = 1-df/(got[sensitive]+df); source_r2 = 1-df/(want[sensitive]+df)
            np.testing.assert_allclose(native_r2, source_r2, atol=64*np.finfo(float).eps, rtol=0)
            oracle = 'near-perfect squared-correlation identity'
            differences.append(dict(seed=seed, kind=kind, name=name, df=df,
                                    native_scores=got[sensitive].tolist(), source_scores=want[sensitive].tolist(),
                                    max_r2_difference=float(np.max(np.abs(native_r2-source_r2)))))
        else:
            np.testing.assert_allclose(got, want, atol=tol, rtol=tol, equal_nan=True,
                                       err_msg=f'seed={seed} kind={kind} {name}')
        checks.append(dict(seed=seed, kind=kind, name=name, passed=True,
                           oracle=oracle))
        if name.endswith('pvalues'):
            normal = np.isfinite(want) & (want >= np.finfo(float).tiny) & (want < 1)
            if normal.any():
                assert np.all(got[normal] > 0)
                np.testing.assert_allclose(np.log(got[normal]), np.log(want[normal]),
                                           atol=1.e-7, rtol=3.e-8, err_msg=f'{seed} {kind} {name} log tails')
                checks.append(dict(seed=seed, kind=kind, name=name+'_log_tails', passed=True, oracle='sklearn'))


def inverse_result(model, z):
    if z.shape[1]: return model.inverse_transform(z), False
    try:
        model.inverse_transform(z)
    except ValueError as error:
        assert '0 feature(s)' in str(error), str(error)
    else:
        raise AssertionError('source dense inverse unexpectedly accepted zero columns')
    return np.zeros((len(z), model.n_features_in_)), True


def record(model, x, prefix, expected, identities):
    z = model.transform(x); inverse, empty = inverse_result(model, z)
    if empty: identities.add(prefix+'inverse')
    values = dict(scores=model.scores_, mask=model.get_support().astype(int),
                  indices=model.get_support(indices=True), transform=z, inverse=inverse,
                  meta=[model.n_features_in_, model.get_support().sum()])
    if model.pvalues_ is not None: values['pvalues'] = model.pvalues_
    if isinstance(model, SelectPercentile): model.set_params(percentile=0)
    elif isinstance(model, GenericUnivariateSelect): model.set_params(param=0)
    else: model.set_params(alpha=0)
    values['changed_mask'] = model.get_support().astype(int)
    values['changed_inverse'], empty = inverse_result(model, model.transform(x))
    if empty: identities.add(prefix+'changed_inverse')
    expected.update({prefix+key: value for key, value in values.items()})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--executable', type=Path); args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable: executable = args.executable.resolve()
    else:
        proc = subprocess.run(['fpm', 'run', '--flag', FLAGS, '--target', 'reference_univariate_selection_driver', '--runner'],
                              cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        paths = [ROOT/line.strip().strip('"') for line in proc.stdout.splitlines()
                 if line.strip().strip('"').endswith(('reference_univariate_selection_driver', 'reference_univariate_selection_driver.exe'))]
        paths = [p.resolve() for p in paths if p.is_file()]; assert len(paths) == 1, proc.stdout
        executable = paths[0]
    checks = []; differences = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for seed in [0, 7, 42, 2026]:
            for kind, cls in enumerate([SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect], 1):
                rng = np.random.RandomState(seed); lines = []; expected = {}; identities = set(); regression_df = {}; op = 0
                for family in ['f_classif', 'chi2', 'f_regression', 'custom']:
                    for case in range(23 if family == 'custom' else (15 if family == 'f_regression' else 14)):
                        n, p = 43, 9; centered = True; mode = 'percentile'; all_k = False; has_p = True
                        if case == 1: n, p = 17, 3
                        if case == 2: n, p = 11, 19
                        if case == 3: n, p = 31, 1
                        if case == 13: n, p = 1203, 5
                        if family == 'custom' and case >= 14: n, p = 8, (1 if case == 21 else 2)
                        x = rng.normal(size=(n, p)); y = rng.randint(0, 3, n)
                        task = 1; func = family; direct = direct_p = None
                        if family == 'f_classif':
                            classes = 2 if case == 4 else (7 if case == 5 else 3)
                            y = np.arange(n) % classes; rng.shuffle(y)
                            x[:, 0] += (1 if case != 13 else 12)*y
                            if case == 6: x[:, -1] = 0
                            if case == 7: y[:] = 1
                            if case == 8: y = np.arange(n)
                            if case == 9: x[:, :] = np.arange(p)
                            if case == 10: x = np.round(x)
                            if case == 11: x[:, -1] = x[:, 0]
                            if case == 12: x *= .0001
                        elif family == 'chi2':
                            x = rng.poisson(2, size=(n, p)).astype(float); y = np.arange(n) % (2 if case == 4 else 3)
                            x[:, 0] += y*2
                            if case == 5: y = np.arange(n) % 7
                            if case == 6: x[:, -1] = 0
                            if case == 7: y[:] = 1
                            if case == 8: y = np.arange(n)
                            if case == 9: x[:] = 0
                            if case == 10: x *= .13
                            if case == 11: x[:, -1] = x[:, 0]
                            if case == 12: x *= 100
                            if case == 13: y = np.arange(n) % 101
                        elif family == 'f_regression':
                            task = 2; y = 1.3+2*x[:, 0]+rng.normal(size=n)*.7
                            if case == 4: centered = False
                            if case == 5: x[:, -1] = 0
                            if case == 6: y[:] = 0
                            if case == 7: x[:] = 1
                            if case == 8: n = 3; x = x[:n]; y = y[:n]
                            if case == 9: x[:, -1] = x[:, 0]
                            if case == 10: x *= .0001
                            if case == 11: x *= 10000
                            if case == 12: y = rng.normal(size=n)
                            if case == 13: y = 2*x[:, 0]+.0001*rng.normal(size=n)
                            if case == 14:
                                n = 17; x = np.column_stack([np.arange(n), np.sin(np.arange(n)), np.zeros(n)])
                                p = 3; y = x[:, 0].copy()
                        else:
                            task = 3; func = 'auto'; direct = np.round(rng.normal(size=p), 1)
                            direct_p = rng.uniform(size=p)
                            if case == 0: direct[:] = 1
                            if case == 1: direct[0] = np.nan; direct_p[0] = np.nan
                            if case == 2: direct[:] = np.nan; direct_p[:] = np.nan
                            if case == 3: direct[:] = np.inf; direct_p[:] = 0
                            if case == 4: direct[:3] = [-np.inf, 0, np.inf]
                            if case == 5: direct[:3] = [-np.finfo(float).max, 0, np.finfo(float).max]
                            if case == 6: direct_p = np.arange(1, p+1)*(.05/p)
                            if case == 7: direct_p[:] = .05
                            if case == 8: direct_p[:] = .05/p
                            if case == 9: direct_p[:] = 1
                            if case == 10: direct_p[:] = 0
                            if case == 11: direct[:] = -np.inf
                            if case == 12 and kind in (1, 5): has_p = False
                            if case == 13: direct_p = np.logspace(-300, -.1, p)
                            if case == 14: direct[:] = [-np.finfo(float).max, np.finfo(float).max]
                            if case == 15: direct[:] = np.finfo(float).max
                            if case == 16: direct[:] = [np.nan, np.inf]
                            if case == 17: direct[:] = [-np.inf, 2]
                            if case == 18: direct[:] = [2, np.inf]
                            if case == 19: direct[:] = np.inf
                            if case == 20: direct[:] = -np.inf
                            if case == 21: direct[:] = np.nan
                            if case == 22: direct[:] = [-np.finfo(float).max, np.finfo(float).max]
                        value = [10, 0, 100, 33.3, 50, 1, 75][case % 7] if kind == 1 else [.05, 0, 1, .01, .2][case % 5]
                        if kind == 5:
                            mode = ['percentile', 'k_best', 'fpr', 'fdr', 'fwe'][case % 5]
                            if mode == 'percentile': value = [0, 10, 50, 100][case % 4]
                            if mode == 'k_best': value = [0, 1, p, p+3][case % 4]
                            if case == 11: mode = 'k_best'; all_k = True; value = 0
                            if case == 12: mode = 'percentile'; value = 50
                        if family == 'custom' and case >= 14 and kind in (1, 5):
                            mode = 'percentile'; value = 25 if case == 22 else 50
                        if task == 3:
                            s = direct.copy(); pv = direct_p.copy()
                            score = (lambda X, y, a=s, b=pv: (a.copy(), b.copy())) if has_p else (lambda X, y, a=s: a.copy())
                        elif family == 'f_regression':
                            score = lambda X, y, c=centered: f_regression(X, y, center=c)
                        else: score = f_classif if family == 'f_classif' else chi2
                        if kind == 1: model = cls(score_func=score, percentile=value)
                        elif kind == 5: model = cls(score_func=score, mode=mode, param='all' if all_k else value)
                        else: model = cls(score_func=score, alpha=value)
                        model.fit(x, y); op += 1; record(model, x, f'{op}_', expected, identities)
                        if family == 'f_regression': regression_df[f'{op}_scores'] = n-(2 if centered else 1)
                        lines.extend([f'{kind} {task} {n} {p} {value} {int(has_p)}',
                                      f'{mode} {func} {"T" if centered else "F"} {"T" if all_k else "F"}'])
                        if task == 3:
                            lines.append(' '.join(f'{v:.17g}' for v in direct))
                            if has_p: lines.append(' '.join(f'{v:.17g}' for v in direct_p))
                        lines.extend(' '.join(f'{v:.17g}' for v in row) for row in np.column_stack([x, y]))
                inp = RESULTS/'univariate_input.txt'; out = RESULTS/'univariate_output.txt'
                inp.write_text(str(op)+'\n'+'\n'.join(lines)+'\n')
                proc = subprocess.run([str(executable), str(inp), str(out)], cwd=ROOT, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                assert proc.returncode == 0, (seed, kind, proc.stdout)
                actual = {}
                for line in out.read_text().splitlines():
                    name, *values = line.split(); actual[name] = np.array(values, dtype=float)
                compare(actual, expected, identities, regression_df, checks, differences, seed, kind)
                print(f'seed {seed} kind {kind}: {op} cases passed', flush=True)
    (RESULTS/'univariate_selection.json').write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks, numerical_differences=differences), indent=2)+'\n')
    print(f'{len(checks)} univariate selection result groups passed')


if __name__ == '__main__': main()
