"""Permutation importance against sklearn, including every scorer input and RNG trace."""
from pathlib import Path
import argparse
import copy
import json
import os
import subprocess
import warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
import numpy as np
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile=None):
    options = ['--profile', profile] if profile else ['--flag', FLAGS]
    run = subprocess.run(['fpm', 'run', *options, '--target', 'reference_inspection_driver', '--runner'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    paths = [ROOT/line.strip().strip('"') for line in run.stdout.splitlines()
        if line.strip().strip('"').endswith(('reference_inspection_driver', 'reference_inspection_driver.exe'))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, run.stdout
    return paths[0]


class AnalyticModel:
    def fit(self, x, y=None): return self
    def predict(self, x):
        clean = np.nan_to_num(x, nan=0.)
        columns = np.arange(1, clean.shape[1]+1)
        return np.column_stack([clean@columns, (clean**2/columns).sum(axis=1)])


def fixtures(seed):
    rng = np.random.default_rng(seed)
    for mode in range(6):
        for case in range(23):
            if case in [8, 15, 19] and mode != 0:
                continue
            if case == 6 and mode not in [0, 1, 2]:
                continue
            n = 1 if case == 6 else 10 if case in [20, 22] else 100 if case in [9, 10, 21] else 40
            p = 1 if case == 17 else 3
            repeats = 1 if case == 1 else 7 if case in [7, 12, 13, 14, 15] else 5
            metrics = 1 if case in [0, 3, 6] else 3
            x = rng.normal(size=(n, p))
            if case == 7: x[:, -1] = 3.
            y = AnalyticModel().predict(x)+rng.normal(scale=.1, size=(n, 2))
            if mode >= 3:
                ids = np.arange(n) % 3
                y[:, 0] = np.array([-7, 4, 19])[ids]
                x[:, 0] += ids*1.1
            if case in [8, 19]: x[::4, -1] = np.nan
            weighted = case in [4, 5, 19]
            w = rng.uniform(.3, 2., n) if weighted else np.ones(n)
            if case in [5, 19]: w[0] = -.1
            count, fractional, fraction = -1, False, 1.
            if case in [2, 9, 19]: count = 1
            if case == 3: count = n//3
            if case in [4, 11, 16]:
                fractional = True; fraction = {4:.5, 11:.001, 16:1.}[case]
            if case in [5, 10]: count = n-1
            if case == 18: count = n
            if case in [20, 21]: fractional = 2; fraction = .7
            if case == 22: fractional = True; fraction = .7
            special = {12:1, 13:2, 14:3, 15:4}.get(case, 0)
            model = [AnalyticModel(), LinearRegression(), Ridge(alpha=.7),
                LogisticRegression(C=.7, tol=1.e-10, max_iter=500, solver='newton-cg'),
                GaussianNB(), KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='brute')][mode]
            yield dict(seed=seed, mode=mode, case=case, repeats=repeats, metrics=metrics, weighted=weighted,
                count=count, fractional=fractional, fraction=fraction, special=special), x, y, w, model


def response(model, x, mode):
    if mode >= 3: return model.predict_proba(x)
    return np.asarray(model.predict(x)).reshape(len(x), -1)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--executable', type=Path); parser.add_argument('--profile')
    parser.add_argument('--seed', action='append', type=int); args = parser.parse_args()
    executable = args.executable.resolve() if args.executable else resolve(args.profile)
    RESULTS.mkdir(exist_ok=True); checks = []; fixture_count = 0
    for seed in args.seed or [0, 7, 42, 2026, 314159]:
        for cfg, x, y, w, model in fixtures(seed):
            mode, case, repeats, metrics = cfg['mode'], cfg['case'], cfg['repeats'], cfg['metrics']
            n, p = x.shape; context = (seed, mode, case)
            data, output = RESULTS/'inspection_input.txt', RESULTS/'inspection_actual.txt'
            with data.open('w') as stream:
                stream.write(' '.join(map(str, [n, p, repeats, seed, cfg['count'], metrics, int(cfg['weighted']),
                    mode, cfg['special'], int(cfg['fractional']), cfg['fraction']]))+'\n')
                for a in [w[None], x, y]: np.savetxt(stream, a, fmt='%.17g')
            run = subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert run.returncode == 0, (context, run.stdout)
            traces = []
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(x, y[:, 0] if mode else y)
                before_model = copy.deepcopy(model.__dict__)
                def scorer(estimator, features, rows, sample_weight=None):
                    prediction = response(estimator, features, mode) if len(rows) else None
                    rows = np.asarray(rows, int)
                    values = np.array([-1., -2., -3.])
                    if len(rows):
                        if mode <= 2:
                            error = prediction-y[rows-1, :prediction.shape[1]]
                            point = np.column_stack([-(error**2).sum(axis=1), -abs(error).sum(axis=1), prediction.sum(axis=1)])
                        else:
                            labels = estimator.classes_[prediction.argmax(axis=1)]
                            target = y[rows-1, 0]
                            hot = target[:, None] == estimator.classes_
                            point = np.column_stack([labels == target, -((prediction-hot)**2).sum(axis=1), prediction[:, -1]])
                        values = np.average(point, axis=0, weights=sample_weight)
                    special = cfg['special']
                    if special in [1, 2, 3]: values[2] = [np.nan, np.inf, -np.inf][special-1]
                    if special == 4:
                        call = len(traces)+1
                        values[0] = (.75 if call == 1 else -.75)*np.finfo(float).max
                        values[1] = .75*np.finfo(float).max if call % 2 else 0.
                        values[2] = -np.inf if call == 1 else 0.
                    traces.append((features.copy(), rows.copy(), None if sample_weight is None else sample_weight.copy(), values.copy()))
                    return float(values[0]) if metrics == 1 else {f'metric_{i}': value for i, value in enumerate(values)}
                kwargs = dict(n_repeats=repeats, random_state=seed, scoring=scorer, n_jobs=1)
                if cfg['fractional']: kwargs['max_samples'] = float(np.float32(cfg['fraction'])) if cfg['fractional'] == 2 else cfg['fraction']
                elif cfg['count'] != -1: kwargs['max_samples'] = cfg['count']
                if cfg['weighted']: kwargs['sample_weight'] = w
                expected = permutation_importance(model, x, np.arange(1, n+1), **kwargs)
                if metrics == 1: expected = [expected]
                else: expected = [expected[f'metric_{i}'] for i in range(metrics)]
            lines = iter(output.read_text().splitlines())
            def vector(dtype=float): return np.fromstring(next(lines), sep=' ', dtype=dtype)
            def matrix(rows, cols): return np.array([vector() for _ in range(rows)]).reshape(rows, cols)
            def record(name, oracle='sklearn'):
                checks.append(dict(seed=seed, mode=mode, case=case, name=name, oracle=oracle, passed=True))
            tol = 3.e-12 if mode == 0 else 3.e-7
            assert len(traces) == 1+p*repeats
            for call, (features, rows, weights, scores) in enumerate(traces):
                count, has_weights = vector(int)
                assert count == len(rows) and bool(has_weights) == (weights is not None)
                np.testing.assert_array_equal(vector(int), rows)
                if weights is not None: np.testing.assert_array_equal(vector(), weights)
                np.testing.assert_array_equal(matrix(count, p), features)
                np.testing.assert_allclose(vector(), scores, atol=tol, rtol=tol, equal_nan=True, err_msg=str((context, call)))
                record('callback_'+str(call))
            actual_metrics, columns, actual_repeats, calls = vector(int)
            assert (actual_metrics, columns, actual_repeats, calls) == (metrics, p, repeats, len(traces))
            record('metadata')
            for k, expected_metric in enumerate(expected):
                mean, std, raw = vector(), vector(), matrix(p, repeats)
                for field, actual in [('importances_mean', mean), ('importances_std', std), ('importances', raw)]:
                    np.testing.assert_allclose(actual, expected_metric[field], atol=2*tol, rtol=2*tol,
                        equal_nan=True, err_msg=str((context, k, field)))
                    record(str(k)+'_'+field)
            width = 2 if mode == 0 else len(model.classes_) if mode >= 3 else 1
            native_before, native_after = matrix(n, width), matrix(n, width)
            np.testing.assert_array_equal(native_before, native_after)
            np.testing.assert_allclose(native_before, response(model, x, mode), atol=tol, rtol=tol, equal_nan=True)
            record('model_response'); record('model_preserved', 'input_identity')
            if mode in [1, 2, 3, 4]:
                rows = (len(model.classes_) if mode == 4 or mode == 3 and len(model.classes_) > 2 else 1)
                coef, intercept = matrix(rows, p), vector()
                np.testing.assert_array_equal(coef, matrix(rows, p)); np.testing.assert_array_equal(intercept, vector())
                expected_coef = model.theta_ if mode == 4 else np.atleast_2d(model.coef_)
                expected_intercept = model.class_prior_ if mode == 4 else np.atleast_1d(model.intercept_)
                np.testing.assert_allclose(coef, expected_coef, atol=tol, rtol=tol)
                np.testing.assert_allclose(intercept, expected_intercept, atol=tol, rtol=tol)
                record('model_parameters'); record('model_parameters_preserved', 'input_identity')
            for key, value in before_model.items():
                if isinstance(value, np.ndarray): np.testing.assert_array_equal(value, model.__dict__[key])
            for actual, wanted in [(matrix(n, p), x), (matrix(n, 2), y), (vector(), w)]:
                np.testing.assert_array_equal(actual, wanted)
            assert list(lines) == []; record('inputs_preserved', 'input_identity'); fixture_count += 1
        print(f'seed {seed}: permutation importance passed', flush=True)
    (RESULTS/'inspection.json').write_text(json.dumps(dict(sklearn=sklearn.__version__, fixtures=fixture_count, checks=checks), indent=2)+'\n')
    print(f'{len(checks)} permutation importance groups across {fixture_count} fixtures passed')


if __name__ == '__main__': main()
