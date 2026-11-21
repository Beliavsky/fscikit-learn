"""Brute partial dependence grids, prediction traces and ICE against sklearn 1.6.1."""
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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import partial_dependence
from sklearn.inspection._partial_dependence import _grid_from_X
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile=None):
    options = ['--profile', profile] if profile else ['--flag', FLAGS]
    run = subprocess.run(['fpm', 'run', *options, '--target', 'reference_dependence_driver', '--runner'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    paths = [ROOT/line.strip().strip('"') for line in run.stdout.splitlines()
        if line.strip().strip('"').endswith(('reference_dependence_driver', 'reference_dependence_driver.exe'))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, run.stdout
    return paths[0]


class AnalyticModel(RegressorMixin, BaseEstimator):
    def __init__(self, special=0): self.special = special
    def fit(self, x, y): self.fitted_ = True; return self
    def predict(self, x):
        z = np.nan_to_num(x, nan=0.); columns = np.arange(1, x.shape[1]+1)
        out = np.column_stack([z@columns+z[:, 0]*z[:, -1], (z**2/columns).sum(axis=1)])
        if self.special in [1, 2, 3]: out[:, -1] = [np.nan, np.inf, -np.inf][self.special-1]
        if self.special == 4: out[:, -1] = -np.inf; out[0, -1] = np.inf
        return out


def fixtures(seed):
    rng = np.random.default_rng(seed)
    for mode in range(10):
        for case in range(23):
            if case in range(13, 20) and mode != 0: continue
            n, p = 24, 3; m = 1 if case == 11 else 2 if case == 12 else 24
            train = rng.normal(size=(n, p)); x = rng.normal(size=(m, p))
            target = AnalyticModel().predict(train)[:, 0]+rng.normal(scale=.1, size=n)
            if mode >= 3:
                k = 2 if mode in [4, 6, 9] else 3; ids = np.arange(n) % k
                target = np.array([-7, 4, 19])[ids]; train[:, 0] += ids*1.2
                x[:, 0] += np.arange(m) % k
            features = [1]; resolution = 4; limits = [.05, .95]; kind = 0; cat = 0; special = 0
            if case == 1: resolution = 5; limits = [0., 1.]; kind = 2
            if case == 2: features = [1, 3]; resolution = 3; limits = [0., 1.]; kind = 1
            if case == 3: features = [3, 1]; limits = [.1, .9]; kind = 2
            if case == 4: features = [2]
            if case == 5: features = [1, 2]; resolution = 3; kind = 2
            if case == 6:
                features = [3, 1]; resolution = 3; cat = 1; kind = 2
                x[:, 2] = np.arange(m) % 3-1
            if case == 7: features = [2]; resolution = 2; cat = 2; x[:, 1] = np.arange(m) % 4
            if case == 8: features = [2]; x[:, 1] = np.arange(m) % 2; kind = 2
            if case == 9:
                features = [2]; limits = [.25, .75]
                x[:, 1] = np.r_[np.zeros(10), np.ones(7), np.full(6, 2.), 3.]
            if case == 10: features = [1, 1]; resolution = 3; kind = 2
            if case in [11, 12]: kind = 2
            if case == 13: x[::7, 0] = np.nan; resolution = 3; kind = 2
            if case == 14:
                features = [2]; resolution = 2; cat = 4; kind = 2
                x[:, 1] = np.arange(m) % 4; x[::7, 1] = np.nan
            if case == 15: x[:, 0] = np.nan; kind = 2
            if case in [16, 17, 18, 19]: special = case-15; kind = 2
            if case == 20: features = [3, 1, 2]; resolution = 3; kind = 2
            if case == 21: features = [1, 2]; resolution = 3; kind = 1
            if case == 22: cat = 3
            weighted = case in [4, 5, 17, 21]
            w = rng.uniform(.3, 2., m) if weighted else np.ones(m)
            if case in [4, 17]: w[0] = 0
            if case == 5: w[0] = -.1
            if case == 21: w = -np.arange(1, m+1, dtype=float)
            model = [AnalyticModel(special), LinearRegression(), Ridge(alpha=.7),
                LogisticRegression(C=.7, tol=1.e-10, max_iter=500, solver='newton-cg'),
                LogisticRegression(C=.7, tol=1.e-10, max_iter=500, solver='newton-cg'),
                LogisticRegression(C=.7, tol=1.e-10, max_iter=500, solver='newton-cg'),
                LogisticRegression(C=.7, tol=1.e-10, max_iter=500, solver='newton-cg'), GaussianNB(),
                KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='brute'),
                LinearSVC(C=.7, tol=1.e-10, max_iter=10000, dual=False)][mode]
            yield dict(seed=seed, mode=mode, case=case, features=features, resolution=resolution, limits=limits,
                kind=kind, cat=cat, special=special, weighted=weighted), train, target, x, w, model


def method_name(mode):
    return 'predict' if mode <= 2 else 'decision_function' if mode in [5, 6, 9] else 'predict_proba'


def resolved_response(raw, mode, n):
    if mode == 4: raw = raw[:, 1]
    return np.asarray(raw).reshape(n, -1)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--executable', type=Path); parser.add_argument('--profile')
    parser.add_argument('--seed', action='append', type=int); args = parser.parse_args()
    executable = args.executable.resolve() if args.executable else resolve(args.profile)
    RESULTS.mkdir(exist_ok=True); checks = []; source_errors = []; fixture_count = 0
    for seed in args.seed or [0, 7, 42, 2026, 314159]:
        for cfg, train, target, x, w, model in fixtures(seed):
            mode, case = cfg['mode'], cfg['case']; context = (seed, mode, case)
            n, p = train.shape; m = len(x); features = np.array(cfg['features'])-1; axes = len(features)
            data, output = RESULTS/'dependence_input.txt', RESULTS/'dependence_actual.txt'
            with data.open('w') as stream:
                stream.write(' '.join(map(str, [mode, case, n, p, m, axes, cfg['resolution'], cfg['kind'],
                    cfg['cat'], int(cfg['weighted']), cfg['special']]))+'\n')
                for a in [np.array(cfg['features'])[None], np.array(cfg['limits'])[None], w[None], train, target[None], x]:
                    np.savetxt(stream, a, fmt='%.17g')
            run = subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert run.returncode == 0, (context, run.stdout)
            traces = []
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(train, target); before_state = copy.deepcopy(model.__dict__)
                method = method_name(mode); original = getattr(model, method)
                wanted_before = resolved_response(original(x), mode, m)
                def traced(features):
                    raw = original(features)
                    traces.append((features.copy(), resolved_response(raw, mode, len(features)).copy()))
                    return raw
                # Preserve source response-method name checks and real classifier/regressor tags.
                traced.__name__ = method; setattr(model, method, traced)
                categories = None if cfg['cat'] == 0 else np.array([False, False, True]) if cfg['cat'] == 1 else [1] if cfg['cat'] in [2, 4] else [99]
                kwargs = dict(grid_resolution=cfg['resolution'], percentiles=tuple(cfg['limits']), method='brute',
                    kind=['average', 'individual', 'both'][cfg['kind']], categorical_features=categories)
                if mode in [5, 6, 9]: kwargs['response_method'] = 'decision_function'
                if cfg['weighted']: kwargs['sample_weight'] = w
                is_cat = [bool(categories[j]) for j in features] if cfg['cat'] == 1 else [j in (categories or []) for j in features]
                grid, grid_values = _grid_from_X(x[:, features], cfg['limits'], is_cat, cfg['resolution'])
                expected = None; failure = None
                try:
                    expected = partial_dependence(model, x, features, **kwargs)
                except ValueError as exc:
                    assert mode in [4, 6, 9] and (m == 2 or len(grid) == 2), (context, str(exc))
                    assert 'cannot reshape' in str(exc), (context, str(exc))
                    failure = str(exc)
                    source_errors.append(dict(seed=seed, mode=mode, case=case, n_samples=m, grid_points=len(grid), message=failure))
                delattr(model, method)
                assert len(traces) == len(grid), (context, len(traces), len(grid))
                if failure:
                    # Source has already evaluated each grid point correctly; only its post-response reshape fails.
                    expected_average = np.array([np.average(response, axis=0, weights=w if cfg['weighted'] else None)
                        for _, response in traces]).T
                    expected_individual = np.array([response for _, response in traces]).transpose(2, 1, 0)
                else:
                    grid_values = expected.grid_values
                    expected_average = expected.average.reshape(wanted_before.shape[1], -1) if cfg['kind'] != 1 else None
                    expected_individual = expected.individual.reshape(wanted_before.shape[1], m, -1) if cfg['kind'] != 0 else None
            lines = iter(output.read_text().splitlines())
            def vector(dtype=float): return np.fromstring(next(lines), sep=' ', dtype=dtype)
            def matrix(rows, columns): return np.array([vector() for _ in range(rows)]).reshape(rows, columns)
            def record(name, oracle='sklearn'):
                checks.append(dict(seed=seed, mode=mode, case=case, name=name, oracle=oracle, passed=True))
            tolerance = 3.e-12 if mode == 0 else 3.e-7
            for j, (features_, response) in enumerate(traces):
                actual_features = matrix(m, p)
                untouched = [i for i in range(p) if i not in features]
                np.testing.assert_array_equal(actual_features[:, untouched], x[:, untouched])
                np.testing.assert_allclose(actual_features, features_, atol=3.e-12, rtol=3.e-12, equal_nan=True)
                np.testing.assert_allclose(matrix(m, wanted_before.shape[1]), response, atol=tolerance, rtol=tolerance,
                    equal_nan=True, err_msg=str((context, j)))
                record('response_'+str(j))
            dimensions, outputs, points, has_average, has_individual, calls = vector(int)
            assert (dimensions, outputs, points, calls) == (axes, wanted_before.shape[1], len(grid), len(traces))
            assert bool(has_average) == (cfg['kind'] != 1) and bool(has_individual) == (cfg['kind'] != 0)
            record('metadata')
            for j, values in enumerate(grid_values):
                assert int(vector(int)[0]) == len(values)
                np.testing.assert_allclose(vector(), values, atol=3.e-12, rtol=3.e-12, equal_nan=True)
                record('axis_'+str(j))
            oracle = 'source_binary_shape_identity' if failure else 'sklearn'
            if has_average:
                np.testing.assert_allclose(matrix(outputs, points), expected_average, atol=tolerance, rtol=tolerance, equal_nan=True,
                    err_msg=str((context, 'average')))
                record('average', oracle)
            if has_individual:
                individual = np.array([matrix(m, points) for _ in range(outputs)])
                np.testing.assert_allclose(individual, expected_individual, atol=tolerance, rtol=tolerance, equal_nan=True,
                    err_msg=str((context, 'individual')))
                record('individual', oracle)
            before, after = matrix(m, outputs), matrix(m, outputs)
            np.testing.assert_array_equal(before, after)
            np.testing.assert_allclose(before, wanted_before, atol=tolerance, rtol=tolerance, equal_nan=True)
            record('model_response'); record('model_preserved', 'input_identity')
            rows, columns = vector(int); coef_before, intercept_before = matrix(rows, columns), vector()
            np.testing.assert_array_equal(coef_before, matrix(rows, columns)); np.testing.assert_array_equal(intercept_before, vector())
            record('parameters_preserved', 'input_identity')
            if rows:
                coef = model.theta_ if mode == 7 else np.atleast_2d(model.coef_)
                intercept = model.class_prior_ if mode == 7 else np.atleast_1d(model.intercept_)
                np.testing.assert_allclose(coef_before, coef, atol=tolerance, rtol=tolerance)
                np.testing.assert_allclose(intercept_before, intercept, atol=tolerance, rtol=tolerance)
                record('model_parameters')
            for key, value in before_state.items():
                if isinstance(value, np.ndarray): np.testing.assert_array_equal(value, model.__dict__[key])
            for actual, wanted in [(matrix(n, p), train), (vector(), target), (matrix(m, p), x), (vector(), w)]:
                np.testing.assert_array_equal(actual, wanted)
            assert list(lines) == []; record('inputs_preserved', 'input_identity'); fixture_count += 1
        print(f'seed {seed}: partial dependence passed', flush=True)
    (RESULTS/'partial_dependence.json').write_text(json.dumps(dict(sklearn=sklearn.__version__, fixtures=fixture_count,
        checks=checks, source_shape_errors=source_errors), indent=2)+'\n')
    print(f'{len(checks)} partial dependence groups across {fixture_count} fixtures passed')


if __name__ == '__main__': main()
