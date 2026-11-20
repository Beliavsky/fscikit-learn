"""Compare native OMP CV to sklearn 1.6.1, observing actual fold paths."""
from pathlib import Path
import argparse
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.linear_model import OrthogonalMatchingPursuitCV
import sklearn.linear_model._omp as omp_module
from sklearn.model_selection import KFold, PredefinedSplit, TimeSeriesSplit, GroupKFold, ShuffleSplit
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def run(executable, lines):
    inp = RESULTS / 'omp_cv_input.txt'; out = RESULTS / 'omp_cv_output.txt'
    inp.write_text('\n'.join(lines)+'\n')
    process = subprocess.run([str(executable), str(inp), str(out)], cwd=ROOT,
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.returncode == 0, process.stdout
    actual = {}
    for line in out.read_text().splitlines():
        name, *values = line.split(); actual[name] = np.array(values, dtype=float)
    return actual


def reference(x, y, query, folds, intercept, max_iter):
    recorded = []
    original = omp_module._omp_path_residues
    def observe(*args, **kwargs):
        result = original(*args, **kwargs); recorded.append(result.copy()); return result
    omp_module._omp_path_residues = observe
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            model = OrthogonalMatchingPursuitCV(cv=folds, max_iter=max_iter,
                                              fit_intercept=intercept, n_jobs=1).fit(x, y)
    finally:
        omp_module._omp_path_residues = original
    common = min(len(path) for path in recorded)
    errors = np.array([np.mean(path[:common]**2, axis=1) for path in recorded]).T
    values = dict(coef=model.coef_, intercept=[model.intercept_], train=model.predict(x),
                  query=model.predict(query), score=[model.score(x, y)],
                  mse=errors, best_mse=[errors.mean(axis=1).min()],
                  steps=[len(path) for path in recorded],
                  train_sizes=[len(a) for a, b in folds], test_sizes=[len(b) for a, b in folds],
                  meta=[model.n_nonzero_coefs_, model.n_iter_, model.n_features_in_, len(folds)])
    return values


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--executable', type=Path); args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable:
        executable = args.executable.resolve()
    else:
        process = subprocess.run(['fpm', 'run', '--flag', FLAGS, '--target', 'reference_omp_cv_driver', '--runner'],
                                 cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        paths = [ROOT / line.strip().strip('"') for line in process.stdout.splitlines()
                 if line.strip().strip('"').endswith(('reference_omp_cv_driver', 'reference_omp_cv_driver.exe'))]
        paths = [p.resolve() for p in paths if p.is_file()]
        assert len(paths) == 1, process.stdout
        executable = paths[0]
    checks = []; equivalences = []
    for seed, n, p in [(0, 38, 7), (7, 31, 9), (42, 19, 24), (2026, 47, 6)]:
        rng = np.random.RandomState(seed)
        base = rng.normal(size=(n, p))*np.linspace(.7, 1.4, p)+np.linspace(-.3, .5, p)
        weights = np.zeros(p); weights[[0, 2, 4]] = [2, -1.5, .8]
        for case in range(22):
            expected = {}; lines = ['2']; designs = {}
            for op in range(2):
                x = base.copy(); y = 2+x@weights+.2*rng.normal(size=n)
                query = rng.normal(size=(5, p)); intercept = case != 1; limit = None
                cv = 5; custom = 0; fold_map = None
                if case == 2: limit = 1
                if case == 3: limit = 2
                if case == 4: limit = p
                if case == 5: cv = 2
                if case == 6: cv = n
                if case == 7:
                    fold_map = np.array([[-1, 8, -2][i % 3] for i in range(n)]); custom = 1
                if case == 8:
                    fold_map = np.full(n, -1); fold_map[-5:] = 99; custom = 1
                if case == 9:
                    x[:, -1] = x[:, 0]; limit = p
                if case == 10:
                    x[:, -1] = 3; limit = p
                if case == 11:
                    x[:, -2:] = x[:, :2] @ np.array([[1, .3], [.2, 1]])
                    query[:, -2:] = query[:, :2] @ np.array([[1, .3], [.2, 1]]); limit = p
                if case == 12: y = 2+x[:, 0]*2-x[:, 2]*1.5
                if case == 13:
                    x = x[:, :2]; query = query[:, :2]; y = 2+x@np.array([2., -1.])+.1*rng.normal(size=n)
                if case == 14: x *= .001; y = 2+x@weights+.001*rng.normal(size=n)
                if case == 15: x *= 1000; y = 2+x@weights+20*rng.normal(size=n)
                if case == 21:
                    x = rng.normal(size=(95, 60)); w = np.zeros(60); w[[0, 2, 4]] = [2, -1.5, .8]
                    y = 2+x@w+.2*rng.normal(size=95); query = rng.normal(size=(5, 60))
                if case == 16:
                    folds = list(TimeSeriesSplit(n_splits=3, gap=1, max_train_size=12).split(x)); custom = 2
                elif case == 17:
                    folds = list(GroupKFold(3).split(x, y, groups=np.arange(n) % 6)); custom = 2
                elif case == 18:
                    folds = list(ShuffleSplit(4, test_size=.2, random_state=seed).split(x)); custom = 2
                elif case == 19:
                    if op:
                        x = x[:-5, :3]; y = y[:-5]; query = query[:, :3]
                    folds = list(KFold(3).split(x)); custom = 2
                elif custom == 1:
                    folds = list(PredefinedSplit(fold_map).split(x))
                else:
                    folds = list(KFold(cv).split(x))
                if case == 20:
                    x[:, -1] = x[:, 0]+.01*x[:, -1]
                    query[:, -1] = query[:, 0]+.01*query[:, -1]; limit = p
                if op: intercept = not intercept
                designs[op+1] = (x.copy(), query.copy(), intercept, folds, y.copy())
                values = reference(x, y, query, folds, intercept, limit)
                expected.update({f'{op+1}_{key}': value for key, value in values.items()})
                lines.append(f'{len(x)} {x.shape[1]} {len(query)} {cv} {limit or 0} {"T" if intercept else "F"} {custom} {len(folds)}')
                if custom == 1: lines.append(' '.join(map(str, fold_map)))
                if custom == 2:
                    for train, test in folds:
                        a = np.zeros(len(x), dtype=int); b = a.copy(); a[train] = 1; b[test] = 1
                        lines.extend([' '.join(map(str, a)), ' '.join(map(str, b))])
                lines.extend(' '.join(f'{v:.17g}' for v in row) for row in np.column_stack([x, y]))
                lines.extend(' '.join(f'{v:.17g}' for v in row) for row in query)
            actual = run(executable, lines)
            assert actual.keys() == expected.keys(), (seed, case)
            for name, value in expected.items():
                want = np.asarray(value).ravel(order='F'); assert actual[name].shape == want.shape, (seed, case, name)
                tol = 0 if name.endswith(('meta', 'steps', 'train_sizes', 'test_sizes')) else 3.e-8
                oracle = 'sklearn'
                if name.endswith('_steps') and not np.array_equal(actual[name], want):
                    prefix = name.split('_')[0]+'_'
                    x, query, intercept, folds, y = designs[int(name.split('_')[0])]
                    common = len(np.asarray(expected[prefix+'mse']))
                    assert np.min(actual[name]) == np.min(want) == common
                    assert np.all(actual[name] >= common) and np.all(actual[name] <= x.shape[1])
                    for f in np.flatnonzero(actual[name] != want):
                        train = folds[f][0]; a = x[train]
                        centered = a-a.mean(axis=0) if intercept else a
                        rank = np.linalg.matrix_rank(centered)
                        if min(actual[name][f], want[f]) < rank:
                            assert case == 12 and intercept
                            np.testing.assert_allclose(y, 2+2*x[:, 0]-1.5*x[:, 2], atol=1.e-12, rtol=0)
                    oracle = 'unused path-tail iteration difference'
                    equivalences.append(dict(seed=seed, case=case, name=name,
                                             native=actual[name].tolist(), source=want.tolist(), common_steps=common))
                elif case == 11 and name.endswith('_coef') and not np.allclose(actual[name], want, atol=tol, rtol=tol):
                    x, query, intercept, folds, y = designs[int(name.split('_')[0])]
                    center = x.mean(axis=0) if intercept else np.zeros(x.shape[1])
                    assert np.linalg.matrix_rank(x-center) < x.shape[1]
                    difference = actual[name]-want
                    np.testing.assert_allclose((x-center)@difference, 0, atol=tol, rtol=0)
                    np.testing.assert_allclose((query-center)@difference, 0, atol=tol, rtol=0)
                    oracle = 'rank-deficient coefficient equivalence'
                    equivalences.append(dict(seed=seed, case=case, name=name,
                                             max_training_difference=float(np.max(np.abs((x-center)@difference))),
                                             max_query_difference=float(np.max(np.abs((query-center)@difference)))))
                else:
                    np.testing.assert_allclose(actual[name], want, atol=tol, rtol=tol,
                                               err_msg=f'seed={seed} case={case} {name}')
                checks.append(dict(seed=seed, case=case, name=name, passed=True, oracle=oracle))
            print(f'seed {seed} case {case}: passed', flush=True)
    source_failures = []
    a = np.column_stack([[-3., -2., -1., 1., 2., 3.], [0., 1., 2., 0., -1., -2.], np.ones(6)])
    for name, x, y in [('zero target', a, np.zeros(6)), ('constant X', np.ones((6, 3)), np.arange(6.))]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            try:
                OrthogonalMatchingPursuitCV(cv=2, max_iter=2).fit(x, y)
            except ValueError as error:
                assert 'argmin of an empty sequence' in str(error), str(error)
                source_failures.append(dict(case=name, source_error=str(error), passed=True))
            else:
                raise AssertionError(f'{name}: expected source empty-path failure')
    (RESULTS/'omp_cv.json').write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks,
        numerical_differences=equivalences, source_empty_paths=source_failures), indent=2)+'\n')
    print(f'{len(checks)} OMP CV result groups passed')


if __name__ == '__main__':
    main()
