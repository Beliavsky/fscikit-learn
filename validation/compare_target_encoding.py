"""Numeric TargetEncoder, including source cross-fitting, versus sklearn 1.6.1."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import KFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile):
    options = ['--profile', profile] if profile == 'release' else ['--flag', FLAGS]
    run = subprocess.run(['fpm', 'run', *options, '--target', 'reference_target_encoding_driver', '--runner'],
                         cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    paths = [ROOT / s.strip().strip('"') for s in run.stdout.splitlines()
             if s.strip().strip('"').endswith(('reference_target_encoding_driver', 'reference_target_encoding_driver.exe'))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, run.stdout
    return paths[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--profile', choices=['checked', 'release'], default='checked')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    executable = args.executable.resolve() if args.executable else resolve(args.profile)
    checks, source_warnings = [], []
    fixtures = 0

    def run(seed, case, x, y, query, kind, smooth, cv, shuffle, categories, cross):
        nonlocal fixtures
        model = TargetEncoder(target_type=kind, smooth=smooth, cv=cv, shuffle=shuffle,
                              random_state=seed if shuffle else None, categories=categories)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            expected = {}
            if cross:
                expected['cross'] = model.fit_transform(x, y)
                splitter = KFold if model.target_type_ == 'continuous' else StratifiedKFold
                folds = splitter(cv, shuffle=shuffle, random_state=seed if shuffle else None)
                assignment = np.zeros(len(y), dtype=int)
                for j, (_, test) in enumerate(folds.split(x, y), 1):
                    assignment[test] = j
                expected['fold'] = assignment
            else:
                model.fit(x, y)
            expected.update(train=model.transform(x), query=model.transform(query),
                            shape=model.transform(query).shape, mean=np.atleast_1d(model.target_mean_),
                            classes=[] if model.classes_ is None else model.classes_,
                            type=[['continuous', 'binary', 'multiclass'].index(model.target_type_)],
                            counts=[len(c) for c in model.categories_],
                            categories=np.concatenate(model.categories_), encodings=np.concatenate(model.encodings_))
            expected['snapshot'] = expected['query']
            # Refit preserves the smoothing policy while resetting categories and learned width.
            model.set_params(categories='auto', target_type='continuous', smooth='auto' if smooth == 'auto' else 19.)
            model.fit([[1], [1], [2]], [1., 3., 8.])
            expected['refit'] = model.transform([[1], [2], [3]])
        source_warnings.extend(dict(seed=seed, case=case, category=w.category.__name__, message=str(w.message))
                               for w in caught)
        lines = [f'{len(x)} {x.shape[1]} {len(query)} {int(cross)}',
                 f'{kind} {int(smooth == "auto")} {0 if smooth == "auto" else smooth} {cv} '
                 f'{int(shuffle)} {seed} {int(not isinstance(categories, str))}']
        lines += [' '.join(map(str, row)) for row in x]
        lines += [' '.join(map(str, y))]
        lines += [' '.join(map(str, row)) for row in query]
        if not isinstance(categories, str):
            for col in categories:
                lines += [str(len(col)), ' '.join(map(str, col))]
        inp = RESULTS / f'target_encoding_{args.profile}_input.txt'
        out = RESULTS / f'target_encoding_{args.profile}_output.txt'
        inp.write_text('\n'.join(lines) + '\n')
        proc = subprocess.run([str(executable), str(inp), str(out)], cwd=ROOT,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.returncode == 0, (seed, case, proc.stdout)
        actual = {}
        for line in out.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
        for name, want in expected.items():
            want = np.asarray(want).ravel(order='F')
            if name in ['fold', 'shape', 'counts', 'type', 'classes', 'categories']:
                np.testing.assert_array_equal(actual[name], want, err_msg=f'{seed} {case} {name}')
            else:
                np.testing.assert_allclose(actual[name], want, rtol=2.e-12, atol=2.e-13,
                                           equal_nan=True, err_msg=f'{seed} {case} {name}')
            checks.append(dict(seed=seed, case=case, name=name, passed=True, oracle='sklearn'))
        fixtures += 1

    for seed in [0, 7, 42, 2026, 314159]:
        rng = np.random.RandomState(seed)
        case = 0
        for c in [0, 1, 2, 3, 7]:
            n, p = 43, 3
            x = rng.randint(-2, 5, size=(n, p)).astype(float)
            x[::5, 0] = np.nan
            x[:, 1] = rng.permutation(np.arange(n))  # all singleton categories
            x[:, 2] = -7  # constant category
            query = np.array([[np.nan, -999., -7], [99., 8., -7], [-1., 32., 33.]])
            y = (rng.normal(size=n) * 7 + 1.2 if c == 0 else
                 rng.permutation(np.resize(np.arange(c) * 13 - 21, n)))
            for kind in (['auto', 'continuous'] if c == 0 else
                         ['auto', 'continuous', 'binary'] + (['multiclass'] if c != 2 else [])):
                for smooth in ['auto', 0., 1., 1000.]:
                    for shuffle in [False, True]:
                        case += 1
                        run(seed, case, x, y, query, kind, smooth, 3, shuffle, 'auto', True)
        # Explicit category lists include unused and omitted training categories and NaNs.
        for smooth in ['auto', 0., .5, 19.]:
            for kind in ['continuous', 'binary', 'multiclass']:
                for cross in [False, True]:
                    case += 1
                    x = np.column_stack([rng.randint(0, 4, 31), rng.randint(-3, 2, 31)]).astype(float)
                    x[::3, 0] = np.nan
                    y = rng.normal(size=31) if kind == 'continuous' else np.resize([-8, 4, 77], 31)
                    run(seed, case, x, y, np.array([[np.nan, 0], [8, -3], [1, 99.]]), kind, smooth,
                        5, True, [np.array([0., 1., 9., np.nan]), np.array([-3., -1., 0., 1.])], cross)
        # Rare classes, first-appearance class ordering, constant and binary folds, and cv=n.
        for labels in [[99] * 10 + [-3] * 2 + [4], [-3] * 9 + [4], [7] * 10]:
            for smooth in ['auto', 0., 2.]:
                for shuffle in [False, True]:
                    case += 1
                    x = np.arange(len(labels), dtype=float)[:, None] % 3
                    run(seed, case, x, np.array(labels), [[99.]], 'auto', smooth, 5, shuffle, 'auto', True)
        for smooth in ['auto', 0., 100.]:
            case += 1
            run(seed, case, np.array([[1.], [np.nan], [1.], [2.]]), np.array([3.2, 1.5, 7.1, -2.3]),
                np.array([[np.nan], [99.]]), 'continuous', smooth, 4, True, 'auto', True)
            case += 1
            run(seed, case, np.array([[np.nan]]), np.array([7.]), [[1.]], 'auto', smooth, 5, True, 'auto', False)
        for smooth in ['auto', 0., 1.e100]:
            case += 1
            run(seed, case, np.array([[1.], [1.], [2.], [3.]]), np.array([.2, .7, .2, .7]),
                [[99.]], 'binary', smooth, 2, True, 'auto', False)
            case += 1
            run(seed, case, np.array([[1.], [1.], [2.], [3.]]), np.array([1.e30, 2.e30, -1.e30, 4.e30]),
                [[99.]], 'auto', smooth, 2, True, 'auto', True)
        print(f'seed {seed}: target encodings passed', flush=True)
    source_root = Path(inspect.getfile(TargetEncoder)).parent
    paths = [source_root / '_target_encoder.py', source_root / '_target_encoder_fast.pyx',
             source_root / '_encoders.py',
             source_root.parent / 'model_selection/_split.py']
    report = dict(sklearn=sklearn.__version__, profile=args.profile, fixtures=fixtures, checks=checks,
                  source_warnings=source_warnings,
                  source_sha256={str(p.relative_to(source_root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in paths})
    (RESULTS / f'target_encoding_{args.profile}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(checks)} target-encoding groups across {fixtures} fixtures passed')


if __name__ == '__main__':
    main()
