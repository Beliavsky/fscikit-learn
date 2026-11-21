"""Numeric OrdinalEncoder forward, grouping and inverse contracts versus sklearn."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils._encode import _get_counts

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile):
    options = ['--profile', profile] if profile == 'release' else ['--flag', FLAGS]
    proc = subprocess.run(['fpm', 'run', *options, '--target', 'reference_ordinal_encoding_driver', '--runner'],
                          cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    paths = [ROOT / line.strip().strip('"') for line in proc.stdout.splitlines()
             if line.strip().strip('"').endswith(('reference_ordinal_encoding_driver', 'reference_ordinal_encoding_driver.exe'))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, proc.stdout
    return paths[0]


def inverse_values(values):
    numeric = np.full(values.shape, np.nan)
    unknown = np.zeros(values.shape, dtype=int)
    infrequent = np.zeros(values.shape, dtype=int)
    for index, value in np.ndenumerate(values):
        if value is None:
            unknown[index] = 1
        elif isinstance(value, str):
            assert value == 'infrequent_sklearn'
            infrequent[index] = 1
        else:
            numeric[index] = value
    return dict(inverse=numeric, unknown=unknown, infrequent=infrequent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--profile', choices=['checked', 'release'], default='checked')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    executable = args.executable.resolve() if args.executable else resolve(args.profile)
    checks, source_warnings, source_errors = [], [], []
    fixtures = 0

    def run(seed, case, x, query, *, categories='auto', unknown=-1, missing=np.nan,
            min_frequency=None, max_categories=None, inverse_codes=None):
        nonlocal fixtures
        model = OrdinalEncoder(categories=categories,
                               handle_unknown='error' if unknown is None else 'use_encoded_value',
                               unknown_value=unknown, encoded_missing_value=missing,
                               min_frequency=min_frequency, max_categories=max_categories)
        x, query = np.asarray(x, dtype=float), np.asarray(query, dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            train = model.fit_transform(x)
            q = model.transform(query)
            rare = (model.infrequent_categories_ if model._infrequent_enabled else [None] * x.shape[1])
            expected = dict(train=train, query=q, categories=np.concatenate(model.categories_),
                            counts=[len(c) for c in model.categories_], rare=np.concatenate([
                                [] if c is None else c for c in rare]),
                            rare_counts=[0 if c is None else len(c) for c in rare], feature_count=[x.shape[1]],
                            category_counts=np.concatenate([_get_counts(x[:, j], model.categories_[j])
                                                            for j in range(x.shape[1])]))
            expected['snapshot'] = model.set_params(max_categories=1, min_frequency=999).transform(query)
            codes = q.copy() if inverse_codes is None else np.asarray(inverse_codes, dtype=float)
            try:
                decoded = model.inverse_transform(codes.copy())
            except (IndexError, ValueError) as error:
                inverse_mode = -1
                source_errors.append(dict(seed=seed, case=case, category=type(error).__name__, message=str(error)))
            else:
                inverse_mode = 1
                expected.update(inverse_values(decoded), preserved_codes=codes)
                model.set_params(categories='auto', handle_unknown='error', unknown_value=None,
                                 encoded_missing_value=np.nan, min_frequency=None, max_categories=None)
                expected['refit'] = model.fit_transform([[7.], [7.], [np.nan]])
        source_warnings.extend(dict(seed=seed, case=case, category=w.category.__name__, message=str(w.message))
                               for w in caught)
        freq_count = min_frequency if isinstance(min_frequency, int) else 0
        freq_fraction = min_frequency if isinstance(min_frequency, float) else 0
        lines = [f'{len(x)} {x.shape[1]} {len(query)} {int(not isinstance(categories, str))} {inverse_mode} '
                 f'{0 if inverse_codes is None else len(codes)}',
                 f'{"error" if unknown is None else "use_encoded_value"} {int(unknown is not None)} '
                 f'{0 if unknown is None else unknown} '
                 f'{missing} {freq_count} {freq_fraction} {max_categories or 0}']
        lines += [' '.join(map(str, row)) for row in x]
        lines += [' '.join(map(str, row)) for row in query]
        if not isinstance(categories, str):
            for values in categories:
                lines += [str(len(values)), ' '.join(map(str, values))]
        if inverse_codes is not None:
            lines += [' '.join(map(str, row)) for row in codes]
        inp = RESULTS / f'ordinal_encoding_{args.profile}_input.txt'
        out = RESULTS / f'ordinal_encoding_{args.profile}_output.txt'
        inp.write_text('\n'.join(lines) + '\n')
        proc = subprocess.run([str(executable), str(inp), str(out)], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if inverse_mode < 0:
            assert proc.returncode != 0, (seed, case, 'source inverse error not reproduced')
            assert ('OrdinalEncoder inverse index is out of bounds' in proc.stdout or
                    'OrdinalEncoder inverse code is not a finite index' in proc.stdout), (seed, case, proc.stdout)
            assert 'Index ' not in proc.stdout and 'Segmentation fault' not in proc.stdout, proc.stdout
            checks.append(dict(seed=seed, case=case, name='source_inverse_error', passed=True, oracle='sklearn'))
        else:
            assert proc.returncode == 0, (seed, case, proc.stdout)
        actual = {}
        for line in out.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        assert actual.keys() == expected.keys(), (seed, case, actual.keys(), expected.keys())
        for name, want in expected.items():
            want = np.asarray(want).ravel(order='F')
            np.testing.assert_array_equal(actual[name], want, err_msg=f'{seed} {case} {name}')
            checks.append(dict(seed=seed, case=case, name=name, passed=True,
                               oracle='input_identity' if name == 'preserved_codes' else 'sklearn'))
        fixtures += 1

    for seed in [0, 7, 42, 2026, 314159]:
        rng = np.random.RandomState(seed)
        case = 0
        for nan in [False, True]:
            x = np.column_stack([rng.randint(-3, 5, 43), rng.randint(0, 4, 43), np.ones(43)]).astype(float)
            if nan:
                x[::5, 0] = np.nan
                x[:, 2] = np.nan
            query = np.vstack([x[:11], [99., 99., 99.]])
            for freq in [None, 1, 3, 20, .1, .8]:
                for maximum in [None, 1, 2, 4, 8, 19]:
                    for unknown, missing in [(-1, np.nan), (np.nan, -2), (-9, -8)]:
                        case += 1
                        run(seed, case, x, query, unknown=unknown, missing=missing,
                            min_frequency=freq, max_categories=maximum)
        for maximum in [None, 1, 2, 3, 4]:
            for unknown in [None, -1, np.nan]:
                case += 1
                x = np.repeat([1., 2., 3., 4.], [3, 3, 3, 3])[:, None]
                run(seed, case, rng.permutation(x), x, unknown=unknown, max_categories=maximum)
        for freq in [None, 1, .2]:
            for maximum in [None, 1, 3]:
                case += 1
                x = np.array([[0., -1], [1., 0], [2., 1], [3., 2], [np.nan, 99.]])
                run(seed, case, x, [[np.nan, -2.], [7., 0.], [1., 3.]], min_frequency=freq,
                    max_categories=maximum, categories=[np.array([-1., 0., 1., 9., np.nan]), np.array([-2., 0., 3.])])
        # Sentinel collisions admitted by the source grouped-cardinality check.
        for missing in [np.nan, 0, 1, 9]:
            for unknown in [0, 1, 9, np.nan]:
                case += 1
                run(seed, case, [[0.], [1.], [np.nan]], [[0.], [1.], [np.nan], [99.]],
                    unknown=unknown, missing=missing, max_categories=1)
        # Source inverse permits negative indices and truncates fractional codes.
        for unknown in [None, -9, np.nan]:
            case += 1
            run(seed, case, [[1.], [2.], [3.]], [[1.], [2.]], unknown=unknown,
                inverse_codes=[[-3.9], [-2.8], [-.8], [.8], [1.9], [2.9]])
        for bad_code in [3., -4., np.nan]:
            case += 1
            run(seed, case, [[1.], [2.], [3.]], [[1.]], unknown=None, inverse_codes=[[bad_code]])
        case += 1
        run(seed, case, [[np.nan]], [[np.nan], [99.]], unknown=np.nan, missing=np.nan, max_categories=1)
        print(f'seed {seed}: ordinal encodings passed', flush=True)
    path = Path(inspect.getfile(OrdinalEncoder))
    paths = [path, path.parent.parent / 'utils/_encode.py']
    report = dict(sklearn=sklearn.__version__, profile=args.profile, fixtures=fixtures, checks=checks,
                  source_warnings=source_warnings, source_errors=source_errors,
                  source_sha256={str(p.relative_to(path.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in paths})
    (RESULTS / f'ordinal_encoding_{args.profile}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(checks)} ordinal-encoding groups across {fixtures} fixtures passed; {len(source_errors)} source inverse errors')


if __name__ == '__main__':
    main()
