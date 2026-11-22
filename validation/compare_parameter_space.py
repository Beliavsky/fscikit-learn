"""ParameterGrid/ParameterSampler: ordering, mixed scalar values and RNG branches."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.model_selection import ParameterGrid, ParameterSampler
from compare_function_transformer import ROOT, resolve


class Uniform:
    def rvs(self, random_state):
        return random_state.uniform()


def fixtures():
    spaces = [dict(z=[1, 2, 3], a=[False, True], b=[.1, 2.]), {},
              [dict(beta=['x', 'long_name'], alpha=[1, 2]), {}, dict(alpha=[True, False])],
              dict(index=list(range(100))), dict(a=list(range(1000)), b=list(range(1000))),
              dict(z=[1, 2, 3], alpha=Uniform()),
              [dict(z=[1, 2], alpha=Uniform()), {}, dict(a=['cat', 'dog'])]]
    for seed in [0, 42, 2026, 2147483647]:
        for family, space in enumerate(spaces):
            iterations = [1, 2, 5, 10, 99, 100, 105] if family == 3 else [1, 3, 9]
            if family == 4:
                iterations = [1, 3]  # million-point grid remains lazy
            if family < 4:
                yield dict(seed=seed, family=family, sample=0, n_iter=1), space
            for n_iter in iterations:
                yield dict(seed=seed, family=family, sample=1, n_iter=n_iter), space


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--profile', choices=['checked','release'], default='checked')
    parser.add_argument('--executable', type=Path); args = parser.parse_args()
    exe = args.executable.resolve() if args.executable else resolve(args.profile, 'reference_parameter_space_driver')
    checks = []; count = 0; warnings_seen = {}
    for c, space in fixtures():
        grids = space if isinstance(space, list) else [space]
        payload = f"{c['sample']} {c['n_iter']} {c['seed']} {len(grids)}\n"
        for grid in grids:
            payload += str(len(grid))+'\n'
            for name, values in grid.items():
                payload += name+'\n'
                if isinstance(values, Uniform):
                    payload += '1 0\n'; continue
                payload += f'0 {len(values)}\n'
                for value in values:
                    kind = 3 if isinstance(value, bool) else 2 if isinstance(value, int) else 4 if isinstance(value, str) else 1
                    payload += f'{kind}\n'+(str(int(value)) if kind == 3 else str(value))+'\n'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            want = list(ParameterSampler(space, c['n_iter'], random_state=c['seed'])) if c['sample'] else list(ParameterGrid(space))
        for warning in caught:
            key = str(warning.message); warnings_seen[key] = warnings_seen.get(key, 0)+1
        run = subprocess.run([str(exe)], input=payload, text=True, capture_output=True)
        assert run.returncode == 0, (c, run.stdout, run.stderr)
        lines = iter(run.stdout.splitlines()); width = int(next(lines)); actual = []
        for _ in range(width):
            params = {}
            for _ in range(int(next(lines))):
                name = next(lines); kind = int(next(lines)); value = next(lines)
                params[name] = float(value) if kind == 1 else bool(int(value)) if kind == 3 else int(value) if kind == 2 else value
            actual.append(params)
        assert next(lines, None) is None
        assert len(actual) == len(want), c
        for index, (got, expected) in enumerate(zip(actual, want)):
            assert list(got) == sorted(expected), (c, index, got, expected)
            assert got.keys() == expected.keys(), c
            for name, value in expected.items():
                if isinstance(value, float):
                    np.testing.assert_allclose(got[name], value, atol=1e-15, rtol=1e-15, err_msg=str(c))
                else:
                    assert type(got[name]) is type(value) and got[name] == value, (c, got, expected)
            checks.append(dict(**c, candidate=index, passed=True))
        count += 1
    source = Path(sklearn.__file__).parent
    report = dict(profile=args.profile, fixtures=count, sklearn=sklearn.__version__, numpy=np.__version__, checks=checks,
                  warnings=warnings_seen, native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in
                                 ['model_selection/_search.py', 'utils/_random.pyx']})
    (ROOT/'validation/results').mkdir(exist_ok=True)
    (ROOT/'validation/results'/f'parameter_space_{args.profile}.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'{len(checks)} parameter-space groups / {count} fixtures passed')


if __name__ == '__main__':
    main()
