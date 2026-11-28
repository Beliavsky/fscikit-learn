"""Direct sparse LIBSVM comparisons; retain every numerical or structural gap."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import subprocess
import warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.svm import SVR, NuSVR, OneClassSVM
import sklearn.svm._libsvm_sparse as backend


def matrix(values, layout):
    data, indices, ptr = [], [], [0]
    for row in values:
        for j in (range(len(row)) if layout == 0 else reversed(range(len(row)))):
            if layout == 3:
                data.append(row[j]); indices.append(j)
            elif row[j] != 0:
                if layout == 2:
                    data.extend([.25 * row[j], .75 * row[j], 0.]); indices.extend([j, j, j])
                else:
                    data.append(row[j]); indices.append(j)
        ptr.append(len(data))
    return sp.csr_matrix((data, indices, ptr), shape=values.shape)


def arrays(c):
    rng = np.random.RandomState(c['seed'])
    x = rng.normal(size=(18, c['p'])); x[rng.uniform(size=x.shape) < .45] = 0
    q = rng.normal(size=(9, c['p'])); q[rng.uniform(size=q.shape) < .4] = 0
    if c['family'] == 3:
        x = np.abs(x) + .7; q = np.abs(q) + .9
    y = np.sin(x[:, 0]) + .3 * x[:, 1] - .1 * x[:, 2]**2
    w = rng.uniform(.3, 2., len(x)); w[::7] = 0
    return matrix(x, c['layout']), matrix(q, c['layout']), y, w


def oracle(c, x, q, y, w):
    options = dict(kernel=['rbf', 'linear', 'poly'][c['kernel'] - 1], degree=2,
                   coef0=.4, tol=1e-8, max_iter=200000, shrinking=False,
                   gamma=('scale' if c['rule'] == 0 else 'auto') if c['gamma'] == -1 else c['gamma'])
    if c['family'] != 3:
        options['C'] = c['c']
    options['epsilon' if c['family'] == 1 else 'nu'] = c['epsilon'] if c['family'] == 1 else c['nu']
    model = [SVR, NuSVR, OneClassSVM][c['family'] - 1](**options)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        model.fit(x.copy(), y if c['family'] != 3 else None,
                  sample_weight=w if c['weighted'] else None)
    predict = model.decision_function if c['family'] == 3 else model.predict
    support = model.support_vectors_
    result = dict(query=predict(q.copy()), dense_query=predict(q.toarray()),
                  parameters=[model.intercept_[0], model._gamma], support=model.support_ + 1,
                  dual=model.dual_coef_.toarray().ravel(), sv_indptr=support.indptr + 1,
                  sv_indices=support.indices + 1, sv_data=support.data, storage=[0, 0],
                  converged=[int(model.fit_status_ == 0)])
    if c['family'] == 3:
        result.update(raw=model.score_samples(q.copy()), labels=model.predict(q.copy()))
    else:
        result['score'] = [model.score(x.copy(), y)]
    if c['kernel'] == 2:
        result['coef'] = model.coef_.toarray().ravel()
    return result, [str(w.message) for w in recorded]


def csr_payload(a):
    return f'{a.shape[0]} {a.shape[1]} {a.nnz}\n' + '\n'.join([
        ' '.join(map(str, a.indptr + 1)), ' '.join(map(str, a.indices + 1)),
        ' '.join(format(v, '.17g') for v in a.data)]) + '\n'


def payload(c, a):
    x, q, y, w = a
    controls = [c[k] for k in ['family', 'kernel', 'rule', 'weighted', 'gamma', 'c', 'nu', 'epsilon']]
    return ' '.join(map(str, controls)) + '\n' + csr_payload(x) + csr_payload(q) + \
        ' '.join(format(v, '.17g') for v in y) + '\n' + ' '.join(format(v, '.17g') for v in w) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    cases = [dict(family=f, kernel=k, rule=r, weighted=w, layout=l, seed=s,p=p,
                  gamma=-1., c=1.3, nu=.4, epsilon=.1)
             for f, k, r, w, l, s, p in itertools.product(range(1, 4), range(1, 4), range(2), range(2), range(4), [7, 42, 2026],[7,17,43])]
    inputs = [arrays(c) for c in cases]
    expected = [oracle(c, *a) for c, a in zip(cases, inputs)]
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), args.executable, root / 'src/sklearn_kernel_svm.f90',
             root / 'app/reference_kernel_svm_sparse_driver.f90', Path(backend.__file__),
             Path(sklearn.__file__).parent / 'svm/_base.py',
             Path(sklearn.__file__).parent / 'svm/src/libsvm/svm.cpp']
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    hashes = {str(p): sha(p) for p in files}
    run = subprocess.run([str(args.executable)], input=str(len(cases)) + '\n' +
                         ''.join(payload(c, a) for c, a in zip(cases, inputs)), capture_output=True, text=True)
    assert run.returncode == 0, (run.stdout[-1500:], run.stderr[-1500:])
    actual, index, fields = [], -1, set()
    lines = iter(run.stdout.splitlines())
    checks = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'BEGIN':
            index += 1; fields = set(); actual.append({}); assert int(parts[1]) == index + 1
        elif parts[0] == 'END':
            assert fields == expected[index][0].keys()
        else:
            field, size = parts[0], int(parts[1]); assert field not in fields; fields.add(field)
            value = np.fromstring(next(lines), sep=' '); assert len(value) == size
            want = np.asarray(expected[index][0][field]).ravel()
            actual[index][field] = value.tolist()
            exact = field in ['support', 'sv_indptr', 'sv_indices', 'sv_data', 'storage', 'converged', 'labels']
            tolerance = 2e-4 if field in ['dual', 'coef'] else 3e-5
            passed = value.shape == want.shape and (np.array_equal(value, want) if exact else
                np.allclose(value, want, atol=tolerance, rtol=tolerance, equal_nan=True))
            if field == 'parameters' and value.shape == want.shape:
                passed = passed and np.isclose(value[1], want[1], atol=2e-14, rtol=2e-13)
            checks.append(dict(case=index, field=field, passed=bool(passed),
                               oracle='native_storage_contract' if field == 'storage' else 'source',
                               max_abs_error=float(np.max(np.abs(value - want), initial=0)) if value.shape == want.shape else None))
    assert index + 1 == len(cases)
    assert hashes == {str(p): sha(p) for p in files}
    gaps = [c for c in checks if not c['passed']]
    result = dict(cases=cases, checks=checks, gaps=gaps, actual=actual,
                  expected=[{k: np.asarray(v).tolist() for k, v in e.items()} for e, _ in expected],
                  source_warnings=[w for _, w in expected], hashes=hashes,
                  versions=dict(sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__),
                  strict_parity=not gaps)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(len(cases), 'fixtures;', len(checks), 'groups;', len(gaps), 'gaps', flush=True)
    if gaps:
        print('First gaps:', gaps[:12])
    raise SystemExit(bool(gaps) and not args.diagnostic)


if __name__ == '__main__':
    main()
