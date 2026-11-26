"""Compare DBSCAN labels, core rows and owned components with sklearn 1.6.1."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, subprocess, warnings
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from scipy.sparse import csr_matrix, issparse
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances


def fixtures():
    rng = np.random.RandomState(514)
    for n, p in [(1, 2), (9, 2), (17, 4), (31, 18)]:
        x = np.round(rng.normal(size=(n, p)), 1)
        x[rng.random_sample(x.shape) < .5] = 0
        if n > 1:
            x[-1] = x[0]
        metrics = [('euclidean', 2), ('manhattan', 2), ('chebyshev', 2),
                   ('minkowski', .5), ('minkowski', 1.5), ('minkowski', 3),
                   ('cosine', 2), ('haversine', 2), ('precomputed', 2)]
        for (metric, power), algorithm, storage, weighting in itertools.product(
                metrics, ['auto', 'brute', 'kd_tree', 'ball_tree'], range(4), range(4)):
            dense = pairwise_distances(x) if metric == 'precomputed' else x.copy()
            data = csr_matrix(dense)
            if storage == 2:
                for i in range(n):
                    a, b = data.indptr[i:i+2]
                    data.indices[a:b] = data.indices[a:b][::-1]
                    data.data[a:b] = data.data[a:b][::-1]
            if storage == 3:
                # Duplicate entries and explicit zero coordinates are retained.
                values, indices, indptr = [], [], [0]
                for i in range(n):
                    for j in range(data.indptr[i], data.indptr[i+1]):
                        values.extend([data.data[j] * .25, data.data[j] * .75])
                        indices.extend([data.indices[j]] * 2)
                    values.append(0.)
                    indices.append(i if metric == 'precomputed' else 0)
                    indptr.append(len(values))
                data = csr_matrix((values, indices, indptr), shape=dense.shape)
            weights = None
            if weighting == 1:
                weights = np.ones(n)
            elif weighting == 2:
                weights = np.resize([2., -1., .5, 0., 3.], n)
            elif weighting == 3:
                weights = np.resize([1e16, 1., -1e16, 1., 2., -2., 0.], n)
            yield dict(n=n, p=dense.shape[1], metric=metric, power=power,
                       algorithm=algorithm, storage=storage, weighting=weighting,
                       eps=1.5 if metric != 'cosine' else .5, min_samples=2,
                       leaf_size=3), dense, data if storage else dense, weights
    # Duplicate diagonal entries trigger SciPy's whole-matrix canonicalization.
    # A partial diagonal also exercises setdiag's COO insertion and row order.
    for pattern, minimum, weighting in itertools.product(range(3), [1, 2, 3], range(3)):
        if pattern == 0:
            data = csr_matrix(([.1]*6, [0, 0, 1, 0, 1, 1], [0, 3, 6, 6]), shape=(3, 3))
        elif pattern == 1:
            data = csr_matrix(([.1, .2, .1, .1], [0, 1, 0, 2], [0, 2, 3, 4]), shape=(3, 3))
        else:
            data = csr_matrix(([.2, .1, .3, .1, .2], [1, 0, 0, 1, 2], [0, 3, 5, 5]), shape=(3, 3))
        weights = [None, np.ones(3), np.array([2., -1., 3.])][weighting]
        yield dict(n=3, p=3, metric='precomputed', power=2, algorithm='auto',
                   storage=1, weighting=weighting, pattern=pattern, eps=.3,
                   min_samples=minimum, leaf_size=3), data.toarray(), data, weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--executable', required=True, type=Path)
    ap.add_argument('--report', required=True, type=Path)
    a = ap.parse_args()
    records, accepted, payloads = [], [], []
    for case, dense, x, weights in fixtures():
        c = case
        text = (f'{c["n"]} {c["p"]} {int(bool(c["storage"]))} {int(weights is not None)} '
                f'{c["metric"]} {c["algorithm"]} {c["eps"]} {c["power"]} '
                f'{c["min_samples"]} {c["leaf_size"]}\n')
        text += '\n'.join(' '.join(map(str, row)) for row in dense) + '\n'
        text += ' '.join(map(str, np.ones(c['n']) if weights is None else weights)) + '\n'
        if issparse(x):
            text += (str(x.nnz) + '\n' + ' '.join(map(str, x.indptr + 1)) + '\n' +
                     ' '.join(map(str, x.indices + 1)) + '\n' + ' '.join(map(str, x.data)) + '\n')
        record = dict(case=c, differences=[])
        records.append(record)
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            try:
                model = DBSCAN(eps=c['eps'], min_samples=c['min_samples'], metric=c['metric'],
                               p=c['power'], algorithm=c['algorithm'], leaf_size=c['leaf_size'])
                model.fit(x, sample_weight=weights)
            except (ValueError, TypeError) as error:
                record['source_error'] = dict(type=type(error).__name__, message=str(error))
                native = subprocess.run([str(a.executable)], input=text, capture_output=True, text=True)
                record['native_returncode'] = native.returncode
                record['native_output'] = native.stdout + native.stderr
                if not native.returncode or 'fscikit-learn:' not in record['native_output']:
                    record['differences'].append('Source rejection not matched')
                continue
        components = model.components_
        record['source'] = dict(labels=model.labels_.tolist(), core=(model.core_sample_indices_ + 1).tolist(),
                                components=(components.toarray() if issparse(components) else components).ravel().tolist())
        if issparse(components):
            record['source'].update(data=components.data.tolist(), indices=(components.indices + 1).tolist(),
                                    indptr=(components.indptr + 1).tolist())
        accepted.append(record)
        payloads.append(text)
    payload = ''.join(payloads)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.with_suffix('.in').write_text(payload)
    native = subprocess.run([str(a.executable)], input=payload, capture_output=True, text=True)
    a.report.with_suffix('.stdout').write_text(native.stdout)
    a.report.with_suffix('.stderr').write_text(native.stderr)
    lines = native.stdout.splitlines()
    for i, record in enumerate(accepted):
        chunk = lines[7*i:7*i+7]
        if len(chunk) != 7:
            record['differences'].append('missing native output')
            continue
        record['native'] = dict(metadata=chunk[0])
        for label, line in zip(['labels', 'core', 'components', 'data', 'indices', 'indptr'], chunk[1:]):
            values = np.fromstring(line, sep=' ')
            record['native'][label] = values.tolist()
            want = record['source'].get(label, [])
            try:
                if label in ['components', 'data']:
                    np.testing.assert_allclose(values, want, rtol=2e-12, atol=2e-12, equal_nan=True)
                else:
                    np.testing.assert_array_equal(values, want)
            except AssertionError:
                record['differences'].append(label)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = dict(configurations=len(records), source_successes=len(accepted),
                  source_rejections=len(records)-len(accepted), native_returncode=native.returncode,
                  failed_configurations=sum(bool(r['differences']) for r in records), records=records,
                  sklearn=sklearn.__version__, numpy=np.__version__,
                  input_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                  sha256={str(p): sha(p) for p in [Path(__file__), a.executable, Path('src/sklearn_dbscan.f90')]})
    a.report.write_text(json.dumps(result, indent=2) + '\n')
    print({k: v for k, v in result.items() if k not in ['records', 'sha256']})
    assert not result['failed_configurations'] and not native.returncode


if __name__ == '__main__':
    main()
