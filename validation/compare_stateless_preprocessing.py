"""Normalizer/Binarizer and dense/CSR free functions against sklearn 1.6.1."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import warnings

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy
import sklearn
from sklearn.preprocessing import Normalizer, Binarizer, normalize, binarize
from sklearn.preprocessing import _data
from compare_nmf_cd import fields

NORMS = ['l1', 'l2', 'max']
REJECTIONS = {
    'norm': 'unknown normalization norm', 'axis': 'axis must be zero or one',
    'threshold_nan': 'threshold must be finite', 'threshold_inf': 'threshold must be finite',
    'sparse_negative': 'threshold below zero', 'sparse_l1_norms': 'implemented only for max',
    'sparse_l2_norms': 'implemented only for max', 'empty_rows': 'must be nonempty',
    'empty_features': 'must be nonempty', 'dense_nan': 'NaN or infinity', 'dense_inf': 'NaN or infinity',
    'sparse_nan': 'NaN or infinity', 'sparse_inf': 'NaN or infinity',
    'csr_pointer': 'CSR pointer endpoints', 'csr_index': 'CSR column index', 'csr_empty': 'must be nonempty',
    'norm_fit': 'unknown normalization norm', 'binary_fit': 'threshold must be finite',
    'norm_query_width': 'feature count mismatch', 'binary_query_width': 'feature count mismatch',
    'sparse_query_width': 'feature count mismatch', 'fit_name_count': 'fit name count mismatch',
    'fit_name_unallocated': 'feature name is unallocated', 'unfitted_names': 'feature names require fit',
    'unfitted_supplied_names': 'feature names require fit',
    'get_name_count': 'feature count mismatch', 'get_name_text': 'input feature names differ',
    'get_name_length': 'input feature names differ', 'get_name_unallocated': 'feature name is unallocated',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sparse_variant(x, storage, rng):
    if storage == 1:
        return csr_matrix(x)
    indptr, indices, data = [0], [], []
    for row in x:
        items = []
        for j, v in enumerate(row):
            items.extend([(j, v/2), (j, v/2)])
        items.append((0, 0.))
        if storage == 3:
            rng.shuffle(items)
        indices.extend(j for j, _ in items)
        data.extend(v for _, v in items)
        indptr.append(len(data))
    return csr_matrix((data, indices, indptr), shape=x.shape)


def payload_matrix(x):
    if isinstance(x, csr_matrix):
        return (str(x.nnz)+'\n'+' '.join(map(str, x.indptr+1))+'\n'+
                (' '.join(map(str, x.indices+1))+'\n'+' '.join(format(v, '.17g') for v in x.data)+'\n' if x.nnz else ''))
    return ' '.join(format(v, '.17g') for v in x.ravel(order='F'))+'\n'


def names_matrix(names):
    result = np.full((max(map(len, names), default=0), len(names)), -1.)
    for j, name in enumerate(names):
        result[:len(name), j] = list(map(ord, name))
    return result


def append_result(expected, prefix, x, sparse):
    if not sparse:
        expected.append((prefix, np.asarray(x)))
    else:
        x = x.tocsr()
        expected.extend([(prefix, x.toarray()), (prefix+'_shape', np.array(x.shape)[:, None]),
                         (prefix+'_indptr', x.indptr[:, None]+1), (prefix+'_indices', x.indices[:, None]+1),
                         (prefix+'_data', x.data[:, None])])


def assert_close(native, source, fixture, field):
    assert native.shape == source.shape, (fixture, field, native.shape, source.shape)
    np.testing.assert_array_equal(np.isnan(native), np.isnan(source), err_msg=f'{fixture} {field} NaNs')
    np.testing.assert_array_equal(np.isposinf(native), np.isposinf(source), err_msg=f'{fixture} {field} +Inf')
    np.testing.assert_array_equal(np.isneginf(native), np.isneginf(source), err_msg=f'{fixture} {field} -Inf')
    finite = np.isfinite(source)
    np.testing.assert_allclose(native[finite], source[finite], atol=3.e-14, rtol=3.e-13,
                               err_msg=f'{fixture} {field}')
    return float(np.max(abs(native[finite]-source[finite]), initial=0.))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--profile', choices=['development', 'checked', 'release'], default='development')
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    exe = args.executable.resolve()
    fixtures = []
    for seed, shape, norm, kind, axis, storage, regime in itertools.product(
            [0, 42], [(7, 4), (4, 7), (9, 17)], range(3), [0, 2], [0, 1], range(4),
            ['ordinary', 'zeros', 'tiny', 'small', 'subnormal', 'large', 'huge']):
        if kind == 2 and axis == 0:
            continue
        fixtures.append((seed, shape, norm, kind, axis, storage, regime, 0.))
    for seed, shape, threshold, kind, storage, regime in itertools.product(
            [0, 42], [(7, 4), (4, 7)], [-1., 0., .5, 100.], [1, 3], range(4), ['ordinary', 'zeros']):
        if storage and threshold < 0:
            continue
        fixtures.append((seed, shape, 1, kind, 1, storage, regime, threshold))
    for norm, kind, storage, boundary in itertools.product(range(3), [0, 2], range(4), [-1, 0, 1]):
        fixtures.append((boundary, (3, 1), norm, kind, 1, storage, 'scale_boundary', 0.))
    for kind, axis, regime in itertools.product([0, 2], [0, 1],
            ['overflow_duplicates', 'cancelled_duplicates', 'ill_conditioned_duplicates']):
        if kind == 2 and axis == 0:
            continue
        fixtures.append((0, (1, 3), 2, kind, axis, 3, regime, 0.))
    checks, inventories = [], []
    nonfinite = []
    for index, (seed, shape, norm, kind, axis, storage, regime, threshold) in enumerate(fixtures):
        rng = np.random.RandomState(max(seed, 0))
        n, p = shape
        x = rng.normal(size=shape)
        q = rng.normal(size=(3, p))
        if regime == 'zeros':
            x[0] = 0;x[:, -1] = 0;q[0] = 0;q[:, -1] = 0
        elif regime in ['tiny', 'small', 'subnormal', 'large', 'huge']:
            factor = {'tiny': 1.e-17, 'small': 1.e-160, 'subnormal': 1.e-320, 'large': 1.e200, 'huge': 1.e308}[regime]
            x = np.clip(x, -.5, .5)*factor
            q = np.clip(q, -.5, .5)*factor
        elif regime == 'scale_boundary':
            value = 10*np.finfo(float).eps
            if seed:
                value = np.nextafter(value, 0. if seed < 0 else np.inf)
            x[:] = value;q[:] = -value
        if regime.endswith('_duplicates'):
            columns, values = {
                'overflow_duplicates': ([0, 0, 0, 1], [1.e308, 1.e308, -1.e308, 2.]),
                'cancelled_duplicates': ([0, 0, 0, 1], [3., -3., 0., 2.]),
                'ill_conditioned_duplicates': ([0, 1, 0, 1, 0, 1]*8,
                                                [1.e16, 2., 1., 2., -1.e16, 2.]*8),
            }[regime]
            x = csr_matrix((values, columns, [0, len(values)]), shape=(1, 3));q = x.copy()
        elif storage:
            x = sparse_variant(x, storage, rng)
            q = sparse_variant(q, storage, rng)
        named = bool(index % 2) and regime not in ['large', 'huge'] and not regime.endswith('_duplicates')
        return_norm = int(kind == 0 and (not storage or norm == 2))
        names = [f'axis {j}' for j in range(p)]
        expected = []
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            if kind < 2:
                if kind == 0:
                    result = normalize(x, norm=NORMS[norm], axis=axis, return_norm=bool(return_norm))
                    if return_norm:
                        result, scales = result
                else:
                    result = binarize(x, threshold=threshold)
                append_result(expected, 'output', result, storage)
                if return_norm:
                    expected.append(('norms', scales[:, None]))
            else:
                model = Normalizer(norm=NORMS[norm]) if kind == 2 else Binarizer(threshold=threshold)
                append_result(expected, 'train_before_fit', model.transform(x), storage)
                append_result(expected, 'query_before_fit', model.transform(q), storage)
                if named:
                    # Fitting only records width/names; transform retains raw CSR
                    # duplicate storage even when names are learned from a frame.
                    model.fit(pd.DataFrame(x.toarray() if storage else x, columns=names))
                    result = model.transform(x)
                else:
                    result = model.fit_transform(x)
                append_result(expected, 'fit_transform', result, storage)
                append_result(expected, 'query', model.transform(q), storage)
                expected.append(('names_fitted', names_matrix(model.get_feature_names_out())))
                if named:
                    expected.append(('names_supplied', names_matrix(model.get_feature_names_out(names))))
                model.fit(x)
                expected.append(('names_refit', names_matrix(model.get_feature_names_out())))
                expected.append(('metadata', np.array([0, model.n_features_in_, 0])[:, None]))
        header = [kind, int(storage != 0), norm, axis, return_norm, n, p, len(q.toarray() if storage else q), int(named), threshold]
        proc = subprocess.run([str(exe)], input=' '.join(map(str, header))+'\n'+payload_matrix(x)+payload_matrix(q),
                              text=True, capture_output=True)
        assert proc.returncode == 0, (index, header, regime, proc.stdout, proc.stderr)
        actual = fields(proc.stdout)
        assert len(actual) == len(expected), (index, len(actual), len(expected), proc.stdout)
        fields_seen = []
        for value, (field, reference) in zip(actual, expected):
            reference = np.asarray(reference)
            error = assert_close(value, reference, index, field)
            if field.endswith(('_shape', '_indptr', '_indices')) or field.startswith('names_') or field == 'metadata':
                np.testing.assert_array_equal(value, reference)
            checks.append(dict(fixture=index, field=field, max_abs_error=error))
            fields_seen.append(field)
            if np.any(~np.isfinite(reference)):
                nonfinite.append(dict(fixture=index, field=field, nan=int(np.isnan(reference).sum()),
                                      positive_inf=int(np.isposinf(reference).sum()), negative_inf=int(np.isneginf(reference).sum())))
        checks.append(dict(fixture=index, field='input_preservation', max_abs_error=0.))
        fields_seen.append('input_preservation')
        inventories.append(dict(fixture=index, kind=kind, storage=storage, norm=NORMS[norm], axis=axis,
                                regime=regime, named=named, fields=fields_seen))
        if index % 200 == 0:
            print(f'{index}/{len(fixtures)} fixtures passed', flush=True)
    for name, message in REJECTIONS.items():
        proc = subprocess.run([str(exe), name], text=True, capture_output=True)
        assert proc.returncode != 0 and message in proc.stdout+proc.stderr, (name, proc.stdout, proc.stderr)
        assert 'SIGFPE' not in proc.stderr and 'Fortran runtime error' not in proc.stderr, (name, proc.stderr)
    report = dict(profile=args.profile, sklearn_version=sklearn.__version__, fixtures=len(fixtures), checks=checks,
                  inventories=inventories, nonfinite_reference_results=nonfinite, rejections=list(REJECTIONS),
                  executable=str(exe), executable_sha256=digest(exe), validator_sha256=digest(__file__),
                  scipy_version=scipy.__version__, numpy_version=np.__version__,
                  source_sha256={name: digest(Path(sklearn.__file__).parent/name) for name in
                    ['preprocessing/_data.py', 'utils/sparsefuncs.py', 'utils/sparsefuncs_fast.pyx', 'utils/extmath.py']},
                  scipy_source_sha256={name: digest(Path(scipy.__file__).parent/name) for name in
                    ['sparse/_compressed.py', 'sparse/_data.py']})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'PASS: {len(fixtures)} fixtures, {len(checks)} groups, {len(REJECTIONS)} rejections')


if __name__ == '__main__':
    main()
