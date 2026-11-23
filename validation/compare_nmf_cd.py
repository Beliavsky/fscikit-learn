"""Dense coordinate-descent NMF against sklearn's public API and Cython sweeps."""
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
os.environ['LOKY_MAX_CPU_COUNT'] = '2'
import numpy as np
import sklearn
from sklearn.decomposition import NMF
from sklearn.decomposition import _nmf

ROOT = Path(__file__).resolve().parents[1]
METHODS = [None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom']
REJECTIONS = {
    'solver': 'unknown NMF solver', 'initializer': 'unknown NMF initializer',
    'components': 'invalid NMF controls', 'iterations': 'invalid NMF controls',
    'seed': 'invalid NMF controls', 'tol_negative': 'NMF tolerance/regularization',
    'tol_nan': 'NMF controls must be finite', 'alpha_w_negative': 'NMF tolerance/regularization',
    'alpha_h_negative': 'NMF tolerance/regularization', 'alpha_inf': 'NMF controls must be finite',
    'l1_range': 'NMF l1_ratio', 'l1_nan': 'NMF controls must be finite',
    'data_negative': 'NMF requires nonnegative', 'data_nan': 'X contains NaN or infinity', 'data_empty': 'empty',
    'missing_w': 'NMF custom initialization requires', 'missing_h': 'NMF custom initialization requires',
    'w_shape': 'NMF initializer shape mismatch', 'h_shape': 'NMF initializer shape mismatch',
    'w_nan': 'NMF initializer must be finite', 'h_inf': 'NMF initializer must be finite',
    'w_negative': 'NMF initializer must be nonnegative', 'h_negative': 'NMF initializer must be nonnegative',
    'w_zero': 'NMF initializer must not be all zero', 'h_zero': 'NMF initializer must not be all zero',
    'nn_rank': 'NMF NNDSVD requires', 'unfitted': 'NMF is not fitted',
    'query_width': 'NMF query feature count mismatch', 'query_negative': 'NMF requires nonnegative',
    'zero_h_transform': 'NMF transform components must not be all zero',
    'data_range': 'NMF data exceeds working range', 'factor_range': 'NMF factors exceed working range',
    'regularization_range': 'NMF regularization exceeds working range',
    'cd_beta': 'NMF cd solver requires Frobenius beta_loss=2',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fields(output):
    tokens = iter(output.split())
    result = []
    while True:
        try:
            n = int(next(tokens))
        except StopIteration:
            return result
        p = int(next(tokens))
        result.append(np.array([float(next(tokens)) for _ in range(n*p)]).reshape((n, p), order='F'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--report', type=Path, default=ROOT/'validation/results/nmf_cd.json')
    parser.add_argument('--profile', choices=['development', 'checked', 'release'], default='development')
    args = parser.parse_args()
    executable = args.executable.resolve()
    checks = []
    roundoff_stopping = []
    fixtures = []
    for seed, shape, k, method, shuffle, reg, stopping in itertools.product(
            [0, 42, 2026], [(7, 4), (4, 7), (10, 5)], [1, 2], range(6), [False, True],
            [(0., -1., 0.), (.03, -1., .2), (.02, .05, .8)], [(1, 0.), (7, 0.), (200, 1.e-5)]):
        fixtures.append((seed, shape, k, method, shuffle, reg, stopping, 'ordinary'))
    for seed, method, special in itertools.product([0, 42], [0, 1, 5],
                                                 ['auto', 'overcomplete', 'zero', 'zero_column', 'zero_factor']):
        fixtures.append((seed, (6, 4), 2, method, True, (.01, .02, .5), (30, 1.e-6), special))
    for seed, method in itertools.product([0, 42], [0, 1]):
        fixtures.append((seed, (4, 7), 2, method, True, (0., -1., 0.), (20, 1.e-6), 'auto'))
    for seed, shuffle in itertools.product([0, 42], [False, True]):
        fixtures.append((seed, (6, 4), 2, 5, shuffle, (0., -1., 0.), (20, 1.e-6), 'zero_factor'))
    for seed, shuffle, reg, method in itertools.product(
            [0, 42], [False, True], [(.03, -1., 1.), (.02, .05, 0.),
                                    (.03, 0., .5), (0., .02, .5), (1.e4, -1., 1.)], [1, 5]):
        fixtures.append((seed, (8, 5), 2, method, shuffle, reg, (80, 1.e-6), 'regularization_edges'))
    for index, (seed, shape, k, method, shuffle, reg, stopping, special) in enumerate(fixtures):
        rng = np.random.RandomState(seed)
        n, p = shape
        x = .1 + abs(rng.normal(size=shape))
        q = .1 + abs(rng.normal(size=(3, p)))
        requested_k = k
        if special == 'auto':
            requested_k = 0
            if method != 5:
                k = p
        if special == 'overcomplete':
            requested_k = k = p+2
        if special == 'zero':
            x[:] = 0
            q[:] = 0
            method = 1  # source NNDSVD on zero matrices is undefined
        if special == 'zero_column':
            x[:, 0] = 0
        w0 = .1+rng.uniform(size=(n, k))
        h0 = .1+rng.uniform(size=(k, p))
        if special == 'zero_factor' and method == 5:
            w0[:, 0] = 0
            h0[0, :] = 0
        aw, ah, ratio = reg
        steps, tol = stopping
        source = NMF(n_components='auto' if requested_k == 0 else requested_k,
                     solver='cd', init=METHODS[method], random_state=seed, shuffle=shuffle,
                     alpha_W=aw, alpha_H='same' if ah == -1 else ah, l1_ratio=ratio,
                     max_iter=steps, tol=tol)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sw = source.fit_transform(x, W=w0.copy() if method == 5 else None,
                                      H=h0.copy() if method == 5 else None)
        sh = source.components_.copy()
        # Replay the actual upstream compiled coordinate kernel to certify both
        # the first projected-gradient violation and the final stopping test.
        if method == 5:
            rw, rh = w0.copy(), h0.copy()
        else:
            rw, rh = _nmf._initialize_nmf(x, k, init=METHODS[method], random_state=seed)
        ht = np.ascontiguousarray(rh.T)
        rw = np.ascontiguousarray(rw)
        random = np.random.RandomState(seed)
        l1w, l2w = p*aw*ratio, p*aw*(1-ratio)
        h_alpha = aw if ah == -1 else ah
        l1h, l2h = n*h_alpha*ratio, n*h_alpha*(1-ratio)
        for iteration in range(source.n_iter_):
            violation = _nmf._update_coordinate_descent(x, rw, ht, l1w, l2w, shuffle, random)
            violation += _nmf._update_coordinate_descent(x.T, ht, rw, l1h, l2h, shuffle, random)
            if iteration == 0:
                initial = violation
        np.testing.assert_allclose(rw, sw, atol=1.e-12, rtol=1.e-12)
        np.testing.assert_allclose(ht.T, sh, atol=1.e-12, rtol=1.e-12)
        converged = initial == 0 or violation/initial <= tol
        query_enabled = bool(np.max(sh) > 0)
        expected = [sw, sh, [[source.reconstruction_err_]],
                    np.array([source.n_iter_, k, p])[:, None],
                    np.array([violation, initial])[:, None], [[int(converged)]], sw@sh]
        if query_enabled:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sq = source.transform(q)
            expected += [sq, sq@sh]
        else:
            try:
                source.transform(q)
            except ValueError as error:
                assert 'zero' in str(error)
            else:
                raise AssertionError('upstream accepted all-zero transform components')
        header = [n, p, len(q), requested_k, k, steps, seed, int(shuffle), method,
                  int(query_enabled), aw, ah, ratio, tol]
        def payload(header):
            arrays = [x, q] + ([w0, h0] if method == 5 else [])
            return ' '.join(map(str, header))+'\n'+'\n'.join(
                ' '.join(format(v, '.17g') for v in a.ravel(order='F')) for a in arrays)+'\n'
        proc = subprocess.run([str(executable)], input=payload(header), text=True, capture_output=True)
        assert proc.returncode == 0, (index, header, proc.stderr)
        actual = fields(proc.stdout)
        assert len(actual) == len(expected), (index, proc.stdout)
        native_violation, native_initial = actual[4].ravel()
        native_converged = native_initial == 0 or native_violation/native_initial <= tol
        assert actual[5][0, 0] == int(native_converged), (index, 'native stopping identity')
        native_iterations = int(actual[3][0, 0])
        if native_iterations != source.n_iter_ or native_converged != converged:
            # BLAS/FMA reduction order can turn a residual at machine precision
            # into exact zero. Retain these cases and bound BOTH residuals;
            # An already stationary initializer also has an ill-conditioned
            # violation ratio. Nonstationary iteration mismatches still fail.
            def gradient_scale(a, b):
                return (np.sum(abs(x@b.T))+np.sum(abs(a)@(abs(b)@abs(b.T)))
                        + l1w*a.size+l2w*np.sum(abs(a))
                        + np.sum(abs(x.T@a))+np.sum(abs(b.T)@(abs(a.T)@abs(a)))
                        + l1h*b.size+l2h*np.sum(abs(b)))
            def stationary_violation(a, b):
                ga = a@(b@b.T)-x@b.T+l1w+l2w*a
                gb = (a.T@a)@b-a.T@x+l1h+l2h*b
                return float(np.sum(abs(np.where(a == 0, np.minimum(ga, 0), ga)))
                             + np.sum(abs(np.where(b == 0, np.minimum(gb, 0), gb))))
            scale = max(gradient_scale(sw, sh), gradient_scale(actual[0], actual[1]))
            bound = 64*np.finfo(float).eps*(k+2)*scale
            stationarity = max(stationary_violation(sw, sh), stationary_violation(actual[0], actual[1]))
            assert (tol == 0 or max(initial, native_initial) <= bound) and \
                max(violation, native_violation, stationarity) <= bound, (
                index, 'unexplained stopping difference', violation, native_violation, stationarity, bound)
            roundoff_stopping.append(dict(fixture=index, source_iterations=int(source.n_iter_),
                native_iterations=native_iterations, source_violation=float(violation),
                native_violation=float(native_violation), stationary_violation=stationarity,
                source_initial=float(initial), native_initial=float(native_initial), tolerance=tol,
                arithmetic_scale=float(scale), bound=float(bound)))
            expected[3][0, 0] = native_iterations
            expected[5] = [[int(native_converged)]]
        names = ['W', 'H', 'error', 'metadata', 'violation', 'converged', 'reconstruction', 'query_W', 'query_reconstruction']
        for name, value, reference in zip(names, actual, expected):
            try:
                np.testing.assert_allclose(value, reference, rtol=3.e-7, atol=3.e-8)
            except AssertionError as error:
                raise AssertionError((index, header, special, name, str(error))) from error
            checks.append({'fixture': index, 'field': name, 'max_abs_error': float(np.max(abs(value-reference)))})
        w, h = actual[:2]
        assert np.min(w) >= 0 and np.min(h) >= 0
        np.testing.assert_allclose(actual[6], w@h, atol=1.e-10, rtol=1.e-10)
        def objective(a, b):
            return (.5*np.sum((x-a@b)**2)+l1w*np.sum(a)+.5*l2w*np.sum(a*a)
                    + l1h*np.sum(b)+.5*l2h*np.sum(b*b))
        np.testing.assert_allclose(objective(w, h), objective(sw, sh), rtol=2.e-8, atol=2.e-8)
        checks.append({'fixture': index, 'field': 'objective_nonnegativity_reconstruction',
                       'max_abs_error': float(abs(objective(w, h)-objective(sw, sh)))})
        if not query_enabled:
            header[9] = 1
            fail = subprocess.run([str(executable)], input=payload(header), text=True, capture_output=True)
            assert fail.returncode != 0 and 'NMF transform components must not be all zero' in fail.stdout+fail.stderr, (
                index, header, special, fail.returncode, fail.stdout, fail.stderr)
            checks.append({'fixture': index, 'field': 'source_zero_component_transform_rejection', 'max_abs_error': 0.})
        if index % 200 == 0:
            print(f'{index}/{len(fixtures)} fixtures passed', flush=True)
    rejected = []
    for name, message in REJECTIONS.items():
        proc = subprocess.run([str(executable), name], text=True, capture_output=True)
        assert proc.returncode != 0 and message.lower() in (proc.stdout+proc.stderr).lower(), (
            name, proc.stdout, proc.stderr)
        assert 'SIGFPE' not in proc.stderr and 'Fortran runtime error' not in proc.stderr, (name, proc.stderr)
        rejected.append(name)
    report = {'sklearn_version': sklearn.__version__, 'profile': args.profile,
              'fixtures': len(fixtures), 'checks': checks,
              'rejections': rejected, 'roundoff_stopping_certificates': roundoff_stopping,
              'executable': str(executable), 'executable_sha256': digest(executable),
              'validator_sha256': digest(__file__), 'source_sha256': {
                  name: digest(Path(_nmf.__file__).with_name(name)) for name in ['_nmf.py', '_cdnmf_fast.pyx']}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'PASS: {len(fixtures)} fixtures, {len(checks)} reference groups, {len(rejected)} rejections')


if __name__ == '__main__':
    main()
