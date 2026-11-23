"""Check the dense free NMF API against sklearn 1.6.1's public function."""
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
import sklearn
from sklearn.decomposition import non_negative_factorization, _nmf
from compare_nmf_cd import fields
from compare_nmf_mu import independent_objective

METHODS = [None, 'random', 'custom', 'nndsvd', 'nndsvda', 'nndsvdar']
REJECTIONS = {
    'missing_h': 'NMF fixed factorization requires H',
    'h_width': 'NMF fixed H shape mismatch', 'h_rank': 'NMF fixed H shape mismatch',
    'empty_h': 'NMF component count must be positive',
    'negative_h': 'NMF transform components must be nonnegative',
    'nan_h': 'NMF transform components must be finite',
    'zero_h': 'NMF transform components must not be all zero',
    'long_init': 'unknown NMF initializer', 'long_solver': 'unknown NMF solver',
    'custom_missing_w': 'NMF custom initialization requires W and H',
    'invalid_fixed_init': 'unknown NMF initializer',
    'cd_beta': 'NMF cd solver requires Frobenius beta_loss=2',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--profile', choices=['development', 'checked', 'release'], default='development')
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    exe = args.executable.resolve()
    checks = []
    certificates = []
    fixtures = []
    for seed, shape, rank, method, solver, fixed, reg, steps in itertools.product(
            [0, 42], [(7, 4), (4, 7)], [1, 2], range(6), ['cd', 'mu'], [False, True],
            [(0., -1., 0.), (.03, .02, .5)], [1, 9]):
        fixtures.append((seed, shape, rank, method, solver, fixed, reg, steps, 2., 'ordinary'))
    for seed, beta, fixed, reg, steps in itertools.product(
            [0, 42], [-1., 0., .5, 1., 1.5, 2.5, 3.], [False, True],
            [(0., -1., 0.), (.02, -1., 1.)], [1, 17]):
        fixtures.append((seed, (6, 4), 2, 2, 'mu', fixed, reg, steps, beta, 'beta'))
    # Fixed solves ignore W even when its shape or values would be invalid.
    # Named NNDSVD initialization is also ignored when H is fixed and overcomplete.
    for solver, method, special in itertools.product(
            ['cd', 'mu'], range(6), ['ignored_w', 'overcomplete', 'zero_column']):
        fixtures.append((42, (6, 4), 5 if special == 'overcomplete' else 2,
                         method, solver, True, (.03, -1., .4), 11, 2., special))
    for index, (seed, shape, rank, method, solver, fixed, reg, steps, beta, special) in enumerate(fixtures):
        rng = np.random.RandomState(seed)
        n, p = shape
        x = .2 + rng.uniform(size=shape)*3
        w0 = .1 + rng.uniform(size=(n, rank))
        h0 = .1 + rng.uniform(size=(rank, p))
        if special == 'ignored_w':
            w0[:] = np.nan
        if special == 'zero_column':
            h0[:, -1] = 0
        # Alternate explicit and automatic rank requests for both branches.
        k = 0 if index % 2 == 0 and (fixed or method <= 2 or n >= p) else rank
        aw, ah, ratio = reg
        before_h = h0.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sw, sh, sit = non_negative_factorization(
                x, W=w0.copy(), H=h0.copy(), n_components='auto' if k == 0 else k,
                init=METHODS[method], update_H=not fixed, solver=solver, beta_loss=beta,
                tol=0., max_iter=steps, alpha_W=aw, alpha_H='same' if ah == -1 else ah, l1_ratio=ratio,
                random_state=seed, shuffle=True)
        header = [n, p, k, rank, steps, seed, 1, method, int(solver == 'mu'), int(fixed),
                  beta, 0., aw, ah, ratio]
        payload = ' '.join(map(str, header))+'\n'+'\n'.join(
            ' '.join(format(v, '.17g') for v in a.ravel(order='F')) for a in [x, w0, h0])+'\n'
        proc = subprocess.run([str(exe)], input=payload, text=True, capture_output=True)
        assert proc.returncode == 0, (index, header, proc.stdout, proc.stderr)
        actual = fields(proc.stdout)
        assert len(actual) == 3, (index, proc.stdout)
        nit = int(actual[2][0, 0])
        if nit != sit:
            # CD tol=0 may stop on an exact zero on one reduction path and
            # continue at machine precision on another. Certify both solutions.
            assert solver == 'cd' and 1 <= nit <= steps
            def projected_gradient(a, b):
                lw, lh = p*aw, n*(aw if ah == -1 else ah)
                ga = a@(b@b.T)-x@b.T+lw*ratio+lw*(1-ratio)*a
                terms = [np.where(a == 0, np.minimum(ga, 0), ga)]
                scale = np.sum(abs(a)@(abs(b)@abs(b.T)))+np.sum(abs(x@b.T))
                scale += lw*ratio*a.size+lw*(1-ratio)*np.sum(abs(a))
                if not fixed:
                    gb = (a.T@a)@b-a.T@x+lh*ratio+lh*(1-ratio)*b
                    terms.append(np.where(b == 0, np.minimum(gb, 0), gb))
                    scale += np.sum((abs(a.T)@abs(a))@abs(b))+np.sum(abs(a.T@x))
                    scale += lh*ratio*b.size+lh*(1-ratio)*np.sum(abs(b))
                return sum(float(np.sum(abs(t))) for t in terms), float(scale)
            source_residual, source_scale = projected_gradient(sw, sh)
            native_residual, native_scale = projected_gradient(*actual[:2])
            bound = 64*np.finfo(float).eps*(sh.shape[0]+2)*max(source_scale, native_scale)
            assert max(source_residual, native_residual) <= bound, (index, source_residual, native_residual, bound)
            certificates.append(dict(fixture=index, source_iterations=int(sit), native_iterations=nit,
                                     source_residual=source_residual, native_residual=native_residual, bound=bound))
        for name, value, reference in zip(['W', 'H', 'iterations'], actual, [sw, sh, [[sit]]]):
            if name == 'iterations' and nit != sit:
                checks.append(dict(fixture=index, field=name, max_abs_error=float(abs(nit-sit)), certified=True))
                continue
            try:
                np.testing.assert_allclose(value, reference, rtol=3.e-7, atol=3.e-8 if name != 'iterations' else 0.)
            except AssertionError as error:
                raise AssertionError((index, header, special, name, str(error))) from error
            checks.append(dict(fixture=index, field=name, max_abs_error=float(np.max(abs(value-reference)))))
        w, h = actual[:2]
        assert np.min(w) >= 0 and np.min(h) >= 0
        if fixed:
            np.testing.assert_array_equal(h, before_h)
        objective = independent_objective(x, w, h, beta, aw, ah, ratio)
        source_objective = independent_objective(x, sw, sh, beta, aw, ah, ratio)
        np.testing.assert_allclose(objective, source_objective, rtol=3.e-7, atol=3.e-8)
        checks.append(dict(fixture=index, field='objective_nonnegativity_fixed_H',
                           max_abs_error=float(abs(objective-source_objective))))
        if index % 200 == 0:
            print(f'{index}/{len(fixtures)} fixtures passed', flush=True)
    for name, message in REJECTIONS.items():
        proc = subprocess.run([str(exe), name], text=True, capture_output=True)
        assert proc.returncode != 0 and message in proc.stdout+proc.stderr, (name, proc.stdout, proc.stderr)
        assert 'SIGFPE' not in proc.stderr and 'Fortran runtime error' not in proc.stderr
    report = dict(profile=args.profile, sklearn_version=sklearn.__version__, fixtures=len(fixtures),
                  checks=checks, rejections=list(REJECTIONS), roundoff_stopping_certificates=certificates,
                  executable=str(exe),
                  executable_sha256=digest(exe), validator_sha256=digest(__file__),
                  source_sha256=digest(_nmf.__file__),
                  objective_validator_sha256=digest(Path(__file__).with_name('compare_nmf_mu.py')))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'PASS: {len(fixtures)} fixtures, {len(checks)} groups, {len(REJECTIONS)} rejections')


if __name__ == '__main__':
    main()
