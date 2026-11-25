"""Direct/Gram orthogonal matching pursuit against pinned sklearn column solves."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import os
import subprocess
import warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import sklearn
import scipy
from sklearn.linear_model import orthogonal_mp, orthogonal_mp_gram

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'validation/results'


def fixtures():
    cases = []
    for seed, n, p in [(7, 47, 8), (42, 31, 12), (2026, 11, 17), (91, 30, 1)]:
        rng = np.random.default_rng(seed)
        original = rng.normal(size=(n, p)); original /= np.linalg.norm(original, axis=0)
        for variant in ['regular', 'duplicate', 'zero_column', 'scaled', 'zero_target', 'sparse', 'ragged']:
            x = original.copy(); y = rng.normal(size=(n, 3))
            if variant == 'duplicate' and p > 1:
                x[:, -1] = x[:, 0]
            if variant == 'zero_column':
                x[:, -1] = 0
            if variant == 'scaled':
                x *= np.linspace(.25, 4., p)
            if variant == 'zero_target':
                y[:] = 0
            if variant in ['sparse', 'ragged']:
                coef = np.zeros((p, 3)); coef[0] = [2., 3., 1.]
                if p > 2:
                    coef[2, [0, 2]] = [-1., 4.]
                y = x @ coef
                if variant == 'ragged':
                    y[:, 0] = 0
            for vector, targets in [(1, y[:, :1]), (0, y[:, :1]), (0, y)]:
                for kind, precompute in [(0, 0), (0, 1), (0, 2), (1, 0)]:
                    for count, tol in [(1, None), (min(3, p), None), (p, None), (None, 0.),
                                       (None, .1), (p+10, 1e6), (None, None)]:
                        if kind == 1 and count is None and tol is None and p < 10:
                            continue  # Source Gram default is floor(.1*p), including its zero-count failure.
                        gram = kind == 1
                        cases.append(dict(seed=seed, variant=variant, x=x.T@x if gram else x,
                                          y=x.T@targets if gram else targets, kind=kind, vector=vector,
                                          count=count, tol=tol, norms=(targets**2).sum(axis=0) if gram and tol is not None else None,
                                          precompute=precompute, keep=(count != 1), design=x, response=targets))
    return cases


def oracle(case):
    x, y, kind = case['x'], case['y'], case['kind']
    gram = kind == 1 or case['precompute'] == 1 or (case['precompute'] == 2 and len(x) > x.shape[1])
    g = x if kind == 1 else x.T@x
    xy = y if kind == 1 else x.T@y
    count = case['count']
    if kind == 0 and count is None and case['tol'] is None:
        count = max(x.shape[1]//10, 1)
    paths, coefs, steps, premature = [], [], [], []
    for k in range(y.shape[1]):
        opts = dict(return_path=case['keep'], return_n_iter=True)
        if count is not None:
            opts['n_nonzero_coefs'] = count
        if case['tol'] is not None:
            opts['tol'] = case['tol']
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if gram:
                if case['tol'] is not None:
                    opts['norms_squared'] = np.array([case['norms'][k] if kind == 1 else np.sum(y[:, k]**2)], dtype=float)
                value, iterations = orthogonal_mp_gram(g.copy(), xy[:, k].copy(), **opts)
            else:
                value, iterations = orthogonal_mp(x.copy(), y[:, k].copy(), precompute=False, **opts)
        if case['keep']:
            path = np.asarray(value).reshape(x.shape[1], iterations)
            coefs.append(path[:, -1] if iterations else np.zeros(x.shape[1]))
        else:
            path = np.empty((x.shape[1], 0)); coefs.append(np.asarray(value).reshape(x.shape[1]))
        paths.append(path); steps.append(iterations)
        premature.append(int(any('premature' in str(w.message) for w in caught)))
    length = max(steps) if case['keep'] else 0
    padded = np.zeros((x.shape[1], y.shape[1], length))
    if case['keep']:
        for k, path in enumerate(paths):
            padded[:, k, :path.shape[1]] = path
    return dict(shape=[x.shape[1], y.shape[1], length], coef=np.column_stack(coefs), path=padded,
                iterations=steps, premature=premature, backend=[int(gram)], preserved=[1])


def encode(cases):
    lines = [str(len(cases))]
    for c in cases:
        x, y, norm = c['x'], c['y'], c['norms']
        lines += [f"{len(x)} {x.shape[1]} {y.shape[1]} {c['kind']} {c['vector']} "
                  f"{int(c['count'] is not None)} {c['count'] or 1} {int(c['tol'] is not None)} "
                  f"{c['tol'] or 0} {int(c['keep'])} {c['precompute']} {len(norm) if norm is not None else -1}"]
        if norm is not None:
            lines += [' '.join(map(str, norm))]
        lines += [' '.join(map(str, row)) for row in x]
        lines += [' '.join(map(str, row)) for row in y]
    return '\n'.join(lines)+'\n'


def certify_full_rank_span(case, actual, expected, prefix):
    """Only roundoff-dependent extra steps after spanning a wide dictionary."""
    x, y = case['design'], case['response']
    source_steps = np.array(expected['iterations'])
    native_steps = actual[prefix+'iterations'].astype(int)
    if np.array_equal(native_steps, source_steps) or len(x) >= x.shape[1]:
        return None
    rank = np.linalg.matrix_rank(x)
    changed = native_steps != source_steps
    assert np.all(np.minimum(native_steps, source_steps)[changed] >= rank)
    p, t = expected['shape'][:2]
    native_coef = actual[prefix+'coef'].reshape(p, t, order='F')
    native_path = actual[prefix+'path'].reshape(p, t, -1, order='F')
    for k, (na, ns) in enumerate(zip(native_steps, source_steps)):
        length = min(na, ns)
        np.testing.assert_allclose(native_path[:, k, :length], expected['path'][:, k, :length], rtol=3e-8, atol=3e-8)
        np.testing.assert_array_equal(native_path[:, k, na:], 0)
        if na:
            np.testing.assert_array_equal(native_coef[:, k], native_path[:, k, na-1])
        else:
            np.testing.assert_array_equal(native_coef[:, k], 0)
        if not changed[k]:
            np.testing.assert_allclose(native_coef[:, k], expected['coef'][:, k], rtol=3e-8, atol=3e-8)
            continue
        # Both endpoints must solve the full-row-rank least-squares problem to
        # backward error, and every additional native knot must retain that fit.
        for coefficient in [expected['coef'][:, k], *native_path[:, k, rank-1:na].T]:
            bound = 256*x.shape[0]*x.shape[1]*np.finfo(float).eps * (
                np.linalg.norm(x, 2)*np.linalg.norm(coefficient)+np.linalg.norm(y[:, k]))
            assert np.linalg.norm(x@coefficient-y[:, k]) <= bound
        np.testing.assert_allclose(x@native_coef[:, k], x@expected['coef'][:, k], rtol=3e-8, atol=3e-8)
    certificate = dict(rank=int(rank), native_iterations=native_steps.tolist(), source_iterations=source_steps.tolist())
    expected.update(shape=[p, t, native_path.shape[2]], coef=native_coef, path=native_path, iterations=native_steps)
    return certificate


def wrapper_adapters():
    x = np.eye(3); y = np.array([1., 2., 0.])
    value = orthogonal_mp(x, y, n_nonzero_coefs=1, precompute=True, return_n_iter=True)
    assert isinstance(value, np.ndarray)
    ragged = np.column_stack([0*y, y])
    failures = {}
    for name, function in [('direct', orthogonal_mp), ('gram', orthogonal_mp_gram)]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                function(x, ragged, n_nonzero_coefs=2, return_path=True)
            except IndexError as error:
                failures[name] = type(error).__name__
    assert len(failures) == 2
    return dict(precomputed_return_n_iter_source_type=type(value).__name__,
                ragged_multi_target_path_errors=failures,
                native='owned full paths, per-target lengths, zero padding and retained iteration counts')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--profile', default='checked')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    cases = fixtures()
    inp, out = RESULTS/f'omp_paths_{args.profile}.in', RESULTS/f'omp_paths_{args.profile}.out'
    inp.write_text(encode(cases))
    proc = subprocess.run([str(args.executable.resolve()), str(inp), str(out)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout+proc.stderr
    actual = {line.split()[0]: np.array(line.split()[1:], dtype=float) for line in out.read_text().splitlines()}
    checks, ragged, premature, maximum, zero_tol_flags, duplicate_certificates = [], 0, 0, 0., [], []
    rank_certificates = []
    for i, c in enumerate(cases, 1):
        expected = oracle(c)
        certificate = certify_full_rank_span(c, actual, expected, f'case{i}_')
        if certificate:
            rank_certificates.append(dict(case=i, **certificate))
        ragged += int(c['keep'] and len(set(expected['iterations'])) > 1)
        premature += sum(expected['premature'])
        for name, want in expected.items():
            got = actual[f'case{i}_{name}']; want = np.asarray(want).ravel(order='F')
            if name in ['coef', 'path']:
                if c['variant'] == 'duplicate' and c['x'].shape[1] > 1 and want.size:
                    p, t, length = expected['shape']
                    shape = (p, t) if name == 'coef' else (p, t, length)
                    a, b = got.reshape(shape, order='F'), want.reshape(shape, order='F')
                    # Equal atoms admit interchangeable coefficient allocation.
                    np.testing.assert_allclose(np.einsum('ij,j...->i...', c['x'], a),
                                               np.einsum('ij,j...->i...', c['x'], b), rtol=3e-8, atol=3e-8)
                    a = a.copy(); b = b.copy(); a[0] += a[-1]; b[0] += b[-1]
                    got, want = a[:-1].ravel(order='F'), b[:-1].ravel(order='F')
                    duplicate_certificates.append(dict(case=i, field=name))
                np.testing.assert_allclose(got, want, rtol=3e-8, atol=3e-8,
                                           err_msg=str((i, {k: v for k, v in c.items()
                                                           if k not in ['x', 'y', 'norms', 'design', 'response']}, name)))
                if want.size:
                    maximum = max(maximum, float(np.max(abs(got-want)/(1+abs(want)))))
            elif name == 'premature' and c['tol'] == 0 and not np.array_equal(got, want):
                # Public sklearn returns no warning-state array. At exact zero
                # tolerance its warning can depend on residual roundoff even
                # when every coefficient, path knot and iteration count agrees.
                zero_tol_flags.append(dict(case=i, native=got.tolist(), source=want.tolist()))
            else:
                np.testing.assert_array_equal(got, want, err_msg=f'case {i} {name}')
            checks.append(dict(case=i, field=name, passed=True,
                               oracle='full_rank_span_certificate' if certificate else 'sklearn_column_solve'))
    assert len(actual) == len(checks)
    assert ragged > 0 and premature > 0
    source = Path(inspect.getfile(inspect.unwrap(orthogonal_mp)))
    report = dict(sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__, profile=args.profile,
                  configurations=len(cases), groups=len(checks), checks=checks, ragged_configurations=ragged,
                  premature_targets=premature, max_scaled_error=maximum,
                  zero_tolerance_warning_roundoff=zero_tol_flags,
                  duplicate_column_certificates=duplicate_certificates,
                  full_rank_span_certificates=rank_certificates,
                  wrapper_adapters=wrapper_adapters(), input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),
                  configuration_inventory=[{**{k: v for k, v in c.items()
                                              if k not in ['x', 'y', 'norms', 'design', 'response']},
                                            'x_shape': list(c['x'].shape), 'y_shape': list(c['y'].shape),
                                            'x_sha256': hashlib.sha256(c['x'].tobytes()).hexdigest(),
                                            'y_sha256': hashlib.sha256(c['y'].tobytes()).hexdigest()}
                                           for c in cases],
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                  sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT/'src/sklearn_omp_paths.f90', ROOT/'app/reference_omp_paths_driver.f90', Path(__file__)]})
    (args.report or RESULTS/f'omp_paths_{args.profile}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'{len(checks)} OMP groups across {len(cases)} configurations passed; {ragged} ragged paths')


if __name__ == '__main__':
    main()
