"""Dense elastic-net and Lasso regularization paths against local sklearn."""
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
import scipy
import sklearn
from sklearn.linear_model import enet_path, lasso_path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'validation/results'


def fixtures():
    result = []
    for seed, n, p in [(7, 37, 7), (42, 21, 12), (2026, 9, 14), (91, 17, 1)]:
        rng = np.random.default_rng(seed)
        base = rng.normal(size=(n, p))
        for variant in ['regular', 'correlated', 'zero_column', 'zero_response']:
            x = base.copy(); y = rng.normal(size=(n, 3))
            if variant == 'correlated' and p > 1:
                x[:, 1:] += .7*x[:, :1]
            if variant == 'zero_column':
                x[:, -1] = 0
            if variant == 'zero_response':
                y[:] = 0
            for vector, yy in [(True, y[:, :1]), (False, y[:, :1]), (False, y)]:
                for mix, lasso in [(.5, False), (1., False), (.25, True), (0., False)]:
                    for mode in ['false', 'true', 'auto', 'given']:
                        for selection, limit, tol in [('cyclic', 1000, 1e-4), ('random', 1000, 1e-8),
                                                      ('random', 3, 1e-4), ('cyclic', 0, 1e-4)]:
                            positive = vector and mix == 1
                            alphas = np.array([.03, .5, .03, 0.]) if limit < 1000 or mix == 0 else None
                            initial = rng.normal(size=(yy.shape[1], p))*.02 if limit in [0, 3] else None
                            g = x.T@x if mode == 'given' else None
                            xy = x.T@yy if mode in ['given', 'false'] else None
                            if mode == 'false':
                                xy = 2*xy  # Must be discarded by _pre_fit.
                            result.append(dict(seed=seed, variant=variant, x=x, y=yy, vector=vector,
                                               lasso=lasso, mix=mix, eps=.001, n_alphas=5, tol=tol,
                                               limit=limit, selection=selection, positive=positive,
                                               mode='auto' if mode == 'given' else mode,
                                               alphas=alphas, initial=initial, gram=g, xy=xy))
    # Empty/one-point/increasing/tiny automatic grids and exact zero tolerance.
    for points, eps, yy in [(0, .001, np.ones((3, 1))), (1, .001, np.ones((3, 1))),
                            (4, 2., np.ones((3, 1))), (3, .001, np.zeros((3, 1)))]:
        result.append(dict(seed=0, variant='grid', x=np.eye(3), y=yy, vector=True, lasso=False,
                           mix=.5, eps=eps, n_alphas=max(points, 1), tol=0., limit=3, selection='cyclic', positive=False,
                           mode='auto', alphas=np.array([]) if points == 0 else None,
                           initial=None, gram=None, xy=None))
    return result


def encode(cases):
    lines = [str(len(cases))]
    for c in cases:
        x, y = c['x'], c['y']; alpha = c['alphas']
        lines += [f"{len(x)} {x.shape[1]} {y.shape[1]} {int(c['vector'])} {int(c['lasso'])} "
                  f"{len(alpha) if alpha is not None else -1} {int(c['initial'] is not None)} "
                  f"{int(c['gram'] is not None)} {int(c['xy'] is not None)}",
                  f"{c['mix']} {c['eps']} {c['n_alphas']} {c['tol']} {c['limit']} {c['seed']} "
                  f"{int(c['positive'])} {c['selection']} {c['mode']}"]
        for a in [x, y, alpha, c['initial'], c['gram'], c['xy']]:
            if a is not None:
                if np.ndim(a) == 1:
                    lines.append(' '.join(map(str, a)))
                else:
                    lines += [' '.join(map(str, row)) for row in a]
    return '\n'.join(lines)+'\n'


def oracle(c):
    x, y = c['x'], c['y']
    multi = not c['vector']
    options = dict(eps=c['eps'], n_alphas=c['n_alphas'], alphas=c['alphas'], positive=c['positive'],
                   max_iter=c['limit'], tol=c['tol'], selection=c['selection'], random_state=c['seed'],
                   precompute=c['gram'] if c['gram'] is not None else {'auto': 'auto', 'true': True, 'false': False}[c['mode']],
                   return_n_iter=True)
    if not c['lasso']:
        options['l1_ratio'] = c['mix']
    if c['initial'] is not None:
        options['coef_init'] = np.array(c['initial'] if multi else c['initial'][0], order='F', copy=True)
    if c['xy'] is not None:
        options['Xy'] = c['xy'] if multi else c['xy'][:, 0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        alpha, coefficients, gaps, iterations = (lasso_path if c['lasso'] else enet_path)(
            np.asfortranarray(x), np.asfortranarray(y) if multi else y[:, 0].copy(), **options)
    coefficients = np.asarray(coefficients).reshape(y.shape[1], x.shape[1], len(alpha))
    tolerance = c['tol']*np.sum(y*y)/len(x)
    converged = (gaps <= tolerance) if multi else (gaps < tolerance)
    if c['limit'] == 0:
        converged[:] = False
    backend = 2 if multi else int(c['gram'] is not None or c['mode'] == 'true' or (c['mode'] == 'auto' and len(x) > x.shape[1]))
    return dict(shape=list(coefficients.shape), alphas=alpha, coefficients=coefficients, gaps=gaps,
                iterations=iterations, converged=converged.astype(int), tolerance=[tolerance], backend=[backend], preserved=[1]), caught


def certify_stopping_roundoff(c, actual, expected, prefix):
    """Certify only optimal models whose stopping metadata differs numerically."""
    x, y = c['x'], c['y']; n, p = x.shape
    native_coef = actual[prefix+'coefficients'].reshape(expected['shape'], order='F')
    records = []
    mix = 1. if c['lasso'] else c['mix']
    correlation = x.T@y
    zero_dual = (np.max(correlation) if c['positive'] else
                 np.max(abs(correlation)) if c['vector'] else np.max(np.linalg.norm(correlation, axis=1)))
    eps = np.finfo(float).eps
    for j, alpha in enumerate(expected['alphas']):
        mismatch = (actual[prefix+'iterations'][j] != expected['iterations'][j] or
                    actual[prefix+'converged'][j] != expected['converged'][j] or
                    not np.isclose(actual[prefix+'gaps'][j], expected['gaps'][j], rtol=3e-8, atol=3e-8))
        if not mismatch:
            continue
        l1, l2 = n*alpha*mix, n*alpha*(1-mix)
        coefficients = [native_coef[:, :, j].T, expected['coefficients'][:, :, j].T]
        norm_y, norm_x = np.linalg.norm(y), np.linalg.norm(x)
        zero_bound = 512*eps*max(1., norm_y)/max(1., norm_x)
        if all(np.linalg.norm(w) <= zero_bound for w in coefficients):
            kkt_bound = 512*eps*max(1., norm_x*norm_y, l1)
            assert zero_dual <= l1+kkt_bound, (prefix, j, zero_dual, l1)
            proof = dict(kind='zero_model_kkt', coefficient_bound=zero_bound,
                         correlation_norm=float(zero_dual), l1=float(l1), kkt_roundoff_bound=kkt_bound)
        else:
            assert l1 == 0, (prefix, j, 'nonzero L1 model requires direct metadata agreement')
            optimality = []
            for w in coefficients:
                residual = y-x@w
                if l2 > 0:
                    # Exact ridge primal/dual gap, independent of the L1 scaling
                    # that becomes discontinuous at a numerically zero gradient.
                    q = x.T@residual
                    if c['positive']:
                        assert np.min(w) >= 0
                        gap = np.sum((l2*w-np.maximum(q, 0))**2)/(2*l2)+np.sum(w*np.maximum(-q, 0))
                    else:
                        gap = np.sum((l2*w-q)**2)/(2*l2)
                elif c['positive']:
                    from scipy.optimize import nnls
                    assert np.min(w) >= 0
                    optimum = np.column_stack([nnls(x, y[:, k])[0] for k in range(y.shape[1])])
                    gap = abs(np.sum(residual**2)-np.sum((y-x@optimum)**2))/2
                else:
                    optimum = np.linalg.lstsq(x, y, rcond=None)[0]
                    gap = np.sum((x@w-x@optimum)**2)/2
                bound = max(c["tol"]*norm_y**2, 1e-18*max(1., norm_y**2, l2*np.sum(w*w)))
                assert gap <= bound, (prefix, j, gap, bound)
                optimality.append(dict(gap=float(gap), bound=float(bound)))
            proof = dict(kind='zero_l1_optimality', l2=float(l2), native_source_certificates=optimality)
        records.append(dict(alpha_index=j, alpha=float(alpha), proof=proof,
                            native_iterations=int(actual[prefix+'iterations'][j]), source_iterations=int(expected['iterations'][j]),
                            native_gap=float(actual[prefix+'gaps'][j]), source_gap=float(expected['gaps'][j]),
                            native_converged=int(actual[prefix+'converged'][j]), source_converged=int(expected['converged'][j])))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--profile', default='checked')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    cases = fixtures(); inp = RESULTS/f'coordinate_paths_{args.profile}.in'; out = inp.with_suffix('.out')
    inp.write_text(encode(cases))
    proc = subprocess.run([str(args.executable.resolve()), str(inp), str(out)], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stdout+proc.stderr
    actual = {l.split()[0]: np.array(l.split()[1:], float) for l in out.read_text().splitlines()}
    checks, source_warnings, maximum, certificates = [], [], 0., []
    for i, c in enumerate(cases, 1):
        expected, caught = oracle(c)
        certified = certify_stopping_roundoff(c, actual, expected, f'case{i}_')
        certificates.extend(dict(case=i, **record) for record in certified)
        source_warnings.extend(dict(case=i, category=w.category.__name__, message=str(w.message)) for w in caught)
        for name, want in expected.items():
            want = np.asarray(want).ravel(order='F'); got = actual[f'case{i}_{name}']
            if name in ['gaps', 'iterations', 'converged'] and certified:
                retained = np.ones(len(want), dtype=bool)
                retained[[record['alpha_index'] for record in certified]] = False
                got, want = got[retained], want[retained]
            context = (i, c['seed'], c['variant'], c['mix'], c['vector'], c['lasso'], c['selection'], c['mode'], name)
            if name in ['shape', 'iterations', 'converged', 'backend', 'preserved']:
                np.testing.assert_array_equal(got, want, err_msg=str(context))
            else:
                np.testing.assert_allclose(got, want, rtol=3e-8, atol=3e-8, err_msg=str(context))
                if want.size:
                    maximum = max(maximum, float(np.max(abs(got-want)/(1+abs(want)))))
            checks.append(dict(case=i, field=name, passed=True,
                               certified_stopping_indices=[r['alpha_index'] for r in certified]
                               if name in ['gaps', 'iterations', 'converged'] else []))
    assert len(actual) == len(checks)
    source = Path(inspect.getfile(inspect.unwrap(enet_path)))
    report = dict(sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__, profile=args.profile,
                  configurations=len(cases), groups=len(checks), checks=checks, max_scaled_error=maximum,
                  stopping_certificates=certificates,
                  source_warnings=source_warnings, input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  cython_sha256=hashlib.sha256((Path(sklearn.__file__).parent/'linear_model/_cd_fast.pyx').read_bytes()).hexdigest(),
                  executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                  native_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [ROOT/'src/sklearn_coordinate_paths.f90', ROOT/'src/sklearn_multitask.f90',
                                           ROOT/'app/reference_coordinate_paths_driver.f90', Path(__file__)]},
                  configuration_inventory=[{**{k: v for k, v in c.items()
                                              if k not in ['x', 'y', 'alphas', 'initial', 'gram', 'xy']},
                                            'x_shape': list(c['x'].shape), 'y_shape': list(c['y'].shape),
                                            'explicit_alphas': c['alphas'].tolist() if c['alphas'] is not None else None,
                                            'initial_sha256': hashlib.sha256(c['initial'].tobytes()).hexdigest()
                                                if c['initial'] is not None else None,
                                            'gram_sha256': hashlib.sha256(c['gram'].tobytes()).hexdigest()
                                                if c['gram'] is not None else None,
                                            'xy_sha256': hashlib.sha256(c['xy'].tobytes()).hexdigest()
                                                if c['xy'] is not None else None,
                                            'x_sha256': hashlib.sha256(c['x'].tobytes()).hexdigest(),
                                            'y_sha256': hashlib.sha256(c['y'].tobytes()).hexdigest()}
                                           for c in cases])
    (args.report or RESULTS/f'coordinate_paths_{args.profile}.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'{len(checks)} coordinate-path groups across {len(cases)} configurations passed')


if __name__ == '__main__':
    main()
