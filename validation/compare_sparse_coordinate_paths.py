"""CSR coordinate paths: Source CSC storage, weights and implicit centering."""
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
from scipy import sparse
import sklearn
from sklearn.linear_model import enet_path, lasso_path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'validation/results'


def raw_layout(x, mode):
    data, indices, ptr = [], [], [0]
    for row in x:
        columns = list(range(len(row))) if mode == 'zeros' else np.flatnonzero(row).tolist()
        if mode == 'unsorted':
            columns = columns[::-1]
        for j in columns:
            if mode == 'duplicate':
                data += [.4*row[j], .6*row[j]]; indices += [j, j]
            else:
                data.append(row[j]); indices.append(j)
        ptr.append(len(data))
    return sparse.csr_matrix((np.array(data, float), np.array(indices, np.int32), np.array(ptr, np.int32)), shape=x.shape)


def fixtures():
    cases = []
    for seed, n, p in [(7, 48, 7), (42, 17, 12), (2026, 23, 1)]:
        rng = np.random.default_rng(seed)
        base = rng.normal(size=(n, p)); base[rng.random(size=(n, p)) < .65] = 0
        if p > 1:
            base[:, -1] = 0
        for layout in ['canonical', 'zeros', 'unsorted', 'duplicate']:
            x = raw_layout(base, layout); y = rng.normal(size=n)
            for weighting in ['none', 'ones', 'fractional', 'zeros', 'signed']:
                weights = None
                if weighting != 'none':
                    weights = np.ones(n)
                    if weighting == 'fractional':
                        weights = rng.uniform(.1, 2., size=n); weights[::4] = 0
                    if weighting == 'zeros':
                        weights[:] = 0
                    if weighting == 'signed':
                        weights[-1] = -.001
                for centering in ['none', 'mean', 'shifted']:
                    offset, scale = None, None
                    if centering != 'none':
                        mean = np.asarray(x.mean(axis=0)).ravel()
                        if centering == 'shifted':
                            mean = np.linspace(-.2, .3, p)
                        scale = np.linspace(.5, 1.5, p)
                        if centering == 'shifted':
                            scale[::2] *= -1
                        offset = mean*scale
                    for mix, lasso in [(.5, False), (.25, True), (0., False)]:
                        for selection, limit in [('cyclic', 300), ('random', 300), ('random', 3), ('cyclic', 0)]:
                            if layout == 'duplicate' or weighting == 'signed':
                                limit = min(limit, 3)
                            positive = lasso and centering != 'shifted'
                            alpha = np.array([.03, .4, .03, 0.]) if limit < 300 else None
                            if mix == 0:
                                alpha = np.array([.4, .1])
                            initial = rng.normal(size=p)*.02 if limit in [0, 3] else None
                            if initial is not None and positive:
                                initial = abs(initial)
                            cases.append(dict(seed=seed, layout=layout, weighting=weighting, centering=centering,
                                              x=x, y=y, mix=mix, lasso=lasso, weights=weights,
                                              offset=offset, scale=scale, alphas=alpha, initial=initial,
                                              selection=selection, limit=limit, positive=positive,
                                              precompute='true' if weighting == 'ones' else 'auto',
                                              gram=np.array([[np.nan]]) if weighting == 'fractional' else None,
                                              xy=np.array([99.]) if centering == 'mean' else None,
                                              eps=.01, points=4, tol=1e-8))
    # A design whose dense expansion would contain 268 million real64 entries.
    x = sparse.csr_matrix((np.array([1.]), (np.array([0]), np.array([0]))), shape=(32768, 8192))
    y = np.zeros(x.shape[0]); y[0] = 1
    cases.append(dict(seed=0, layout='large_sparse', weighting='none', centering='none', x=x, y=y,
                      mix=.5, lasso=False, weights=None, offset=None, scale=None, alphas=None, initial=None,
                      selection='cyclic', limit=10, positive=False, precompute='auto', gram=None, xy=None,
                      eps=.25, points=3, tol=1e-8))
    for empty in [False, True]:
        cases.append(dict(seed=0, layout='empty_storage', weighting='none', centering='shifted',
                          x=sparse.csr_matrix((3, 1)), y=np.array([2., 2., 2.]), mix=.5, lasso=False,
                          weights=None, offset=np.array([2.]), scale=np.array([2.]),
                          alphas=np.array([]) if empty else np.array([.1]), initial=None,
                          selection='cyclic', limit=10, positive=False, precompute='false', gram=None,
                          xy=None, eps=.001, points=3, tol=1e-8))
    # Source ignores X_scale completely unless X_offset is supplied.
    c = cases[0].copy(); c.update(scale=np.array([np.nan]), precompute='false', layout='ignored_scale')
    cases.append(c)
    cases.append(dict(seed=0, layout='unused_raw_correlation', weighting='zeros', centering='none',
                      x=sparse.csr_matrix(np.array([[1e150]])), y=np.array([1e170]), mix=.5, lasso=False,
                      weights=np.array([0.]), offset=None, scale=None, alphas=np.array([.1]), initial=None,
                      selection='cyclic', limit=3, positive=False, precompute='auto', gram=None, xy=None,
                      eps=.001, points=3, tol=1e-8))
    for points, eps, target in [(1, .001, np.array([3., -2., 1.])),
                                (3, 2., np.array([3., -2., 1.])), (3, .001, np.zeros(3))]:
        cases.append(dict(seed=0, layout='grid_boundary', weighting='none', centering='none',
                          x=sparse.eye(3, format='csr'), y=target, mix=.5, lasso=False, weights=None,
                          offset=None, scale=None, alphas=None, initial=None, selection='cyclic', limit=3,
                          positive=False, precompute='auto', gram=None, xy=None, eps=eps, points=points, tol=1e-8))
    return cases


def encode(cases):
    lines = [str(len(cases))]
    for c in cases:
        x = c['x']; a = c['alphas']
        lines += [f"{x.shape[0]} {x.shape[1]} {x.nnz} {int(c['lasso'])} {len(a) if a is not None else -1} "
                  f"{int(c['initial'] is not None)} {int(c['weights'] is not None)} {int(c['offset'] is not None)} "
                  f"{len(c['scale']) if c['scale'] is not None else -1} {int(c['gram'] is not None)} {int(c['xy'] is not None)}",
                  f"{c['mix']} {c['eps']} {c['points']} {c['tol']} {c['limit']} {c['seed']} "
                  f"{int(c['positive'])} {c['selection']} {c['precompute']}"]
        for values in [x.indptr+1, x.indices+1, x.data, c['y'], a, c['initial'], c['weights'],
                       c['offset'], c['scale'], c['gram'], c['xy']]:
            if values is not None:
                lines.append(' '.join(map(str, np.asarray(values).ravel())))
    return '\n'.join(lines)+'\n'


def oracle(c):
    params = dict(alphas=c['alphas'], eps=c['eps'], n_alphas=c['points'], tol=c['tol'],
                  max_iter=c['limit'], selection=c['selection'], random_state=c['seed'],
                  positive=c['positive'], return_n_iter=True,
                  precompute=c['gram'] if c['gram'] is not None else {'auto': 'auto', 'true': True, 'false': False}[c['precompute']])
    for field, name in [('initial', 'coef_init'), ('weights', 'sample_weight'), ('offset', 'X_offset'),
                        ('scale', 'X_scale'), ('xy', 'Xy')]:
        if c[field] is not None:
            params[name] = np.array(c[field], copy=True)
    if not c['lasso']:
        params['l1_ratio'] = c['mix']
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        alphas, coef, gaps, iterations = (lasso_path if c['lasso'] else enet_path)(c['x'].copy(), c['y'].copy(), **params)
    weights = np.ones(len(c['y'])) if c['weights'] is None else c['weights']
    tolerance = c['tol']*np.dot(c['y'], weights*c['y'])/len(c['y'])
    converged = (gaps < tolerance).astype(int)
    if c['limit'] == 0:
        converged[:] = 0
    return dict(shape=[1, c['x'].shape[1], len(alphas)], alphas=alphas,
                coefficients=np.asarray(coef).reshape(1, c['x'].shape[1], len(alphas)),
                gaps=gaps, iterations=iterations, converged=converged, tolerance=[tolerance], backend=[3], preserved=[1]), caught


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--profile', default='checked')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    cases = fixtures(); RESULTS.mkdir(exist_ok=True)
    inp = RESULTS/f'sparse_coordinate_{args.profile}.in'; out = inp.with_suffix('.out')
    inp.write_text(encode(cases))
    proc = subprocess.run([str(args.executable.resolve()), str(inp), str(out)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout+proc.stderr
    actual = {l.split()[0]: np.array(l.split()[1:], float) for l in out.read_text().splitlines()}
    checks, source_warnings, maximum = [], [], 0.
    for i, c in enumerate(cases, 1):
        expected, caught = oracle(c)
        source_warnings.extend(dict(case=i, category=w.category.__name__, message=str(w.message)) for w in caught)
        for name, want in expected.items():
            got = actual[f'case{i}_{name}']; want = np.asarray(want).ravel(order='F')
            context = (i, c['layout'], c['weighting'], c['centering'], c['mix'], c['lasso'], c['selection'], c['limit'], name)
            if name in ['shape', 'iterations', 'converged', 'backend', 'preserved']:
                np.testing.assert_array_equal(got, want, err_msg=str(context))
            else:
                np.testing.assert_allclose(got, want, rtol=3e-8, atol=3e-8, err_msg=str(context))
                if want.size:
                    maximum = max(maximum, float(np.max(abs(got-want)/(1+abs(want)))))
            checks.append(dict(case=i, field=name, passed=True))
    assert len(actual) == len(checks)
    source = Path(inspect.getfile(inspect.unwrap(enet_path)))
    report = dict(sklearn=sklearn.__version__, scipy=scipy.__version__, numpy=np.__version__, profile=args.profile,
                  configurations=len(cases), groups=len(checks), checks=checks, source_warnings=source_warnings,
                  max_scaled_error=maximum, input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  cython_sha256=hashlib.sha256((Path(sklearn.__file__).parent/'linear_model/_cd_fast.pyx').read_bytes()).hexdigest(),
                  executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                  native_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [ROOT/'src/sklearn_coordinate_paths.f90', ROOT/'src/sklearn_sparse.f90',
                                           ROOT/'app/reference_sparse_coordinate_driver.f90', Path(__file__)]},
                  configuration_inventory=[{**{k: v for k, v in c.items()
                                              if k not in ['x', 'y', 'alphas', 'initial', 'weights', 'offset', 'scale', 'gram', 'xy']},
                                            'shape': list(c['x'].shape), 'stored': int(c['x'].nnz),
                                            'canonical': bool(c['x'].has_canonical_format),
                                            'overfull_columns': int(np.count_nonzero(np.diff(c['x'].tocsc().indptr) > c['x'].shape[0])),
                                            'dense_bytes': int(c['x'].shape[0])*c['x'].shape[1]*8,
                                            'input_arrays_sha256': {name: hashlib.sha256(values.tobytes()).hexdigest()
                                                for name, values in [('pointers', c['x'].indptr), ('indices', c['x'].indices),
                                                                     ('data', c['x'].data), ('y', c['y']), ('alphas', c['alphas']),
                                                                     ('initial', c['initial']), ('weights', c['weights']),
                                                                     ('offset', c['offset']), ('scale', c['scale']),
                                                                     ('gram', c['gram']), ('xy', c['xy'])] if values is not None}}
                                           for c in cases])
    (args.report or RESULTS/f'sparse_coordinate_{args.profile}.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'{len(checks)} sparse coordinate groups across {len(cases)} configurations passed')


if __name__ == '__main__':
    main()
