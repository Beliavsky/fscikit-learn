"""GraphicalLassoCV selection, warm-start paths and final fits against sklearn 1.6.1."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import os
import subprocess
import warnings
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import sklearn
from sklearn.covariance import GraphicalLassoCV, EmpiricalCovariance, empirical_covariance, log_likelihood
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def certify_endpoint(cov, precision, test_cov, source_cov, source_precision, source_test, score):
    """Backward error of the empirical inverse; only the unselected zero endpoint."""
    p = len(cov)
    eps = np.finfo(float).eps
    norm = lambda a: np.linalg.norm(a, 2)
    np.testing.assert_allclose(cov, source_cov, rtol=3.e-13, atol=3.e-14)
    np.testing.assert_allclose(test_cov, source_test, rtol=3.e-13, atol=3.e-14)
    eigenvalues = np.linalg.eigvalsh(source_cov)
    cutoff = p * eps * np.max(abs(eigenvalues))
    retained = abs(eigenvalues) > cutoff
    rank = int(retained.sum())
    condition = np.max(abs(eigenvalues)) / np.min(abs(eigenvalues[retained])) if rank else 1.
    certificate = dict(rank=rank, features=p, effective_condition=float(condition))
    if rank == p:
        # Inverse residuals must be backward stable, not merely close to one another.
        left = norm(precision @ cov - np.eye(p))
        right = norm(source_cov @ source_precision - np.eye(p))
        assert left <= 128*p*eps*norm(cov)*norm(precision)
        assert right <= 128*p*eps*norm(source_cov)*norm(source_precision)
        # Exact inverse perturbation identity, with residual terms on both sides.
        bound = (norm(precision)*norm(source_cov-cov)*norm(source_precision) +
                 left*norm(source_precision) + norm(precision)*right)
        difference = norm(precision-source_precision)
        assert difference <= bound + 128*p*eps*max(norm(precision),norm(source_precision))
        target = log_likelihood(test_cov, precision)
        np.testing.assert_allclose(score, target, rtol=3.e-8, atol=3.e-8)
        certificate.update(kind='inverse_backward_error', native_residual=float(left),
                           source_residual=float(right), precision_difference=float(difference),
                           precision_bound=float(bound), conditioned_score=float(target))
        return float(target), certificate
    # Verify all four Moore-Penrose equations and the source precision comparison.
    limit = 256*p*eps*max(1.,condition)
    for s, inverse in [(cov,precision),(source_cov,source_precision)]:
        assert norm(s@inverse@s-s) <= limit*max(norm(s),np.finfo(float).tiny)
        assert norm(inverse@s@inverse-inverse) <= limit*max(norm(inverse),np.finfo(float).tiny)
        assert norm(s@inverse-(s@inverse).T) <= limit
        assert norm(inverse@s-(inverse@s).T) <= limit
    np.testing.assert_allclose(precision, source_precision, rtol=3.e-8, atol=3.e-8)
    assert np.linalg.matrix_rank(precision, tol=128*p*eps*norm(precision)) == rank
    assert np.isfinite(score) or np.isneginf(score)
    # A singular mathematical precision acquires tiny eigenvalues in floating-point
    # storage. Partial-pivot LU has growth <= 2**(p-1); Weyl bounds each singular
    # value after its roundoff perturbation. Hence this upper bound on log|det|.
    gamma = (3*p*eps)/(1-3*p*eps)
    # || |L||U| ||_2 <= p**2 * growth * ||A||_2 for partial pivoting.
    delta = gamma * 2.**(p-1) * p**2 * norm(precision)
    upper = np.log(np.linalg.svd(precision,compute_uv=False)+delta).sum()
    if np.isfinite(score):
        determinant = 2*score+np.sum(test_cov*precision)+p*np.log(2*np.pi)
        assert determinant <= upper+128*p*eps*max(1.,abs(determinant))
    certificate.update(kind='rank_deficient_determinant', logdet_upper_bound=float(upper),
                       native_score=float(score), source_on_native_score=float(log_likelihood(test_cov,precision)))
    return float(score), certificate


def resolve(profile):
    opts = ['--profile', profile] if profile == 'release' else ['--flag', FLAGS]
    proc = subprocess.run(['fpm', 'run', *opts, '--target', 'reference_graphical_cv_driver', '--runner'],
                          cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    paths = [ROOT / s.strip().strip('"') for s in proc.stdout.splitlines()
             if s.strip().strip('"').endswith(('reference_graphical_cv_driver', 'reference_graphical_cv_driver.exe'))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, proc.stdout
    return paths[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--profile', choices=['checked', 'release'], default='checked')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    executable = args.executable.resolve() if args.executable else resolve(args.profile)
    checks, source_warnings, endpoint_certificates = [], [], []
    fixtures = 0

    def run(seed, case, x, query, *, alphas=4, refinements=4, cv=5, max_iter=300,
            centered=False, tol=1.e-8, enet_tol=1.e-10):
        nonlocal fixtures
        splitter = KFold(cv) if isinstance(cv, int) else cv
        folds = list(splitter.split(x)) if isinstance(cv, int) else cv
        model = GraphicalLassoCV(alphas=alphas, n_refinements=refinements, cv=cv, max_iter=max_iter,
                                assume_centered=centered, tol=tol, enet_tol=enet_tol)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(x)
            scores = np.column_stack([model.cv_results_[f'split{i}_test_score'] for i in range(len(folds))])
            costs = np.asarray(model.costs_).reshape(-1, 2)
            # The empirical alpha-zero endpoint is appended after selection.
            means = model.cv_results_['mean_test_score'][:-1]
            eligible = np.where(means >= .1 / np.finfo(float).eps, np.nan, means)
            best = -np.inf
            for value in eligible:
                if value >= best:
                    best = value
            expected = dict(alpha=[model.alpha_], best=[best], grid=model.cv_results_['alphas'], splits=scores,
                            mean=model.cv_results_['mean_test_score'], std=model.cv_results_['std_test_score'],
                            location=model.location_, covariance=model.covariance_, precision=model.precision_,
                            costs=costs, n_iter=[model.n_iter_], features=[model.n_features_in_],
                            converged=[int(len(costs) > 0 and abs(costs[-1, 1]) < tol)],
                            score=[model.score(query)], mahalanobis=model.mahalanobis(query))
            endpoints = [EmpiricalCovariance().fit(x[train]) for train, _ in folds]
            expected['endpoint_cov'] = np.stack([e.covariance_ for e in endpoints], axis=2)
            expected['endpoint_precision'] = np.stack([e.precision_ for e in endpoints], axis=2)
            expected['endpoint_test'] = np.stack([empirical_covariance(x[test] - e.location_, assume_centered=True)
                                                 for e, (_, test) in zip(endpoints, folds)], axis=2)
        source_warnings.extend(dict(seed=seed, case=case, category=w.category.__name__, message=str(w.message))
                               for w in caught)
        explicit = not isinstance(alphas, int)
        lines = [f'{len(x)} {x.shape[1]} {len(query)} {len(alphas) if explicit else 0} '
                 f'{0 if isinstance(cv, int) else len(folds)}',
                 f'{4 if explicit else alphas} {refinements} {cv if isinstance(cv, int) else 5} '
                 f'{max_iter} {int(centered)} {tol} {enet_tol} {np.finfo(float).eps}']
        if explicit:
            lines += [' '.join(map(str, alphas))]
        lines += [' '.join(map(str, row)) for row in x]
        lines += [' '.join(map(str, row)) for row in query]
        if not isinstance(cv, int):
            for train, test in folds:
                lines += [f'{len(train)} {len(test)}', ' '.join(map(str, train + 1)), ' '.join(map(str, test + 1))]
        inp = RESULTS / f'graphical_cv_{args.profile}_input.txt'
        out = RESULTS / f'graphical_cv_{args.profile}_output.txt'
        inp.write_text('\n'.join(lines) + '\n')
        proc = subprocess.run([str(executable), str(inp), str(out)], cwd=ROOT,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.returncode == 0, (seed, case, proc.stdout)
        actual = {}
        for line in out.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.asarray(values, dtype=float)
        for i, (train, test) in enumerate(folds, 1):
            expected[f'train{i}'] = train + 1
            expected[f'test{i}'] = test + 1
        conditioned = set()
        native_scores = actual['splits'].reshape(scores.shape,order='F')
        native_precision = actual['endpoint_precision'].reshape(expected['endpoint_precision'].shape,order='F')
        native_cov = actual['endpoint_cov'].reshape(expected['endpoint_cov'].shape,order='F')
        native_test = actual['endpoint_test'].reshape(expected['endpoint_test'].shape,order='F')
        for f in range(len(folds)):
            same_score = np.isclose(native_scores[-1,f],scores[-1,f],rtol=3.e-8,atol=3.e-8,equal_nan=True)
            same_precision = np.allclose(native_precision[:,:,f],expected['endpoint_precision'][:,:,f],
                                         rtol=3.e-8,atol=3.e-8,equal_nan=True)
            if same_score and same_precision:
                continue
            target,certificate = certify_endpoint(native_cov[:,:,f],native_precision[:,:,f],native_test[:,:,f],
                expected['endpoint_cov'][:,:,f],expected['endpoint_precision'][:,:,f],expected['endpoint_test'][:,:,f],
                native_scores[-1,f])
            certificate.update(seed=seed,case=case,fold=f,source_score=float(scores[-1,f]))
            endpoint_certificates.append(certificate)
            scores[-1,f] = target
            if not same_precision:
                expected['endpoint_precision'][:,:,f] = native_precision[:,:,f]
                conditioned.add('endpoint_precision')
            conditioned.update(['splits','mean','std'])
        if conditioned:
            with np.errstate(invalid='ignore'):
                expected['mean'] = scores.mean(axis=1)
                expected['std'] = scores.std(axis=1)
        assert actual.keys() == expected.keys(), (seed, case, actual.keys(), expected.keys())
        for name, want in expected.items():
            want = np.asarray(want).ravel(order='F')
            if name in ['n_iter', 'features', 'converged'] or name.startswith(('train', 'test')):
                np.testing.assert_array_equal(actual[name], want, err_msg=f'{seed} {case} {name}')
            else:
                np.testing.assert_allclose(actual[name], want, rtol=3.e-8, atol=3.e-8,
                                           equal_nan=True, err_msg=f'{seed} {case} {name}')
            checks.append(dict(seed=seed, case=case, name=name, passed=True,
                               oracle='source_conditioned_empirical_endpoint' if name in conditioned else 'sklearn'))
        fixtures += 1

    for seed in [0, 7, 42, 2026, 314159]:
        rng = np.random.RandomState(seed)
        case = 0
        for p in [2, 4, 7]:
            x = rng.normal(size=(73, p))
            x[:, 1] += .7 * x[:, 0]
            if p > 2:
                x[:, 2] += .4 * x[:, 1]
            x += np.linspace(-.5, 1., p)
            query = rng.normal(size=(11, p)) + .2
            for centered in [False, True]:
                for grid, refinements in [(2, 1), (4, 4), ([.01, 1., .1, .1], 3), ([10., 5., 2.], 4)]:
                    case += 1
                    run(seed, case, x, query, alphas=grid, refinements=refinements, cv=3, centered=centered)
            for limit in [0, 1, 5, 10]:
                case += 1
                run(seed, case, x, query, alphas=[1., .3, .1], cv=3, max_iter=limit)
            # Explicit folds preserve ordering, overlap, duplicates and non-partitions.
            folds = [(np.r_[np.arange(50)[::-1], 0, 0], np.arange(30, 60)),
                     (np.arange(20, 73), np.r_[np.arange(20), 3])]
            case += 1
            run(seed, case, x, query, cv=folds, refinements=2)
            case += 1
            run(seed, case, x, query, cv=[folds[0]], alphas=[.5, .2, .1])
        # Near-collinear covariance: verify the ill-conditioned empirical endpoint.
        x = rng.normal(size=(14, 8)); x[:, 1] = x[:, 0] + .002 * rng.normal(size=14)
        for grid in [4, [.5, .1, .01, .001]]:
            case += 1
            run(seed, case, x, rng.normal(size=(5, 8)), alphas=grid, cv=3, refinements=3, max_iter=500)
        x = rng.normal(size=(14,12))
        for grid in [4,[.5,.1,.01,.001]]:
            case += 1
            run(seed,case,x,rng.normal(size=(5,12)),alphas=grid,cv=3,refinements=3,max_iter=500)
        # Every positive candidate can fail on a one-row fold while the final
        # full-data model remains estimable; the smaller alpha wins -inf ties.
        x = rng.normal(size=(17,3))
        case += 1
        run(seed,case,x,rng.normal(size=(5,3)),alphas=[1.,.1],cv=[(np.array([0]),np.arange(1,17))])
        case += 1
        x = rng.normal(size=(2,2))
        run(seed,case,x,rng.normal(size=(3,2)),alphas=[1.,.1],cv=2)
        case += 1
        x = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        run(seed,case,x,x,alphas=4,refinements=2,cv=[(np.arange(4),np.arange(4))])
        for scale in [1.e-3,1.e3]:
            case += 1
            x = rng.normal(size=(47,4))*scale
            x[:,1] += .7*x[:,0]
            run(seed,case,x,rng.normal(size=(5,4))*scale,cv=3,refinements=2)
        print(f'seed {seed}: graphical CV passed', flush=True)
    path = Path(inspect.getfile(GraphicalLassoCV)); root = path.parent.parent
    sources = [path, path.parent / '_empirical_covariance.py', root / 'model_selection/_split.py',
               root / 'linear_model/_cd_fast.pyx']
    report = dict(sklearn=sklearn.__version__, profile=args.profile, fixtures=fixtures, checks=checks,
                  source_warnings=source_warnings, endpoint_certificates=endpoint_certificates,
                  source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    for certificate in endpoint_certificates:
        for name,value in certificate.items():
            if isinstance(value,float) and not np.isfinite(value):
                certificate[name] = str(value)
    (RESULTS / f'graphical_cv_{args.profile}.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'{len(checks)} graphical-CV groups across {fixtures} fixtures passed')


if __name__ == '__main__':
    main()
