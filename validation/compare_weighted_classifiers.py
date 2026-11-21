"""Weighted logistic and Gaussian NB, including every streaming state."""
from pathlib import Path
import argparse
import copy
import json
import subprocess
import warnings
import numpy as np
import sklearn
from scipy.special import expit, logsumexp
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile=None):
    selection = ['--profile', profile] if profile else ['--flag', FLAGS]
    run = subprocess.run(['fpm', 'run', *selection, '--target', 'reference_weighted_classifiers_driver', '--runner'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    paths = [ROOT / line.strip().strip('"') for line in run.stdout.splitlines()
        if line.strip().strip('"').endswith(('reference_weighted_classifiers_driver.exe', 'reference_weighted_classifiers_driver'))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, run.stdout
    return paths[0]


def fixtures(seed):
    rng = np.random.default_rng(seed)
    for k in [1, 2, 3, 4]:
        labels = np.array([11, -7, 4, 0])[:k]
        for p in [1, 4]:
            n = 8*k+11
            centers = rng.normal(size=(k, p))
            y0 = labels[np.arange(n) % k]
            x0 = centers[np.arange(n) % k]+rng.normal(size=(n, p))
            z0 = rng.normal(size=(11, p))*2
            for kind in range(17):
                x, y, z = x0.copy(), y0.copy(), z0.copy()
                w = np.ones(n); weighted = kind not in [0, 9]
                if kind in [2, 8, 10, 11, 12, 14]:
                    w = rng.uniform(.1, 3, n); w[::5] = 0
                if kind == 3:
                    w = rng.integers(0, 5, n).astype(float)
                if kind == 4:
                    w = 10.**rng.uniform(-4, 4, n)
                if kind == 5:
                    w = rng.uniform(.2, 2, n)*1.e-12
                if kind == 6:
                    w[y == labels[-1]] = 0
                if kind == 7:
                    w[:] = 0
                declared = labels[::-1].copy()
                priors = None
                if kind in [8, 11, 12]:
                    count = k + int(kind == 12)
                    priors = rng.uniform(.1, 1, count)
                    if kind == 8 and k > 1:
                        priors[0] = 0
                    priors /= priors.sum()
                    if kind == 11:
                        priors *= 1.000005  # Source np.isclose accepts this without renormalizing.
                smoothing = 1.e-9
                cuts = []
                refit = False
                if kind >= 9 and kind not in [13, 15, 16]:
                    order = np.argsort(y, kind='stable'); x, y, w = x[order], y[order], w[order]
                    cuts = [max(2, n//4), n//2, n//2+1, n]
                    refit = True
                    if kind in [10, 12]:
                        cuts = [1, 4, n//2, n]
                    if kind == 9:
                        smoothing = 0.
                    if kind in [10, 11, 14]:
                        smoothing = .02
                    if kind == 11:
                        w[:cuts[0]] *= 1.e-12
                    if kind == 12:
                        declared = np.r_[declared, 999, declared[0]]
                        refit = False
                    if kind == 14:
                        x[:n//2] *= .001; x[n//2:] *= 20
                if kind in [13, 15]:
                    x[:] = 0; z[:] = 0; w = rng.uniform(.1, 3, n)*(1.e-100 if kind == 13 else 1.e150)
                if kind == 16:
                    w = rng.uniform(.1, 3, n)*1.e150
                yield x, y, z, w, declared, priors, cuts, dict(kind=kind, k=k, p=p, weighted=weighted,
                    smoothing=smoothing, refit=refit, logistic=k>1 and w.sum()>0,
                    intercept=(kind%3 != 1 or kind in [13, 15]) and kind != 16, c=1.e-310 if kind in [15, 16] else [.3, 1., 2.][kind%3])

    if seed == 314159:
        pinned = json.loads((ROOT/'validation/fixtures/logistic_tiny_weight_line_search.json').read_text())
        x, y, z, w = (np.array(pinned[name]) for name in ['x', 'y', 'query', 'weights'])
        yield x, y.astype(int), z, w, np.array(pinned['classes']), None, [], dict(kind=17, k=2, p=4,
            weighted=True, smoothing=1.e-9, refit=False, logistic=True, intercept=True, c=pinned['c'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--profile')
    args = parser.parse_args()
    executable = args.executable.resolve() if args.executable else resolve(args.profile)
    RESULTS.mkdir(exist_ok=True)
    checks, limits, gauges, recoveries = [], [], [], []
    cases = 0
    for seed in [0, 7, 42, 2026, 314159]:
        for x, y, z, w, declared, priors, cuts, cfg in fixtures(seed):
            n, p = x.shape; m = len(z); unique = np.unique(declared)
            data, output = RESULTS/'weighted_classifiers_input.txt', RESULTS/'weighted_classifiers_actual.txt'
            with data.open('w') as stream:
                stream.write(f'{n} {p} {m} {len(unique)} {len(declared)} {len(cuts)} {int(cfg["weighted"])} '
                    f'{int(priors is not None)} {int(cfg["logistic"])} {int(cfg["intercept"])} '
                    f'{int(cfg["refit"])} {cfg["c"]} {cfg["smoothing"]}\n')
                np.savetxt(stream, y[None], fmt='%d'); np.savetxt(stream, declared[None], fmt='%d')
                np.savetxt(stream, w[None], fmt='%.17g')
                np.savetxt(stream, (priors if priors is not None else np.ones(len(unique))/len(unique))[None], fmt='%.17g')
                if cuts:
                    np.savetxt(stream, np.array(cuts)[None], fmt='%d')
                np.savetxt(stream, x, fmt='%.17g'); np.savetxt(stream, z, fmt='%.17g')
            run = subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            context = (seed, cfg)
            assert run.returncode == 0, (context, run.stdout)
            lines = iter(output.read_text().splitlines())
            def vector(dtype=float):
                return np.fromstring(next(lines), sep=' ', dtype=dtype)
            def matrix(rows):
                return np.array([vector() for _ in range(rows)])
            def record(name, oracle='sklearn'):
                checks.append(dict(seed=seed, classes=cfg['k'], features=p, kind=cfg['kind'],
                    name=name, oracle=oracle, passed=True))
            states = []
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                nb = GaussianNB(priors=priors, var_smoothing=cfg['smoothing'])
                if cuts:
                    first = 0
                    for stage, last in enumerate(cuts):
                        nb.partial_fit(x[first:last], y[first:last], classes=declared if stage == 0 else None,
                            sample_weight=w[first:last] if cfg['weighted'] else None)
                        states.append(copy.deepcopy(nb)); first = last
                    if cfg['refit']:
                        nb.fit(x, y, sample_weight=w if cfg['weighted'] else None); states.append(copy.deepcopy(nb))
                else:
                    nb.fit(x, y, sample_weight=w if cfg['weighted'] else None); states.append(nb)
                for stage, expected in enumerate(states):
                    k, actual_p = vector(int); assert actual_p == p
                    actual = dict(classes=vector(int), counts=vector(), mean=matrix(k), variance=matrix(k),
                        prior=vector(), epsilon=vector(), probability=matrix(m), log_probability=matrix(m), prediction=vector(int))
                    wanted = dict(classes=expected.classes_, counts=expected.class_count_, mean=expected.theta_,
                        variance=expected.var_, prior=expected.class_prior_, epsilon=[expected.epsilon_],
                        probability=expected.predict_proba(z), log_probability=expected.predict_log_proba(z),
                        prediction=expected.predict(z))
                    for name, value in wanted.items():
                        if name in ['classes', 'prediction']:
                            np.testing.assert_array_equal(actual[name], value, err_msg=str((context, stage, name)))
                        else:
                            tolerance = 3.e-8 if 'probability' in name else 4.e-11
                            np.testing.assert_allclose(actual[name], value, atol=0 if name in ['counts', 'epsilon'] else tolerance, rtol=tolerance,
                                equal_nan=True, err_msg=str((context, stage, name)))
                        record(f'gaussian_{stage}_{name}')
                if cfg['logistic']:
                    q = int(vector(int)[0]); coef, intercept = matrix(q), vector()
                    decision, probability, prediction = matrix(m), matrix(m), vector(int)
                    iterations, converged = next(lines).split()
                    assert 0 < int(iterations) <= 500 and converged in ['T', 'F']
                    expected = LogisticRegression(C=cfg['c'], fit_intercept=cfg['intercept'],
                        tol=1.e-12, max_iter=1000, solver='newton-cg').fit(x, y,
                        sample_weight=w if cfg['weighted'] else None)
                    classes = expected.classes_; k = len(classes); mass = w.sum()
                    def certificate(beta, bias):
                        raw = x@beta.T+bias
                        penalty_gradient = beta/cfg['c']/mass
                        if k == 2:
                            target = (y == classes[1]).astype(float)
                            fitted = expit(raw[:, 0]); residual = (fitted-target)*w/mass
                            gradient = np.r_[x.T@residual+penalty_gradient[0],
                                residual.sum() if cfg['intercept'] else 0]
                            losses = np.logaddexp(0, raw[:, 0])-target*raw[:, 0]
                        else:
                            target = (y[:, None] == classes).astype(float)
                            fitted = np.exp(raw-logsumexp(raw, axis=1, keepdims=True))
                            residual = (fitted-target)*w[:, None]/mass
                            gradient = np.r_[(residual.T@x+penalty_gradient).ravel(),
                                residual.sum(axis=0) if cfg['intercept'] else np.zeros(k)]
                            losses = logsumexp(raw, axis=1)-np.sum(target*raw, axis=1)
                        loss = (w/mass)@losses+.5*np.sum(beta*penalty_gradient)
                        return float(max(abs(gradient))), float(loss)
                    native_gradient, native_loss = certificate(coef, intercept)
                    assert native_gradient <= 3.e-8, (context, native_gradient, converged)
                    source_gradient, source_loss = certificate(expected.coef_, expected.intercept_)
                    positive_support = all(w[y == c].sum() > 0 for c in classes)
                    source_oracle = 'sklearn'
                    if positive_support and source_gradient > max(1.e-8, 100*max(native_gradient, 1.e-12)):
                        recovered = LogisticRegression(C=cfg['c'], fit_intercept=cfg['intercept'],
                            tol=1.e-12, max_iter=1000, solver='newton-cholesky').fit(x, y,
                            sample_weight=w if cfg['weighted'] else None)
                        recovered_gradient, recovered_loss = certificate(recovered.coef_, recovered.intercept_)
                        assert recovered_gradient <= 3.e-8 and native_loss <= source_loss+1.e-12
                        assert abs(native_loss-recovered_loss) <= 1.e-10
                        recoveries.append(dict(seed=seed, classes=cfg['k'], features=p, kind=cfg['kind'],
                            native_gradient=native_gradient, source_cg_gradient=source_gradient,
                            source_cholesky_gradient=recovered_gradient, native_loss=native_loss,
                            source_cg_loss=source_loss, source_cholesky_loss=recovered_loss))
                        expected = recovered; source_oracle = 'sklearn_newton_cholesky'
                    record('logistic_gradient', 'independent_optimality')
                    np.testing.assert_allclose(coef, expected.coef_, atol=3.e-6, rtol=3.e-6, err_msg=str(context))
                    record('logistic_coef', source_oracle)
                    positive_support = all(w[y == c].sum() > 0 for c in classes)
                    if positive_support:
                        source_intercept = expected.intercept_.copy()
                        source_decision = expected.decision_function(z).reshape(m, -1)
                        oracle = source_oracle
                        if q > 1 and not np.allclose(intercept, source_intercept, atol=3.e-6, rtol=3.e-6):
                            offset = source_intercept.mean()
                            source_intercept -= offset; source_decision -= offset
                            assert abs(intercept.sum()) <= 1.e-10
                            gauges.append(dict(seed=seed, classes=cfg['k'], features=p, kind=cfg['kind'],
                                source_offset=float(offset)))
                            oracle = 'source_softmax_gauge'
                        np.testing.assert_allclose(intercept, source_intercept, atol=3.e-6, rtol=3.e-6, err_msg=str(context))
                        np.testing.assert_allclose(decision, source_decision, atol=3.e-6, rtol=3.e-6, err_msg=str(context))
                        record('logistic_intercept', oracle); record('logistic_decision', oracle)
                    else:
                        limits.append(dict(seed=seed, classes=cfg['k'], features=p, kind=cfg['kind'],
                            gradient=native_gradient))
                        record('logistic_intercept', 'limiting_optimum')
                        record('logistic_decision', 'limiting_optimum')
                    np.testing.assert_allclose(probability, expected.predict_proba(z), atol=3.e-6, rtol=3.e-6, err_msg=str(context))
                    np.testing.assert_array_equal(prediction, expected.predict(z), err_msg=str(context))
                    record('logistic_probability', source_oracle); record('logistic_prediction', source_oracle)
            np.testing.assert_array_equal(vector(int), y); np.testing.assert_array_equal(matrix(n), x)
            np.testing.assert_array_equal(matrix(m), z); np.testing.assert_array_equal(vector(), w)
            assert list(lines) == []
            record('input_preserved', 'input_identity'); cases += 1
        print(f'seed {seed}: weighted/streaming classifiers passed', flush=True)
    (RESULTS/'weighted_classifiers.json').write_text(json.dumps(dict(sklearn=sklearn.__version__, fixtures=cases,
        checks=checks, limiting_optima=limits, softmax_gauges=gauges, source_solver_recoveries=recoveries), indent=2)+'\n')
    print(f'{len(checks)} weighted/streaming classifier groups across {cases} fixtures passed')


if __name__ == '__main__':
    main()
