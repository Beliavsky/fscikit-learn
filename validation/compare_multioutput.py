"""Independent-target fits, streaming clones, ragged probabilities and scores."""
from pathlib import Path
import argparse
import copy
import json
import os
import subprocess
import warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
import numpy as np
import sklearn
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, LogisticRegression, SGDRegressor, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.utils._array_api import _average

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile=None):
    selection = ['--profile', profile] if profile else ['--flag', FLAGS]
    run = subprocess.run(['fpm', 'run', *selection, '--target', 'reference_multioutput_driver', '--runner'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    paths = [ROOT/line.strip().strip('"') for line in run.stdout.splitlines()
        if line.strip().strip('"').endswith(('reference_multioutput_driver.exe', 'reference_multioutput_driver'))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, run.stdout
    return paths[0]


def fixtures(seed):
    rng = np.random.default_rng(seed)
    for mode in [0, 1]:
        for base in range(6 if mode == 0 else 8):
            for kind in range(10):
                stream = kind in [3, 4, 5, 6, 9]
                weighted = kind in [2, 4, 5]
                if mode == 0:
                    if stream and base != 3 or weighted and base != 3 or kind in [7, 8, 9]:
                        continue
                else:
                    if stream and base not in [1, 2] or weighted and base not in [0, 1, 2]:
                        continue
                    if kind in [6, 8, 9] and base != 2:
                        continue
                    if kind == 7 and base in [0, 2]:
                        continue
                n, m, t = 48, 13, 3
                if kind == 1:
                    m, t = 1, 1
                p = 1 if (mode == 0 and base == 5 or mode == 1 and base == 4) else 3
                x = rng.normal(size=(n, p))
                query = rng.normal(size=(m, p))
                if p == 1:
                    x = np.linspace(-3, 3, n).reshape(-1, 1)
                    query *= 2
                weights = rng.uniform(.1, 2, n) if weighted else np.ones(n)
                if weighted:
                    weights[::7] = 0
                score_weights = rng.uniform(.1, 3, m); score_weights[::4] = 0
                if not np.any(score_weights):
                    score_weights[0] = 1
                average = 3 if kind in [1, 6] else 0
                intercept = not (mode == 1 and average > 0)
                probability = kind not in [8, 9]
                cuts = [1, 9, 25, n] if stream else []
                if mode == 0:
                    coef = rng.normal(size=(p, t))
                    y = x@coef+rng.normal(size=(n, t))*.1+np.arange(t)
                    qy = query@coef+np.arange(t)
                    if t > 1:
                        qy[:, -1] = 4  # An imperfect constant query target exercises force_finite.
                    if base == 5:
                        y = np.column_stack([np.sin(x[:, 0]*(j+1))+j for j in range(t)])
                    estimator = [LinearRegression(), Ridge(alpha=.7, solver='svd'),
                        ElasticNet(alpha=.03, l1_ratio=.4, tol=1.e-12, max_iter=20000),
                        SGDRegressor(learning_rate='constant', eta0=.02, alpha=.001, max_iter=4,
                            tol=None, shuffle=False, average=average, fit_intercept=intercept, random_state=0),
                        KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='brute'),
                        DecisionTreeRegressor(max_depth=3, random_state=0)][base]
                else:
                    label_lists = [np.array([-7, 11]), np.array([0, 4, 19]), np.array([-9, 2, 5, 31])][:t]
                    y = np.column_stack([labels[np.arange(n) % len(labels)] for labels in label_lists])
                    qy = np.column_stack([labels[rng.integers(0, len(labels), m)] for labels in label_lists])
                    if base == 4:
                        y = np.column_stack([labels[np.minimum(np.arange(n)*len(labels)//n, len(labels)-1)]
                                             for labels in label_lists])
                    if base >= 5:
                        x = rng.poisson(2, size=(n, p)).astype(float)
                        query = rng.poisson(2, size=(m, p)).astype(float)
                    if kind == 7:
                        y[:, -1] = 77; qy[:, -1] = 77
                    estimator = [LogisticRegression(C=.7, tol=1.e-10, max_iter=500, solver='newton-cg'),
                        GaussianNB(), SGDClassifier(loss='log_loss' if probability else 'hinge',
                            learning_rate='constant', eta0=.02, alpha=.001, max_iter=4, tol=None,
                            shuffle=False, average=average, fit_intercept=intercept, random_state=0),
                        KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='brute'),
                        DecisionTreeClassifier(max_depth=4, random_state=0), MultinomialNB(alpha=.4),
                        BernoulliNB(alpha=.4), ComplementNB(alpha=.4, norm=True)][base]
                yield x, y, query, qy, weights, score_weights, cuts, estimator, dict(mode=mode, base=base,
                    kind=kind, weighted=weighted, refit=kind == 5, shrink=kind == 5, probability=probability,
                    average=average, intercept=intercept)


def certify_cart(nodes, x, y, query):
    """Certify greedy squared-error splits using source stumps and SSE scans."""
    assert x.shape[1] == 1
    ties = 0
    visited = set()
    def sse(values):
        return np.sum((values-values.mean())**2)
    def walk(index, rows, depth):
        nonlocal ties
        assert index not in visited and 0 <= index < len(nodes)
        visited.add(index)
        left, right, feature, count, recorded_depth = nodes[index, :5].astype(int)
        threshold, value, impurity = nodes[index, 5:]
        assert count == len(rows) and depth == recorded_depth
        np.testing.assert_allclose([value, impurity], [y[rows].mean(), sse(y[rows])/len(rows)], atol=3.e-12, rtol=3.e-12)
        stump = DecisionTreeRegressor(max_depth=1, random_state=0).fit(x[rows], y[rows])
        if left == 0:
            assert right == 0 and (depth == 3 or len(rows) < 2 or stump.tree_.node_count == 1)
            return
        assert depth < 3 and feature == 1 and left > 0 and right > 0
        chosen = x[rows, 0] <= threshold
        assert chosen.any() and not chosen.all()
        ordered = np.sort(np.unique(x[rows, 0]))
        candidates = (ordered[:-1]+ordered[1:])/2
        costs = []
        for cut in candidates:
            mask = x[rows, 0] <= cut
            costs.append(sse(y[rows[mask]])+sse(y[rows[~mask]]))
        chosen_cost = sse(y[rows[chosen]])+sse(y[rows[~chosen]])
        bound = 128*np.finfo(float).eps*max(1., np.sum(y[rows]**2))
        assert chosen_cost <= min(costs)+bound
        source_mask = x[rows, 0].astype(np.float32) <= stump.tree_.threshold[0]
        assert abs(chosen_cost-(sse(y[rows[source_mask]])+sse(y[rows[~source_mask]]))) <= bound
        if not np.array_equal(chosen, source_mask):
            ties += 1
        walk(left-1, rows[chosen], depth+1); walk(right-1, rows[~chosen], depth+1)
    walk(0, np.arange(len(x)), 0)
    assert len(visited) == len(nodes)
    prediction = []
    for row in query:
        index = 0
        while nodes[index, 0] != 0:
            index = int(nodes[index, 0 if row[0] <= nodes[index, 5] else 1])-1
        prediction.append(nodes[index, 6])
    return np.array(prediction), ties


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--executable', type=Path); parser.add_argument('--profile')
    args = parser.parse_args()
    executable = args.executable.resolve() if args.executable else resolve(args.profile)
    RESULTS.mkdir(exist_ok=True); checks = []; score_certificates = []; cart_certificates = []; class_tie_certificates = []; cases = 0
    for seed in [0, 7, 42, 2026, 314159]:
        for x, y, query, qy, w, sw, cuts, estimator, cfg in fixtures(seed):
            n, p = x.shape; m, t = qy.shape; mode = cfg['mode']; context = (seed, cfg)
            classes = [np.unique(y[:, j])[::-1] if mode else np.array([0, 1]) for j in range(t)]
            data, output = RESULTS/'multioutput_input.txt', RESULTS/'multioutput_actual.txt'
            with data.open('w') as stream:
                stream.write(' '.join(map(str, [mode, cfg['base'], n, p, m, t, len(cuts), int(cfg['weighted']),
                    int(cfg['refit']), int(cfg['shrink']), int(cfg['probability']), cfg['average'], int(cfg['intercept'])]))+'\n')
                np.savetxt(stream, w[None], fmt='%.17g'); np.savetxt(stream, sw[None], fmt='%.17g')
                if cuts:
                    np.savetxt(stream, np.array(cuts)[None], fmt='%d')
                for labels in classes:
                    stream.write(str(len(labels))+'\n'); np.savetxt(stream, labels[None], fmt='%d')
                for a in [x, query]:
                    np.savetxt(stream, a, fmt='%.17g')
                for a in [y, qy]:
                    np.savetxt(stream, a, fmt='%d' if mode else '%.17g')
            run = subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert run.returncode == 0, (context, run.stdout)
            lines = iter(output.read_text().splitlines())
            def vector(dtype=float):
                return np.fromstring(next(lines), sep=' ', dtype=dtype)
            def matrix(rows, dtype=float):
                return np.array([vector(dtype) for _ in range(rows)])
            def record(name, stage=0, oracle='sklearn'):
                checks.append(dict(seed=seed, mode=mode, base=cfg['base'], kind=cfg['kind'], stage=stage,
                    name=name, oracle=oracle, passed=True))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                estimator.fit(x, y[:, 0]); before = estimator.predict(query)
                wrapper = MultiOutputClassifier(estimator) if mode else MultiOutputRegressor(estimator)
                states = []
                if cuts:
                    first = 0
                    for stage, last in enumerate(cuts):
                        kwargs = dict(sample_weight=w[first:last] if cfg['weighted'] else None)
                        if mode:
                            kwargs['classes'] = classes
                        wrapper.partial_fit(x[first:last], y[first:last], **kwargs)
                        states.append(copy.deepcopy(wrapper)); first = last
                    if cfg['shrink'] and t > 1:
                        wrapper.partial_fit(x, y[:, :t-1]); states.append(copy.deepcopy(wrapper))
                    if cfg['refit']:
                        wrapper.fit(x, y, sample_weight=w if cfg['weighted'] else None)
                        states.append(copy.deepcopy(wrapper))
                else:
                    wrapper.fit(x, y, sample_weight=w if cfg['weighted'] else None); states.append(wrapper)
                for stage, expected in enumerate(states):
                    count, features, top = vector(int)
                    assert count == len(expected.estimators_) and features == p
                    assert top == (len(expected.classes_) if hasattr(expected, 'classes_') else 0)
                    record('metadata', stage)
                    prediction = matrix(m, int if mode else float)
                    score = vector()
                    coefficients, tree_nodes = [], []
                    if not mode:
                        for j, model in enumerate(expected.estimators_):
                            header = vector(int); flag = int(header[0])
                            if flag == 1:
                                coefficients.append((j, vector()))
                            elif flag == 2:
                                tree_nodes.append((j, matrix(int(header[1]))))
                    reference_prediction = expected.predict(query)
                    prediction_oracle = 'sklearn'
                    if tree_nodes:
                        reconstructed = np.empty_like(prediction)
                        ties = 0
                        for j, nodes in tree_nodes:
                            reconstructed[:, j], local_ties = certify_cart(nodes, x, y[:, j], query)
                            ties += local_ties
                        np.testing.assert_allclose(prediction, reconstructed, atol=3.e-12, rtol=3.e-12)
                        if not np.allclose(prediction, reference_prediction, atol=3.e-8, rtol=3.e-8):
                            assert ties > 0
                            cart_certificates.append(dict(seed=seed, base=cfg['base'], kind=cfg['kind'], stage=stage, ties=ties))
                            reference_prediction = reconstructed
                            prediction_oracle = 'source_cart_tie_certificate'
                    if mode:
                        if not np.array_equal(prediction, reference_prediction):
                            assert cfg['base'] == 7
                            for row, column in np.argwhere(prediction != reference_prediction):
                                model = expected.estimators_[column]
                                scores = model.predict_log_proba(query[row:row+1])[0]
                                chosen = np.flatnonzero(model.classes_ == prediction[row, column])
                                assert len(chosen) == 1
                                bound = 64*np.finfo(float).eps*max(1., np.max(abs(scores)))
                                assert scores.max()-scores[chosen[0]] <= bound
                                class_tie_certificates.append(dict(seed=seed, kind=cfg['kind'], stage=stage,
                                    row=int(row), column=int(column), gap=float(scores.max()-scores[chosen[0]]), bound=float(bound)))
                            reference_prediction = prediction.copy()
                            prediction_oracle = 'source_complement_nb_tie_certificate'
                        np.testing.assert_array_equal(prediction, reference_prediction, err_msg=str((context, stage)))
                    else:
                        np.testing.assert_allclose(prediction, reference_prediction, atol=3.e-8, rtol=3.e-8,
                            err_msg=str((context, stage)))
                    record('prediction', stage, prediction_oracle)
                    if mode:
                        wanted = [np.mean(np.all(reference_prediction == qy[:, :count], axis=1))]
                    else:
                        wanted = [r2_score(qy[:, :count], reference_prediction),
                                  r2_score(qy[:, :count], reference_prediction, sample_weight=sw)]
                    score_oracle = prediction_oracle
                    if not mode and not np.allclose(score, wanted, atol=3.e-8, rtol=3.e-8, equal_nan=True):
                        target = qy[:, :count]; center = target[0]
                        constant = np.ptp(target, axis=0) == 0
                        assert np.any(constant), (context, stage)
                        source_mean = _average(target, axis=0, weights=sw)
                        source_denominator = np.sum(sw[:, None]*(target-source_mean)**2, axis=0)
                        bound = 64*np.finfo(float).eps**2*sw.sum()*np.maximum(1., abs(center))**2
                        assert np.all(source_denominator[constant] <= bound[constant])
                        shifted = [r2_score(target-center, reference_prediction-center),
                            r2_score(target-center, reference_prediction-center, sample_weight=sw)]
                        np.testing.assert_allclose(score, shifted, atol=3.e-8, rtol=3.e-8,
                            equal_nan=True, err_msg=str((context, stage, 'translation invariant R2')))
                        score_certificates.append(dict(seed=seed, mode=mode, base=cfg['base'], kind=cfg['kind'],
                            stage=stage, constant_columns=np.flatnonzero(constant).tolist(),
                            source_denominator=source_denominator[constant].tolist(), bound=bound[constant].tolist()))
                        score_oracle = 'source_constant_r2_identity'
                    else:
                        np.testing.assert_allclose(score, wanted, atol=3.e-8, rtol=3.e-8, equal_nan=True,
                            err_msg=str((context, stage, 'score')))
                    record('score', stage, score_oracle)
                    if mode and cfg['probability']:
                        source_probs = expected.predict_proba(query)
                        for j, (model, source_probability) in enumerate(zip(expected.estimators_, source_probs)):
                            k = int(vector(int)[0]); labels = vector(int); probability = matrix(m)
                            assert k == len(model.classes_)
                            np.testing.assert_array_equal(labels, model.classes_)
                            np.testing.assert_allclose(probability, source_probability, atol=3.e-7, rtol=3.e-7,
                                equal_nan=True, err_msg=str((context, stage, j)))
                            record(f'probability_{j}', stage)
                    if not mode:
                        for j, actual in coefficients:
                            model = expected.estimators_[j]
                            time = model.t_ if hasattr(model, 't_') else 0
                            np.testing.assert_allclose(actual, np.r_[model.coef_, model.intercept_, time],
                                atol=3.e-8, rtol=3.e-8, err_msg=str((context, stage, j)))
                            record(f'coef_intercept_time_{j}', stage)
                original_before, original_after = vector(), vector()
                np.testing.assert_array_equal(original_before, original_after)
                if mode:
                    np.testing.assert_array_equal(original_before, before)
                else:
                    if cfg['base'] == 5:
                        before = prediction[:, 0]
                    np.testing.assert_allclose(original_before, before, atol=3.e-8, rtol=3.e-8, err_msg=str(context))
                record('base_preserved', oracle='input_identity')
            np.testing.assert_array_equal(matrix(n), x); np.testing.assert_array_equal(matrix(m), query)
            np.testing.assert_array_equal(vector(), w); np.testing.assert_array_equal(matrix(n, int if mode else float), y)
            assert list(lines) == []; record('data_preserved', oracle='input_identity'); cases += 1
        print(f'seed {seed}: multioutput passed', flush=True)
    (RESULTS/'multioutput.json').write_text(json.dumps(dict(sklearn=sklearn.__version__, fixtures=cases,
        checks=checks, constant_score_certificates=score_certificates, cart_certificates=cart_certificates, class_tie_certificates=class_tie_certificates), indent=2)+'\n')
    print(f'{len(checks)} multioutput groups across {cases} fixtures passed')


if __name__ == '__main__':
    main()
