"""Voting ensembles against sklearn 1.6.1 and source combination of native base responses."""
from pathlib import Path
import argparse
import copy
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import (LogisticRegression, SGDClassifier, Perceptron, PassiveAggressiveClassifier,
    RidgeClassifier, RidgeClassifierCV, LinearRegression, Ridge, SGDRegressor, ElasticNet, Lasso,
    PassiveAggressiveRegressor)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.dummy import DummyClassifier, DummyRegressor
from compare_classifier_chains import certify_tree, certify_complement_labels

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile=None):
    opts = ['--profile', profile] if profile else ['--flag', FLAGS]
    run = subprocess.run(['fpm', 'run', *opts, '--target', 'reference_voting_driver', '--runner'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    candidates = [ROOT/line.strip().strip('"') for line in run.stdout.splitlines()
        if line.strip().strip('"').endswith(('reference_voting_driver', 'reference_voting_driver.exe'))]
    candidates = [p.resolve() for p in candidates if p.is_file()]
    assert len(candidates) == 1, run.stdout
    return candidates[0]


def base_model(mode, code, case):
    if mode < 2:
        return [LogisticRegression(C=.7, tol=1.e-10, max_iter=500, solver='newton-cg'), GaussianNB(),
            SGDClassifier(loss='modified_huber' if case == 3 else 'log_loss', learning_rate='constant',
                eta0=.02, alpha=.001, max_iter=4, tol=None, shuffle=False, average=3 if case == 3 else False, random_state=0),
            KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='brute'),
            DecisionTreeClassifier(max_depth=2, min_samples_leaf=4, random_state=0),
            MultinomialNB(alpha=.4), BernoulliNB(alpha=.4), ComplementNB(alpha=.4, norm=True),
            Perceptron(eta0=.1, max_iter=4, tol=None, shuffle=False, random_state=0),
            PassiveAggressiveClassifier(C=.7, max_iter=4, tol=None, shuffle=False, random_state=0),
            RidgeClassifier(alpha=.7), LinearSVC(C=.7, tol=1.e-10, max_iter=10000, dual=False),
            SVC(C=.7, gamma=.4, tol=1.e-8, max_iter=100000, shrinking=False),
            RidgeClassifierCV(alphas=[.1, .7, 2.]),
            DummyClassifier(strategy='constant' if case == 3 else 'prior', constant=0)][code]
    return [LinearRegression(), Ridge(alpha=.7), SGDRegressor(learning_rate='constant', eta0=.02,
        alpha=.001, max_iter=4, tol=None, shuffle=False, average=3 if case == 3 else False, random_state=0),
        KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='brute'),
        DecisionTreeRegressor(max_depth=2, min_samples_leaf=4, random_state=0),
        ElasticNet(alpha=.05, l1_ratio=.3, tol=1.e-10, max_iter=10000),
        Lasso(alpha=.05, tol=1.e-10, max_iter=10000),
        PassiveAggressiveRegressor(C=.7, max_iter=4, tol=None, shuffle=False, random_state=0),
        KernelRidge(kernel='rbf', gamma=.4, alpha=.7),
        DummyRegressor(strategy='constant' if case == 3 else 'mean', constant=2.)][code]


def fixtures(seed):
    rng = np.random.default_rng(seed)
    for mode in range(3):
        for base in range(15 if mode < 2 else 10):
            if mode == 1 and base in [8, 9, 10, 11, 12, 13]:
                continue
            for case in range(9):
                if case == 4 and base not in ([0, 1, 2, 10, 11, 13] if mode < 2 else [2]):
                    continue
                if case == 5 and mode < 2 and base not in [1, 3, 4, 5, 6, 7, 14]:
                    continue
                if case == 8 and (base != 1 or mode == 2):
                    continue
                n, p, m = 48, 3, (1 if case == 6 else 11)
                k = 1 if case == 5 else 2 if case == 1 else 3
                ids = np.arange(n) % k; classes = np.array([-13, 4, 29])[:k]
                centers = rng.normal(size=(k, p))*1.3
                x = rng.normal(size=(n, p))*.6+centers[ids]
                qids = np.arange(m) % k; q = rng.normal(size=(m, p))*.6+centers[qids]
                y = classes[ids] if mode < 2 else x@np.array([.7, -1.2, .4])+.2*rng.normal(size=n)
                truth = classes[qids] if mode < 2 else q@np.array([.7, -1.2, .4])+.2*rng.normal(size=m)
                if mode == 2 and case == 5:
                    y = np.full(n, 2.); truth = np.full(m, 2.)
                if mode < 2 and base in [5, 6, 7]:
                    rates = np.exp(np.clip(centers*.4, -1., 1.5))
                    x = rng.poisson(2*rates[ids]).astype(float); q = rng.poisson(2*rates[qids]).astype(float)
                codes = [base]; drops = [0]; vw = [1.]
                if case in [2, 3, 4, 7]:
                    other = (1 if base != 1 else 0) if mode < 2 else (2 if case == 4 else 1)
                    codes = [base, -1, other, base]; drops = [0, 1, 0, 0]
                    vw = [-1., 123., 2., 1.] if case == 3 else [2., 123., .5, 1.]
                if case == 8:
                    codes = [base, base]; drops = [0, 0]; vw = [1., 2.]
                weighted = case in [4, 8]
                w = rng.uniform(.1, 2., n) if weighted else np.ones(n)
                if weighted:
                    w[::7] = 0
                if case == 8:
                    w[:] = 0
                sw = rng.uniform(.2, 2., m); sw[0] = -.1
                if case == 6:
                    sw[:] = -.5
                if case == 1 and mode == 0:
                    vw = [0.]  # hard voting permits zero active weight mass
                yield dict(mode=mode, base=base, case=case, seed=seed, weighted=weighted,
                    vote_weights=case not in [0, 5, 6], refit=case == 7, codes=codes, drops=drops, weights=vw), x, y, q, truth, w, sw


def certify_regression_tree(nodes, x, y, query):
    """Certify each native greedy split against exhaustive SSE and source stumps."""
    visited = set(); ties = 0
    def sse(a):
        return np.sum((a-a.mean())**2)
    def walk(index, rows, depth):
        nonlocal ties
        assert 0 <= index < len(nodes) and index not in visited
        visited.add(index)
        left, right, feature, count, recorded_depth = nodes[index, :5].astype(int)
        threshold, value, impurity = nodes[index, 5:]
        assert count == len(rows) and depth == recorded_depth
        np.testing.assert_allclose([value, impurity], [y[rows].mean(), sse(y[rows])/len(rows)],
            atol=3.e-12, rtol=3.e-12)
        stump = DecisionTreeRegressor(max_depth=1, min_samples_leaf=4, random_state=0).fit(x[rows], y[rows])
        if left == 0:
            assert right == 0 and (depth == 2 or stump.tree_.node_count == 1)
            return
        assert depth < 2 and left > 0 and right > 0 and 1 <= feature <= x.shape[1]
        mask = x[rows, feature-1] <= threshold
        assert mask.sum() >= 4 and (~mask).sum() >= 4
        costs = []; candidates = []
        for column in range(x.shape[1]):
            ordered = np.sort(np.unique(x[rows, column]))
            for cut in (ordered[:-1]+ordered[1:])/2:
                partition = x[rows, column] <= cut
                if partition.sum() >= 4 and (~partition).sum() >= 4:
                    cost = sse(y[rows[partition]])+sse(y[rows[~partition]])
                    costs.append(cost); candidates.append((cost, column, partition.copy()))
        loss = sse(y[rows[mask]])+sse(y[rows[~mask]])
        bound = 128*np.finfo(float).eps*max(1., np.sum(y[rows]**2))
        assert loss <= min(costs)+bound
        source_feature = stump.tree_.feature[0]
        assert source_feature >= 0
        source_mask = x[rows, source_feature].astype(np.float32) <= stump.tree_.threshold[0]
        assert abs(loss-(sse(y[rows[source_mask]])+sse(y[rows[~source_mask]]))) <= bound
        alternatives = sum(abs(value-loss) <= bound and (column != feature-1 or not np.array_equal(partition, mask))
                           for value, column, partition in candidates)
        if alternatives:
            ties += 1
        if feature-1 != source_feature or not np.array_equal(mask, source_mask):
            assert alternatives > 0
        walk(left-1, rows[mask], depth+1); walk(right-1, rows[~mask], depth+1)
    walk(0, np.arange(len(x)), 0)
    assert len(visited) == len(nodes)
    prediction = []
    for row in query:
        index = 0
        while nodes[index, 0] != 0:
            feature = int(nodes[index, 2])-1
            index = int(nodes[index, 0 if row[feature] <= nodes[index, 5] else 1])-1
        prediction.append(nodes[index, 6])
    return np.array(prediction), ties



def certify_classifier_prediction(actual, source, q):
    if isinstance(source, ComplementNB):
        witnesses = certify_complement_labels(actual.astype(int), source, q)
    else:
        raw = source.decision_function(q) if hasattr(source, 'decision_function') else source.predict_proba(q)
        if raw.ndim == 1: raw = np.column_stack([-raw, raw])
        witnesses = []
        for row in np.flatnonzero(actual != wanted):
            chosen = np.flatnonzero(source.classes_ == actual[row])[0]
            gap = raw[row].max()-raw[row, chosen]
            bound = 64*np.finfo(float).eps*max(1., abs(raw[row]).max())
            assert 0 <= gap <= bound, (context, slot, row, gap, bound)
            witnesses.append(dict(row=int(row), gap=float(gap), bound=float(bound)))
    return witnesses


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--executable', type=Path); parser.add_argument('--profile')
    parser.add_argument('--seed', action='append', type=int); args = parser.parse_args()
    executable = args.executable.resolve() if args.executable else resolve(args.profile)
    RESULTS.mkdir(exist_ok=True); checks = []; certificates = []; fixture_count = 0
    for seed in args.seed or [0, 7, 42, 2026, 314159]:
        for cfg, x, y, q, truth, w, sw in fixtures(seed):
            mode, base, case = cfg['mode'], cfg['base'], cfg['case']; context = (seed, mode, base, case)
            n, p = x.shape; m = len(q); b = len(cfg['codes']); active = np.flatnonzero(np.array(cfg['drops']) == 0)
            data, output = RESULTS/'voting_input.txt', RESULTS/'voting_actual.txt'
            with data.open('w') as stream:
                stream.write(' '.join(map(str, [mode, n, p, m, b, int(cfg['weighted']), int(cfg['vote_weights']),
                    int(cfg['refit']), seed, case]))+'\n')
                for a in [cfg['codes'], cfg['drops'], cfg['weights'], w, sw]:
                    np.savetxt(stream, np.atleast_2d(a), fmt='%.17g')
                np.savetxt(stream, x, fmt='%.17g'); np.savetxt(stream, y[None], fmt='%.17g')
                np.savetxt(stream, q, fmt='%.17g'); np.savetxt(stream, truth[None], fmt='%.17g')
            run = subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert run.returncode == 0, (context, run.stdout)
            lines = iter(output.read_text().splitlines())
            def vector(dtype=float): return np.fromstring(next(lines), sep=' ', dtype=dtype)
            def matrix(rows): return np.array([vector() for _ in range(rows)])
            def record(name, stage=0, oracle='sklearn'):
                checks.append(dict(seed=seed, mode=mode, base=base, case=case, stage=stage, name=name, oracle=oracle, passed=True))
            def read_state():
                k = int(vector(int)[0]); labels = vector(int) if k else np.array([], int)
                flag, rows, cols = vector(int)
                if flag in [12, 13]:
                    values = [vector()]
                else:
                    values = [matrix(cols if flag in [5, 7] else rows)]
                    if flag in [1, 3, 4, 9, 11, 6, 14]: values.append(vector())
                    elif flag == 2: values.extend([matrix(rows), vector(), vector()])
                    elif flag == 8: values.extend([vector(), vector()])
                return flag, labels, values
            templates = []; native_before = []
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                encoded = np.searchsorted(np.unique(y), y) if mode < 2 else y
                pre_y = (encoded+1) % len(np.unique(y)) if mode < 2 else y+3
                for j, code in enumerate(cfg['codes']):
                    if cfg['drops'][j]:
                        templates.append((f'base_{j+1:02}', 'drop')); continue
                    estimator = base_model(mode, code, case); estimator.fit(x, pre_y)
                    templates.append((f'base_{j+1:02}', estimator))
                    state, pred = read_state(), vector(); native_before.append((state, pred))
                    wanted = estimator.predict(q); oracle = 'sklearn'
                    if state[0] in [5, 7]:
                        if state[0] == 5:
                            cp, stats = certify_tree((state[1], state[2][0]), x, pre_y, q)
                            certified = state[1][cp.argmax(axis=1)]; ties = stats['ties']
                        else:
                            certified, ties = certify_regression_tree(state[2][0], x, pre_y, q)
                        np.testing.assert_allclose(pred, certified, atol=3.e-12, rtol=3.e-12)
                        if not np.allclose(pred, wanted, atol=3.e-7, rtol=3.e-7):
                            assert ties > 0; wanted = certified; oracle = 'source_template_cart_certificate'
                            certificates.append(dict(seed=seed, mode=mode, base=base, case=case, stage=-1,
                                slot=j, kind='template_tree', ties=ties))
                    if mode < 2 and not np.array_equal(pred, wanted):
                        witness = certify_classifier_prediction(pred.astype(int), estimator, q)
                        assert witness; wanted = pred; oracle = 'source_template_label_boundary'
                        certificates.append(dict(seed=seed, mode=mode, base=base, case=case, stage=-1,
                            slot=j, kind='template_label_boundary', witnesses=witness))
                    np.testing.assert_allclose(pred, wanted, atol=3.e-7 if mode == 2 else 0,
                        rtol=3.e-7 if mode == 2 else 0, err_msg=str((context, j, 'template')))
                    record('template_prediction_'+str(j), oracle=oracle)
                cls = VotingClassifier if mode < 2 else VotingRegressor
                kwargs = dict(weights=cfg['weights'] if cfg['vote_weights'] else None)
                if mode < 2: kwargs['voting'] = 'hard' if mode == 0 else 'soft'
                wrapper = cls(templates, **kwargs)
                for stage in range(1+cfg['refit']):
                    if stage:
                        y = np.where(y == np.unique(y)[0], -31, 83) if mode < 2 else 1.3*y+2
                    wrapper.fit(x, y, **(dict(sample_weight=w) if cfg['weighted'] else {}))
                    k, fitted_count, width = vector(int)
                    labels = vector(int) if k else np.array([], int)
                    assert fitted_count == len(active) and width == p
                    if mode < 2: np.testing.assert_array_equal(labels, wrapper.classes_)
                    np.testing.assert_array_equal(vector(int), active+1); record('metadata', stage)
                    prediction, transformed = vector(), matrix(m)
                    probability = matrix(m) if mode == 1 else None
                    tensors = [matrix(m) for j in active] if mode == 1 else None
                    scores = vector(); states = []
                    for j in active:
                        assert next(lines).strip() == f'base_{j+1:02}'
                        states.append(read_state())
                    np.testing.assert_array_equal(vector(), prediction); record('copy_prediction', stage, 'input_identity')
                    proxy = copy.deepcopy(wrapper); changed = False
                    for slot, (j, source, state) in enumerate(zip(active, wrapper.estimators_, states)):
                        flag, head_labels, values = state; alternative = False
                        if mode < 2: np.testing.assert_array_equal(head_labels, source.classes_)
                        actual = tensors[slot] if mode == 1 else transformed[:, slot]
                        wanted = source.predict_proba(q) if mode == 1 else source.predict(q)
                        if flag in [5, 7]:
                            if flag == 5:
                                certified, stats = certify_tree((head_labels, values[0]), x, np.searchsorted(wrapper.classes_, y), q)
                                certified = certified if mode == 1 else head_labels[certified.argmax(axis=1)]
                                ties = stats['ties']
                            else:
                                certified, ties = certify_regression_tree(values[0], x, y, q)
                            np.testing.assert_allclose(actual, certified, atol=3.e-12, rtol=3.e-12)
                            if not np.allclose(actual, wanted, atol=3.e-7, rtol=3.e-7, equal_nan=True):
                                assert ties > 0; alternative = True
                                certificates.append(dict(seed=seed, mode=mode, base=base, case=case, stage=stage,
                                    slot=slot, kind='tree', ties=ties)); wanted = certified
                            record('tree_certificate_'+str(slot), stage, 'source_cart_certificate')
                        if mode == 0 and not np.array_equal(actual, wanted) and not alternative:
                            witnesses = certify_classifier_prediction(actual.astype(int), source, q)
                            assert witnesses; alternative = True; wanted = actual
                            certificates.append(dict(seed=seed, mode=mode, base=base, case=case, stage=stage,
                                slot=slot, kind='label_boundary', witnesses=witnesses))
                        np.testing.assert_allclose(actual, wanted, atol=3.e-7 if mode else 0,
                            rtol=3.e-7 if mode else 0, equal_nan=True, err_msg=str((context, stage, slot)))
                        record('base_response_'+str(slot), stage, 'source_certified_base' if alternative else 'sklearn')
                        changed |= alternative
                        if mode == 1:
                            proxy.estimators_[slot].predict_proba = lambda X, a=actual: a.copy()
                        else:
                            proxy.estimators_[slot].predict = lambda X, a=actual.astype(int) if mode == 0 else actual: a.copy()
                        if flag in [1, 3, 9, 11]:
                            np.testing.assert_allclose(values[0], np.atleast_2d(source.coef_), atol=3.e-7, rtol=3.e-7)
                            tail = np.r_[source.intercept_, source.t_] if flag in [3, 11] else np.atleast_1d(source.intercept_)
                            np.testing.assert_allclose(values[1], tail, atol=3.e-7, rtol=3.e-7)
                            record('linear_state_'+str(slot), stage)
                        elif flag == 2:
                            for a, expected in zip(values, [source.theta_, source.var_, source.class_prior_, source.class_count_]):
                                np.testing.assert_allclose(a, expected, atol=3.e-8, rtol=3.e-8, equal_nan=True)
                            record('gaussian_state_'+str(slot), stage)
                        elif flag == 4:
                            for a, expected in zip(values, [source.feature_log_prob_, source.class_log_prior_]):
                                np.testing.assert_allclose(a, expected, atol=3.e-8, rtol=3.e-8)
                            record('count_nb_state_'+str(slot), stage)
                        elif flag == 6:
                            np.testing.assert_array_equal(values[0], source._fit_X)
                            yy = source.classes_[source._y] if mode < 2 else source._y
                            np.testing.assert_array_equal(values[1], yy); record('knn_state_'+str(slot), stage)
                        elif flag == 12:
                            np.testing.assert_allclose(values[0], source.class_prior_, atol=3.e-12, rtol=3.e-12)
                            record('dummy_state_'+str(slot), stage)
                        elif flag == 13:
                            np.testing.assert_allclose(values[0], source.constant_.ravel(), atol=3.e-12, rtol=3.e-12)
                            record('dummy_state_'+str(slot), stage)
                        elif flag == 14:
                            np.testing.assert_array_equal(values[0], source.X_fit_)
                            np.testing.assert_allclose(values[1], source.dual_coef_, atol=3.e-8, rtol=3.e-8)
                            record('kernel_ridge_state_'+str(slot), stage)
                    np.testing.assert_allclose(transformed, proxy.transform(q), atol=3.e-12, rtol=3.e-12, equal_nan=True)
                    np.testing.assert_allclose(prediction, proxy.predict(q), atol=3.e-12 if mode == 2 else 0,
                        rtol=3.e-12 if mode == 2 else 0, equal_nan=True)
                    np.testing.assert_allclose(scores, [proxy.score(q, truth), proxy.score(q, truth, sample_weight=sw)],
                        atol=3.e-12, rtol=3.e-12, equal_nan=True)
                    if mode == 1:
                        np.testing.assert_allclose(probability, proxy.predict_proba(q), atol=3.e-12, rtol=3.e-12, equal_nan=True)
                        proxy.flatten_transform = False
                        np.testing.assert_allclose(tensors, proxy.transform(q), atol=3.e-12, rtol=3.e-12, equal_nan=True)
                    record('source_combination_identity', stage, 'source_native_base_identity')
                    # Native base agreement above plus this independent source reduction checks the wrapper.
                    if not changed:
                        expected = wrapper.predict(q)
                        if mode != 2 and not np.array_equal(prediction, expected):
                            assert mode == 1
                            raw = wrapper.predict_proba(q); witness = []
                            for row in np.flatnonzero(prediction != expected):
                                selected = np.flatnonzero(wrapper.classes_ == prediction[row])[0]
                                gap = raw[row].max()-raw[row, selected]
                                bound = 64*np.finfo(float).eps*max(1., abs(raw[row]).max())
                                assert 0 <= gap <= bound, (context, gap, bound)
                                witness.append(dict(row=int(row), gap=float(gap), bound=float(bound)))
                            certificates.append(dict(seed=seed, mode=mode, base=base, case=case, stage=stage,
                                kind='soft_boundary', witnesses=witness)); changed = True
                        else:
                            np.testing.assert_allclose(prediction, expected, atol=3.e-7, rtol=3.e-7, equal_nan=True)
                    record('wrapper_prediction', stage, 'source_certified_reduction' if changed else 'sklearn')
                for index, j in enumerate(active):
                    before_state, before_pred = native_before[index]; after_state, after_pred = read_state(), vector()
                    assert before_state[0] == after_state[0]
                    np.testing.assert_array_equal(before_state[1], after_state[1])
                    assert len(before_state[2]) == len(after_state[2])
                    for a, expected in zip(before_state[2], after_state[2]): np.testing.assert_array_equal(a, expected)
                    np.testing.assert_array_equal(before_pred, after_pred); record('template_preserved_'+str(j), oracle='input_identity')
            for actual, expected in [(matrix(n), x), (matrix(m), q), (vector(), w), (vector(), cfg['weights']), (vector(), sw), (vector(), y)]:
                np.testing.assert_array_equal(actual, expected)
            assert list(lines) == []; record('inputs_preserved', oracle='input_identity'); fixture_count += 1
        print(f'seed {seed}: voting ensembles passed', flush=True)
    (RESULTS/'voting.json').write_text(json.dumps(dict(sklearn=sklearn.__version__, fixtures=fixture_count,
        checks=checks, certificates=certificates), indent=2)+'\n')
    print(f'{len(checks)} voting groups across {fixture_count} fixtures passed')


if __name__ == '__main__':
    main()
