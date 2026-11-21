"""ClassifierChain responses, source CV plans, fitted state and ordered prediction."""
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
from sklearn import config_context
from sklearn.base import clone
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'
METHODS = ['predict', 'predict_proba', 'predict_log_proba', 'decision_function',
           ['decision_function', 'predict_proba', 'predict']]


def resolve(profile=None):
    options = ['--profile', profile] if profile else ['--flag', FLAGS]
    run = subprocess.run(['fpm', 'run', *options, '--target', 'reference_classifier_chain_driver', '--runner'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    paths = [ROOT/line.strip().strip('"') for line in run.stdout.splitlines()
        if line.strip().strip('"').endswith(('reference_classifier_chain_driver.exe', 'reference_classifier_chain_driver'))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, run.stdout
    return paths[0]


def fixtures(seed):
    rng = np.random.default_rng(seed)
    for base in range(8):
        for case in range(25):
            if case in [5, 6] and base not in [0, 1, 2, 6]:
                continue
            if case in [7, 8] and base not in [0, 2]:
                continue
            if case == 10 and base not in [0, 1, 2]:
                continue
            if case in [15, 16] and base in [0, 2]:
                continue
            if case in [18, 19] and base != 2:
                continue
            n, p, m, t = 48, 3, 13, 3
            if case in [14, 23]:
                t = 1
            if case == 22:
                m = 1
            method = {2: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3, 9: 4, 10: 1, 11: 1,
                13: 1, 15: 1, 18: 1, 19: 4, 20: 1, 21: 1, 24: 1}.get(case, 0)
            cv = {2: 3, 3: 3, 6: 3, 8: 3, 9: 2, 10: 3, 12: 3, 13: 3, 14: 99,
                15: 3, 17: 3, 18: 3, 19: 3, 22: 2, 23: 99}.get(case, 0)
            order_kind = 1 if case in [1, 4, 6, 8, 12, 17, 21] else 2 if case in [2, 5, 7, 9, 10, 13, 18, 19, 22] else 0
            if case in [11, 20, 24]:
                order_kind = {11: 3, 20: 4, 24: 5}[case]
            x = rng.normal(size=(n, p)); query = rng.normal(size=(m, p))
            coefficients = rng.normal(size=(p, t))
            y = (x@coefficients+.7*rng.normal(size=(n, t)) > 0).astype(int)
            truth = (query@coefficients+.7*rng.normal(size=(m, t)) > 0).astype(int)
            if case in [12, 23]:
                labels = np.array([0, 2, 5]); y = labels[(np.arange(n)[:, None]+np.arange(t)) % 3]
                truth = labels[rng.integers(0, 3, size=(m, t))]
            if case == 13:
                y = np.where(y == 0, -7, 11); truth = np.where(truth == 0, -7, 11)
            if case == 15:
                y[:, 0] = 0; y[0, 0] = 1; truth[:, 0] = 0
            if case == 16:
                y[:, -1] = 1; truth[:, -1] = 1
            if base >= 5:
                x = rng.poisson(2, size=(n, p)).astype(float); query = rng.poisson(2, size=(m, p)).astype(float)
            weighted = case == 10
            w = rng.uniform(.1, 2, n) if weighted else np.ones(n)
            if weighted:
                w[::7] = 0
            sw = rng.uniform(.1, 2, m)
            if case == 21:
                sw[0] = -.2
            hinge = case == 19; average = 3 if case == 18 else 0
            modified_huber = case == 18 and seed in [7, 2026]
            estimator = [LogisticRegression(C=.7, tol=1.e-10, max_iter=500, solver='newton-cg'),
                GaussianNB(), SGDClassifier(loss='hinge' if hinge else 'modified_huber' if modified_huber else 'log_loss', learning_rate='constant',
                    eta0=.02, alpha=.001, max_iter=4, tol=None, shuffle=False, average=average, random_state=0),
                KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='brute'),
                DecisionTreeClassifier(max_depth=2, min_samples_leaf=4, random_state=0),
                MultinomialNB(alpha=.4), BernoulliNB(alpha=.4), ComplementNB(alpha=.4, norm=True)][base]
            probability = case not in [12, 16, 19, 23]
            decision = base in [0, 2] and case not in [12, 23]
            score = case != 12
            flags = int(probability)+2*int(decision)+4*int(score)+8*int(hinge)+16*int(average > 0)+32*int(modified_huber)
            plans = []
            for column in range(t):
                if t == 1 or cv == 0 and order_kind < 3:
                    plans.append([])
                elif order_kind >= 3:
                    ids = np.arange(n)
                    folds = [(ids[ids % 3 != k], ids[ids % 3 == k]) for k in range(3)]
                    if order_kind == 4:
                        folds = [(ids, test) for train, test in folds]
                    if order_kind == 5:
                        folds = [(np.r_[train, train[:2]], test) for train, test in folds]
                    plans.append(folds)
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        plans.append(list(StratifiedKFold(n_splits=cv).split(x, y[:, column])))
            yield dict(seed=seed, base=base, case=case, method=method, cv=cv, order_kind=order_kind,
                weighted=weighted, flags=flags, refit=case == 17), x, y, query, truth, w, sw, plans, estimator


def certify_tree(state, x, y, query):
    labels, nodes = state
    stats = dict(ties=0)
    visited = set()
    def cost(rows):
        count = np.array([np.sum(y[rows] == label) for label in labels])
        return len(rows)*(1-np.sum((count/len(rows))**2))
    def walk(index, rows, depth):
        assert 0 <= index < len(nodes) and index not in visited
        visited.add(index)
        left, right, feature, count, recorded_depth = nodes[index, :5].astype(int)
        threshold, impurity = nodes[index, 5:7]; probability = nodes[index, 7:]
        assert count == len(rows) and depth == recorded_depth
        wanted = [np.mean(y[rows] == label) for label in labels]
        np.testing.assert_allclose(probability, wanted, atol=3.e-12, rtol=3.e-12)
        np.testing.assert_allclose(impurity, cost(rows)/len(rows), atol=3.e-12, rtol=3.e-12)
        stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=4, random_state=0).fit(x[rows], y[rows])
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
                    candidate_cost = cost(rows[partition])+cost(rows[~partition])
                    costs.append(candidate_cost); candidates.append((candidate_cost, column, partition.copy()))
        loss = cost(rows[mask])+cost(rows[~mask]); bound = 128*np.finfo(float).eps*len(rows)
        assert loss <= min(costs)+bound
        alternatives = sum(abs(value-loss) <= bound and (column != feature-1 or not np.array_equal(partition, mask))
                           for value, column, partition in candidates)
        if alternatives:
            stats['ties'] += 1
        sf = stump.tree_.feature[0]
        if sf >= 0:
            source_mask = x[rows, sf].astype(np.float32) <= stump.tree_.threshold[0]
            source_loss = cost(rows[source_mask])+cost(rows[~source_mask])
        else:
            source_loss = cost(rows)
        assert loss <= source_loss+bound
        assert abs(loss-source_loss) <= bound
        if sf != feature-1 or not np.array_equal(mask, source_mask):
            assert alternatives > 0
        walk(left-1, rows[mask], depth+1); walk(right-1, rows[~mask], depth+1)
    walk(0, np.arange(len(x)), 0)
    assert len(visited) == len(nodes)
    probability = []
    for row in query:
        index = 0
        while nodes[index, 0] != 0:
            feature = int(nodes[index, 2])-1
            index = int(nodes[index, 0 if row[feature] <= nodes[index, 5] else 1])-1
        probability.append(nodes[index, 7:])
    return np.array(probability), stats


def tree_response(labels, probability, method, global_labels=None):
    if method == 'predict':
        return labels[probability.argmax(axis=1)]
    if global_labels is None:
        assert len(labels) == 2
        values = probability[:, 1]
    else:
        position = np.flatnonzero(labels == global_labels[1])
        values = probability[:, position[0]] if len(position) else np.zeros(len(probability))
    if method == 'predict_log_proba':
        with np.errstate(divide='ignore'):
            values = np.log(values)
        if global_labels is not None and not len(position):
            values[:] = -np.finfo(float).max
    return values


def certify_tree_chain(states, fold_states, design, x, y, query, expected, plans):
    n, p = x.shape; outputs = len(states); m = len(query)
    qdesign = np.zeros((m, p+outputs)); qdesign[:, :p] = query
    prediction = np.zeros((m, outputs), dtype=int); probability = np.zeros((m, outputs))
    stats = dict(ties=0); cv_values = {}
    def accumulate(local):
        for key in stats:
            stats[key] += local[key]
    for j, column in enumerate(expected.order_):
        flag, state = states[j]; assert flag == 5
        labels, nodes = state
        proba, local = certify_tree(state, design[:, :p+j], y[:, column], qdesign[:, :p+j]); accumulate(local)
        prediction[:, column] = tree_response(labels, proba, 'predict')
        if len(labels) == 2:
            probability[:, column] = proba[:, 1]
        qdesign[:, p+j] = tree_response(labels, proba, expected.chain_method_)
        if j < outputs-1:
            current = plans[column] if expected.cv is not None else []
            assert len(current) == len(fold_states[j])
            if current:
                cv_values[j] = np.zeros(n)
            for (train, test), (flag, fold_state) in zip(current, fold_states[j]):
                assert flag == 5
                proba, local = certify_tree(fold_state, design[train, :p+j], y[train, column], design[test, :p+j])
                accumulate(local)
                cv_values[j][test] = tree_response(fold_state[0], proba, expected.chain_method_, labels)
            if current:
                np.testing.assert_allclose(design[:, p+j], cv_values[j], atol=3.e-12, rtol=3.e-12)
    return prediction, probability, cv_values, stats


def certify_complement_labels(actual, model, x):
    expected = model.predict(x); rows = np.flatnonzero(actual != expected); witnesses = []
    scores = model.predict_log_proba(x)
    for row in rows:
        column = np.flatnonzero(model.classes_ == actual[row])
        assert len(column) == 1
        bound = 64*np.finfo(float).eps*max(1., np.max(abs(scores[row])))
        gap = scores[row].max()-scores[row, column[0]]
        assert 0 <= gap <= bound, (row, gap, bound)
        witnesses.append(dict(row=int(row), gap=float(gap), bound=float(bound)))
    return witnesses


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--executable', type=Path); parser.add_argument('--profile')
    args = parser.parse_args()
    executable = args.executable.resolve() if args.executable else resolve(args.profile)
    RESULTS.mkdir(exist_ok=True); checks = []; certificates = []; fixture_count = 0
    for seed in [0, 7, 42, 2026, 314159]:
        for cfg, x, y, query, truth, w, sw, plans, estimator in fixtures(seed):
            n, p = x.shape; m, t = truth.shape; base = cfg['base']; context = (seed, base, cfg['case'])
            data, output = RESULTS/'classifier_chain_input.txt', RESULTS/'classifier_chain_actual.txt'
            with data.open('w') as stream:
                stream.write(' '.join(map(str, [n, p, m, t, base, cfg['method'], cfg['order_kind'], cfg['cv'], seed,
                    int(cfg['weighted']), cfg['flags'], int(cfg['refit'])]))+'\n')
                np.savetxt(stream, w[None], fmt='%.17g'); np.savetxt(stream, sw[None], fmt='%.17g')
                for a in [x, y, query, truth]:
                    np.savetxt(stream, a, fmt='%.17g')
                for folds in plans:
                    stream.write(str(len(folds))+'\n')
                    for train, test in folds:
                        stream.write(f'{len(train)} {len(test)}\n')
                        np.savetxt(stream, (train+1)[None], fmt='%d'); np.savetxt(stream, (test+1)[None], fmt='%d')
            run = subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert run.returncode == 0, (context, cfg, run.stdout)
            lines = iter(output.read_text().splitlines())
            def vector(dtype=float):
                return np.fromstring(next(lines), sep=' ', dtype=dtype)
            def matrix(rows, dtype=float):
                return np.array([vector(dtype) for _ in range(rows)])
            def record(name, stage=0, oracle='sklearn'):
                checks.append(dict(seed=seed, base=base, case=cfg['case'], stage=stage, name=name, oracle=oracle, passed=True))
            def read_state():
                header = vector(int); flag = header[0]
                if flag == 0:
                    return flag, None
                if flag == 5:
                    labels = vector(int); nodes = matrix(header[2])
                    return flag, (labels, nodes)
                if flag == 6:
                    return flag, [matrix(header[1]), vector(int)]
                values = [matrix(header[1])]
                if flag == 2:
                    values.append(matrix(header[1])); values.extend([vector(), vector()])
                else:
                    values.append(vector())
                return flag, values
            with warnings.catch_warnings(), config_context(enable_metadata_routing=cfg['weighted']):
                warnings.simplefilter('ignore')
                if cfg['weighted']:
                    estimator.set_fit_request(sample_weight=True)
                estimator.fit(x, y[:, 0]); before = estimator.predict(query)
                order = np.arange(t)[::-1] if cfg['order_kind'] == 1 else 'random' if cfg['order_kind'] == 2 else None
                cv = plans[0] if cfg['order_kind'] >= 3 else cfg['cv'] or None
                chain = ClassifierChain(estimator, order=order, cv=cv, chain_method=METHODS[cfg['method']], random_state=seed)
                chain.fit(x, y, **(dict(sample_weight=w) if cfg['weighted'] else {})); states = [copy.deepcopy(chain)]
                if cfg['refit']:
                    chain.set_params(order=None, cv=None); chain.fit(x, y[:, :1]); states.append(chain)
                for stage, expected in enumerate(states):
                    features, outputs = vector(int); selected = next(lines).strip()
                    assert features == p and outputs == len(expected.estimators_) and selected == expected.chain_method_
                    np.testing.assert_array_equal(vector(int), expected.order_+1)
                    for model in expected.estimators_:
                        k = int(vector(int)[0]); labels = vector(int)
                        assert k == len(model.classes_); np.testing.assert_array_equal(labels, model.classes_)
                    record('metadata_order_classes_method', stage)
                    predicted = matrix(m, int)
                    probability = matrix(m) if cfg['flags'] & 1 else None
                    decision = matrix(m) if cfg['flags'] & 2 else None
                    scores = vector() if cfg['flags'] & 4 else None
                    native_states = [read_state() for _ in range(outputs)]
                    native_folds = []
                    if base == 4:
                        for j in range(outputs-1):
                            count = int(vector(int)[0]); native_folds.append([read_state() for _ in range(count)])
                    design = matrix(n)
                    assert design.shape == (n, p+outputs-1)
                    np.testing.assert_array_equal(design[:, :p], x)
                    conditioned = [clone(estimator).fit(design[:, :p+j], y[:, column],
                        **(dict(sample_weight=w) if cfg['weighted'] and stage == 0 else {}))
                        for j, column in enumerate(expected.order_)]
                    complement_probability = np.zeros((m, outputs)); complement_witnesses = []
                    if base == 7:
                        if expected.cv is not None and selected == 'predict':
                            for j, column in enumerate(expected.order_[:-1]):
                                for fold_index, (train, test) in enumerate(plans[column]):
                                    source_fold = clone(estimator).fit(design[train, :p+j], y[train, column])
                                    witness = certify_complement_labels(design[test, p+j], source_fold, design[test, :p+j])
                                    complement_witnesses.extend([dict(output=j, fold=fold_index, **value) for value in witness])
                        qdesign = query.copy()
                        for j, column in enumerate(expected.order_):
                            source = conditioned[j]
                            witness = certify_complement_labels(predicted[:, column], source, qdesign)
                            complement_witnesses.extend([dict(output=j, **value) for value in witness])
                            if len(source.classes_) == 2:
                                complement_probability[:, column] = source.predict_proba(qdesign)[:, 1]
                            if selected == 'predict':
                                feature = predicted[:, column]
                            else:
                                feature = getattr(source, selected)(qdesign)[:, 1]
                            qdesign = np.column_stack([qdesign, feature])
                    tree_prediction = tree_probability = tree_features = tree_stats = None
                    if base == 4:
                        tree_prediction, tree_probability, tree_features, tree_stats = certify_tree_chain(
                            native_states, native_folds, design, x, y, query, expected, plans)
                        np.testing.assert_array_equal(predicted, tree_prediction)
                        if probability is not None:
                            np.testing.assert_allclose(probability, tree_probability, atol=3.e-12, rtol=3.e-12)
                        record('tree_chain_certificate', stage, 'source_cart_certificate')
                    def tree_exception(name):
                        assert base == 4 and sum(tree_stats.values()) > 0, (context, stage, name)
                        certificates.append(dict(seed=seed, base=base, case=cfg['case'], stage=stage,
                            name=name, **tree_stats))
                        return 'source_cart_chain_certificate'
                    def complement_exception(name):
                        assert base == 7 and complement_witnesses, (context, stage, name)
                        certificates.append(dict(seed=seed, base=base, case=cfg['case'], stage=stage,
                            name=name, complement_ties=complement_witnesses))
                        return 'source_complement_chain_certificate'
                    reference_prediction = expected.predict(query); prediction_oracle = 'sklearn'
                    if not np.array_equal(predicted, reference_prediction):
                        if base == 7:
                            prediction_oracle = complement_exception('prediction'); reference_prediction = predicted
                        else:
                            prediction_oracle = tree_exception('prediction'); reference_prediction = tree_prediction
                    np.testing.assert_array_equal(predicted, reference_prediction, err_msg=str((context, stage)))
                    record('prediction', stage, prediction_oracle)
                    if probability is not None:
                        wanted = expected.predict_proba(query); oracle = 'sklearn'
                        if not np.allclose(probability, wanted, atol=3.e-7, rtol=3.e-7, equal_nan=True):
                            if base == 7:
                                oracle = complement_exception('probability'); wanted = complement_probability
                            else:
                                oracle = tree_exception('probability'); wanted = tree_probability
                        np.testing.assert_allclose(probability, wanted, atol=3.e-7, rtol=3.e-7,
                            equal_nan=True, err_msg=str((context, stage))); record('probability', stage, oracle)
                    if decision is not None:
                        np.testing.assert_allclose(decision, expected.decision_function(query), atol=3.e-7, rtol=3.e-7,
                            err_msg=str((context, stage))); record('decision', stage)
                    if scores is not None:
                        wanted = [accuracy_score(truth[:, :outputs], reference_prediction),
                            accuracy_score(truth[:, :outputs], reference_prediction, sample_weight=sw)]
                        np.testing.assert_allclose(scores, wanted, atol=3.e-12, rtol=3.e-12)
                        record('subset_score', stage, prediction_oracle)
                    # Conditioned fits isolate CV-feature roundoff from model fitting/routing.
                    for j, column in enumerate(expected.order_):
                        source = conditioned[j]
                        flag, values = native_states[j]
                        if flag in [1, 3]:
                            np.testing.assert_allclose(values[0], source.coef_, atol=3.e-7, rtol=3.e-7,
                                err_msg=str((context, stage, j, 'coef')))
                            tail = np.r_[source.intercept_, source.t_] if flag == 3 else source.intercept_
                            np.testing.assert_allclose(values[1], tail, atol=3.e-7, rtol=3.e-7)
                        elif flag == 2:
                            for actual, wanted in zip(values, [source.theta_, source.var_, source.class_prior_, source.class_count_]):
                                np.testing.assert_allclose(actual, wanted, atol=3.e-8, rtol=3.e-8,
                                    err_msg=str((context, stage, j, 'Gaussian moments')))
                        elif flag == 4:
                            for actual, wanted in zip(values, [source.feature_log_prob_, source.class_log_prior_]):
                                np.testing.assert_allclose(actual, wanted, atol=3.e-8, rtol=3.e-8)
                        elif flag == 5:
                            np.testing.assert_array_equal(values[0], source.classes_)
                        elif flag == 6:
                            np.testing.assert_array_equal(values[0], source._fit_X)
                            np.testing.assert_array_equal(values[1], source.classes_[source._y])
                        else:
                            raise AssertionError(('unverified native state', flag))
                        state_oracle = 'source_cart_certificate' if flag == 5 else 'input_identity' if flag == 6 else 'source_conditioned_fit'
                        record('conditioned_state_'+str(j), stage, state_oracle)
                        if j < outputs-1:
                            feature_oracle = 'input_identity' if expected.cv is None else 'source_conditioned_cv'
                            if expected.cv is None:
                                np.testing.assert_array_equal(design[:, p+j], y[:, column])
                            else:
                                source_column = np.zeros(n)
                                for train, test in plans[column]:
                                    fold_model = clone(estimator).fit(design[train, :p+j], y[train, column])
                                    raw = getattr(fold_model, selected)(design[test, :p+j])
                                    if raw.ndim > 1:
                                        global_labels = np.unique(y[:, column])
                                        position = np.flatnonzero(fold_model.classes_ == global_labels[1])
                                        if selected in ['predict_proba', 'predict_log_proba'] and not len(position):
                                            raw = np.full(len(test), 0. if selected == 'predict_proba' else -np.finfo(float).max)
                                        else:
                                            raw = raw[:, position[0] if selected in ['predict_proba', 'predict_log_proba'] else 1]
                                    if base == 7 and selected == 'predict' and not np.array_equal(design[test, p+j], raw):
                                        witness = certify_complement_labels(design[test, p+j], fold_model, design[test, :p+j])
                                        assert witness
                                        certificates.append(dict(seed=seed, base=base, case=cfg['case'], stage=stage,
                                            name='CV_feature_'+str(j), complement_ties=witness))
                                        raw = design[test, p+j]; feature_oracle = 'source_complement_chain_certificate'
                                    source_column[test] = raw
                                if base == 4 and not np.allclose(design[:, p+j], source_column,
                                        atol=3.e-7, rtol=3.e-7, equal_nan=True):
                                    feature_oracle = tree_exception('chain_feature_'+str(j)); source_column = tree_features[j]
                                np.testing.assert_allclose(design[:, p+j], source_column, atol=3.e-7, rtol=3.e-7,
                                    equal_nan=True, err_msg=str((context, stage, j, 'CV features')))
                            record('chain_feature_'+str(j), stage, feature_oracle)
                original_before, original_after = matrix(m), matrix(m)
                np.testing.assert_array_equal(original_before, original_after)
                if base == 4:
                    flag, state = read_state(); assert flag == 5
                    proba, stats = certify_tree(state, x, y[:, 0], query)
                    certified = state[0][proba.argmax(axis=1)]
                    np.testing.assert_array_equal(original_before[:, 0], certified)
                    if not np.array_equal(original_before[:, 0], before):
                        assert sum(stats.values()) > 0
                        certificates.append(dict(seed=seed, base=base, case=cfg['case'], stage='template',
                            name='prediction', **stats)); before = certified
                if base == 7 and not np.array_equal(original_before[:, 0], before):
                    witness = certify_complement_labels(original_before[:, 0], estimator, query)
                    assert witness
                    certificates.append(dict(seed=seed, base=base, case=cfg['case'], stage='template',
                        name='prediction', complement_ties=witness)); before = original_before[:, 0]
                np.testing.assert_array_equal(original_before[:, 0], before, err_msg=str(context))
                record('template_preserved', oracle='input_identity')
            np.testing.assert_array_equal(matrix(n), x); np.testing.assert_array_equal(matrix(m), query)
            np.testing.assert_array_equal(matrix(n, int), y); np.testing.assert_array_equal(vector(), w)
            assert list(lines) == []; record('input_preserved', oracle='input_identity'); fixture_count += 1
        print(f'seed {seed}: classifier chains passed', flush=True)
    (RESULTS/'classifier_chains.json').write_text(json.dumps(dict(sklearn=sklearn.__version__, fixtures=fixture_count,
        checks=checks, certificates=certificates), indent=2)+'\n')
    print(f'{len(checks)} classifier-chain groups across {fixture_count} fixtures passed')


if __name__ == '__main__':
    main()
