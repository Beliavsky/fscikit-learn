"""OVR, OVO and output-code reductions against pinned sklearn and native-head reconstructions."""
from pathlib import Path
import argparse
import copy
import json
import os
import subprocess
import warnings
from itertools import combinations
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
import numpy as np
import sklearn
from sklearn import config_context
from sklearn.base import is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier, _ConstantPredictor, _predict_binary
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, LinearRegression, Ridge, SGDRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics.pairwise import rbf_kernel
from compare_classifier_chains import certify_tree, certify_complement_labels

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile=None):
    options = ['--profile', profile] if profile else ['--flag', FLAGS]
    run = subprocess.run(['fpm', 'run', *options, '--target', 'reference_multiclass_driver', '--runner'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    paths = [ROOT/line.strip().strip('"') for line in run.stdout.splitlines()
        if line.strip().strip('"').endswith(('reference_multiclass_driver.exe', 'reference_multiclass_driver'))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, run.stdout
    return paths[0]


def fixtures(seed):
    rng = np.random.default_rng(seed)
    for mode in range(3):
        for base in range(19):
            if mode == 2 and base in [13, 14, 15, 16, 17]:
                continue
            for case in range(21):
                if case == 2 and mode == 1:
                    continue
                if case == 4 and base not in [0, 1, 2, 10, 11, 15, 18]:
                    continue
                if case in [5, 6, 7] and base != 2:
                    continue
                if case in [8, 9] and (base not in [1, 2] or mode == 2):
                    continue
                if case in [10, 11, 12, 17, 19] and mode != 0:
                    continue
                if case in [13, 14, 20] and (mode != 0 or base not in [1, 2]):
                    continue
                if case == 15 and (base not in [1, 2] or base == 1 and mode == 2):
                    continue
                if case == 16 and (base != 1 or mode == 2):
                    continue
                n, p, m = (3 if case == 16 else 54), 3, (1 if case == 18 else 9)
                kind = int(case in [10, 11, 12, 13, 14, 17, 19, 20])
                t = 3 if kind and case not in [19, 20] else 1
                k = 1 if case == 2 else 2 if case in [1, 19, 20] else 4 if case == 3 else 3
                labels = np.array([-9, 4, 23, 71])[:k]
                class_index = (np.arange(n)+k-1) % k; truth_index = np.arange(m) % k
                centers = rng.normal(size=(k, p))*1.2
                x = rng.normal(size=(n, p))*.8+centers[class_index]
                query = rng.normal(size=(m, p))*.8+centers[truth_index]
                y = labels[class_index, None]; truth = labels[truth_index, None]
                if kind and t > 1:
                    coefficients = rng.normal(size=(p, t))
                    y = (x@coefficients+.5*rng.normal(size=(n, t)) > 0).astype(int)
                    truth = (query@coefficients+.5*rng.normal(size=(m, t)) > 0).astype(int)
                    y[0, :] = 0; y[1, :] = 1
                    if case == 11:
                        y[:, 0] = 0; truth[:, 0] = 0
                    if case == 12:
                        y[:] = [0, 1, 0]; truth[:] = [0, 1, 0]
                    if case == 17:
                        y = np.where(y == 0, -7, 11); truth = np.where(truth == 0, -7, 11)
                if base in [5, 6, 7]:
                    rates = np.exp(np.clip(centers*.4, -1., 1.5))
                    x = rng.poisson(2*rates[class_index]).astype(float)
                    query = rng.poisson(2*rates[truth_index]).astype(float)
                weighted = case in [4, 9, 14, 15]
                w = rng.uniform(.1, 2, n) if weighted else np.ones(n)
                if weighted:
                    w[::7] = 0
                if case == 15:
                    w[:] = 0
                sw = rng.uniform(.1, 2, m)
                if case == 18:
                    sw[:] = -.5  # Signed query accuracy weights are accepted by source.
                cuts = [1, 6, 22, n] if case in [8, 9, 20] else [14, 29, n] if case in [13, 14] else [n] if case == 16 else []
                initial_fit = case in [13, 14]; refit = case in [8, 9, 13, 14, 20]
                declared = np.arange(t) if kind and t > 1 else np.r_[labels, 100] if cuts else labels
                declared = np.r_[declared[::-1], declared[0]]
                average = 3 if case == 5 or case == 8 and base == 2 and seed in [7, 2026] else 0
                loss = 2 if case == 6 else 1 if case == 7 else 0
                estimator = [LogisticRegression(C=.7, tol=1.e-10, max_iter=500, solver='newton-cg'),
                    GaussianNB(), SGDClassifier(loss=['log_loss', 'hinge', 'modified_huber'][loss],
                        learning_rate='constant', eta0=.02, alpha=.001, max_iter=4, tol=None,
                        shuffle=False, average=average, fit_intercept=not(cuts and average), random_state=0),
                    KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='brute'),
                    DecisionTreeClassifier(max_depth=2, min_samples_leaf=4, random_state=0),
                    MultinomialNB(alpha=.4), BernoulliNB(alpha=.4), ComplementNB(alpha=.4, norm=True),
                    Perceptron(eta0=.1, max_iter=4, tol=None, shuffle=False, random_state=0),
                    PassiveAggressiveClassifier(C=.7, max_iter=4, tol=None, shuffle=False, random_state=0),
                    RidgeClassifier(alpha=.7, solver='svd'),
                    LinearSVC(C=.7, tol=1.e-10, max_iter=10000, dual=False, random_state=0),
                    SVC(C=.7, gamma=.4, tol=1.e-8, max_iter=100000, shrinking=False),
                    LinearRegression(), Ridge(alpha=.7, solver='svd'),
                    SGDRegressor(learning_rate='constant', eta0=.02, alpha=.001, max_iter=4, tol=None, shuffle=False),
                    KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='brute'),
                    DecisionTreeRegressor(max_depth=2, min_samples_leaf=4, random_state=0),
                    RidgeClassifierCV(alphas=[.1, .7, 2.])][base]
                code_size = {1: .75, 2: 3., 3: 2.25, 18: 1/3}.get(case, 1.5)
                yield dict(seed=seed, mode=mode, base=base, case=case, kind=kind, average=average, loss=loss,
                    weighted=weighted, initial_fit=initial_fit, refit=refit, code_size=code_size), x, y, query, truth, w, sw, cuts, declared, estimator


def fitted(model):
    try:
        check_is_fitted(model); return True
    except NotFittedError:
        return False


def dense(a):
    return a.toarray() if hasattr(a, 'toarray') else np.asarray(a)


def source_matrix(a, m):
    return dense(a).reshape(m, -1)


def native_proxy(expected, heads):
    proxy = copy.deepcopy(expected)
    for model, head in zip(proxy.estimators_, heads):
        if not head['status']:
            continue
        model.predict = lambda X, r=head['predict']: r.copy()
        if hasattr(model, 'decision_function'):
            model.decision_function = lambda X, r=head['confidence']: r.copy()
        if head['probability'] is not None:
            probability = np.column_stack([1-head['probability'], head['probability']])
            model.predict_proba = lambda X, r=probability: r.copy()
    return proxy


def head_training(mode, expected, x, y, head):
    if mode == 1:
        a, b = list(combinations(expected.classes_, 2))[head]
        rows = np.flatnonzero((y[:, 0] == a) | (y[:, 0] == b))
        return x[rows], (y[rows, 0] == b).astype(int)
    if mode == 2:
        mapping = {label: i for i, label in enumerate(expected.classes_)}
        indices = np.array([mapping[label] for label in y[:, 0]])
        return x, expected.code_book_[indices, head].astype(int)
    if expected.multilabel_:
        return x, y[:, head]
    if len(expected.classes_) == 1:
        return x, np.zeros(len(y), int)
    positive = expected.classes_[1] if len(expected.classes_) == 2 else expected.classes_[head]
    return x, (y[:, 0] == positive).astype(int)


def binary_tree_certificate(flag, labels, nodes, train_x, target, query):
    if flag == 5:
        probability, stats = certify_tree((labels, nodes), train_x, target, query)
        prediction = labels[probability.argmax(axis=1)]; confidence = probability[:, 1]
        return prediction, confidence, confidence, stats
    labels = np.unique(target); assert len(labels) == 2
    span = float(labels[1]-labels[0]); converted = np.zeros((len(nodes), 9))
    converted[:, :6] = nodes[:, :6]
    converted[:, 6] = 2*nodes[:, 7]/span**2
    converted[:, 8] = (nodes[:, 6]-labels[0])/span
    converted[:, 7] = 1-converted[:, 8]
    # Squared error on two-valued targets is span**2/2 times Gini loss.
    probability, stats = certify_tree((labels, converted), train_x, target, query)
    prediction = labels[0]+span*probability[:, 1]
    return prediction, prediction, None, stats


def certify_label_boundary(actual, model, query):
    expected = model.predict(query); rows = np.flatnonzero(actual != expected); witnesses = []
    classes = model.classes_
    if hasattr(model, 'decision_function'):
        score = np.ravel(model.decision_function(query)); threshold = 0.
    else:
        score = model.predict_proba(query)[:, 1]; threshold = .5
    for row in rows:
        assert actual[row] in classes
        bound = 64*np.finfo(float).eps*max(1., abs(score[row]))
        gap = abs(score[row]-threshold)
        assert np.isfinite(gap) and gap <= bound, (row, actual[row], expected[row], gap, bound)
        witnesses.append(dict(row=int(row), gap=float(gap), bound=float(bound)))
    return witnesses


def certify_parent_boundary(mode, actual, expected, query):
    wanted = source_matrix(expected.predict(query), len(query)); rows = np.argwhere(actual != wanted)
    witnesses = []
    if mode == 0:
        score = np.column_stack([np.ravel(_predict_binary(e, query)) for e in expected.estimators_])
    elif mode == 1:
        score = source_matrix(expected.decision_function(query), len(query))
    else:
        responses = np.column_stack([np.ravel(_predict_binary(e, query)) for e in expected.estimators_])
    for row, column in rows:
        if mode == 0 and expected.multilabel_:
            first = expected.estimators_[0]
            threshold = 0. if hasattr(first, 'decision_function') and is_classifier(first) else .5
            gap = abs(score[row, column]-threshold); scale = max(1., abs(score[row, column]))
        elif mode != 2 and len(expected.classes_) == 2:
            first = expected.estimators_[0]
            threshold = 0. if hasattr(first, 'decision_function') and is_classifier(first) else .5
            gap = abs(score[row, 0]-threshold); scale = max(1., abs(score[row, 0]))
        elif mode != 2:
            index = np.flatnonzero(expected.classes_ == actual[row, 0]); assert len(index) == 1
            gap = np.nanmax(score[row])-score[row, index[0]]; scale = max(1., np.nanmax(abs(score[row])))
        else:
            costs = np.sum((responses[row]-expected.code_book_)**2, axis=1)
            index = np.flatnonzero(expected.classes_ == actual[row, 0]); assert len(index) == 1
            gap = costs[index[0]]-costs.min(); scale = max(1., costs.max())
            source_index = np.flatnonzero(expected.classes_ == wanted[row, 0]); assert len(source_index) == 1
            source_gap = costs[source_index[0]]-costs.min()
            assert 0 <= source_gap <= 64*np.finfo(float).eps*scale
        bound = 64*np.finfo(float).eps*scale
        assert np.isfinite(gap) and 0 <= gap <= bound, (row, column, gap, bound)
        witnesses.append(dict(row=int(row), column=int(column), gap=float(gap), bound=float(bound)))
    return witnesses


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--executable', type=Path); parser.add_argument('--profile')
    parser.add_argument('--seed', type=int, action='append')
    args = parser.parse_args()
    executable = args.executable.resolve() if args.executable else resolve(args.profile)
    RESULTS.mkdir(exist_ok=True); checks = []; certificates = []; fixture_count = 0
    for seed in args.seed or [0, 7, 42, 2026, 314159]:
        for cfg, x, y, query, truth, w, sw, cuts, declared, estimator in fixtures(seed):
            n, p = x.shape; m, t = truth.shape; mode = cfg['mode']; context = (seed, mode, cfg['base'], cfg['case'])
            data, output = RESULTS/'multiclass_input.txt', RESULTS/'multiclass_actual.txt'
            with data.open('w') as stream:
                stream.write(' '.join(map(str, [mode, cfg['base'], cfg['kind'], n, p, m, t, len(cuts), int(cfg['weighted']),
                    seed, cfg['average'], cfg['loss'], int(cfg['initial_fit']), int(cfg['refit']), cfg['code_size']]))+'\n')
                stream.write(str(len(declared))+'\n'); np.savetxt(stream, declared[None], fmt='%d')
                np.savetxt(stream, w[None], fmt='%.17g'); np.savetxt(stream, sw[None], fmt='%.17g')
                if cuts:
                    np.savetxt(stream, np.array(cuts)[None], fmt='%d')
                for a in [x, y, query, truth]:
                    np.savetxt(stream, a, fmt='%.17g')
            run = subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert run.returncode == 0, (context, cfg, run.stdout)
            lines = iter(output.read_text().splitlines())
            def vector(dtype=float):
                return np.fromstring(next(lines), sep=' ', dtype=dtype)
            def matrix(rows, dtype=float):
                return np.array([vector(dtype) for _ in range(rows)])
            def record(name, stage=0, oracle='sklearn'):
                checks.append(dict(seed=seed, mode=mode, base=cfg['base'], case=cfg['case'], stage=stage,
                    name=name, oracle=oracle, passed=True))
            def read_state():
                k = int(vector(int)[0]); labels = vector(int) if k else np.array([], int)
                flag, rows, columns = vector(int)
                values = [matrix(columns if flag in [5, 7] else rows)]
                if flag in [1, 3, 4, 9, 11, 6]:
                    values.append(vector())
                elif flag == 2:
                    values.extend([matrix(rows), vector(), vector()])
                elif flag == 8:
                    values.extend([vector(), vector()])
                return flag, labels, values
            template_state = read_state()
            with warnings.catch_warnings(), config_context(enable_metadata_routing=cfg['weighted']):
                warnings.simplefilter('ignore')
                if cfg['weighted']:
                    estimator.set_fit_request(sample_weight=True)
                    if cuts:
                        estimator.set_partial_fit_request(sample_weight=True)
                pre_y = np.arange(1, n+1) % 2; estimator.fit(x, pre_y)
                if mode == 0:
                    wrapper = OneVsRestClassifier(estimator)
                elif mode == 1:
                    wrapper = OneVsOneClassifier(estimator)
                else:
                    wrapper = OutputCodeClassifier(estimator, code_size=cfg['code_size'], random_state=seed)
                source_y = y if cfg['kind'] else y[:, 0]
                states = []; first = 0
                for stage, last in enumerate(cuts or [n]):
                    kwargs = dict(sample_weight=w[first:last]) if cfg['weighted'] else {}
                    if not cuts or stage == 0 and cfg['initial_fit']:
                        wrapper.fit(x[first:last], source_y[first:last], **kwargs)
                    else:
                        wrapper.partial_fit(x[first:last], source_y[first:last], classes=declared, **kwargs)
                    states.append(copy.deepcopy(wrapper)); first = last
                if cfg['refit']:
                    wrapper.fit(x, source_y, **(dict(sample_weight=w) if cfg['weighted'] else {})); states.append(wrapper)
                for stage, expected in enumerate(states):
                    features, classes_count, head_count, multi, ready, has_proba, has_margin = vector(int)
                    labels = vector(int)
                    assert features == p and classes_count == len(expected.classes_) and head_count == len(expected.estimators_)
                    assert bool(multi) == (mode == 0 and expected.multilabel_)
                    assert bool(ready) == all(fitted(e) for e in expected.estimators_)
                    np.testing.assert_array_equal(labels, expected.classes_); record('metadata', stage)
                    if mode == 2:
                        np.testing.assert_array_equal(matrix(classes_count), expected.code_book_); record('code_book', stage)
                    prediction = probability = decision = scores = None
                    if ready:
                        scores = vector(); prediction = matrix(m, int)
                        if has_proba:
                            probability = matrix(m)
                        if has_margin:
                            decision = matrix(m)
                    heads = []
                    for j, source in enumerate(expected.estimators_):
                        status, value, width = vector(int)
                        assert bool(status) == fitted(source)
                        if status == 0:
                            assert width == 0; record('unfitted_head_'+str(j), stage)
                            heads.append(dict(status=status)); continue
                        assert width == p
                        assert (status == 1) == isinstance(source, _ConstantPredictor)
                        pred, confidence = vector(), vector()
                        hp = int(vector(int)[0]); native_probability = vector() if hp else None
                        state = None if status == 1 else read_state()
                        if status == 1:
                            assert value == int(source.y_[0])
                        wanted_confidence = np.ravel(_predict_binary(source, query))
                        wanted_prediction = source.predict(query)
                        wanted_probability = source.predict_proba(query)[:, 1] if hp else None
                        component_alternative = False; response_oracle = 'sklearn'
                        if state is not None and state[0] in [5, 7]:
                            train_x, target = head_training(mode, expected, x, y, j)
                            flag, head_labels, values = state
                            certified_pred, certified_confidence, certified_probability, stats = binary_tree_certificate(
                                flag, head_labels, values[0], train_x, target, query)
                            np.testing.assert_allclose(pred, certified_pred, atol=3.e-12, rtol=3.e-12)
                            np.testing.assert_allclose(confidence, certified_confidence, atol=3.e-12, rtol=3.e-12)
                            if hp:
                                np.testing.assert_allclose(native_probability, certified_probability, atol=3.e-12, rtol=3.e-12)
                            if not np.allclose(confidence, wanted_confidence, atol=3.e-7, rtol=3.e-7) or not np.allclose(
                                    pred, wanted_prediction, atol=3.e-7 if is_regressor(source) else 0,
                                    rtol=3.e-7 if is_regressor(source) else 0):
                                assert stats['ties'] > 0
                                component_alternative = True; response_oracle = 'source_binary_cart_certificate'
                                certificates.append(dict(seed=seed, mode=mode, base=cfg['base'], case=cfg['case'],
                                    stage=stage, head=j, kind='tree', **stats))
                                wanted_prediction, wanted_confidence, wanted_probability = certified_pred, certified_confidence, certified_probability
                        np.testing.assert_allclose(confidence, wanted_confidence, atol=3.e-7, rtol=3.e-7,
                            equal_nan=True, err_msg=str((context, stage, j, 'confidence')))
                        if not is_regressor(source) and not np.array_equal(pred, wanted_prediction):
                            witness = certify_label_boundary(pred, source, query)
                            assert witness; component_alternative = True; response_oracle = 'source_label_boundary'
                            certificates.append(dict(seed=seed, mode=mode, base=cfg['base'], case=cfg['case'],
                                stage=stage, head=j, kind='head_boundary', witnesses=witness)); wanted_prediction = pred
                        np.testing.assert_allclose(pred, wanted_prediction, atol=3.e-7 if is_regressor(source) else 0,
                            rtol=3.e-7 if is_regressor(source) else 0, equal_nan=True,
                            err_msg=str((context, stage, j, 'head prediction')))
                        if mode == 1 and is_regressor(source) and not component_alternative:
                            source_pred = source.predict(query)
                            changed_vote = ((pred == 0) != (source_pred == 0)) | ((pred == 1) != (source_pred == 1))
                            if changed_vote.any():
                                gaps = abs(pred[changed_vote]-source_pred[changed_vote])
                                bounds = 64*np.finfo(float).eps*np.maximum(1., abs(source_pred[changed_vote]))
                                assert np.all(gaps <= bounds), (context, stage, j, gaps, bounds)
                                component_alternative = True; response_oracle = 'source_vote_boundary'
                                certificates.append(dict(seed=seed, mode=mode, base=cfg['base'], case=cfg['case'],
                                    stage=stage, head=j, kind='vote_boundary', gaps=gaps.tolist(), bounds=bounds.tolist()))
                        if hp:
                            np.testing.assert_allclose(native_probability, wanted_probability,
                                atol=3.e-7, rtol=3.e-7, equal_nan=True)
                        record('head_response_'+str(j), stage, response_oracle)
                        if state is not None:
                            flag, head_labels, values = state
                            if hasattr(source, 'classes_'):
                                np.testing.assert_array_equal(head_labels, source.classes_)
                            else:
                                assert not len(head_labels)
                            if flag in [1, 3, 9, 11]:
                                np.testing.assert_allclose(values[0], np.atleast_2d(source.coef_), atol=3.e-7, rtol=3.e-7)
                                tail = np.r_[source.intercept_, source.t_] if flag in [3, 11] else np.atleast_1d(source.intercept_)
                                np.testing.assert_allclose(values[1], tail, atol=3.e-7, rtol=3.e-7)
                            elif flag == 2:
                                for actual, wanted in zip(values, [source.theta_, source.var_, source.class_prior_, source.class_count_]):
                                    np.testing.assert_allclose(actual, wanted, atol=3.e-8, rtol=3.e-8, equal_nan=True)
                            elif flag == 4:
                                for actual, wanted in zip(values, [source.feature_log_prob_, source.class_log_prior_]):
                                    np.testing.assert_allclose(actual, wanted, atol=3.e-8, rtol=3.e-8)
                            elif flag == 6:
                                np.testing.assert_array_equal(values[0], source._fit_X)
                                yy = source._y if is_regressor(source) else source.classes_[source._y]
                                np.testing.assert_array_equal(values[1], yy)
                            elif flag in [5, 7]:
                                assert 'stats' in locals()
                            elif flag == 8:
                                train_x, target = head_training(mode, expected, x, y, j)
                                np.testing.assert_array_equal(values[0], train_x)
                                signed = np.where(target == source.classes_[1], 1., -1.)
                                dual = values[1]; alpha = dual*signed; gram = rbf_kernel(train_x, gamma=.4)
                                assert alpha.min() >= -1.e-9 and alpha.max() <= .7+1.e-9
                                assert abs(dual.sum()) <= 3.e-8
                                margin = signed*(gram@dual+values[2][0])
                                low = alpha < 1.e-7; high = alpha > .7-1.e-7; free = ~low & ~high
                                violation = max(np.max(np.maximum(1-margin[low], 0), initial=0),
                                    np.max(np.maximum(margin[high]-1, 0), initial=0),
                                    np.max(abs(margin[free]-1), initial=0))
                                assert violation <= 3.e-6, (context, stage, j, 'SVC KKT', violation)
                                source_dual = np.zeros(len(train_x)); source_dual[source.support_] = source.dual_coef_[0]
                                objective = alpha.sum()-.5*dual@gram@dual
                                optimum = (source_dual*signed).sum()-.5*source_dual@gram@source_dual
                                assert objective >= optimum-3.e-7*max(1., abs(optimum))
                            else:
                                raise AssertionError(('unexpected native head state', flag))
                            state_oracle = 'source_cart_certificate' if flag in [5, 7] else 'source_svm_optimality' if flag == 8 else 'sklearn'
                            record('head_state_'+str(j), stage, state_oracle)
                        heads.append(dict(status=status, predict=pred, confidence=confidence, probability=native_probability, alternative=component_alternative))
                    if ready:
                        proxy = native_proxy(expected, heads)
                        reconstructed = source_matrix(proxy.predict(query), m); identity_oracle = 'source_native_head_identity'
                        if not np.array_equal(prediction, reconstructed):
                            assert mode == 2, (context, stage, 'unexpected reduction mismatch')
                            witnesses = certify_parent_boundary(mode, prediction, proxy, query); assert witnesses
                            certificates.append(dict(seed=seed, mode=mode, base=cfg['base'], case=cfg['case'],
                                stage=stage, kind='native_decoder_boundary', witnesses=witnesses))
                            identity_oracle = 'source_native_decoder_boundary'; reconstructed = prediction
                        np.testing.assert_array_equal(prediction, reconstructed, err_msg=str((context, stage, 'native-head reduction')))
                        record('reduction_identity', stage, identity_oracle)
                        prediction_oracle = 'sklearn'
                        original_prediction = source_matrix(expected.predict(query), m)
                        if not np.array_equal(prediction, original_prediction):
                            if any(head.get('alternative', False) for head in heads):
                                prediction_oracle = 'source_certified_head_reduction'
                            else:
                                witnesses = certify_parent_boundary(mode, prediction, expected, query)
                                assert witnesses; prediction_oracle = 'source_decoder_boundary'
                                certificates.append(dict(seed=seed, mode=mode, base=cfg['base'], case=cfg['case'],
                                    stage=stage, kind='decoder_boundary', witnesses=witnesses))
                            original_prediction = prediction
                        np.testing.assert_array_equal(prediction, original_prediction, err_msg=str((context, stage, 'prediction')))
                        record('prediction', stage, prediction_oracle)
                        if probability is not None:
                            np.testing.assert_allclose(probability, proxy.predict_proba(query), atol=3.e-12, rtol=3.e-12, equal_nan=True)
                            wanted = expected.predict_proba(query); oracle = 'sklearn'
                            if not np.allclose(probability, wanted, atol=3.e-7, rtol=3.e-7, equal_nan=True):
                                assert any(head.get('alternative', False) for head in heads)
                                wanted = proxy.predict_proba(query); oracle = 'source_certified_head_reduction'
                            np.testing.assert_allclose(probability, wanted, atol=3.e-7, rtol=3.e-7, equal_nan=True)
                            record('probability', stage, oracle)
                        if decision is not None:
                            np.testing.assert_allclose(decision, source_matrix(proxy.decision_function(query), m),
                                atol=3.e-12, rtol=3.e-12, equal_nan=True)
                            wanted = source_matrix(expected.decision_function(query), m); oracle = 'sklearn'
                            if not np.allclose(decision, wanted, atol=3.e-7, rtol=3.e-7, equal_nan=True):
                                assert any(head.get('alternative', False) for head in heads)
                                wanted = source_matrix(proxy.decision_function(query), m); oracle = 'source_certified_head_reduction'
                            np.testing.assert_allclose(decision, wanted, atol=3.e-7, rtol=3.e-7, equal_nan=True)
                            record('decision', stage, oracle)
                        target = truth if multi else truth[:, 0]
                        scorer = expected if prediction_oracle == 'sklearn' else copy.deepcopy(proxy)
                        if prediction_oracle != 'sklearn':
                            native_decoded = prediction if multi else prediction[:, 0]
                            scorer.predict = lambda X, r=native_decoded: r.copy()
                        np.testing.assert_allclose(scores, [scorer.score(query, target),
                            scorer.score(query, target, sample_weight=sw)], atol=3.e-12, rtol=3.e-12)
                        record('score', stage, prediction_oracle)
                    else:
                        try:
                            expected.predict(query)
                        except NotFittedError:
                            pass
                        else:
                            raise AssertionError(('expected unfitted pair', context, stage))
                        record('unfitted_prediction', stage)
                np.testing.assert_array_equal(vector(), vector())
                after_state = read_state()
                assert template_state[0] == after_state[0]
                np.testing.assert_array_equal(template_state[1], after_state[1])
                assert len(template_state[2]) == len(after_state[2])
                for before, after in zip(template_state[2], after_state[2]):
                    np.testing.assert_array_equal(before, after)
                record('template_preserved', oracle='input_identity')
            np.testing.assert_array_equal(matrix(n), x); np.testing.assert_array_equal(matrix(m), query)
            np.testing.assert_array_equal(matrix(n, int), y); np.testing.assert_array_equal(vector(), w)
            assert list(lines) == []; record('input_preserved', oracle='input_identity'); fixture_count += 1
        print(f'seed {seed}: multiclass reductions passed', flush=True)
    (RESULTS/'multiclass.json').write_text(json.dumps(dict(sklearn=sklearn.__version__, fixtures=fixture_count,
        checks=checks, certificates=certificates), indent=2)+'\n')
    print(f'{len(checks)} multiclass groups across {fixture_count} fixtures passed')


if __name__ == '__main__':
    main()
