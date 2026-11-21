"""RegressorChain ordering, cross-validated features, weights and fitted state."""
from pathlib import Path
import argparse
import copy
from sklearn.base import clone
import json
import os
import subprocess
import warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
import numpy as np
import sklearn
from sklearn.multioutput import RegressorChain
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile=None):
    selection = ['--profile', profile] if profile else ['--flag', FLAGS]
    run = subprocess.run(['fpm', 'run', *selection, '--target', 'reference_chain_driver', '--runner'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    paths = [ROOT/line.strip().strip('"') for line in run.stdout.splitlines()
        if line.strip().strip('"').endswith(('reference_chain_driver.exe', 'reference_chain_driver'))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, run.stdout
    return paths[0]


def certify_tree(nodes, x, y, query):
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
        stump = DecisionTreeRegressor(max_depth=1, min_samples_leaf=5, random_state=0).fit(x[rows], y[rows])
        if left == 0:
            assert right == 0 and (depth == 2 or stump.tree_.node_count == 1)
            return
        assert depth < 2 and left > 0 and right > 0 and 1 <= feature <= x.shape[1]
        mask = x[rows, feature-1] <= threshold
        assert mask.sum() >= 5 and (~mask).sum() >= 5
        costs = []
        for column in range(x.shape[1]):
            ordered = np.sort(np.unique(x[rows, column]))
            for cut in (ordered[:-1]+ordered[1:])/2:
                partition = x[rows, column] <= cut
                if partition.sum() >= 5 and (~partition).sum() >= 5:
                    costs.append(sse(y[rows[partition]])+sse(y[rows[~partition]]))
        loss = sse(y[rows[mask]])+sse(y[rows[~mask]])
        bound = 128*np.finfo(float).eps*max(1., np.sum(y[rows]**2))
        assert loss <= min(costs)+bound
        source_feature = stump.tree_.feature[0]
        assert source_feature >= 0
        source_mask = x[rows, source_feature].astype(np.float32) <= stump.tree_.threshold[0]
        assert abs(loss-(sse(y[rows[source_mask]])+sse(y[rows[~source_mask]]))) <= bound
        if feature-1 != source_feature or not np.array_equal(mask, source_mask):
            ties += 1
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


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--executable', type=Path); parser.add_argument('--profile')
    args = parser.parse_args()
    executable = args.executable.resolve() if args.executable else resolve(args.profile)
    RESULTS.mkdir(exist_ok=True); checks = []; certificates = []; sgd_rounding = []; fixtures = 0
    for seed in [0, 7, 42, 2026, 314159]:
        rng = np.random.default_rng(seed)
        for base in range(6):
            for kind in range(16):
                if kind in [7, 11] and base != 3:
                    continue
                n, p, m, t = 64, 3, 17, 3
                if kind in [9, 15]:
                    t = 1
                if kind == 10:
                    m = 1
                x = rng.normal(size=(n, p)); query = rng.normal(size=(m, p))
                coef = rng.normal(size=(p, t))
                y = x@coef+.3*rng.normal(size=(n, t))+np.arange(t)
                if t > 1:
                    y[:, 1] += .5*y[:, 0]**2
                truth = query@coef+.3*rng.normal(size=(m, t))+np.arange(t)
                if t > 1:
                    truth[:, 1] += .5*truth[:, 0]**2
                weighted = kind in [7, 11]; refit = kind == 8
                w = rng.uniform(.1, 3, n) if weighted else np.ones(n)
                if weighted:
                    w[::7] = 0
                sw = rng.uniform(.1, 2, m)
                order_kind = 1 if kind in [1, 5, 8, 10] else 2 if kind in [2, 6, 7] else 0
                cv = {4: 3, 5: 4, 6: 2, 7: 3, 8: 3, 10: 2}.get(kind, 0)
                if kind == 15:
                    cv = 99
                if kind in [3, 11]:
                    order_kind = 3
                if kind in [12, 13, 14]:
                    order_kind = kind-8
                estimator = [LinearRegression(), Ridge(alpha=.7, solver='svd'),
                    ElasticNet(alpha=.03, l1_ratio=.4, tol=1.e-12, max_iter=20000),
                    SGDRegressor(learning_rate='constant', eta0=.02, alpha=.001,
                        max_iter=4, tol=None, shuffle=False),
                    KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='brute'),
                    DecisionTreeRegressor(max_depth=2, min_samples_leaf=5, random_state=0)][base]
                order = np.arange(t)[::-1] if order_kind == 1 else 'random' if order_kind == 2 else None
                splits = None
                if order_kind >= 3:
                    ids = np.arange(n)
                    splits = [(ids[ids % 3 != k], ids[ids % 3 == k]) for k in range(3)]
                    if order_kind == 4:
                        splits = [(ids, test) for train, test in splits]
                    if order_kind == 5:
                        splits = [(train[(train+1) % 2 == 0], test) for train, test in splits]
                    if order_kind == 6:
                        splits = [(np.r_[train, train[:2]], test) for train, test in splits]
                chain = RegressorChain(estimator, order=order, cv=splits if splits is not None else cv or None,
                    random_state=seed)
                if seed == 0 and base == 3 and kind == 7:
                    saved = json.loads((ROOT/'validation/fixtures/regressor_chain_lazy_l2.json').read_text())
                    assert saved['cv'] == cv and saved['order_kind'] == order_kind
                    for name, value in [('x', x), ('y', y), ('query', query), ('truth', truth),
                                        ('sample_weight', w), ('score_weight', sw)]:
                        np.testing.assert_array_equal(value, saved[name], err_msg='saved lazy-L2 regression fixture')
                data, output = RESULTS/'chain_input.txt', RESULTS/'chain_actual.txt'
                with data.open('w') as stream:
                    stream.write(' '.join(map(str, [n, p, m, t, base, order_kind, cv, seed, int(weighted), int(refit)]))+'\n')
                    for a in [w[None], sw[None], x, y, query, truth]:
                        np.savetxt(stream, a, fmt='%.17g')
                run = subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                assert run.returncode == 0, (seed, base, kind, run.stdout)
                lines = iter(output.read_text().splitlines())
                def vector(dtype=float):
                    return np.fromstring(next(lines), sep=' ', dtype=dtype)
                def matrix(rows):
                    return np.array([vector() for _ in range(rows)])
                def record(name, stage=0, oracle='sklearn'):
                    checks.append(dict(seed=seed, base=base, kind=kind, stage=stage, name=name,
                        oracle=oracle, passed=True))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    estimator.fit(x, y[:, 0]); before = estimator.predict(query)
                    chain.fit(x, y, **(dict(sample_weight=w) if weighted else {}))
                    states = [copy.deepcopy(chain)]
                    if refit:
                        chain.set_params(order=None, cv=None)
                        chain.fit(x, y[:, :1]); states.append(chain)
                    for stage, expected in enumerate(states):
                        context = (seed, base, kind, stage)
                        features, outputs = vector(int)
                        assert features == p and outputs == len(expected.estimators_)
                        np.testing.assert_array_equal(vector(int), expected.order_+1)
                        record('metadata_order', stage)
                        actual_prediction, actual_score = matrix(m), vector()
                        coefficients = []
                        for j, model in enumerate(expected.estimators_):
                            flag = int(vector(int)[0])
                            if flag:
                                coefficients.append((j, vector()))
                        reference_prediction = expected.predict(query); oracle = 'sklearn'
                        conditioned = None
                        if base == 3:
                            design = matrix(n)
                            assert design.shape == (n, p+outputs-1)
                            np.testing.assert_array_equal(design[:, :p], x)
                            if expected.cv is None:
                                np.testing.assert_array_equal(design[:, p:], y[:, expected.order_[:-1]])
                            else:
                                if isinstance(expected.cv, int):
                                    current_folds = list(KFold(n_splits=expected.cv).split(x)) if outputs > 1 else []
                                else:
                                    current_folds = expected.cv
                                for j, column in enumerate(expected.order_[:-1]):
                                    source_feature = np.zeros(n)
                                    for train, test in current_folds:
                                        source_fold = clone(estimator).fit(design[train, :p+j], y[train, column])
                                        source_feature[test] = source_fold.predict(design[test, :p+j])
                                    np.testing.assert_allclose(design[:, p+j], source_feature, atol=3.e-8, rtol=3.e-8,
                                        err_msg=str((context, j, 'OOF features')))
                            conditioned = [clone(estimator).fit(design[:, :p+j], y[:, column],
                                **(dict(sample_weight=w) if weighted and stage == 0 else {}))
                                for j, column in enumerate(expected.order_)]
                            record('sgd_design_certificate', stage, 'source_conditioned_sgd')

                        if base == 5:
                            design = np.zeros((n, p+outputs)); design[:, :p] = x
                            qdesign = np.zeros((m, p+outputs)); qdesign[:, :p] = query
                            if expected.cv is None:
                                design[:, p:] = y[:, expected.order_]
                                current_folds = []
                            elif isinstance(expected.cv, int):
                                current_folds = list(KFold(n_splits=expected.cv).split(x)) if outputs > 1 else []
                            else:
                                current_folds = expected.cv
                            reconstructed = np.zeros((m, outputs)); ties = 0
                            for j, column in enumerate(expected.order_):
                                nodes = matrix(int(vector(int)[0]))
                                values, count = certify_tree(nodes, design[:, :p+j], y[:, column], qdesign[:, :p+j])
                                ties += count; qdesign[:, p+j] = values; reconstructed[:, column] = values
                                fold_count = int(vector(int)[0])
                                assert fold_count == (len(current_folds) if j < outputs-1 else 0)
                                for train, test in current_folds[:fold_count]:
                                    nodes = matrix(int(vector(int)[0]))
                                    values, count = certify_tree(nodes, design[train, :p+j], y[train, column],
                                        design[test, :p+j])
                                    ties += count; design[test, p+j] = values
                            np.testing.assert_allclose(actual_prediction, reconstructed, atol=3.e-12, rtol=3.e-12)
                            record('tree_node_and_chain_certificate', stage, 'source_cart_certificate')
                            if not np.allclose(actual_prediction, reference_prediction, atol=3.e-8, rtol=3.e-8):
                                assert ties > 0
                                certificates.append(dict(seed=seed, base=base, kind=kind, stage=stage, ties=ties))
                                reference_prediction = reconstructed; oracle = 'source_cart_tie_certificate'
                        np.testing.assert_allclose(actual_prediction, reference_prediction, atol=3.e-8, rtol=3.e-8,
                            err_msg=str(context)); record('prediction', stage, oracle)
                        np.testing.assert_allclose(actual_score, [r2_score(truth[:, :outputs], reference_prediction),
                            r2_score(truth[:, :outputs], reference_prediction, sample_weight=sw)],
                            atol=3.e-8, rtol=3.e-8, equal_nan=True, err_msg=str(context)); record('score', stage, oracle)
                        for j, actual in coefficients:
                            model = expected.estimators_[j]; wanted = np.r_[model.coef_, model.intercept_]
                            if hasattr(model, 't_'):
                                wanted = np.r_[wanted, model.t_]
                            parameter_oracle = 'sklearn'
                            if conditioned is not None:
                                if not np.allclose(actual, wanted, atol=3.e-8, rtol=3.e-8):
                                    sgd_rounding.append(dict(seed=seed, base=base, kind=kind, stage=stage, output=j,
                                        max_scaled_error=float(np.max(abs(actual-wanted)/np.maximum(1., abs(wanted))))))
                                source = conditioned[j]
                                wanted = np.r_[source.coef_, source.intercept_, source.t_]
                                parameter_oracle = 'source_conditioned_sgd'
                            np.testing.assert_allclose(actual, wanted, atol=3.e-8, rtol=3.e-8,
                                err_msg=str((context, j))); record('coef_intercept_counter_'+str(j), stage, parameter_oracle)
                    original_before, original_after = vector(), vector()
                    np.testing.assert_array_equal(original_before, original_after)
                    if base == 5:
                        certified, ties = certify_tree(matrix(int(vector(int)[0])), x, y[:, 0], query)
                        np.testing.assert_allclose(original_before, certified, atol=3.e-12, rtol=3.e-12)
                        if not np.allclose(original_before, before, atol=3.e-8, rtol=3.e-8):
                            assert ties > 0
                            certificates.append(dict(seed=seed, base=base, kind=kind, stage='template', ties=ties))
                            before = certified
                    np.testing.assert_allclose(original_before, before, atol=3.e-8, rtol=3.e-8)
                    record('template_preserved', oracle='input_identity')
                np.testing.assert_array_equal(matrix(n), x); np.testing.assert_array_equal(matrix(n), y)
                np.testing.assert_array_equal(matrix(m), query); np.testing.assert_array_equal(vector(), w)
                assert list(lines) == []; record('input_preserved', oracle='input_identity')
                if seed == 0 and base == 3 and kind == 7:
                    record('lazy_l2_regression_fixture', oracle='fixture_identity')
                fixtures += 1
        print(f'seed {seed}: regressor chains passed', flush=True)
    (RESULTS/'chains.json').write_text(json.dumps(dict(sklearn=sklearn.__version__, fixtures=fixtures,
        checks=checks, cart_certificates=certificates, sgd_rounding=sgd_rounding), indent=2)+'\n')
    print(f'{len(checks)} regressor-chain groups across {fixtures} fixtures passed')


if __name__ == '__main__':
    main()
