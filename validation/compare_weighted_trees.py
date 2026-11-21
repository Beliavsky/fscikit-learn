"""Weighted CART and standalone ExtraTrees against installed sklearn trees.

Compares complete fitted nodes as well as held-out/NaN query responses. These
    Source-conditioned stump certificates distinguish roundoff-level CART ties
    from ordinary source comparisons; all fixtures remain in the report.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn_tree_runtime import tree_runtime_provenance
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                          ExtraTreeClassifier, ExtraTreeRegressor)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def certify_cart(nodes, stats, values, x, y, w, q, classes, leaf, split, depth, criterion, decrease,
                 mode, nf, seed):
    """Independently fit sklearn stumps on every native node's training rows.

    A different split is admissible only if its source weighted child impurity
    is within a floating-point accumulation bound of the optimal source stump.
    Predictions are then computed from source-fitted leaf responses.
    """
    z = x.astype(np.float32)
    responses = {}
    source_type = DecisionTreeClassifier if classes is not None else DecisionTreeRegressor
    root_mass = w.sum()
    ties = []
    seen = set()
    features = list(range(x.shape[1]))
    constants = [0]*x.shape[1]
    state = int(np.random.RandomState(seed).randint(0, 2147483647))
    effective = nf or (max(1, int(np.sqrt(x.shape[1]))) if mode == 2 else x.shape[1])

    def draw():
        nonlocal state
        state = state or 1
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        return state & 0x7fffffff

    def candidates(ids, known):
        fi, found, drawn, visited, nconstant = len(features), 0, 0, 0, known
        choices = []
        while fi > nconstant and (visited < effective or visited <= found+drawn):
            visited += 1
            fj = drawn + draw() % (fi-found-drawn)
            if fj < known:
                features[drawn], features[fj] = features[fj], features[drawn]
                drawn += 1
                continue
            fj += found
            f = features[fj]
            low, high = z[ids, f].min(), z[ids, f].max()
            if high <= np.float32(low + np.float32(1.e-7)):
                features[fj], features[nconstant] = features[nconstant], features[fj]
                found += 1; nconstant += 1
                continue
            fi -= 1
            features[fi], features[fj] = features[fj], features[fi]
            if mode >= 2:
                threshold = (float(high)-float(low))*draw()/2147483647+float(low)
                if threshold == high:
                    threshold = float(low)
                choices.append((f, threshold))
            else:
                vals = np.sort(z[ids, f])
                choices.extend((f, float(a)/2+float(b)/2) for a, b in zip(vals[:-1], vals[1:])
                               if b > np.float32(a + np.float32(1.e-7)))
        features[:known] = constants[:known]
        constants[known:nconstant] = features[known:nconstant]
        return choices, nconstant

    def visit(i, ids, level, known=0):
        assert i not in seen
        seen.add(i)
        opts = dict(max_depth=1, min_samples_leaf=leaf, min_samples_split=split, random_state=0)
        if classes is not None:
            opts['criterion'] = ['gini', 'entropy', 'log_loss'][criterion]
        stump = source_type(**opts).fit(x[ids], y[ids], sample_weight=w[ids])
        t = stump.tree_
        np.testing.assert_allclose(stats[i, 1], t.weighted_n_node_samples[0], atol=2.e-11, rtol=2.e-10)
        assert nodes[i, 3] == len(ids) and nodes[i, 4] == level
        expected = np.zeros(len(classes)) if classes is not None else t.value[0, 0]
        if classes is not None:
            expected[np.searchsorted(classes, stump.classes_)] = t.value[0, 0]
        np.testing.assert_allclose(values[i], expected, atol=2.e-11, rtol=2.e-10, equal_nan=True)
        np.testing.assert_allclose(stats[i, 2], t.impurity[0], atol=2.e-11, rtol=2.e-10, equal_nan=True)
        responses[i] = expected
        forced = (level >= depth or len(ids) < split or len(ids)//2 < leaf or
                  stats[i, 2] <= np.finfo(float).eps or w[ids].sum() < 0)
        choices, nconstant = ([], known) if forced else candidates(ids, known)
        costs = []
        for feature, threshold in choices:
            partition = z[ids, feature] <= threshold
            if min(partition.sum(), (~partition).sum()) < leaf:
                continue
            if w[ids[partition]].sum() < 0 or w[ids[~partition]].sum() < 0:
                continue
            cost = 0.
            for subset in [ids[partition], ids[~partition]]:
                child = source_type(**opts).fit(x[subset], y[subset], sample_weight=w[subset]).tree_
                cost += child.weighted_n_node_samples[0] * child.impurity[0]
            if np.isfinite(cost):
                costs.append((feature, threshold, cost))
        source_cost = min((c for _, _, c in costs), default=t.weighted_n_node_samples[0]*t.impurity[0])
        if nodes[i, 0] == 0:
            if not forced and costs:
                gain = t.impurity[0] - source_cost / t.weighted_n_node_samples[0]
                assert gain * w[ids].sum() / root_mass <= decrease + 256*np.finfo(float).eps
            return
        assert not forced
        f = nodes[i, 2] - 1
        assert any(f == feature and abs(stats[i, 0]-threshold) <= 2.e-14*max(1.,abs(threshold))
                   for feature, threshold, _ in costs)
        mask = z[ids, f] <= stats[i, 0]
        assert mask.sum() >= leaf and (~mask).sum() >= leaf
        assert nodes[i, 5] == (mask.sum() > (~mask).sum())
        child_cost = 0.
        for child_ids in [ids[mask], ids[~mask]]:
            child = source_type(**opts).fit(x[child_ids], y[child_ids], sample_weight=w[child_ids]).tree_
            child_cost += child.weighted_n_node_samples[0] * child.impurity[0]
        bound = 256*np.finfo(float).eps*max(1., abs(source_cost), abs(child_cost),
                                         abs(t.weighted_n_node_samples[0]*t.impurity[0]))
        assert abs(child_cost - source_cost) <= bound, (i, child_cost, source_cost, bound)
        ties.append(dict(node=int(i), gap=float(child_cost-source_cost), bound=float(bound)))
        visit(nodes[i, 0]-1, ids[mask], level+1, nconstant)
        visit(nodes[i, 1]-1, ids[~mask], level+1, nconstant)

    visit(0, np.flatnonzero(w), 0)
    assert len(seen) == len(nodes)
    response = []
    for row in q.astype(np.float32):
        i = 0
        while nodes[i, 0] > 0:
            v = row[nodes[i, 2]-1]
            go_left = bool(nodes[i, 5]) if np.isnan(v) else v <= stats[i, 0]
            i = nodes[i, 0 if go_left else 1]-1
        response.append(responses[i])
    return np.array(response), ties


def resolve(profile):
    opts = ['--profile', profile] if profile else ['--flag', FLAGS]
    run = subprocess.run(['fpm', 'run', *opts, '--target',
        'reference_weighted_tree_driver', '--runner'], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    candidates = [ROOT / line.strip().strip('"') for line in run.stdout.splitlines()
                  if line.strip().strip('"').endswith(('reference_weighted_tree_driver',
                                                      'reference_weighted_tree_driver.exe'))]
    candidates = [p.resolve() for p in candidates if p.is_file()]
    assert len(candidates) == 1, run.stdout
    return candidates[0]


def fixtures():
    for seed in range(12):
        rng = np.random.default_rng(seed + 912)
        for mode in range(4):
            for case in range(12):
                n, p = 40, 4
                x = rng.normal(size=(n, p))
                q = rng.normal(size=(13, p))
                y = rng.integers(0, 3, n) * 7 - 4 if mode % 2 == 0 else rng.normal(size=n)
                w = rng.uniform(.2, 3, n)
                depth, leaf, split, nf, weighted, criterion, decrease = 2, 1, 2, 0, 1, 0, 0.
                if case == 1:
                    w[::3] = 0
                elif case == 2:
                    w = rng.integers(0, 5, n).astype(float)
                elif case == 3:
                    leaf, split, depth = 5, 13, 3
                elif case == 4:
                    nf = 2
                elif case == 5:
                    x[:, :2] = 1
                    nf = 1
                elif case == 6:
                    criterion = 1
                elif case == 7:
                    criterion, decrease = 2, .12
                elif case == 8:
                    w[:] = 0
                elif case == 9:
                    w[:] = -1
                elif case == 10:
                    # Omitted weights now use the same float32/source RNG kernel
                    # for both CART and standalone ExtraTrees.
                    weighted = 0
                    w[:] = 1
                elif case == 11:
                    w[::5] *= -.2
                q[0, :] = np.nan
                q[1, 1] = np.nan
                yield (seed, mode, case, x, q, y, w, depth, leaf, split,
                       nf, weighted, criterion, decrease)
                if mode < 2 and case in [0,3,4,5,6,7]:
                    yield (seed, mode, case+100, x, q, y, np.ones(n), depth, leaf, split,
                           nf, 0, criterion, decrease)
            yield (seed, mode, 200, np.array([[0.],[5.e-8]]), np.array([[0.],[2.5e-8],[5.e-8],[np.nan]]),
                   np.array([-4.,3.]) if mode%2==0 else np.array([0.,1.]), np.ones(2), 1,1,2,0,0,0,0.)


def run(exe, profile):
    runtime=tree_runtime_provenance()
    assert runtime["feature_threshold"]==0., "native tree baseline requires runtime FEATURE_THRESHOLD=0"
    groups = 0
    cases = 0
    pure_topology_cases = 0
    certificates = []
    checks = []
    source = [DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor]
    inp = RESULTS / 'weighted_tree_input.txt'
    out = RESULTS / 'weighted_tree_output.txt'
    for seed, mode, case, x, q, y, w, depth, leaf, split, nf, weighted, criterion, decrease in fixtures():
        with inp.open('w') as f:
            f.write(f'{mode} {len(x)} {x.shape[1]} {len(q)} {depth} {leaf} {split} '
                    f'{nf} {seed} {weighted} {criterion} {decrease}\n')
            np.savetxt(f, x, fmt='%.17g')
            np.savetxt(f, y[None, :], fmt='%.17g')
            np.savetxt(f, w[None, :], fmt='%.17g')
            np.savetxt(f, q, fmt='%.17g')
        subprocess.run([str(exe), str(inp), str(out)], check=True, capture_output=True, text=True)
        rows = iter(out.read_text().splitlines())
        count, effective, k = map(int, next(rows).split())
        classes = np.fromstring(next(rows), sep=' ') if k else None
        pred = np.fromstring(next(rows), sep=' ')
        proba = np.array([np.fromstring(next(rows), sep=' ') for _ in q]) if k else None
        importance = np.fromstring(next(rows), sep=' ')
        nodes, stats, values = [], [], []
        for _ in range(count):
            nodes.append(np.fromstring(next(rows), sep=' ', dtype=int))
            stats.append(np.fromstring(next(rows), sep=' '))
            values.append(np.fromstring(next(rows), sep=' '))
        assert not list(rows)
        nodes, stats, values = np.array(nodes), np.array(stats), np.array(values)
        opts = dict(max_depth=depth, min_samples_leaf=leaf, min_samples_split=split,
                    random_state=seed, min_impurity_decrease=decrease,
                    max_features=nf or ('sqrt' if mode == 2 else None))
        if k:
            opts['criterion'] = ['gini', 'entropy', 'log_loss'][criterion]
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            model = source[mode](**opts).fit(x, y, sample_weight=w if weighted else None)
            expected_pred = model.predict(q)
            expected_proba = model.predict_proba(q) if k else None
            expected_importance = model.feature_importances_
        label = f'seed={seed}, mode={mode}, case={case}'
        oracle = 'sklearn'

        def equal(a, b, name, exact=False):
            nonlocal groups
            try:
                np.testing.assert_allclose(a, b, rtol=0 if exact else 2.e-10,
                                           atol=0 if exact else 2.e-11, equal_nan=True)
            except AssertionError as error:
                raise AssertionError(f'{label}: {name}\n{error}') from error
            groups += 1
            checks.append(dict(seed=seed, mode=mode, case=case, name=name, oracle=oracle, passed=True))

        tree = model.tree_
        mismatch = not np.allclose(pred, expected_pred, rtol=2.e-10, atol=2.e-11, equal_nan=True)
        if k:
            mismatch |= not np.allclose(proba, expected_proba, rtol=2.e-10, atol=2.e-11, equal_nan=True)
        mismatch |= not np.allclose(importance, expected_importance, rtol=2.e-10, atol=2.e-11, equal_nan=True)
        if count == tree.node_count:
            mismatch |= not np.array_equal(nodes[:, 2], np.where(tree.feature < 0, 0, tree.feature+1))
        if mismatch:
            oracle = 'source_conditioned_tree_certificate'
            response, ties = certify_cart(nodes, stats, values, x, y, w, q, classes,
                                         leaf, split, depth, criterion, decrease, mode, nf, seed)
            equal(pred, classes[np.argmax(response, axis=1)] if k else response[:, 0],
                  'source-conditioned CART predictions', bool(k))
            if k:
                equal(proba, response, 'source-conditioned CART probabilities')
            contributions = np.zeros(x.shape[1])
            for i in np.flatnonzero(nodes[:, 0]):
                l, r = nodes[i, :2]-1
                contributions[nodes[i, 2]-1] += stats[i, 1]*stats[i, 2]-stats[l, 1]*stats[l, 2]-stats[r, 1]*stats[r, 2]
            contributions /= stats[0, 1]
            if contributions.sum() > 0:
                contributions /= contributions.sum()
            equal(importance, contributions, 'certified feature importance identity')
            certificates.append(dict(seed=seed, mode=mode, case=case, splits=ties))
            cases += 1
            continue

        tree = model.tree_
        # C's incremental moments can put a pure child's impurity just above
        # epsilon while recomputed moments put it below (or vice versa). Prove
        # every collapsed subtree has identical responses before comparing the
        # response-equivalent topology. Never collapse an impure alternative.
        def canonical(left, right, impurity, value):
            kept = []
            def visit(i):
                pure = k and np.count_nonzero(value[i]) == 1 and np.max(value[i]) == 1
                if left[i] >= 0 and pure and abs(impurity[i]) < 1.e-13:
                    stack = [left[i], right[i]]
                    while stack:
                        child = stack.pop()
                        np.testing.assert_array_equal(value[child], value[i])
                        assert abs(impurity[child]) < 1.e-13
                        if left[child] >= 0:
                            stack.extend([left[child], right[child]])
                    kept.append((i, False))
                elif left[i] < 0:
                    kept.append((i, False))
                else:
                    kept.append((i, True)); visit(left[i]); visit(right[i])
            visit(0)
            ids = np.array([i for i, _ in kept])
            internal = np.array([v for _, v in kept], dtype=bool)
            return ids, internal

        native_left = nodes[:, 0] - 1
        native_right = nodes[:, 1] - 1
        ni, nn = canonical(native_left, native_right, stats[:, 2], values)
        si, sn = canonical(tree.children_left, tree.children_right, tree.impurity, tree.value[:, 0, :])
        if len(ni) != count or len(si) != tree.node_count:
            pure_topology_cases += 1
            oracle = 'source_pure_subtree_identity'
        equal(len(ni), len(si), 'response-equivalent node count', True)
        equal(nn, sn, 'response-equivalent topology', True)
        equal(effective, model.max_features_, 'effective feature count', True)
        equal(pred, expected_pred, 'prediction', bool(k))
        equal(importance, expected_importance, 'importance')
        if k:
            equal(classes, model.classes_, 'classes', True)
            equal(proba, expected_proba, 'probabilities')
        equal(nodes[ni[nn], 2], tree.feature[si[sn]] + 1, 'features', True)
        equal(nodes[ni, 3], tree.n_node_samples[si], 'active row counts', True)
        equal(nodes[ni[nn], 5], tree.missing_go_to_left[si[sn]], 'missing routing', True)
        equal(stats[ni[nn], 0], tree.threshold[si[sn]], 'thresholds')
        equal(stats[ni, 1], tree.weighted_n_node_samples[si], 'weighted masses')
        equal(stats[ni, 2], tree.impurity[si], 'impurity')
        equal(values[ni], tree.value[si, 0, :], 'node responses')
        cases += 1
    report = dict(sklearn_version=sklearn.__version__, profile=profile or 'checked',
                  cases=cases, groups=groups, checks=checks,
                  pure_topology_cases=pure_topology_cases,
                  certificates=certificates,
                  tree_runtime=runtime,
                  note='Source comparisons plus recorded source-conditioned split/response and pure-subtree certificates.')
    source_root = Path(sklearn.__file__).parent
    report['source_sha256'] = {name: hashlib.sha256((source_root/name).read_bytes()).hexdigest()
        for name in ['tree/_classes.py', 'tree/_tree.pyx', 'tree/_splitter.pyx',
                     'tree/_partitioner.pyx', 'tree/_criterion.pyx', 'utils/_random.pxd']}
    path = RESULTS / f'weighted_trees_{profile or "checked"}.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['checks', 'certificates']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['release', 'debug'])
    parser.add_argument('--exe', type=Path)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    run(args.exe or resolve(args.profile), args.profile)
