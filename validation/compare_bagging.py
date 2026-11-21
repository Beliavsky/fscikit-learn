"""Bagging sampling, fitted members, aggregation, warm starts and OOB vs sklearn."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import subprocess
import warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import sklearn
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble._bagging import _generate_bagging_indices
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, RidgeClassifier, ElasticNet
from sklearn.metrics import accuracy_score, r2_score
from compare_weighted_trees import certify_cart

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile):
    opts = ['--profile', profile] if profile else ['--flag', FLAGS]
    run = subprocess.run(['fpm', 'run', *opts, '--target', 'reference_bagging_driver', '--runner'],
                         cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    paths = [ROOT / line.strip().strip('"') for line in run.stdout.splitlines()
             if line.strip().strip('"').endswith(('reference_bagging_driver', 'reference_bagging_driver.exe'))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, run.stdout
    return paths[0]


def base_model(mode, code, case=0):
    if code < 0:
        return None
    if code == 6:
        return ExtraTreeClassifier() if mode == 0 else ExtraTreeRegressor()
    if mode == 0:
        return [DecisionTreeClassifier(max_depth=3), ExtraTreeClassifier(max_depth=3),
                KNeighborsClassifier(n_neighbors=1 if case == 12 else 3, algorithm='brute'), GaussianNB(),
                LogisticRegression(C=.7, tol=1.e-10, max_iter=500, solver='newton-cg'),
                RidgeClassifier(alpha=.7)][code]
    return [DecisionTreeRegressor(max_depth=3), ExtraTreeRegressor(max_depth=3),
            KNeighborsRegressor(n_neighbors=1 if case == 12 else 3, algorithm='brute'), LinearRegression(),
            Ridge(alpha=.7), ElasticNet(alpha=.05, l1_ratio=.3, tol=1.e-10, max_iter=10000)][code]


def fixtures():
    for name in ['bagging_ols_rank_deficient', 'bagging_extra_default_example', 'bagging_classifier_example']:
        witness = json.loads((ROOT/f'validation/fixtures/{name}.json').read_text())
        yield witness['cfg'], *(np.array(witness[key]) for key in ['x', 'y', 'w', 'q'])
    for seed in [0, 7, 42, 2026]:
        rng = np.random.default_rng(seed+834)
        for mode in range(2):
            for code in range(-1, 6):
                for case in range(13):
                    if case == 5 and code not in ([-1, 0, 1, 3, 4, 5] if mode == 0 else [-1, 0, 1, 3, 4]):
                        continue
                    if case == 8 and code not in [-1, 0, 1]:
                        continue
                    if case == 9 and code != 2:
                        continue
                    if case in [10, 11] and code not in [-1, 0, 1]:
                        continue
                    if case == 12 and code != 2:
                        continue
                    n, p, m = (100, 100, 7) if case in [8, 9] else (24, 3, 7)
                    x = rng.normal(size=(n, p)); q = rng.normal(size=(m, p))
                    ids = np.arange(n) % 3
                    x[:, 0] += ids*2
                    y = np.array([-7, 4, 19])[ids] if mode == 0 else .7*x[:, 0]-x[:, 1]+rng.normal(size=n)*.2
                    w = np.ones(n)
                    cfg = dict(mode=mode, code=code, case=case, seed=seed, b=5, warm=0, oob=0,
                               boot=1, bootf=0, ms=0, mf=0, sf=0., ff=0., weighted=0)
                    if case == 1:
                        cfg.update(ms=13, mf=2)
                    elif case == 2:
                        cfg.update(boot=0, sf=.65, ff=.5)
                    elif case == 3:
                        cfg.update(bootf=1, mf=p)
                    elif case == 4:
                        cfg.update(oob=1, b=17)
                    elif case == 5:
                        w = rng.uniform(.2, 3, n); w[::5] = 0
                        cfg.update(weighted=1)
                    elif case == 6:
                        cfg.update(b=2, warm=5)
                    elif case == 7:
                        cfg.update(ms=3, mf=2)
                    elif case == 8:
                        cfg.update(boot=0, ms=1, mf=1, b=2)
                    elif case == 9:
                        cfg.update(boot=0, ms=99, mf=99, b=2)
                    elif case == 10:
                        w = rng.uniform(.5, 2, n); w[::7] *= -.2
                        cfg.update(weighted=1)
                    elif case == 11:
                        w[:] = -1
                        cfg.update(weighted=1)
                    elif case == 12:
                        cfg.update(ms=1, oob=1, b=3)
                    yield cfg, x, y, w, q


def run(executable, profile):
    checks, certificates, warm_properties = [], [], []
    cases = 0
    for cfg, x, y, w, q in fixtures():
        n, p = x.shape; m = len(q); mode, code = cfg['mode'], cfg['code']
        inp, out = RESULTS/'bagging_input.txt', RESULTS/'bagging_output.txt'
        with inp.open('w') as f:
            f.write(' '.join(map(str, [n, p, m, mode, code, cfg['b'], cfg['warm'], cfg['oob'],
                cfg['boot'], cfg['bootf'], cfg['ms'], cfg['mf'], cfg['weighted'], cfg['seed']]))+'\n')
            f.write(f"{cfg['sf']} {cfg['ff']}\n")
            for a in [x, y[None], w[None], q]:
                np.savetxt(f, a, fmt='%.17g')
        process = subprocess.run([str(executable), str(inp), str(out)], capture_output=True, text=True)
        assert process.returncode == 0, (cfg, process.stdout, process.stderr)
        rows = iter(out.read_text().splitlines())

        def vector(dtype=float):
            return np.fromstring(next(rows), sep=' ', dtype=dtype)

        def matrix(count):
            return np.array([vector() for _ in range(count)])

        def equal(a, b, name, exact=False, oracle='sklearn'):
            np.testing.assert_allclose(a, b, rtol=0 if exact else 3.e-8,
                                       atol=0 if exact else 3.e-9, equal_nan=True,
                                       err_msg=str((cfg, stage, name)))
            checks.append(dict(seed=cfg['seed'], mode=mode, code=code, case=cfg['case'], stage=stage,
                               name=name, oracle=oracle, passed=True))

        ms = cfg['ms'] or (int(cfg['sf']*n) if cfg['sf'] else n)
        mf = cfg['mf'] or (max(1, int(cfg['ff']*p)) if cfg['ff'] else p)
        ensemble_type = BaggingClassifier if mode == 0 else BaggingRegressor
        model = ensemble_type(estimator=base_model(mode, code, cfg['case']), n_estimators=cfg['b'],
            max_samples=cfg['ms'] or cfg['sf'] or 1., max_features=cfg['mf'] or cfg['ff'] or 1.,
            bootstrap=bool(cfg['boot']), bootstrap_features=bool(cfg['bootf']),
            oob_score=bool(cfg['oob']), random_state=cfg['seed'])
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            model.fit(x, y, sample_weight=w if cfg['weighted'] else None)
            states = [copy.deepcopy(model)]
            if cfg['warm']:
                model.set_params(warm_start=True, n_estimators=cfg['warm'])
                model.fit(x, y, sample_weight=w if cfg['weighted'] else None)
                states.append(model)
        for stage, expected in enumerate(states):
            count, features, k, learned_m, learned_f = vector(int)
            equal([count, features, k, learned_m, learned_f],
                  [len(expected.estimators_), p, len(expected.classes_) if mode == 0 else 0, ms, mf],
                  'fitted metadata', True)
            classes = vector(int) if k else None
            if k:
                equal(classes, expected.classes_, 'classes', True)
            prediction = vector()
            combined = matrix(m) if k else prediction[:, None]
            training = matrix(n) if k else vector()[:, None]
            score = vector()[0]
            if cfg['oob']:
                if k:
                    oob_score = vector()[0]; oob_response = matrix(n)
                else:
                    data = vector(); oob_score, oob_response = data[0], data[1:, None]
            seeds = np.random.RandomState(cfg['seed']).randint(2147483647, size=count)
            query_responses, train_responses, samples_all, oob_train = [], [], [], []
            conditioned = False
            for b, source in enumerate(expected.estimators_):
                seed, fcount, scount = vector(int)
                selected = vector(int)-1; sampled = vector(int)-1
                equal(seed, seeds[b], f'member {b} seed', True)
                ef, es = _generate_bagging_indices(int(seed), bool(cfg['bootf']), bool(cfg['boot']), p, n, mf, ms)
                equal(selected, ef, f'member {b} feature draws', True)
                equal(sampled, es, f'member {b} sample draws', True)
                equal([fcount, scount], [mf, ms], f'member {b} draw counts', True)
                if k:
                    probability = bool(vector(int)[0])
                    if probability:
                        local_classes = vector(int); response = matrix(m); train = matrix(n)
                        equal(local_classes, source.classes_, f'member {b} local classes', True)
                        with np.errstate(all='ignore'):
                            reference = source.predict_proba(q[:, selected]); reference_train = source.predict_proba(x[:, selected])
                    else:
                        response = vector(int)[:, None]; train = vector(int)[:, None]
                        reference = source.predict(q[:, selected])[:, None]
                        reference_train = source.predict(x[:, selected])[:, None]
                else:
                    response = vector()[:, None]; train = vector()[:, None]
                    reference = source.predict(q[:, selected])[:, None]
                    reference_train = source.predict(x[:, selected])[:, None]
                node_count, tree_seed = vector(int)
                mismatch = not (np.allclose(response, reference, rtol=3.e-8, atol=3.e-9, equal_nan=True) and
                                np.allclose(train, reference_train, rtol=3.e-8, atol=3.e-9, equal_nan=True))
                oracle = 'sklearn'
                if node_count < 0:
                    equal(vector(), source.coef_, f'member {b} linear coefficients')
                    equal(vector()[0], source.intercept_, f'member {b} linear intercept')
                    if node_count == -1:
                        equal(tree_seed, source.rank_, f'member {b} weighted OLS rank', True)
                if node_count > 0:
                    importance = vector(); nodes, stats, values = [], [], []
                    for _ in range(node_count):
                        nodes.append(vector(int)); stats.append(vector()); values.append(vector())
                    nodes, stats, values = np.array(nodes), np.array(stats), np.array(values)
                    equal(tree_seed, source.random_state, f'member {b} base seed', True)
                    mismatch |= not np.allclose(importance, source.feature_importances_, rtol=3.e-8, atol=3.e-9, equal_nan=True)
                    if mismatch:
                        current = np.bincount(sampled, minlength=n).astype(float) if cfg['boot'] else np.isin(np.arange(n), sampled).astype(float)
                        current *= w if cfg['weighted'] else 1
                        encoded = np.searchsorted(classes, y) if k else y
                        both, witnesses = certify_cart(nodes, stats, values, x[:, selected], encoded, current,
                            np.vstack([q[:, selected], x[:, selected]]), np.arange(k) if k else None,
                            1, 2, 100 if code in [-1, 6] else 3, 0, 0.,
                            (2 if code in [1, 6] else 0)+(1 if not k else 0), 0, int(tree_seed))
                        equal(response, both[:m], f'member {b} certified response', oracle='source_conditioned_tree_certificate')
                        equal(train, both[m:], f'member {b} certified training', oracle='source_conditioned_tree_certificate')
                        certificates.append(dict(cfg=cfg, stage=stage, member=b, witnesses=witnesses))
                        contribution = np.zeros(len(selected))
                        for node in np.flatnonzero(nodes[:, 0]):
                            left, right = nodes[node, :2]-1
                            contribution[nodes[node, 2]-1] += (stats[node, 1]*stats[node, 2]
                                - stats[left, 1]*stats[left, 2] - stats[right, 1]*stats[right, 2])
                        contribution /= stats[0, 1]
                        if contribution.sum() > 0:
                            contribution /= contribution.sum()
                        equal(importance, contribution, f'member {b} certified importance',
                              oracle='source_conditioned_tree_certificate')
                        conditioned = True; oracle = 'source_conditioned_tree_certificate'
                    else:
                        equal(importance, source.feature_importances_, f'member {b} importance')
                if oracle == 'sklearn':
                    equal(response, reference, f'member {b} response', exact=bool(k and not probability))
                    equal(train, reference_train, f'member {b} training response', exact=bool(k and not probability))
                if k:
                    aligned = np.zeros((m, k)); aligned_train = np.zeros((n, k))
                    if probability:
                        aligned[:, local_classes] = response; aligned_train[:, local_classes] = train
                    else:
                        aligned[np.arange(m), response[:, 0].astype(int)] = 1
                        aligned_train[np.arange(n), train[:, 0].astype(int)] = 1
                    query_responses.append(aligned); train_responses.append(aligned_train)
                    oob_train.append(np.repeat(train, k, axis=1) if probability and train.shape[1] == 1
                                     else train if probability else aligned_train)
                else:
                    query_responses.append(response); train_responses.append(train)
                    oob_train.append(train)
                samples_all.append(sampled)
            oracle = 'source_conditioned_member_aggregation' if conditioned else 'sklearn'
            expected_combined = np.mean(query_responses, axis=0)
            expected_training = np.mean(train_responses, axis=0)
            equal(combined, expected_combined, 'mean member responses', oracle=oracle)
            equal(training, expected_training, 'mean training responses', oracle=oracle)
            if k:
                equal(prediction, classes[np.argmax(expected_combined, axis=1)], 'predicted labels', True, oracle)
                equal(score, accuracy_score(y, classes[np.argmax(expected_training, axis=1)], sample_weight=w),
                      'weighted score', oracle=oracle)
            else:
                equal(score, r2_score(y, expected_training[:, 0], sample_weight=w), 'weighted score', oracle=oracle)
            if not conditioned:
                with np.errstate(all='ignore'), warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    equal(prediction, expected.predict(q), 'ensemble source prediction', bool(k))
                    if k:
                        equal(combined, expected.predict_proba(q), 'ensemble source probability')
            if cfg['oob']:
                total = np.zeros((n, k or 1)); counts = np.zeros(n)
                for response, sampled in zip(oob_train, samples_all):
                    mask = ~np.isin(np.arange(n), sampled)
                    total[mask] += response[mask]; counts[mask] += 1
                if k:
                    with np.errstate(all='ignore'):
                        expected_oob = total / total.sum(axis=1, keepdims=True)
                    expected_score = accuracy_score(np.searchsorted(classes, y), np.argmax(total, axis=1))
                else:
                    expected_oob = total / np.maximum(counts, 1)[:, None]
                    expected_score = r2_score(y, expected_oob[:, 0])
                equal(oob_response, expected_oob, 'OOB member reduction', oracle=oracle)
                equal(oob_score, expected_score, 'OOB score reduction', oracle=oracle)
                if not conditioned:
                    equal(oob_score, expected.oob_score_, 'source OOB score')
                    equal(oob_response, expected.oob_decision_function_ if k else expected.oob_prediction_[:, None], 'source OOB responses')
            if stage == 1:
                # sklearn 1.6.1 exposes samples only for the latest warm batch.
                observed = len(expected.estimators_samples_)
                if observed != count:
                    assert observed == cfg['warm']-cfg['b']
                    warm_properties.append(dict(cfg=cfg, observed=observed, fitted=int(count)))
        equal(matrix(n), x, 'training input preserved', True, 'input_identity')
        equal(matrix(m), q, 'query input preserved', True, 'input_identity')
        equal(vector(), y, 'target input preserved', True, 'input_identity')
        equal(vector(), w, 'weight input preserved', True, 'input_identity')
        assert list(rows) == []
        cases += 1
    # Retain the source failure for an OOB base with two local classes but
    # three ensemble classes, separately from successful fitting fixtures.
    source_errors = []
    error_x = np.arange(4.)[:, None]
    try:
        BaggingClassifier(KNeighborsClassifier(n_neighbors=1), n_estimators=1,
            max_samples=2, oob_score=True, random_state=0).fit(error_x, [0, 1, 2, 2])
    except ValueError as error:
        assert 'broadcast' in str(error) and '(2,3)' in str(error) and '(2,2)' in str(error)
        source_errors.append(dict(name='OOB incompatible local classes', error=str(error)))
    else:
        raise AssertionError('expected sklearn 1.6.1 OOB class-shape failure')
    with inp.open('w') as f:
        f.write('4 1 2 0 2 1 0 1 1 0 2 0 0 0\n0 0\n')
        for a in [error_x, np.array([[0, 1, 2, 2]]), np.ones((1, 4)), error_x[:2]]:
            np.savetxt(f, a, fmt='%.17g')
    rejected = subprocess.run([str(executable), str(inp), str(out)], text=True, capture_output=True)
    assert rejected.returncode != 0 and 'bagging OOB probability class shape mismatch' in rejected.stdout+rejected.stderr
    checks.append(dict(seed=0, mode=0, code=2, case=13, stage=0, name='OOB class-shape rejection',
                       oracle='sklearn_error', passed=True))
    source_root = Path(sklearn.__file__).parent
    report = dict(sklearn=sklearn.__version__, profile=profile or 'checked', fixtures=cases, checks=checks,
        cart_certificates=certificates, source_warm_sample_properties=warm_properties, source_errors=source_errors,
        source_sha256={name: hashlib.sha256((source_root/name).read_bytes()).hexdigest()
            for name in ['ensemble/_bagging.py', 'ensemble/_base.py', 'utils/_random.pyx',
                         'linear_model/_base.py', 'linear_model/_ridge.py']})
    (RESULTS/f'bagging_{profile or "checked"}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'{len(checks)} bagging groups across {cases} fixtures passed; {len(certificates)} tree certificates', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['release', 'debug'])
    parser.add_argument('--executable', type=Path)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    run(args.executable.resolve() if args.executable else resolve(args.profile), args.profile)
