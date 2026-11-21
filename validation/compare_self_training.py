"""Self-training histories and delegated predictions versus sklearn 1.6.1."""
from pathlib import Path
import argparse
import json
import subprocess
import warnings
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile=None):
    command = ['fpm', 'run', '--target', 'reference_self_training_driver', '--runner']
    command += ['--profile', profile] if profile else ['--flag', FLAGS]
    run = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=True)
    paths = [ROOT / s.strip().strip('"') for s in run.stdout.splitlines()
             if s.strip().strip('"').endswith(('reference_self_training_driver.exe', 'reference_self_training_driver'))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, run.stdout
    return paths[0]


def certify_history(estimator, x, y, actual, iterations, reason, threshold, criterion, k_best, cap):
    """Certify every native choice with source base fits, including tied cutoffs."""
    transduction = y.copy()
    history = actual['history']
    np.testing.assert_array_equal(history[y != -1], np.zeros(np.sum(y != -1), dtype=int))
    assert np.all((history[y == -1] == -1) | ((history[y == -1] > 0) & (history[y == -1] <= iterations)))
    ties = 0
    selected = np.array([], dtype=int)
    for iteration in range(1, iterations + 1):
        labeled = (history >= 0) & (history < iteration)
        query = np.flatnonzero(~labeled)
        assert len(query) > 0 and (cap == -1 or iteration <= cap)
        model = clone(estimator).fit(x[labeled], transduction[labeled])
        proba = model.predict_proba(x[query])
        confidence = proba.max(axis=1)
        selected = np.flatnonzero(history[query] == iteration)
        if criterion == 'threshold':
            np.testing.assert_array_equal(selected, np.flatnonzero(confidence > threshold))
        else:
            take = min(k_best, len(query))
            assert len(selected) == take
            remaining = np.setdiff1d(np.arange(len(query)), selected)
            if len(remaining):
                low, high = confidence[selected].min(), confidence[remaining].max()
                # Base probability arithmetic can move saturated/tied values by a few ulps.
                bound = 32 * np.finfo(float).eps if not isinstance(estimator, LogisticRegression) else 0.
                assert low + bound >= high, (low, high, bound)
                if abs(low - high) <= bound:
                    ties += 1
        transduction[query[selected]] = model.classes_[proba[selected].argmax(axis=1)]
        if len(selected) == 0:
            assert iteration == iterations
    np.testing.assert_array_equal(actual['transduction'], transduction)
    expected_reason = 'no_change' if iterations > 0 and len(selected) == 0 else ''
    if iterations == cap:
        expected_reason = 'max_iter'
    if np.all(history != -1):
        expected_reason = 'all_labeled'
    assert expected_reason and reason == expected_reason
    model = clone(estimator).fit(x[history != -1], transduction[history != -1])
    return model, ties


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--profile')
    args = parser.parse_args()
    executable = args.executable.resolve() if args.executable else resolve(args.profile)
    RESULTS.mkdir(exist_ok=True)
    checks = []
    tie_cases = []
    case_count = 0
    for seed in [0, 7, 42, 2026, 314159]:
        rng = np.random.default_rng(seed)
        for classes in [1, 2, 3, 4]:
            for base in range(3):
                if classes == 1 and base == 2:
                    continue  # LogisticRegression itself requires at least two classes.
                n, p, m = 40 + classes * 4, classes + 1, 13
                centers = rng.normal(size=(classes, p)) * 1.6
                target = np.arange(n) % classes
                x = centers[target] + rng.normal(size=(n, p))
                query = rng.normal(size=(m, p)) * 2
                labels = np.array([-7, 0, 4, 19])[:classes]
                y = labels[target]
                y[classes * 3:] = -1
                for mode in range(9):
                    criterion = 'threshold' if mode < 5 else 'k_best'
                    threshold = [0., .5, .75, .999999, .8, .75, .75, .75, .75][mode]
                    cap = [10, 1, -1, 1, 0, 2, -1, 1, 10][mode]
                    k_best = [3, 3, 3, 3, 3, 2, 4, n, 3][mode]
                    neighbors = min(classes * 3, 3 if mode % 2 else 5)
                    distance = mode % 2
                    current_y = labels[target] if mode == 8 else y.copy()
                    if base == 0:
                        estimator = GaussianNB()
                    elif base == 1:
                        estimator = KNeighborsClassifier(n_neighbors=neighbors,
                            weights='distance' if distance else 'uniform', algorithm='brute')
                    else:
                        estimator = LogisticRegression(C=.8, tol=1.e-10, max_iter=500, solver='newton-cg')
                    data, output = RESULTS / 'self_training_input.txt', RESULTS / 'self_training_actual.txt'
                    with data.open('w') as stream:
                        stream.write(f'{n} {p} {m} {base} {int(criterion == "k_best")} {cap} '
                                     f'{k_best} {neighbors} {distance} {threshold}\n')
                        np.savetxt(stream, current_y[None], fmt='%d')
                        np.savetxt(stream, x, fmt='%.17g')
                        np.savetxt(stream, query, fmt='%.17g')
                    run = subprocess.run([str(executable), str(data), str(output)], cwd=ROOT,
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    assert run.returncode == 0, run.stdout
                    lines = iter(output.read_text().splitlines())
                    actual_iter = int(next(lines)); actual_reason = next(lines).strip(); k = int(next(lines))
                    actual = dict(classes=np.fromstring(next(lines), sep=' ', dtype=int),
                        transduction=np.fromstring(next(lines), sep=' ', dtype=int),
                        history=np.fromstring(next(lines), sep=' ', dtype=int),
                        prediction=np.fromstring(next(lines), sep=' ', dtype=int))
                    actual['probability'] = np.array([np.fromstring(next(lines), sep=' ') for _ in range(m)])
                    if base != 1:
                        actual['log_probability'] = np.array([np.fromstring(next(lines), sep=' ') for _ in range(m)])
                    if base == 2:
                        actual['decision'] = np.array([np.fromstring(next(lines), sep=' ') for _ in range(m)])
                    actual['input_labels'] = np.fromstring(next(lines), sep=' ', dtype=int)
                    actual['input_features'] = np.array([np.fromstring(next(lines), sep=' ') for _ in range(n)])
                    assert list(lines) == [] and k == classes
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        reference = SelfTrainingClassifier(estimator, threshold=threshold, criterion=criterion,
                            k_best=k_best, max_iter=None if cap == -1 else cap).fit(x, current_y)
                        certified, ties = certify_history(estimator, x, current_y, actual, actual_iter, actual_reason,
                            threshold, criterion, k_best, cap)
                    matches = (np.array_equal(actual['transduction'], reference.transduction_) and
                        np.array_equal(actual['history'], reference.labeled_iter_))
                    if matches:
                        assert (actual_iter, actual_reason) == (reference.n_iter_, reference.termination_condition_)
                        expected_model = reference
                        transduction, history = reference.transduction_, reference.labeled_iter_
                        oracle = 'sklearn'
                    else:
                        assert criterion == 'k_best' and ties > 0
                        expected_model = certified
                        transduction, history = actual['transduction'], actual['history']
                        oracle = 'source_tied_cutoff_certificate'
                        tie_cases.append(dict(seed=seed, classes=classes, base=base, mode=mode, cutoffs=ties))
                    expected = dict(classes=expected_model.classes_, transduction=transduction, history=history,
                        prediction=expected_model.predict(query), probability=expected_model.predict_proba(query),
                        input_labels=current_y, input_features=x)
                    if base != 1:
                        expected['log_probability'] = expected_model.predict_log_proba(query)
                    if base == 2:
                        expected['decision'] = expected_model.decision_function(query).reshape(m, -1)
                    if base == 2 and classes > 2:
                        # Softmax logits have an unidentifiable common intercept.
                        actual['decision'] -= actual['decision'].mean(axis=1, keepdims=True)
                        expected['decision'] -= expected['decision'].mean(axis=1, keepdims=True)
                    for name, wanted in expected.items():
                        if name in ['probability', 'log_probability', 'decision']:
                            np.testing.assert_allclose(actual[name], wanted, atol=2.e-7, rtol=2.e-7,
                                err_msg=f'{seed} {classes} {base} {mode} {name}')
                        else:
                            np.testing.assert_array_equal(actual[name], wanted)
                        checks.append(dict(seed=seed, classes=classes, base=base, mode=mode, name=name,
                            oracle='input_identity' if name.startswith('input_') else oracle, passed=True))
                    checks.extend(dict(seed=seed, classes=classes, base=base, mode=mode, name=name,
                        oracle=oracle, passed=True) for name in ['iterations', 'termination'])
                    case_count += 1
        print(f'seed {seed}: self-training passed', flush=True)
    (RESULTS / 'self_training.json').write_text(json.dumps(dict(sklearn=sklearn.__version__, fixtures=case_count,
        checks=checks, tied_cutoffs=tie_cases), indent=2) + '\n')
    print(f'{len(checks)} self-training groups across {case_count} fixtures passed; {len(tie_cases)} tied-cutoff cases')


if __name__ == '__main__':
    main()
