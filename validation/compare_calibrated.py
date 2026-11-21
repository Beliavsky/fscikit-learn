"""CalibratedClassifierCV workflows and independent fitted-map certificates."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
import numpy as np
import sklearn
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV, _sigmoid_calibration
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile=None):
    command = ['fpm', 'run', '--target', 'reference_calibrated_driver', '--runner']
    command += ['--profile', profile] if profile else ['--flag', FLAGS]
    run = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=True)
    paths = [ROOT / s.strip().strip('"') for s in run.stdout.splitlines()
             if s.strip().strip('"').endswith(('reference_calibrated_driver.exe', 'reference_calibrated_driver'))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, run.stdout
    return paths[0]


def response(model, x):
    if hasattr(model, 'decision_function'):
        return model.decision_function(x).reshape(len(x), -1)
    proba = model.predict_proba(x)
    return proba[:, 1:2] if proba.shape[1] == 2 else proba


def close(a, b, base, context):
    tolerance = 3.e-6 if base == 2 else 2.e-6
    np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance, err_msg=str(context))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--profile')
    args = parser.parse_args()
    executable = args.executable.resolve() if args.executable else resolve(args.profile)
    RESULTS.mkdir(exist_ok=True)
    checks = []
    cases = 0
    prediction_ties = []
    for seed in [0, 7, 42, 2026, 314159]:
        rng = np.random.default_rng(seed)
        for k in [2, 3, 4]:
            n, p, m, nt = 48 + k * 4, k + 1, 17, 32 + k * 4
            centers = rng.normal(size=(k, p)) * 1.2
            labels = np.array([11, -7, 4, 0])[:k]
            target, training_target = np.arange(n) % k, np.arange(nt) % k
            x = centers[target] + rng.normal(size=(n, p))
            tx = centers[training_target] + rng.normal(size=(nt, p))
            y, ty = labels[target], labels[training_target]
            query = rng.normal(size=(m, p)) * 3
            # Deliberately begin with a non-smallest label: source fold encoding is by appearance.
            for base in range(3):
                for method in ['sigmoid', 'isotonic']:
                    for mode in range(11):
                        prefit = mode in [2, 3, 7, 8, 10]
                        weighted = mode in [3, 4, 5, 9, 10]
                        ensemble = mode not in [1, 5]
                        cv = 5 if mode == 6 else 3
                        if weighted and not prefit and base != 1:
                            continue
                        if mode == 7 and base != 2:
                            continue
                        if mode in [9, 10] and method != 'sigmoid':
                            continue
                        weighting = (seed + mode) % 2
                        cy = y.copy()
                        if mode == 3:
                            cy[cy == labels[-1]] = labels[0]  # Missing calibration classes are allowed in prefit mode.
                        if mode == 8:
                            cy[:] = 99  # Label binarization maps unknown calibration labels to all zeros.
                        weights = rng.uniform(.1, 3, n) if weighted else np.ones(n)
                        if weighted:
                            weights[::5] = 0
                        if mode in [9, 10]:
                            weights[:] = 0
                        cx = x * 1000 if mode == 7 else x
                        cq = query * 500 if mode == 7 else query
                        if base == 0:
                            estimator = GaussianNB().fit(tx, ty)
                        elif base == 1:
                            estimator = KNeighborsClassifier(n_neighbors=3,
                                weights='distance' if weighting else 'uniform', algorithm='brute').fit(tx, ty)
                        else:
                            estimator = LogisticRegression(C=.7, max_iter=500, tol=1.e-10, solver='newton-cg').fit(tx, ty)
                        data, output = RESULTS / 'calibrated_input.txt', RESULTS / 'calibrated_actual.txt'
                        with data.open('w') as stream:
                            stream.write(f'{n} {p} {m} {nt} {base} {int(method == "isotonic")} '
                                f'{int(ensemble)} {int(prefit)} {cv} {int(weighted)} {weighting}\n')
                            np.savetxt(stream, cy[None], fmt='%d'); np.savetxt(stream, ty[None], fmt='%d')
                            np.savetxt(stream, weights[None], fmt='%.17g')
                            for a in [cx, tx, cq]:
                                np.savetxt(stream, a, fmt='%.17g')
                        run = subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        assert run.returncode == 0, (seed, k, base, method, mode, run.stdout)
                        lines = iter(output.read_text().splitlines())
                        def vector(dtype=float):
                            return np.fromstring(next(lines), sep=' ', dtype=dtype)
                        def matrix(rows):
                            return np.array([vector() for _ in range(rows)])
                        pairs, actual_k = vector(int)
                        actual_classes, prediction, probability = vector(int), vector(int), matrix(m)
                        folds = vector(int)
                        context = (seed, k, base, method, mode)
                        def record(name, oracle='sklearn'):
                            checks.append(dict(seed=seed, classes=k, base=base, method=method, mode=mode,
                                name=name, oracle=oracle, passed=True))
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            reference = CalibratedClassifierCV(estimator, method=method,
                                cv='prefit' if prefit else cv, ensemble=ensemble).fit(cx, cy,
                                sample_weight=weights if weighted else None)
                        assert pairs == len(reference.calibrated_classifiers_) and actual_k == k
                        np.testing.assert_array_equal(actual_classes, reference.classes_); record('classes')
                        source_probability = reference.predict_proba(cq)
                        close(probability, source_probability, base, context); record('probability')
                        source_prediction = reference.predict(cq)
                        different = prediction != source_prediction
                        if np.any(different):
                            chosen = np.searchsorted(actual_classes, prediction)
                            gap = source_probability.max(axis=1)-source_probability[np.arange(m), chosen]
                            native_gap = np.sort(probability, axis=1)[:, -1]-np.sort(probability, axis=1)[:, -2]
                            assert max(gap[different]) <= 64*np.finfo(float).eps, (context, gap)
                            assert max(native_gap[different]) <= 64*np.finfo(float).eps, (context, native_gap)
                            prediction_ties.append(dict(seed=seed, classes=k, base=base, method=method, mode=mode,
                                samples=np.flatnonzero(different).tolist(), max_gap=float(max(gap[different]))))
                            record('prediction', 'source_argmax_tie')
                        else:
                            record('prediction')
                        source_folds = np.zeros(n, dtype=int)
                        if not prefit:
                            for f, (_, test) in enumerate(StratifiedKFold(cv).split(cx, cy), 1):
                                source_folds[test] = f
                        np.testing.assert_array_equal(folds, source_folds); record('folds')
                        reconstructed = np.zeros_like(probability)
                        if not ensemble and not prefit:
                            source_oof = cross_val_predict(estimator, cx, cy, cv=cv,
                                method='decision_function' if base == 2 else 'predict_proba')
                            if k == 2 and base != 2:
                                source_oof = source_oof[:, 1:2]
                            source_oof = source_oof.reshape(n, -1)
                        for f, pair in enumerate(reference.calibrated_classifiers_, 1):
                            count, q = vector(int)
                            scores, query_scores = matrix(count), matrix(m)
                            subset = np.arange(n) if prefit or not ensemble else np.flatnonzero(folds == f)
                            assert count == len(subset) and q == (1 if k == 2 else k)
                            source_scores = (source_oof if not ensemble and not prefit
                                else response(pair.estimator, cx[subset]))
                            query_reference = response(pair.estimator, cq)
                            close(scores, source_scores, base, context)
                            close(query_scores, query_reference, base, context)
                            record(f'calibration_response_{f}'); record(f'query_response_{f}')
                            current = np.zeros_like(probability)
                            for c in range(q):
                                label = actual_classes[1] if k == 2 else actual_classes[c]
                                target = (cy[subset] == label).astype(float)
                                w = weights[subset]
                                if method == 'sigmoid':
                                    a, b = vector()
                                    prior0, prior1 = w[target == 0].sum(), w[target == 1].sum()
                                    smooth = np.where(target > 0, (prior1 + 1)/(prior1 + 2), 1/(prior0 + 2))
                                    scale = max(abs(scores[:, c]))
                                    scale = scale if scale >= 30 else 1.
                                    fitted = expit(-(a * scores[:, c] + b))
                                    residual = (smooth - fitted) * w / max(1., w.sum())
                                    gradient = np.array([residual @ (scores[:, c]/scale), residual.sum()])
                                    assert max(abs(gradient)) <= 2.1e-9, (context, gradient)
                                    sa, sb = _sigmoid_calibration(scores[:, c], target, w)
                                    def loss(aa, bb):
                                        raw = -(aa * scores[:, c] + bb)
                                        return np.sum(w * (np.logaddexp(0, raw) - smooth * raw))/max(1., w.sum())
                                    assert loss(a, b) <= loss(sa, sb) + 2.e-10, context
                                    values = expit(-(a * query_scores[:, c] + b))
                                else:
                                    knots = int(vector(int)[0]); xs, ys = vector(), vector()
                                    assert len(xs) == len(ys) == knots and np.all(np.diff(xs) > 0)
                                    assert np.all(np.diff(ys) >= 0) and min(ys) >= 0 and max(ys) <= 1, (context, xs, ys)
                                    independent = IsotonicRegression(out_of_bounds='clip').fit(scores[:, c], target, w)
                                    grid = np.unique(np.r_[xs, independent.X_thresholds_, query_scores[:, c]])
                                    np.testing.assert_allclose(np.interp(grid, xs, ys), independent.predict(grid),
                                        atol=3.e-12, rtol=3.e-12, err_msg=str(context))
                                    values = np.interp(query_scores[:, c], xs, ys)
                                current[:, 1 if k == 2 else c] = values
                                record(f'calibrator_{f}_{c}', 'source_calibration_certificate')
                            if k == 2:
                                current[:, 0] = 1-current[:, 1]
                            else:
                                denominator = current.sum(axis=1, keepdims=True)
                                current = np.divide(current, denominator, out=np.full_like(current, 1/k), where=denominator != 0)
                            reconstructed += current/pairs
                        np.testing.assert_allclose(probability, reconstructed, atol=2.e-14, rtol=2.e-14)
                        record('composition', 'independent_identity')
                        before, after = matrix(m), matrix(m)
                        np.testing.assert_array_equal(before, after); record('base_preserved', 'input_identity')
                        close(before, estimator.predict_proba(cq), base, context); record('input_base_probability')
                        np.testing.assert_array_equal(vector(int), cy)
                        np.testing.assert_array_equal(matrix(n), cx); record('data_preserved', 'input_identity')
                        assert list(lines) == []
                        cases += 1
        print(f'seed {seed}: calibration passed', flush=True)
    (RESULTS / 'calibrated.json').write_text(json.dumps(dict(sklearn=sklearn.__version__, fixtures=cases,
        checks=checks, prediction_ties=prediction_ties), indent=2)+'\n')
    print(f'{len(checks)} calibration groups across {cases} fixtures passed')


if __name__ == '__main__':
    main()
