"""SelectFromModel: fitted/prefit importances, norms, thresholds and transforms."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection._base import _get_feature_importances
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, LogisticRegression,
                                 MultiTaskElasticNet, MultiTaskLasso, SGDRegressor, SGDClassifier)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from compare_function_transformer import ROOT, resolve
from model_selector_tree_certificate import certify


def fixtures():
    for seed in [0, 42, 2026]:
        rng = np.random.RandomState(seed)
        for mode in range(12):
            for config in range(12):
                n, p, m = 36 + 3*(config % 3), 5 + config % 2, 7
                r = 3 if mode in [1, 7, 11] else 1
                x = rng.normal(size=(n, p)); q = rng.normal(size=(m, p))
                y = 2*x[:, 1] - 1.5*x[:, 3] + .2*rng.normal(size=n)
                ym = np.c_[y, -y + .4*x[:, 0], .3*y + x[:, 2]][:, :r]
                classes = np.array([-4, 3, 9]) if mode == 4 and config % 3 == 0 else np.array([-4, 9])
                yi = classes[np.argmax(np.c_[x[:, 1], -x[:, 1] + x[:, 3], x[:, 0]][:, :len(classes)], axis=1)]
                if mode not in [4, 6, 9]:
                    yi = np.rint(y).astype(int)
                coefficients = rng.normal(size=(r, p))
                coefficients[:, 0] = 0
                if config in [0, 2, 6, 8]:
                    coefficients[:] = np.resize([0., -2., 2., -2., 0., 2.], coefficients.shape)
                if config == 7:
                    coefficients[:] = 0
                weights = rng.uniform(.2, 2, size=n); weights[:3] = 0
                threshold, value, cap, cap_callback = [
                    ('auto', 0., -1, 0), ('mean', 0., 0, 0), ('median', 0., 2, 0),
                    ('1.25*mean', 0., -1, 0), ('0.75 * median', 0., -1, 1), ('value', 0., -1, 0),
                    ('value', -np.inf, 3, 0), ('value', np.inf, -1, 0), ('auto', 0., p, 0),
                    ('-1*mean', 0., -1, 0), ('value', .05, 2, 0), ('2*median', 0., -1, 1)][config]
                operation = 2 if mode in [0, 1, 10, 11] and config in [3, 6, 7, 10] else config % 2
                target_format = int(config % 3 != 1) if mode in [4, 6] else int(config % 3 == 2)
                if mode in [4, 6] and not target_format:
                    y = yi.astype(float)
                weighted = int(mode in [2, 4, 5, 6, 8, 9] and config % 2 == 0)
                c = dict(seed=seed, mode=mode, config=config, n=n, p=p, m=m, r=r,
                         target_format=target_format, named=config % 2, weighted=weighted, cap=cap,
                         cap_callback=cap_callback, operation=operation, threshold=threshold)
                order = [1., 2., -1., -2., np.inf, -np.inf][config % 6]
                yield c, x, q, y, yi, ym, weights, coefficients, value, order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['checked', 'release'], default='checked')
    parser.add_argument('--executable', type=Path)
    args = parser.parse_args()
    exe = args.executable.resolve() if args.executable else resolve(args.profile, 'reference_model_selector_driver')
    checks, errors, source_warnings = [], {}, {}
    count = 0
    certificates = []
    for c, x, q, y, yi, ym, weights, coefficients, value, order in fixtures():
        mode, config, p = c['mode'], c['config'], c['p']
        prefit = mode in [0, 1, 10, 11]
        if mode in [0, 10, 11]:
            base = LinearRegression(); base.coef_ = coefficients[0].copy(); base.intercept_ = 0.; base.n_features_in_ = p
        elif mode == 1:
            base = MultiTaskElasticNet(); base.coef_ = coefficients.copy(); base.intercept_ = np.zeros(c['r'])
            base.n_features_in_ = p
        else:
            base = {2: lambda: Ridge(alpha=.7, solver='svd'),
                    3: lambda: Lasso(alpha=.08, tol=1e-12),
                    4: lambda: LogisticRegression(C=.8, tol=1e-11, max_iter=200),
                    5: lambda: DecisionTreeRegressor(max_depth=3, random_state=7),
                    6: lambda: DecisionTreeClassifier(max_depth=3, random_state=7),
                    7: lambda: MultiTaskLasso(alpha=.08, tol=1e-12),
                    8: lambda: SGDRegressor(penalty='l2', alpha=.02, learning_rate='constant', eta0=.01,
                                           tol=None, shuffle=False, random_state=0),
                    9: lambda: SGDClassifier(loss='log_loss', penalty='l2', alpha=.02, learning_rate='constant',
                                            eta0=.01, tol=None, shuffle=False, random_state=0)}[mode]()
        calls = dict(importance=0, cap=0)
        def importance(model):
            calls['importance'] += 1
            return coefficients.copy() if mode == 11 else coefficients[0].copy()
        def cap(data):
            calls['cap'] += 1
            return data.shape[1]//2
        model = SelectFromModel(base, prefit=prefit,
                               threshold=None if c['threshold'] == 'auto' else value if c['threshold'] == 'value'
                               else c['threshold'], norm_order=order if not np.isfinite(order) else int(order),
                               max_features=cap if c['cap_callback'] else None if c['cap'] == -1 else c['cap'],
                               importance_getter=importance if mode in [10, 11] else
                               'feature_importances_' if config == 8 and mode in [5, 6] else
                               'coef_' if config == 8 else 'auto')
        names = ['field_'+str(i+1) for i in range(p)]
        data = pd.DataFrame(x, columns=names) if c['named'] else x
        query = pd.DataFrame(q, columns=names) if c['named'] else q
        target = yi if c['target_format'] else y
        if mode in [1, 7]:
            target = ym
        with warnings.catch_warnings(record=True) as caught, np.errstate(all='ignore'):
            warnings.simplefilter('always')
            if mode in [8, 9]:
                for j in range(3):
                    sl = slice(j*len(x)//3, (j+1)*len(x)//3)
                    params = dict(sample_weight=weights[sl]) if c['weighted'] else {}
                    if mode == 9:
                        params['classes'] = [-4, 9]
                    model.partial_fit(data.iloc[sl] if c['named'] else data[sl],
                                      yi[sl] if mode == 9 else y[sl], **params)
                fit = model.transform(data)
            elif c['operation'] == 2:
                fit = model.transform(data)
            else:
                params = dict(sample_weight=weights) if c['weighted'] else {}
                if c['operation'] == 1:
                    fit = model.fit_transform(data, target, **params)
                else:
                    model.fit(data, target, **params); fit = model.transform(data)
            transformed = model.transform(query)
            inverse = model.inverse_transform(transformed) if transformed.shape[1] else np.empty((0, 0))
            indices = model.get_support(indices=True) + 1
            fitted_base = getattr(model, 'estimator_', model.estimator)
            scores = _get_feature_importances(fitted_base, model.importance_getter,
                                             transform_func='norm', norm_order=model.norm_order)
            # Naming requires fit even though direct prefit transforms are supported.
            expected_names = (model.get_feature_names_out() if c['operation'] != 2 else np.array([], dtype=str))
            if c['operation'] == 2:
                from sklearn.feature_selection._from_model import _calculate_threshold
                # Source threshold_ requires estimator_ even though transform supports prefit.
                threshold = _calculate_threshold(base, _get_feature_importances(
                    base, model.importance_getter, transform_func='norm', norm_order=model.norm_order), model.threshold)
            else:
                threshold = model.threshold_
        for warning in caught:
            key = warning.category.__name__+': '+str(warning.message)
            source_warnings[key] = source_warnings.get(key, 0)+1
        header = [c[k] for k in ['mode','n','p','m','r','target_format','named','weighted','cap','cap_callback','operation']]
        payload = ' '.join(map(str, header))+'\n'+c['threshold']+'\n'+f'{value:.17g} {order:.17g}\n'
        payload += ''.join(' '.join(f'{v:.17g}' for v in np.asarray(a).ravel(order='F'))+'\n'
                           for a in [x, q, y, yi, ym, weights, coefficients])
        run = subprocess.run([str(exe)], input=payload, text=True, capture_output=True)
        assert run.returncode == 0, (c, run.stdout, run.stderr)
        lines = iter(run.stdout.splitlines())
        metadata = np.fromstring(next(lines), sep=' ', dtype=int)
        actual_threshold = float(next(lines)); actual_indices = np.fromstring(next(lines), sep=' ', dtype=int)
        actual_scores = np.fromstring(next(lines), sep=' ')
        actuals = {}
        for key in ['fit_transform', 'transform', 'inverse_transform']:
            rows, cols = map(int, next(lines).split())
            actuals[key] = np.array([np.fromstring(next(lines), sep=' ') for _ in range(rows)]).reshape(rows, cols)
        width = int(next(lines)); actual_names = [next(lines) for _ in range(width)]
        actual_calls = list(map(int, next(lines).split()))
        node_count = int(next(lines))
        nodes = [list(map(float, next(lines).split())) for _ in range(node_count)]
        assert next(lines, None) is None
        oracle = 'sklearn'
        if mode in [5, 6]:
            target = yi if c['target_format'] else y
            certificate = certify(x, target, weights if c['weighted'] else np.ones(len(x)),
                                  nodes, actual_scores, mode == 6)
            certificate.update(c)
            certificate['source_importance_difference'] = float(np.max(abs(scores-actual_scores), initial=0))
            if not np.allclose(scores, actual_scores, rtol=3e-9, atol=3e-9):
                oracle = 'certified_native_tree_and_sklearn_selection'
                frozen = LinearRegression(); frozen.coef_ = actual_scores.copy(); frozen.intercept_ = 0.
                frozen.n_features_in_ = p
                model.estimator_ = frozen; model.importance_getter = 'auto'
                with warnings.catch_warnings(), np.errstate(all='ignore'):
                    warnings.simplefilter('ignore')
                    fit = model.transform(data); transformed = model.transform(query)
                    inverse = model.inverse_transform(transformed) if transformed.shape[1] else np.empty((0, 0))
                    indices = model.get_support(indices=True)+1
                    scores = _get_feature_importances(frozen, 'auto', transform_func='norm', norm_order=model.norm_order)
                    expected_names = model.get_feature_names_out(); threshold = model.threshold_
            certificate['selection_oracle'] = oracle
            certificates.append(certificate)
        expected_metadata = [len(indices), 0 if c['operation'] == 2 else p,
                             -1 if c['operation'] == 2 else p//2 if c['cap_callback'] else c['cap']]
        expected = dict(metadata=expected_metadata, support_indices=indices, scores=scores,
                        threshold=threshold, fit_transform=fit, transform=transformed, inverse_transform=inverse,
                        feature_names=expected_names.tolist(), callbacks=[calls['importance'], calls['cap']])
        actuals.update(metadata=metadata, support_indices=actual_indices, scores=actual_scores,
                       threshold=actual_threshold, feature_names=actual_names, callbacks=actual_calls)
        tolerance = 3e-7 if mode == 4 else 3e-9
        for key, want in expected.items():
            actual = actuals[key]
            if key in ['metadata', 'support_indices', 'feature_names', 'callbacks']:
                np.testing.assert_array_equal(actual, want, err_msg=f'{c} {key}')
            else:
                np.testing.assert_allclose(actual, want, rtol=tolerance, atol=tolerance, err_msg=f'{c} {key}')
                a, b = np.asarray(actual), np.asarray(want)
                finite = np.isfinite(a) & np.isfinite(b)
                error = float(np.max(abs(a[finite]-b[finite]), initial=0))
                errors[key] = max(errors.get(key, 0), error)
            checks.append(dict(**c, name=key, oracle=oracle, passed=True))
        count += 1
        if count % 100 == 0:
            print(f'{count} SelectFromModel fixtures passed', flush=True)
    source = Path(sklearn.__file__).parent
    report = dict(profile=args.profile, sklearn=sklearn.__version__, numpy=np.__version__, fixtures=count,
                  checks=checks, max_absolute_errors=errors, source_warnings=source_warnings, tree_certificates=certificates,
                  certificate_sha256=hashlib.sha256((ROOT/'validation/model_selector_tree_certificate.py').read_bytes()).hexdigest(),
                  native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
                                 ['feature_selection/_from_model.py', 'feature_selection/_base.py',
                                  'linear_model/_stochastic_gradient.py', 'tree/_classes.py']})
    (ROOT/'validation/results').mkdir(exist_ok=True)
    (ROOT/'validation/results'/f'model_selector_{args.profile}.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {count} SelectFromModel fixtures passed')


if __name__ == '__main__':
    main()
