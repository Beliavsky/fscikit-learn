"""Actual sklearn validation-stopping parameter and split acceptance/errors."""
from pathlib import Path
import argparse, hashlib, json, subprocess, warnings
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--executable', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    a = ap.parse_args()
    cases = dict(patience_negative=dict(n_iter_no_change=-1), disabled=dict(n_iter_no_change=None),
        fraction_zero=dict(validation_fraction=0.), fraction_one=dict(validation_fraction=1.),
        fraction_negative=dict(validation_fraction=-.1), fraction_nan=dict(validation_fraction=np.nan),
        fraction_inf=dict(validation_fraction=np.inf), tol_negative=dict(tol=-.1), tol_nan=dict(tol=np.nan),
        tol_inf=dict(tol=np.inf), disabled_fraction=dict(n_iter_no_change=None, validation_fraction=1.),
        disabled_tol=dict(n_iter_no_change=None, tol=-1.), empty_train=dict(validation_fraction=.99),
        missing_classes=dict(validation_fraction=.9), small_test=dict(validation_fraction=.01),
        singleton_classes={}, one_class={}, weights_zero={}, weights_nan={}, weights_inf={})
    records = []
    for kind, cls in [('regression', GradientBoostingRegressor), ('classification', GradientBoostingClassifier)]:
        for name, overrides in cases.items():
            n = 100 if name == 'missing_classes' else 20
            x = np.column_stack([np.arange(1., n+1), np.arange(1, n+1) % 3])
            y = x[:, 0]**2 if kind == 'regression' else np.arange(1, n+1) % 2
            w = np.ones(n)
            if kind == 'classification':
                if name == 'singleton_classes': y = np.arange(1, n+1)
                if name == 'one_class': y[:] = 0
                if name == 'missing_classes': y[:] = 2; y[:2] = 0; y[2:4] = 1
            if name == 'weights_zero': w[:] = 0
            if name == 'weights_nan': w[0] = np.nan
            if name == 'weights_inf': w[0] = np.inf
            params = dict(n_estimators=6, n_iter_no_change=2, validation_fraction=.2, random_state=0)
            params.update(overrides)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    cls(**params).fit(x, y, sample_weight=w)
                accepted, error = True, None
            except (ValueError, ZeroDivisionError) as e:
                accepted, error = False, dict(type=type(e).__name__, message=str(e))
            p = subprocess.run([str(a.executable), kind, name], text=True, capture_output=True, timeout=30)
            output = p.stdout + p.stderr
            passed = (p.returncode == 0 and 'FIT ACCEPTED' in output) if accepted else (
                p.returncode != 0 and 'fscikit-learn:' in output and not any(v in output for v in [
                    'SIGFPE', 'SIGSEGV', 'Fortran runtime error']))
            records.append(dict(kind=kind, case=name, accepted=accepted, source_error=error,
                native_returncode=p.returncode, native_output=output, passed=passed))
    files = [Path(__file__), a.executable, Path('app/reference_gb_early_invalid_driver.f90'),
        Path('src/sklearn_classic_gradient_boosting.f90'), Path('src/sklearn_resampling.f90'),
        Path('src/sklearn_search_adapters.f90')]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = dict(configurations=len(records), passed=all(r['passed'] for r in records), records=records,
        sklearn=sklearn.__version__, sha256={str(p): sha(p) for p in files})
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    print(dict(configurations=len(records), passed=report['passed']))
    assert report['passed']


if __name__ == '__main__':
    main()
