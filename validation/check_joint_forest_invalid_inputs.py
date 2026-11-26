"""Joint forest errors compared to sklearn, plus explicit native API guards."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--executable', type=Path, required=True)
ap.add_argument('--report', type=Path, required=True)
a = ap.parse_args()
x = np.ones((4, 2)); y = np.ones((4, 2))
cases = dict(
    empty_outputs=(False, {}, y[:, :0], None),
    extra_empty_outputs=(True, {}, y[:, :0], None),
    target_rows=(False, {}, y[:3], None),
    target_nan=(False, {}, np.array([[1, np.nan], [1, 1], [1, 1], [1, 1]]), None),
    target_inf=(False, {}, np.array([[1, np.inf], [1, 1], [1, 1], [1, 1]]), None),
    weights_length=(False, {}, y, [1]),
    weights_nan=(False, {}, y, [np.nan, 1, 1, 1]),
    weights_inf=(False, {}, y, [np.inf, 1, 1, 1]),
    poisson_negative=(False, dict(criterion='poisson'), np.array([[1, -1], [1, 1], [1, 1], [1, 1]]), None),
    poisson_zero=(False, dict(criterion='poisson'), np.zeros_like(y), None),
    score_outputs=(False, {}, y, None),
    oob_no_bootstrap=(False, dict(bootstrap=False, oob_score=True), y, None),
    extra_oob_default=(True, dict(oob_score=True), y, None),
    max_samples_no_bootstrap=(False, dict(bootstrap=False, max_samples=2), y, None),
    max_samples_range=(False, dict(max_samples=5), y, None))
checks = []
for name, (extra, kwargs, target, w) in cases.items():
    try:
        model = (ExtraTreesRegressor if extra else RandomForestRegressor)(n_estimators=2, **kwargs).fit(x, target, sample_weight=w)
        if name == 'score_outputs':
            model.score(x, y[:, :1])
    except (ValueError, TypeError) as error:
        source_error = dict(type=type(error).__name__, message=str(error))
    else:
        raise AssertionError(('Source accepted', name))
    p = subprocess.run([str(a.executable), name], capture_output=True, text=True)
    out = p.stdout + p.stderr
    passed = p.returncode != 0 and 'fscikit-learn:' in out and 'invalid joint forest input was accepted' not in out and not any(s in out for s in ['SIGFPE', 'SIGSEGV', 'Fortran runtime error'])
    checks.append(dict(case=name, source_error=source_error, passed=passed, native_output=out, native_adapter=False))
for name, message in [('scalar_predict', 'joint forest prediction requires predict_multi'), ('samples_unfitted', 'forest is not fitted')]:
    p = subprocess.run([str(a.executable), name], capture_output=True, text=True)
    out = p.stdout + p.stderr
    checks.append(dict(case=name, native_adapter=True, native_output=out, passed=p.returncode != 0 and message in out))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
report = dict(count=len(checks), checks=checks, passed=all(c['passed'] for c in checks), sha256={str(p): sha(p) for p in [Path(__file__), a.executable, Path('src/sklearn_ensemble.f90')]})
a.report.parent.mkdir(parents=True, exist_ok=True)
a.report.write_text(json.dumps(report, indent=2) + '\n')
print(len(checks), 'joint forest guards passed:', report['passed'])
assert report['passed']
