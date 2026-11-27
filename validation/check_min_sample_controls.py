"""Compare typed minimum-sample validation for ten sklearn estimators."""
from pathlib import Path
import argparse, hashlib, json, subprocess
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeRegressor, ExtraTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor,
    ExtraTreesClassifier, GradientBoostingRegressor, GradientBoostingClassifier)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--executable', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    a = ap.parse_args()
    models = [DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeRegressor, ExtraTreeClassifier,
        RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier,
        GradientBoostingRegressor, GradientBoostingClassifier]
    x = np.arange(1., 11.).reshape(-1, 1)
    records = []
    for k, cls in enumerate(models, 1):
        for name, values in [('min_samples_leaf', [0, -1, 1, 99, 0., -.1, 1., 1.01, .001, .9999999999999999, np.nan, np.inf]),
                             ('min_samples_split', [0, 1, 2, 99, 0., -.1, 1., 1.01, .001, .4, np.nan, np.inf])]:
            for value in values:
                try:
                    cls(**{name: value}, random_state=0).fit(x, (x[:, 0] % 2).astype(int) if k % 2 == 0 else x[:, 0]**2)
                    accepted, error = True, None
                except ValueError as e:
                    accepted, error = False, str(e)
                p = subprocess.run([str(a.executable), str(k), name, str(value)], text=True, capture_output=True, timeout=30)
                output = p.stdout + p.stderr
                passed = ((p.returncode == 0 and 'FIT ACCEPTED' in output) if accepted else
                    (p.returncode != 0 and 'fscikit-learn: minimum' in output and
                     not any(s in output for s in ['SIGFPE', 'SIGSEGV', 'Fortran runtime error'])))
                records.append(dict(estimator=cls.__name__, parameter=name, token=str(value), accepted=accepted,
                    source_error=error, native_returncode=p.returncode, output=output, passed=passed))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = dict(configurations=len(records), passed=all(r['passed'] for r in records), records=records,
        sklearn=sklearn.__version__, sha256={str(p): sha(p) for p in [Path(__file__), a.executable,
            Path('app/reference_min_samples_invalid_driver.f90'), Path('src/sklearn_tree_features.f90'),
            Path('src/sklearn_tree.f90'), Path('src/sklearn_ensemble.f90'), Path('src/sklearn_classic_gradient_boosting.f90'),
            Path('src/sklearn_search_adapters.f90')]})
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(dict(configurations=len(records), passed=report['passed']))
    assert report['passed']


if __name__ == '__main__':
    main()
