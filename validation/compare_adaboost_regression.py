"""AdaBoost.R2 and each bootstrapped CART against installed sklearn."""
from pathlib import Path
import argparse, hashlib
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"


def main():
    RESULTS.mkdir(exist_ok=True)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path)
    parser.add_argument('--report',type=Path,default=RESULTS/'adaboost_regression.json')
    args=parser.parse_args()
    if args.executable:
        executable=args.executable.resolve()
    else:
        resolved = subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                                   "--target", "reference_adaboost_regression_driver", "--runner"],
                                  cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE)
        candidates = [ROOT / line.strip().strip('"') for line in resolved.stdout.splitlines()
                      if line.strip().strip('"').endswith(("reference_adaboost_regression_driver",
                                                         "reference_adaboost_regression_driver.exe"))]
        candidates = [path for path in candidates if path.is_file()]
        assert len(candidates) == 1, resolved.stdout
        executable = candidates[0].resolve()
    checks = [];records=[]
    for seed, n, p in ((7, 57, 3), (42, 71, 4), (2026, 43, 5)):
        rng = np.random.default_rng(seed)
        original = rng.normal(size=(n, p))
        target = np.sin(original[:, 0]) + original[:, 1] ** 2 + .3 * rng.normal(size=n)
        weights = rng.uniform(.2, 2, n); weights[0] = 0
        for case in range(12):
            x, y, w = original.copy(), target.copy(), weights.copy()
            options = dict(n_estimators=9, random_state=seed, learning_rate=1., loss="linear")
            tree = dict(max_depth=3, min_samples_leaf=1, min_samples_split=2,
                        max_features=None, min_impurity_decrease=0.)
            if case == 1: options["loss"] = "square"
            if case == 2: options.update(loss="exponential", learning_rate=.3)
            if case == 3: tree.update(max_depth=1); options["learning_rate"] = 2.
            if case == 4: tree.update(max_depth=4, min_samples_leaf=2, max_features=2)
            if case == 5: tree.update(min_samples_split=8, max_features=1)
            if case == 6: x[:, 1:] = 0
            if case == 7: tree["min_impurity_decrease"] = .04
            if case == 8: y[:] = 4
            if case == 9: x[:] = 0; y = np.where(np.arange(n) % 2, 1., -1.); w[:] = 1
            if case == 10: w[1] = 1.e-100; y[0] = 10000
            if case == 11: x = np.round(x); options["loss"] = "square"
            data = args.report.with_suffix(".in")
            output = args.report.with_suffix(".actual.txt")
            with data.open("w") as stream:
                stream.write(f"{n} {p}\n")
                stream.write(f"9 {seed} {tree['max_depth']} {tree['min_samples_leaf']} "
                             f"{tree['min_samples_split']} {tree['max_features'] or 0}\n")
                stream.write(f"{options['loss']} {options['learning_rate']} {tree['min_impurity_decrease']}\n")
                np.savetxt(stream, np.column_stack([x, y, w]), fmt="%.17g")
            subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, check=True)
            actual = {}
            for line in output.read_text().splitlines():
                name, *values = line.split(); actual[name] = np.array(values, dtype=float)
            model = AdaBoostRegressor(estimator=DecisionTreeRegressor(**tree), **options).fit(x, y, sample_weight=w)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                importance = model.feature_importances_
            if not np.all(np.isfinite(importance)):
                importance = np.zeros(p)  # Native zero-total-importance convention.
            expected = dict(weights=model.estimator_weights_, errors=model.estimator_errors_,
                            importance=importance, shape=[len(model.estimators_), p],
                            train=model.predict(x), query=model.predict(x+.137),
                            staged=np.stack(list(model.staged_predict(x+.137)), axis=1),
                            scores=list(model.staged_score(x, y)), score=[model.score(x, y)])
            for k, estimator in enumerate(model.estimators_, 1):
                t = estimator.tree_
                expected.update({f"tree{k}_predict": estimator.predict(x+.137),
                                 f"tree{k}_meta": [estimator.random_state, estimator.get_depth(), estimator.get_n_leaves()],
                                 f"tree{k}_threshold": np.where(t.children_left < 0, 0, t.threshold),
                                 f"tree{k}_feature": np.where(t.children_left < 0, 0, t.feature+1),
                                 f"tree{k}_value": t.value[:, 0, 0]})
            records.append(dict(seed=seed,case=case,input=data.read_text(),source={k:np.asarray(v).ravel(order='F').tolist() for k,v in expected.items()},native={k:v.tolist() for k,v in actual.items()}))
            assert actual.keys() == expected.keys(), (seed, case, actual.keys() ^ expected.keys())
            for name, value in expected.items():
                want = np.asarray(value).ravel(order="F")
                np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8,
                                           err_msg=f"seed={seed} case={case} {name}")
                checks.append(dict(seed=seed, case=case, name=name, passed=True))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    args.report.write_text(json.dumps(dict(sklearn=sklearn.__version__,checks=checks,records=records,sha256={str(p):sha(p) for p in [Path(__file__),executable,ROOT/'src/sklearn_adaboost_regression.f90']}),indent=2)+'\n')
    print(f"{len(checks)} AdaBoostRegressor result groups passed")


if __name__ == "__main__":
    main()
