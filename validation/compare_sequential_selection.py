"""SequentialFeatureSelector masks and transforms against scikit-learn 1.6.1."""
from pathlib import Path
import json
import subprocess
import numpy as np
import sklearn
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"


def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 48, 7), (42, 37, 6), (2026, 29, 5)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p)) * np.linspace(.7, 1.5, p) + np.linspace(-1, 1, p)
        x[:, -1] += .3*x[:, 1]
        y = 2.3*x[:, 1] - 1.7*x[:, 3] + .3*x[:, 0] + rng.normal(size=n)*.7
        data = RESULTS / f"sequential_{seed}.txt"
        output = RESULTS / f"sequential_actual_{seed}.txt"
        np.savetxt(data, np.column_stack((x, y)), header=f"{n} {p}", comments="", fmt="%.17g")
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_sequential_selection_driver", "--", str(data), str(output)],
                       cwd=ROOT, check=True)
        actual, expected = {}, {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        for j in range(1, 16):
            options = dict(n_features_to_select=2, direction="forward", cv=4)
            estimator = LinearRegression()
            if j in (2, 4, 6, 8, 10, 12, 13, 15):
                options["direction"] = "backward"
            if j in (3, 4, 5, 6, 7, 8):
                options["n_features_to_select"] = "auto"
            if j in (5, 6, 7, 8, 13):
                options["tol"] = {5: .01, 6: -.01}.get(j, 1.e6)
            if j in (9, 10):
                estimator = Ridge(alpha=2, solver="svd")
            elif j == 11:
                estimator = ElasticNet(alpha=.07, l1_ratio=.35, tol=1.e-11, max_iter=10000)
            elif j == 12:
                estimator = Lasso(alpha=.1, tol=1.e-11, max_iter=10000)
            if j == 14:
                options["n_features_to_select"] = 1
            elif j == 15:
                options["n_features_to_select"] = p-1
            model = SequentialFeatureSelector(estimator, **options).fit(x, y)
            z = model.transform(x)
            values = dict(support=model.get_support().astype(int),
                          shape=[model.n_features_in_, model.n_features_to_select_],
                          train=z, query=model.transform(x+.137), inverse=model.inverse_transform(z))
            expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    (RESULTS / "sequential_selection.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} SequentialFeatureSelector result groups passed")


if __name__ == "__main__":
    main()
