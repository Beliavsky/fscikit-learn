"""Compare NaN-aware neighbor imputation and dense missingness indicators."""
from pathlib import Path
import json
import subprocess
import numpy as np
import sklearn
from sklearn.impute import KNNImputer, MissingIndicator
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"

def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 45, 5), (42, 60, 7), (2026, 24, 4)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p)); q = rng.normal(size=(17, p))
        # Two complete continuous anchors prevent ambiguous neighbor cutoffs.
        missing = rng.uniform(size=(n, p)) < .3; missing[:, :2] = False
        x[missing] = np.nan
        x[:, -1] = np.nan
        q[rng.uniform(size=q.shape) < .35] = np.nan
        q[0, :] = np.nan
        data, output = RESULTS/f"knnimpute_{seed}.txt", RESULTS/f"knnimpute_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p} {len(q)}\n")
            np.savetxt(stream, np.nan_to_num(np.vstack([x, q]), nan=-9999), fmt="%.17g")
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_knn_impute_driver", "--", str(data), str(output)],
                       cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        expected = {}
        for j in range(1, 9):
            options = {}
            if j == 1: options["n_neighbors"] = 1
            if j == 2: options["n_neighbors"] = 3
            if j == 3: options.update(n_neighbors=3, weights="distance")
            if j == 4: options.update(n_neighbors=n+10, weights="distance")
            if j == 5: options["add_indicator"] = True
            if j == 6: options["keep_empty_features"] = True
            if j == 7: options.update(keep_empty_features=True, add_indicator=True, weights="distance")
            if j == 8: options.update(n_neighbors=n+10, add_indicator=True)
            model = KNNImputer(**options)
            fitted = model.fit_transform(x); result = model.transform(q)
            features = np.flatnonzero(np.isnan(x).any(axis=0))+1
            values = dict(train=fitted, query=result, shape=[fitted.shape[1], result.shape[1]],
                          valid=~np.isnan(x).all(axis=0), features=features)
            expected.update({f"knn{j}_{name}": value for name, value in values.items()})
        for j in range(1, 4):
            model = MissingIndicator(features="all" if j == 2 else "missing-only", error_on_new=(j == 3))
            fitted = model.fit_transform(x); result = model.transform(x if j == 3 else q)
            values = dict(train=fitted, query=result, features=model.features_+1, shape=result.shape)
            expected.update({f"indicator{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name, actual[name].shape, want.shape)
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    (RESULTS/"knn_impute.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} imputation result groups passed")

if __name__ == "__main__":
    main()
