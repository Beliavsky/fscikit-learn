"""TheilSenRegressor reference checks, with explicit native sampling inputs."""
from pathlib import Path
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model._theil_sen import _lstsq, _spatial_median, _breakdown_point
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 12, 2), (42, 13, 3), (2026, 14, 4)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))
        y = x @ np.linspace(.5, 1.5, p)+.7+rng.normal(scale=.03, size=n)
        y[:2] += [7, -6]
        data, output = RESULTS/f"theil_{seed}.txt", RESULTS/f"theil_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, y]), fmt="%.17g")
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_theil_sen_driver", "--", str(data), str(output)],
                       cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = {}
        for j in range(1, 9):
            train = x.copy()
            options = dict(max_subpopulation=50000, max_iter=2000, tol=1e-8, random_state=31, n_jobs=1)
            if j == 2:
                options["fit_intercept"] = False
            if j == 3:
                options["n_subsamples"] = p+2
            if j in (4, 5):
                options["n_subsamples"] = n
            if j == 5:
                train[:, -1] = train[:, 0]
            if j == 6:
                options.update(max_iter=2, tol=1e-14)
            if j == 7:
                train = x[:, :1]
                options["fit_intercept"] = False
            query = train+.137
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if j == 8:
                    # Native and NumPy random streams differ. Validate the
                    # same selected row subsets using sklearn's own solvers.
                    indices = actual["sampled_subsets"].reshape((17, p+1), order="F").astype(int)-1
                    assert np.all((indices >= 0) & (indices < n))
                    assert all(len(np.unique(row)) == p+1 for row in indices)
                    weights = _lstsq(train, y, indices, True)
                    n_iter, center = _spatial_median(weights, max_iter=2000, tol=1e-8)
                    intercept, coef = center[0], center[1:]
                    count, subsamples = 17, p+1
                    breakdown = _breakdown_point(n, subsamples)
                else:
                    model = TheilSenRegressor(**options).fit(train, y)
                    coef, intercept, n_iter = model.coef_, model.intercept_, model.n_iter_
                    count, breakdown = model.n_subpopulation_, model.breakdown_
                    subsamples = options.get("n_subsamples", min(train.shape[1]+int(model.fit_intercept), n))
            prediction = train @ coef+intercept
            expected.update({f"theil{j}_coef": coef,
                             f"theil{j}_parameters": [intercept, breakdown, count, subsamples, n_iter],
                             f"theil{j}_train": prediction, f"theil{j}_query": query @ coef+intercept,
                             f"theil{j}_score": [r2_score(y, prediction)]})
        assert actual.keys()-{"sampled_subsets"} == expected.keys()
        for name, want in expected.items():
            np.testing.assert_allclose(actual[name], np.asarray(want).ravel(order="F"),
                                       atol=2e-7, rtol=2e-7, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, api=name, passed=True))
        print(f"PASS TheilSenRegressor seed={seed}", flush=True)
    (RESULTS/"theil_sen_report.json").write_text(json.dumps(
        dict(sklearn_version=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} Theil-Sen comparisons passed against sklearn {sklearn.__version__}")


if __name__ == "__main__":
    main()
