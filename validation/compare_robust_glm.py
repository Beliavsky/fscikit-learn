"""Compare robust linear and generalized linear regressors with sklearn 1.6.1."""
from pathlib import Path
import json
import os
import subprocess
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import sklearn
from sklearn.linear_model import (HuberRegressor, RANSACRegressor, LinearRegression,
                                  PoissonRegressor, GammaRegressor, TweedieRegressor)
from sklearn.linear_model._huber import _huber_loss_and_gradient

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def reference(x, yr, yp, yg, yt):
    query = x+.017
    expected = {}
    for prefix, options in (("huber", dict(epsilon=1.35, alpha=.01)),
                            ("huber_no_intercept", dict(epsilon=1.8, alpha=.2, fit_intercept=False)),
                            ("huber_unpenalized", dict(epsilon=2, alpha=0))):
        m = HuberRegressor(max_iter=1000, tol=1e-6, **options).fit(x, yr)
        expected[prefix+"_coef"] = m.coef_
        expected[prefix+"_parameters"] = [m.intercept_, m.scale_]
        expected[prefix+"_predict"] = m.predict(query)
        expected[prefix+"_outliers"] = m.outliers_.astype(float)
        w = np.r_[m.coef_, m.intercept_, m.scale_] if m.fit_intercept else np.r_[m.coef_, m.scale_]
        expected[prefix+"_loss"] = [_huber_loss_and_gradient(w, x, yr, m.epsilon, m.alpha, np.ones(len(yr)))[0]]
    for prefix, options, target in (("ransac", dict(residual_threshold=.5), yr),
                                    ("ransac_squared", dict(residual_threshold=.25, loss="squared_error"), yr),
                                    ("ransac_auto", {}, yr),
                                    ("ransac_no_intercept", dict(residual_threshold=.5,
                                                               estimator=LinearRegression(fit_intercept=False)), yr-1.2)):
        m = RANSACRegressor(random_state=21, stop_probability=.999999, max_trials=1000, **options).fit(x, target)
        # This separated-outlier fixture has the same unique clean consensus
        # under different random streams. Do not compare the trial counts.
        known = np.arange(len(yr)) >= len(yr)//6
        np.testing.assert_array_equal(m.inlier_mask_, known)
        expected[prefix+"_coef"] = m.estimator_.coef_
        expected[prefix+"_parameters"] = [m.estimator_.intercept_, m.score(x, target)]
        expected[prefix+"_predict"] = m.predict(query)
        expected[prefix+"_inliers"] = m.inlier_mask_.astype(float)
    configs = [("poisson", PoissonRegressor(alpha=.3), yp),
               ("poisson_no_intercept", PoissonRegressor(alpha=0, fit_intercept=False), yp),
               ("gamma", GammaRegressor(alpha=.4), yg),
               ("gamma_no_intercept", GammaRegressor(alpha=0, fit_intercept=False), yg)]
    for i, power in enumerate((0, 1, 1.5, 1.9, 2), 1):
        target = yr if power == 0 else yp if power == 1 else yg if power == 2 else yt
        configs.append((f"tweedie_{i}", TweedieRegressor(power=power, alpha=.2), target))
    configs.append(("poisson_capped", PoissonRegressor(alpha=.3, max_iter=1), yp))
    for prefix, m, target in configs:
        m.set_params(solver="newton-cholesky", tol=1e-10, max_iter=1 if prefix.endswith("capped") else 500)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Newton solver did not converge")
            m.fit(x, target)
        expected[prefix+"_coef"] = m.coef_
        expected[prefix+"_parameters"] = [m.intercept_, int(m.n_iter_ < m.max_iter)]
        expected[prefix+"_predict"] = m.predict(query)
        expected[prefix+"_score"] = [m.score(x, target)]
    return expected


def main():
    RESULTS.mkdir(exist_ok=True)
    report = {"sklearn_version": sklearn.__version__, "checks": []}
    for seed, n, p in ((7, 90, 3), (42, 84, 4), (2026, 96, 5)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))*.5 + np.linspace(-.2, .3, p)
        yr = x @ np.linspace(-1, 2, p)+1.2+rng.normal(size=n)*.025
        yr[:n//6] += 20
        rate = np.exp(x @ np.linspace(-.4, .5, p)+.3)
        yp = rng.poisson(rate).astype(float)
        yg = rate*rng.gamma(shape=3, scale=1/3, size=n)
        yt = rate*rng.gamma(shape=3, scale=1/3, size=n)
        yt[::7] = 0
        data, output = RESULTS/f"robust_glm_case_{seed}.txt", RESULTS/f"robust_glm_actual_{seed}.txt"
        with data.open("w") as f:
            f.write(f"{n} {p}\n")
            np.savetxt(f, np.column_stack([x, yr, yp, yg, yt]), fmt="%.17g")
        subprocess.run(["fpm", "run", "--target", "reference_robust_glm_driver", "--", str(data), str(output)],
                       cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = reference(x, yr, yp, yg, yt)
        assert actual.keys() == expected.keys(), actual.keys() ^ expected.keys()
        for name, values in expected.items():
            tolerance = 1e-4 if name.startswith("huber") else 3e-8
            np.testing.assert_allclose(actual[name], np.asarray(values).ravel(), atol=tolerance, rtol=tolerance,
                                       err_msg=f"seed={seed} {name}")
            report["checks"].append({"seed": seed, "api": name, "passed": True})
            print(f"PASS seed={seed} {name}", flush=True)
    (RESULTS/"robust_glm_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"{len(report['checks'])} robust/GLM reference comparisons passed against sklearn {sklearn.__version__}")


if __name__ == "__main__":
    main()
