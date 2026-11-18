"""Reference checks for Bayesian/sparse regression, factors and projections.

Random streams differ. Projection algebra is checked against sklearn using the
actual Fortran-drawn matrices; native tests separately check their distribution.
"""
from pathlib import Path
import json
import os
import subprocess
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import sklearn
from sklearn.linear_model import BayesianRidge, ARDRegression, OrthogonalMatchingPursuit
from sklearn.decomposition import FactorAnalysis
from sklearn.random_projection import (GaussianRandomProjection, SparseRandomProjection,
                                       johnson_lindenstrauss_min_dim)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def reference(x, y, actual):
    q = x + .031
    expected = {}
    # In centered wide data sklearn's thin SVD retains one arbitrary null-space
    # direction in sigma_; independent SVD implementations choose different
    # bases. Use full-row-rank uncentered wide data for elementwise comparisons.
    for prefix, options in (("bayes", dict(fit_intercept=len(x)>=x.shape[1])), ("bayes_no_intercept", dict(fit_intercept=False, alpha_init=.7, lambda_init=1.3)),
                            ("bayes_capped", dict(fit_intercept=False, alpha_init=.7, lambda_init=1.3, max_iter=1))):
        m = BayesianRidge(tol=1e-8, **options).fit(x, y)
        expected[prefix+"_coef"] = m.coef_
        expected[prefix+"_sigma"] = m.sigma_
        expected[prefix+"_predict"], expected[prefix+"_std"] = m.predict(q, return_std=True)
        expected[prefix+"_parameters"] = [m.alpha_, m.lambda_, m.intercept_, m.n_iter_]
    for prefix, options in (("ard", {}), ("ard_pruned", dict(fit_intercept=False, threshold_lambda=200)),
                            ("ard_capped", dict(fit_intercept=False, threshold_lambda=200, max_iter=1))):
        m = ARDRegression(tol=1e-4 if len(x)<x.shape[1] else 1e-8, **({"max_iter": 1000} | options)).fit(x, y)
        expected[prefix+"_coef"] = m.coef_
        expected[prefix+"_sigma"] = m.sigma_
        expected[prefix+"_predict"], expected[prefix+"_std"] = m.predict(q, return_std=True)
        expected[prefix+"_lambda"] = m.lambda_
        expected[prefix+"_parameters"] = [m.alpha_, m.intercept_, m.n_iter_]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Orthogonal matching pursuit ended prematurely")
        m = OrthogonalMatchingPursuit(n_nonzero_coefs=2).fit(x, y)
        expected["omp_coef"] = m.coef_
        expected["omp_predict"] = m.predict(q)
        expected["omp_iter"] = [m.n_iter_]
        m = OrthogonalMatchingPursuit(tol=.5).fit(x, y)
        expected["omp_tol_coef"] = m.coef_
        expected["omp_tol_predict"] = m.predict(q)
        expected["omp_no_intercept"] = OrthogonalMatchingPursuit(n_nonzero_coefs=2, fit_intercept=False).fit(x, y).predict(q)
    for prefix, options in (("factor", dict(n_components=2)), ("factor_zero", dict(n_components=0)),
                            ("factor_capped", dict(n_components=1, max_iter=1, noise_variance_init=np.full(x.shape[1], .8)))):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FactorAnalysis did not converge")
            m = FactorAnalysis(svd_method="lapack", tol=1e-7, **({"max_iter": 3000} | options)).fit(x)
        z = m.transform(q)
        expected[prefix+"_cov"] = m.get_covariance()
        expected[prefix+"_precision"] = m.get_precision()
        expected[prefix+"_latent_gram"] = z @ z.T  # invariant to SVD signs
        expected[prefix+"_noise"] = m.noise_variance_
        expected[prefix+"_score"] = m.score_samples(q)
        expected[prefix+"_loglike"] = m.loglike_
        expected[prefix+"_iter"] = [m.n_iter_]
    for prefix, cls, k in (("gaussian", GaussianRandomProjection, 3),
                           ("sparse", SparseRandomProjection, 3),
                           ("dense_sign", SparseRandomProjection, x.shape[1]+2)):
        # Inject the native matrix to isolate transform/pseudoinverse correctness
        # from the intentionally different RNG. No sklearn fit is needed here.
        m = cls(n_components=k)
        m.components_ = actual[prefix+"_components"].reshape((k, x.shape[1]), order="F")
        m.n_features_in_ = x.shape[1]
        z = m.transform(x)
        expected[prefix+"_transform"] = z
        expected[prefix+"_inverse"] = m.inverse_transform(z)
        if prefix == "gaussian":
            expected["gaussian_pinv"] = np.linalg.pinv(m.components_)
    expected["jl_bound"] = [johnson_lindenstrauss_min_dim(len(x), eps=.4)]
    return expected


def main():
    RESULTS.mkdir(exist_ok=True)
    report = {"sklearn_version": sklearn.__version__, "checks": []}
    for seed, n, p in ((7, 80, 6), (42, 60, 5), (2026, 8, 12)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p)) + np.linspace(-.3, .7, p)
        # Correlated features exercise the factor model and sparse recovery.
        x[:, 2] += .8*x[:, 0]
        y = 2*x[:, 0]-1.5*x[:, 1] + .7 + rng.normal(size=n)*.05
        data, output = RESULTS/f"regproj_case_{seed}.txt", RESULTS/f"regproj_actual_{seed}.txt"
        with data.open("w") as f:
            f.write(f"{n} {p}\n")
            np.savetxt(f, np.column_stack([x, y]), fmt="%.17g")
        subprocess.run(["fpm", "run", "--target", "reference_regression_projection_driver", "--", str(data), str(output)],
                       cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = reference(x, y, actual)
        matrix_inputs = {name for name in actual if name.endswith("_components")}
        assert actual.keys() - matrix_inputs == expected.keys(), (actual.keys() - matrix_inputs) ^ expected.keys()
        for name, values in expected.items():
            want = np.asarray(values).ravel(order="F")
            # The small wide ARD fixture magnifies roundoff in sklearn's
            # Woodbury pseudoinverse; direct active-precision Cholesky is more
            # stable here. Keep the relaxed absolute bound local to this case.
            atol = 1e-5 if n<p and name.startswith("ard") else 2e-7
            rtol = 1e-5 if n<p and name.startswith("ard") else 2e-7
            np.testing.assert_allclose(actual[name], want, atol=atol, rtol=rtol, err_msg=f"seed={seed} {name}")
            report["checks"].append({"seed": seed, "api": name, "passed": True})
            print(f"PASS seed={seed} {name}")
    (RESULTS/"regression_projection_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"{len(report['checks'])} regression/factor/projection reference comparisons passed against sklearn {sklearn.__version__}")


if __name__ == "__main__":
    main()
