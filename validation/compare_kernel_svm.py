"""SVR/NuSVR/OneClassSVM comparisons against sklearn's LIBSVM implementation."""
from pathlib import Path
import json
import os
import subprocess
import argparse
import hashlib
import tempfile
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import sklearn
from sklearn.svm import SVR, NuSVR, OneClassSVM
from sklearn.metrics.pairwise import rbf_kernel
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def reference(x, y, w):
    n = len(x)
    expected = {}
    g, gq = rbf_kernel(x, gamma=.7), rbf_kernel(x+.137, x, gamma=.7)
    for family, cls, count in (("svr", SVR, 8), ("nu", NuSVR, 7), ("one", OneClassSVM, 7)):
        for j in range(1, count+1):
            options = dict(kernel="rbf", gamma="scale", tol=1e-8, max_iter=200000, shrinking=False,
                           degree=2, coef0=.4)
            if family != "one":
                options["C"] = 2
            if family == "svr":
                options["epsilon"] = .1
            else:
                options["nu"] = .3 if family == "one" else .4
            train, query, target = x, x+.137, y
            if j == 2:
                options["gamma"] = "auto"
                if family == "svr":
                    options.update(C=1, epsilon=.2)
                elif family == "nu":
                    options.update(C=1, nu=.7)
                else:
                    options["nu"] = .5
            if j == 3:
                options["gamma"] = .7
                if family == "svr":
                    options["epsilon"] = .05
                else:
                    options["nu"] = .4 if family == "one" else .3
            if j == 4:
                options["kernel"] = "linear"
                if family == "one":
                    options["nu"] = .23
                    # A nonzero boundary avoids the float-cache degeneracy of
                    # LIBSVM when the origin lies inside the training hull.
                    train, query = np.abs(x)+1, np.abs(x+.137)+1
                else:
                    options["C"] = .7
                    if family == "nu":
                        options["nu"] = .5
            if j == 5:
                options.update(kernel="poly", gamma=.3)
                if family != "one":
                    options["C"] = .7
                if family == "nu":
                    options["nu"] = .8
            if j == 6:
                options["kernel"] = "precomputed"
                train, query = g, gq
                if family != "svr":
                    options["nu"] = .5
            if j == 7:
                if family == "one":
                    options.update(gamma=0, nu=.4)
                else:
                    target = np.full(n, 2.)
            if j == 8:
                options["epsilon"] = 1000
            m = cls(**options)
            if family == "one":
                m.fit(train, sample_weight=w if j == 3 else None)
            else:
                m.fit(train, target, sample_weight=w if j == 3 else None)
            assert m.fit_status_ == 0, (family, j, "sklearn did not converge")
            prefix = f"{family}{j}_"
            if family == "one":
                expected.update({prefix+"train": m.decision_function(train), prefix+"query": m.decision_function(query),
                                 prefix+"raw": m.score_samples(query),
                                 prefix+"parameters": [m.intercept_[0], m.offset_[0], m._gamma]})
            else:
                expected.update({prefix+"train": m.predict(train), prefix+"query": m.predict(query),
                                 prefix+"parameters": [m.intercept_[0], m._gamma], prefix+"score": [m.score(train, target)]})
            # Linear/polynomial kernels can have nonunique dual representations.
            # Their primal decision functions and linear coefficients are checked.
            if j not in (4, 5) and not (family == "one" and j == 7):
                beta = np.zeros(n)
                # sklearn's dense LIBSVM wrapper reports support indices after
                # removing zero-weight rows. Map those back to original rows.
                active = np.flatnonzero(w > 0) if j == 3 else np.arange(n)
                beta[active[m.support_]] = m.dual_coef_[0]
                expected[prefix+"dual"] = beta
            if j == 4:
                expected[prefix+"coef"] = m.coef_[0]
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    binary_hash = hashlib.sha256(args.executable.read_bytes()).hexdigest() if args.executable else None
    RESULTS.mkdir(exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix='fsklearn-kernelsvm-')
    work = Path(temporary.name)
    checks = []
    for seed, n, p in ((7, 36, 3), (42, 40, 4), (2026, 44, 5)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p)) + np.linspace(-.4, .6, p)
        y = np.sin(x[:, 0]) + .4*x[:, 1] - .2*x[:, 2]**2 + rng.normal(size=n)*.08
        w = rng.uniform(.4, 2, n)
        w[::11] = 0
        data, output = work/f"kernelsvm_{seed}.txt", work/f"kernelsvm_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, y, w]), fmt="%.17g")
        command = [str(args.executable)] if args.executable else ["fpm", "run", "--target", "reference_kernel_svm_driver", "--"]
        subprocess.run([*command, str(data), str(output)],
                       cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = reference(x, y, w)
        assert actual.keys()-{"one4_dual_meta"} == expected.keys()
        beta = actual["one4_dual_meta"]
        assert np.all(beta >= -1e-12) and np.all(beta <= 1+1e-12)
        np.testing.assert_allclose(beta.sum(), .23*n, atol=1e-10, rtol=1e-10)
        for name, want in expected.items():
            tolerance = 2e-4 if name.endswith("_dual") else 3e-5
            np.testing.assert_allclose(actual[name], np.asarray(want).ravel(), atol=tolerance, rtol=tolerance,
                                       err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, api=name, passed=True))
        print(f"PASS kernel SVMs seed={seed}", flush=True)
    if args.executable:
        assert binary_hash == hashlib.sha256(args.executable.read_bytes()).hexdigest()
    (args.report or RESULTS/"kernel_svm_report.json").write_text(json.dumps(dict(sklearn_version=sklearn.__version__, checks=checks, executable_sha256=binary_hash),
                                                           indent=2)+"\n")
    print(f"{len(checks)} kernel SVM comparisons passed against sklearn {sklearn.__version__}")
    temporary.cleanup()

if __name__ == "__main__":
    main()
