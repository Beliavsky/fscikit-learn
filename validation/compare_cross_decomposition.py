"""Cross-decomposition numerical contracts against the installed sklearn baseline."""
from pathlib import Path
import json
import os
import subprocess
import warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import sklearn
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, CCA, PLSSVD
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def reference(x, y):
    expected = {}
    for family, cls, count in (("reg", PLSRegression, 4), ("canonical", PLSCanonical, 4), ("cca", CCA, 3)):
        for j in range(1, count+1):
            k = 2 if family == "reg" else min(2, y.shape[1])
            options = dict(n_components=k, scale=j != 2, tol=1e-10, max_iter=2000)
            if (family == "reg" and j == 3) or (family == "canonical" and j == 4) or (family == "cca" and j == 3):
                options["max_iter"] = 1
            if family == "canonical" and j == 3:
                options["algorithm"] = "svd"
            train_y = np.full_like(y, 2) if family == "reg" and j == 4 else y
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = cls(**options).fit(x, train_y)
            xt, yt = m.transform(x, train_y)
            xi, yi = m.inverse_transform(xt, yt)
            prefix = f"{family}{j}_"
            for attr in ("x_weights", "y_weights", "x_loadings", "y_loadings", "x_rotations", "y_rotations",
                         "coef", "intercept"):
                expected[prefix+attr] = getattr(m, attr+"_")
            expected.update({prefix+"x_scores": m._x_scores, prefix+"y_scores": m._y_scores,
                             prefix+"x_transform": xt, prefix+"y_transform": yt,
                             prefix+"x_inverse": xi, prefix+"y_inverse": yi,
                             prefix+"predict": m.predict(x+.031), prefix+"iterations": m.n_iter_})
            if family == "reg":
                expected[prefix+"score"] = [m.score(x, train_y)]
    for j in (1, 2):
        m = PLSSVD(n_components=min(2, y.shape[1]), scale=j == 1).fit(x, y)
        xt, yt = m.transform(x, y)
        prefix = f"svd{j}_"
        expected.update({prefix+"x_weights": m.x_weights_, prefix+"y_weights": m.y_weights_,
                         prefix+"x_transform": xt, prefix+"y_transform": yt, prefix+"query": m.transform(x+.031)})
    return expected


def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p, q in ((7, 80, 6, 3), (42, 72, 4, 1), (2026, 65, 3, 5), (123, 60, 5, 3)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))*np.linspace(.5, 2, p)+np.linspace(-1, 2, p)
        if seed == 123:
            x[:, 2] = 2*x[:, 0]
            x[:, 4] = 2
        y = x @ rng.normal(size=(p, q)) + rng.normal(size=(n, q))*.6 + np.arange(q)
        if seed == 123:
            y[:, 2] = 3
        data, output = RESULTS/f"cross_{seed}.txt", RESULTS/f"cross_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p} {q}\n")
            np.savetxt(stream, np.column_stack([x, y]), fmt="%.17g")
        subprocess.run(["fpm", "run", "--target", "reference_cross_decomposition_driver", "--", str(data), str(output)],
                       cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = reference(x, y)
        assert actual.keys() == expected.keys(), actual.keys() ^ expected.keys()
        for name, want in expected.items():
            np.testing.assert_allclose(actual[name], np.asarray(want).ravel(order="F"), atol=2e-7, rtol=2e-7,
                                       err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, api=name, passed=True))
        print(f"PASS cross-decomposition seed={seed}", flush=True)
    (RESULTS/"cross_decomposition_report.json").write_text(json.dumps(dict(sklearn_version=sklearn.__version__,
                                                                        checks=checks), indent=2)+"\n")
    print(f"{len(checks)} cross-decomposition comparisons passed against sklearn {sklearn.__version__}")

if __name__ == "__main__":
    main()
