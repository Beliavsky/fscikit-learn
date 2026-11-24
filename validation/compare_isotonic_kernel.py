"""Isotonic and kernel-map checks against sklearn 1.6.1, controlling random draws."""
from pathlib import Path
from unittest.mock import patch
import json
import argparse
import os
import subprocess
import warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import sklearn
from sklearn.isotonic import IsotonicRegression, isotonic_regression
from sklearn.kernel_approximation import RBFSampler, Nystroem
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def reference(x, y, w, actual):
    n, p = x.shape
    t = x[:, 0]
    q = np.r_[t+.17, t.min()-2, t.max()+2]
    expected = {}
    for j in range(1, 7):
        options = dict(increasing=j != 2, out_of_bounds="nan" if j == 4 else "clip")
        if j == 3:
            options.update(increasing="auto", y_min=-1, y_max=2)
        weights = None if j == 1 else w
        if j == 6:
            weights = np.zeros(n)
            weights[0] = 1
        train = np.ones(n) if j == 5 else t
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = IsotonicRegression(**options).fit(train, y, sample_weight=weights)
        expected.update({f"iso{j}_fit": m.transform(train), f"iso{j}_query": m.predict(q),
                         f"iso{j}_x": m.X_thresholds_, f"iso{j}_y": m.y_thresholds_,
                         f"iso{j}_parameters": [m.X_min_, m.X_max_, int(m.increasing_)]})
    for name, options in (("default", {}), ("weighted", dict(sample_weight=w+.1)),
                          ("decreasing", dict(sample_weight=w+.1, increasing=False)),
                          ("bounded", dict(sample_weight=w+.1, y_min=-.5, y_max=1.5))):
        expected["pava_"+name] = isotonic_regression(y, **options)
    for j in range(1, 4):
        # Feed the actual Fortran draws through sklearn's transform to check
        # feature algebra independently of generation, now checked by compare_rbf_features.py.
        m = RBFSampler(n_components=150, gamma="scale" if j == 3 else (0 if j == 2 else .7)).fit(x)
        expected[f"rbf{j}_gamma"] = [m._gamma]
        m.random_weights_ = actual[f"rbf{j}_weights"].reshape((p, 150), order="F")
        m.random_offset_ = actual[f"rbf{j}_offset"]
        expected[f"rbf{j}_fit"] = m.transform(x)
        expected[f"rbf{j}_query"] = m.transform(x+.031)
    for j, kernel in enumerate(("rbf", "rbf", "polynomial", "linear", "cosine"), 1):
        k = p-1 if j >= 4 else 6
        indices = actual[f"ny{j}_indices"].astype(int)-1
        assert len(np.unique(indices)) == k and np.all((indices >= 0) & (indices < n))
        order = np.r_[indices, np.setdiff1d(np.arange(n), indices)]
        class FixedOrder:
            def permutation(self, size):
                assert size == n
                return order
        m = Nystroem(kernel=kernel, gamma=.7 if j == 2 else None, coef0=.7, degree=2, n_components=k)
        # Only the landmark draw is fixed; sklearn performs its full fit/SVD.
        with patch("sklearn.kernel_approximation.check_random_state", return_value=FixedOrder()):
            m.fit(x)
        expected[f"ny{j}_components"] = m.components_
        expected[f"ny{j}_normalization"] = m.normalization_
        expected[f"ny{j}_fit"] = m.transform(x)
        expected[f"ny{j}_query"] = m.transform(x+.031)
    return expected


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path);parser.add_argument('--report',type=Path);parser.add_argument('--profile',default='development');args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 60, 3), (42, 72, 4), (2026, 84, 5)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))
        x[:, 0] = rng.integers(-6, 7, n)
        y = .4*x[:, 0]+rng.normal(size=n)*1.5
        w = rng.uniform(.1, 3, n)
        w[::7] = 0
        data, output = RESULTS/f"isokernel_{seed}.txt", RESULTS/f"isokernel_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, y, w]), fmt="%.17g")
        command=[str(args.executable),str(data),str(output)] if args.executable else ['fpm','run','--target','reference_isotonic_kernel_driver','--',str(data),str(output)]
        subprocess.run(command,cwd=ROOT,check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = reference(x, y, w, actual)
        metadata = {key for key in actual if key.endswith(("_weights", "_offset", "_indices"))}
        assert actual.keys()-metadata == expected.keys()
        for name, want in expected.items():
            np.testing.assert_allclose(actual[name], np.asarray(want).ravel(order="F"),
                                       atol=2e-8, rtol=2e-8, equal_nan=True, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, api=name, passed=True))
        print(f"PASS isotonic/kernel maps seed={seed}", flush=True)
    (args.report or RESULTS/"isotonic_kernel_report.json").write_text(json.dumps(dict(sklearn_version=sklearn.__version__,profile=args.profile,checks=checks),
                                                               indent=2)+"\n")
    print(f"{len(checks)} isotonic/kernel reference comparisons passed against sklearn {sklearn.__version__}")

if __name__ == "__main__":
    main()
