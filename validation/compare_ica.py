"""FastICA reference checks with common initial matrices and SVD orientation."""
from pathlib import Path
import json
import os
import subprocess
import warnings
import argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import sklearn
from sklearn.decomposition import FastICA
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path)
    parser.add_argument('--report',type=Path);args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed in (7, 42, 2026):
        rng = np.random.default_rng(seed)
        n, p = 400, 3
        s = np.column_stack([rng.laplace(size=n), rng.uniform(-2, 2, n), rng.standard_t(5, n)])
        x = s @ np.array([[1., .3, .7], [.2, 1.2, .4], [.6, .1, 1.1]]) + [1., -2., .5]
        xc = x-x.mean(axis=0)
        d, v = np.linalg.eigh(xc.T @ xc/n)
        prewhite = xc @ ((v/np.sqrt(d)) @ v.T)
        data, output = RESULTS/f"ica_{seed}.txt", RESULTS/f"ica_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, x, fmt="%.17g")
            np.savetxt(stream, prewhite, fmt="%.17g")
        command=([str(args.executable.resolve()),str(data),str(output)]if args.executable else
                 ["fpm", "run", "--target", "reference_ica_driver", "--", str(data), str(output)])
        subprocess.run(command,
                       cwd=ROOT, check=True)
        lines = iter(output.read_text().splitlines())
        for line in lines:
            a, f, b, c = map(int, line.split())
            k = p-1 if c == 2 and b != 3 else p
            initial = np.full((k, k), .13)
            np.fill_diagonal(initial, 1 + .1*np.arange(1, k+1))
            m = FastICA(n_components=k if b != 3 else None,
                        algorithm=("parallel", "deflation")[a-1], fun=("logcosh", "exp", "cube")[f-1],
                        whiten=("unit-variance", "arbitrary-variance", False)[b-1],
                        tol=1e-7, max_iter=2000 if c == 1 else 1, w_init=initial)
            input_x = prewhite if b == 3 else x
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                z = m.fit_transform(input_x)
            expected = [m.components_, m.mixing_, z, m.transform(input_x+.031), m.inverse_transform(z), [m.n_iter_]]
            for name, want in zip(("components", "mixing", "sources", "transform", "inverse", "iterations"), expected):
                got = np.array(next(lines).split(), dtype=float)
                np.testing.assert_allclose(got, np.asarray(want).ravel(order="F"), atol=2e-6, rtol=2e-6,
                                           err_msg=f"{seed} {a,f,b,c} {name}")
                checks.append(dict(seed=seed, case=[a,f,b,c], api=name, passed=True))
        print(f"PASS FastICA seed={seed}", flush=True)
    (args.report or RESULTS/"ica_report.json").write_text(json.dumps(dict(sklearn_version=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} FastICA reference comparisons passed against sklearn {sklearn.__version__}")

if __name__ == "__main__":
    main()
