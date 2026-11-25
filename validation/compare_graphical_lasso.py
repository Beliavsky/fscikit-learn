"""GraphicalLasso reference checks against the local sklearn baseline."""
from pathlib import Path
import argparse
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.covariance import GraphicalLasso, empirical_covariance

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--mode", choices=["cd", "lars"], default="cd")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 60, 3), (42, 75, 4), (2026, 90, 5)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))
        x[:, 1] += .7*x[:, 0]
        x[:, 2] += .5*x[:, 1]
        x += np.linspace(-.5, .5, p)
        data, output = RESULTS/f"graphical_{seed}.txt", RESULTS/f"graphical_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, x, fmt="%.17g")
        if args.executable:
            subprocess.run([str(args.executable.resolve()), str(data), str(output), args.mode], cwd=ROOT, check=True)
        else:
            subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                            "--target", "reference_graphical_lasso_driver", "--", str(data), str(output), args.mode],
                           cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = {}
        for j in range(1, 9):
            train, query = x.copy(), x+.137
            options = dict(mode=args.mode, alpha=.08, max_iter=300, tol=1e-9, enet_tol=1e-11)
            if j == 2:
                options["alpha"] = .2
            if j == 3:
                options["alpha"] = 10
            if j == 4:
                options["alpha"] = 0
            if j == 5:
                options["assume_centered"] = True
            if j == 6:
                train = empirical_covariance(x)
                options["covariance"] = "precomputed"
            if j == 7:
                train[:, -1] = x[:, 0]+.3*x[:, 1]
                query = train+.137
                options["alpha"] = .2
            if j == 8:
                options.update(max_iter=2, tol=1e-14)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GraphicalLasso(**options).fit(train)
            cost = model.costs_ if j == 4 else model.costs_[-1]
            expected.update({f"graph{j}_location": model.location_, f"graph{j}_covariance": model.covariance_,
                             f"graph{j}_precision": model.precision_, f"graph{j}_cost": cost,
                             f"graph{j}_mahalanobis": model.mahalanobis(query), f"graph{j}_score": [model.score(query)]})
        assert actual.keys() == expected.keys()
        for name, want in expected.items():
            np.testing.assert_allclose(actual[name], np.asarray(want).ravel(order="F"),
                                       atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, api=name, passed=True))
        print(f"PASS GraphicalLasso seed={seed}", flush=True)
    (RESULTS/f"graphical_lasso_{args.mode}_report.json").write_text(json.dumps(
        dict(sklearn_version=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} graphical-lasso comparisons passed against sklearn {sklearn.__version__}")


if __name__ == "__main__":
    main()
