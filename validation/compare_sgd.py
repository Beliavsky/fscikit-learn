"""Ordered SGDRegressor updates, averaging and streaming against sklearn."""
from pathlib import Path
import argparse
import json
import subprocess
import numpy as np
import sklearn
from sklearn.linear_model import SGDRegressor
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--executable', type=Path)
    parser.add_argument('--profile',choices=['checked','release'],default='checked')
    parser.add_argument('--report',type=Path,default=RESULTS/'sgd.json')
    args = parser.parse_args()
    run_options = ["--profile", "release"] if args.profile=="release" else ["--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"]
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 37, 5), (42, 43, 4), (2026, 19, 12)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))*.5
        y = x@rng.normal(size=p)+.8+.2*rng.normal(size=n)
        w = rng.uniform(.2, 2, n); w[0] = 0
        data, output = RESULTS/f"sgd_{seed}.txt", RESULTS/f"sgd_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, y, w]), fmt="%.17g")
        if args.executable:
            subprocess.run([str(args.executable.resolve()), str(data), str(output)], cwd=ROOT, check=True)
        else:
            subprocess.run(["fpm", "run", *run_options,
                            "--target", "reference_sgd_driver", "--", str(data), str(output)], cwd=ROOT, check=True)
        actual, expected = {}, {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        for j in range(1, 13):
            options = dict(max_iter=7, tol=None, eta0=.03, alpha=.01, epsilon=.2, shuffle=False)
            weights = w.copy()
            if j == 2: options.update(penalty=None, learning_rate="constant")
            if j == 3: options["loss"] = "huber"
            if j == 4: options["loss"] = "epsilon_insensitive"
            if j == 5: options["loss"] = "squared_epsilon_insensitive"
            if j == 6: options["average"] = True
            if j == 7: options["average"] = 10
            if j == 8: options["average"] = True
            if j == 9: options.update(max_iter=300, tol=1e-4)
            if j == 10: options["fit_intercept"] = False
            if j == 11: options.update(learning_rate="constant", eta0=.1, alpha=20)
            if j == 12: weights[:] = 0
            model = SGDRegressor(**options)
            if j == 8:
                model.partial_fit(x[:9], y[:9], sample_weight=weights[:9])
                model.partial_fit(x[9:10], y[9:10], sample_weight=weights[9:10])
                model.partial_fit(x[10:], y[10:], sample_weight=weights[10:])
            else: model.fit(x, y, sample_weight=weights)
            values = dict(coef=model.coef_, intercept=model.intercept_, train=model.predict(x),
                          query=model.predict(x+.137), score=[model.score(x, y)], clock=[model.n_iter_, model.t_])
            expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name)
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    args.report.write_text(json.dumps(dict(sklearn=sklearn.__version__,profile=args.profile,checks=checks), indent=2)+"\n")
    print(f"{len(checks)} SGD result groups passed")

if __name__ == "__main__":
    main()
