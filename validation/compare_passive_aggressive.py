"""PA-I/PA-II models against sklearn, using identical unshuffled passes."""
from pathlib import Path
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 45, 3), (42, 54, 4), (2026, 63, 5)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))
        y = x @ np.linspace(-.5, 1.5, p)+.2+rng.normal(scale=.03, size=n)
        scores = np.column_stack([x[:, 0]+.4*x[:, 1], -x[:, 0]+x[:, 1], -x[:, 1]+.2*x[:, 2]])
        labels = np.array([-4, 7, 21])[scores.argmax(axis=1)]
        assert len(np.unique(labels)) == 3
        data, output = RESULTS/f"pa_{seed}.txt", RESULTS/f"pa_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, y, labels]), fmt=["%.17g"]*(p+1)+["%d"])
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_passive_aggressive_driver", "--", str(data), str(output)],
                       cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(1, 9):
                train, target = x.copy(), labels.copy()
                options = dict(C=.5, shuffle=False, tol=None, max_iter=8, n_iter_no_change=3, random_state=0)
                if j == 2:
                    options.update(loss="squared_hinge", C=.3, fit_intercept=False)
                if j == 3:
                    options["class_weight"] = "balanced"
                if j in (4, 5):
                    options["average"] = 1 if j == 4 else 17
                if j == 6:
                    options.update(tol=1e-3, max_iter=100, C=.2)
                if j == 7:
                    target = np.where(labels == labels.min(), -5, 11)
                    options.update(loss="squared_hinge", class_weight="balanced", average=3)
                if j == 8:
                    train[[1, 4]] = 0
                    options["average"] = 7
                query = train+.137
                m = PassiveAggressiveClassifier(**options).fit(train, target)
                expected.update({f"pac{j}_coef": m.coef_, f"pac{j}_intercept": m.intercept_,
                                 f"pac{j}_classes": m.classes_, f"pac{j}_train": m.decision_function(train),
                                 f"pac{j}_query": m.decision_function(query), f"pac{j}_predict": m.predict(query),
                                 f"pac{j}_parameters": [m.n_iter_, m.t_], f"pac{j}_score": [m.score(train, target)]})
            for j in range(1, 9):
                train = x.copy()
                options = dict(C=.5, epsilon=.1, shuffle=False, tol=None, max_iter=8,
                               n_iter_no_change=3, random_state=0)
                if j == 2:
                    options.update(loss="squared_epsilon_insensitive", C=.3, epsilon=.05, fit_intercept=False)
                if j in (3, 4):
                    options["average"] = 1 if j == 3 else 100000
                if j == 5:
                    options.update(tol=1e-3, max_iter=100)
                if j in (6, 7):
                    train[[1, 4]] = 0
                    options["average"] = 5
                if j == 7:
                    options["loss"] = "squared_epsilon_insensitive"
                if j == 8:
                    options.update(epsilon=10000, tol=1e-3, max_iter=100)
                query = train+.137
                m = PassiveAggressiveRegressor(**options).fit(train, y)
                expected.update({f"par{j}_coef": m.coef_, f"par{j}_intercept": m.intercept_,
                                 f"par{j}_train": m.predict(train), f"par{j}_query": m.predict(query),
                                 f"par{j}_parameters": [m.n_iter_, m.t_], f"par{j}_score": [m.score(train, y)]})
        assert actual.keys() == expected.keys()
        for name, want in expected.items():
            np.testing.assert_allclose(actual[name], np.asarray(want).ravel(order="F"),
                                       atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, api=name, passed=True))
        print(f"PASS passive-aggressive models seed={seed}", flush=True)
    (RESULTS/"passive_aggressive_report.json").write_text(json.dumps(
        dict(sklearn_version=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} passive-aggressive comparisons passed against sklearn {sklearn.__version__}")


if __name__ == "__main__":
    main()
