"""Compare joint sparse regression with the installed sklearn reference."""
from pathlib import Path
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskLasso
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"

def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 60, 5), (42, 30, 7), (2026, 9, 14)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))
        x[:, -1] = 2
        if p == 7: x[:, -2] = x[:, 0]
        beta = rng.normal(size=(p, 3)); beta[2:, :] = 0
        y = x @ beta + np.array([1., -2., .5]) + rng.normal(scale=.1, size=(n, 3))
        data, output = RESULTS/f"multitask_{seed}.txt", RESULTS/f"multitask_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p} 3\n")
            np.savetxt(stream, np.column_stack([x, y]), fmt="%.17g")
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_multitask_driver", "--", str(data), str(output)], cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        expected = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(1, 9):
                target = y
                options = dict(alpha=.08, tol=1e-10, max_iter=20000)
                if j == 2: options["l1_ratio"] = .2
                if j == 3: options["l1_ratio"] = 1
                if j == 4: options["fit_intercept"] = False
                if j == 5: options["alpha"] = 100
                if j == 6: options["max_iter"] = 1
                if j == 7: options["warm_start"] = True
                if j == 8: target = y[:, :1]
                model = (MultiTaskLasso if j == 1 else MultiTaskElasticNet)(**options)
                if j == 7:
                    model.alpha = .5; model.fit(x, target); model.alpha = .08
                model.fit(x, target)
                centered = target-target.mean(axis=0) if model.fit_intercept else target
                values = dict(coef=model.coef_, intercept=np.broadcast_to(model.intercept_, (target.shape[1],)),
                              train=model.predict(x), query=model.predict(x+.137), score=[model.score(x, target)],
                              gap=[model.dual_gap_], eps=[model.tol*np.sum(centered**2)], iterations=[model.n_iter_])
                expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name, actual[name].shape, want.shape)
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    (RESULTS/"multitask.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} multi-task result groups passed")

if __name__ == "__main__":
    main()
