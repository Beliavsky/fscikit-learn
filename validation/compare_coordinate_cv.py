"""ElasticNetCV/LassoCV paths and refits against scikit-learn."""
from pathlib import Path
import json
import subprocess
import numpy as np
import sklearn
from sklearn.linear_model import ElasticNetCV, LassoCV
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"

def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 37, 5), (42, 43, 4), (2026, 19, 12)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))
        y = x@rng.normal(size=p)+.8+.5*rng.normal(size=n)
        data, output = RESULTS/f"coordinate_cv_{seed}.txt", RESULTS/f"coordinate_cv_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, y]), fmt="%.17g")
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_coordinate_cv_driver", "--", str(data), str(output)], cwd=ROOT, check=True)
        actual, expected = {}, {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        for j in range(1, 10):
            options = dict(alphas=[.1, 1., .01, .3], tol=1e-12, max_iter=10000, cv=3, n_alphas=12)
            train = x.copy()
            if j == 2: options["l1_ratio"] = [.2, .7, 1.]
            if j == 3: options.update(alphas=None, l1_ratio=[.3, .8], eps=.02)
            if j == 4: options.update(fit_intercept=False, cv=4)
            if j == 5: options.update(alphas=None, n_alphas=1)
            if j == 6: train[:, -1] = train[:, 0]; options["l1_ratio"] = .7
            if j == 8: options["alphas"] = None
            if j == 9: options["fit_intercept"] = False
            model = (ElasticNetCV if j <= 6 else LassoCV)(**options).fit(train, y)
            ratios = np.atleast_1d(options.get("l1_ratio", .5) if j <= 6 else 1.)
            grid = np.asarray(model.alphas_)
            if grid.ndim == 1: grid = np.tile(grid, (len(ratios), 1))
            path = model.mse_path_.reshape(len(ratios), grid.shape[1], options["cv"])
            values = dict(coef=model.coef_, intercept=[model.intercept_],
                          selected=[model.alpha_, getattr(model, "l1_ratio_", 1.), path.mean(axis=2).min()],
                          alphas=grid.T, path=path.transpose(1, 2, 0), train=model.predict(train),
                          query=model.predict(train+.137), score=[model.score(train, y)])
            expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name)
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    (RESULTS/"coordinate_cv.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} coordinate CV result groups passed")

if __name__ == "__main__":
    main()
