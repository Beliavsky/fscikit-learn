"""Least-angle CV grids, residual MSE and AIC/BIC against installed sklearn."""
from pathlib import Path
import argparse
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.linear_model import LarsCV, LassoLarsCV, LassoLarsIC, LassoLars
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import KFold
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"


def single_alpha_reference(x, y, cv, eps):
    # sklearn 1.6.1 duplicates the zero interpolation endpoint for max_n_alphas=1,
    # yielding an empty finite-score grid. Score the same fold paths explicitly.
    paths = [_lars_path_residues(x[train], y[train], x[test], y[test],
                               Gram="auto", method="lasso", eps=eps)
             for train, test in KFold(cv).split(x)]
    grid = np.unique(np.concatenate([path[0] for path in paths]))
    grid = grid[::max(1, int(len(grid)/1))]
    errors = np.empty((len(grid), cv))
    for fold, (alphas, _, _, residuals) in enumerate(paths):
        # A one-value grid consists of the smallest knot from all folds.
        # Each fold is clamped at its minimum-alpha endpoint.
        assert grid[0] <= alphas[-1]
        errors[0, fold] = np.mean(residuals[-1]**2)
    model = LassoLars(alpha=grid[0], eps=eps).fit(x, y)
    model.alpha_, model.cv_alphas_, model.mse_path_ = grid[0], grid, errors
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 48, 7), (42, 37, 9), (2026, 18, 23), (0, 12, 10), (91, 28, 6)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))*np.linspace(.7, 1.4, p)+np.linspace(-.5, .3, p)
        x[:, 2:] += .4*x[:, :1]
        weights = np.zeros(p); weights[[0, 2, 4]] = [1.8, -2.1, 1.3]
        y = 1.7+x@weights+rng.normal(size=n)*.7
        if seed == 0:
            rng = np.random.default_rng(0)
            x = rng.normal(size=(n, p)); x[:, 1:] += .8*x[:, :1]
            y = rng.normal(size=n)
        data = RESULTS / f"lars_selection_{seed}.txt"
        output = RESULTS / f"lars_selection_actual_{seed}.txt"
        np.savetxt(data, np.column_stack((x, y)), header=f"{n} {p}", comments="", fmt="%.17g")
        if args.executable:
            subprocess.run([str(args.executable.resolve()), str(data), str(output)], cwd=ROOT, check=True)
        else:
            subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                            "--target", "reference_lars_selection_driver", "--", str(data), str(output)],
                           cwd=ROOT, check=True)
        actual, expected = {}, {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        for j in range(1, 19):
            if j <= 10:
                options = dict(cv=4 if j in (2, 6, 8) else 3,
                               fit_intercept=j not in (2, 8),
                               max_n_alphas={2: 4, 6: 4, 8: 6, 10: 1}.get(j, 1000),
                               max_iter={3: 2, 4: 0, 9: 3}.get(j, 500))
                train_size = n-int(np.ceil(n/options["cv"]))
                options["eps"] = (.1 if j <= 4 and train_size < p+int(options["fit_intercept"])
                                  and options["max_iter"] >= train_size else np.finfo(float).eps)
                if j <= 4:
                    model = LarsCV(**options)
                else:
                    model = LassoLarsCV(positive=j in (7, 8), **options)
            else:
                options = dict(criterion="bic" if j in (12, 14, 16, 18) else "aic",
                               fit_intercept=j != 14, positive=j in (15, 16),
                               max_iter={17: 2, 18: 0}.get(j, 500),
                               noise_variance={13: .7, 14: .7, 16: .4, 17: .7, 18: .7}.get(j))
                if n <= p+int(options["fit_intercept"]):
                    options["noise_variance"] = .7
                model = LassoLarsIC(**options)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if j == 10:
                    model = single_alpha_reference(x, y, cv=3, eps=options["eps"])
                else:
                    try:
                        model.fit(x, y)
                    except Exception as error:
                        raise RuntimeError(f"sklearn fit failed for seed={seed}, case={j}") from error
            values = dict(coef=model.coef_, intercept=[model.intercept_],
                          shape=[model.n_features_in_, model.n_iter_],
                          predict=model.predict(x), query=model.predict(x+.137),
                          score=[model.score(x, y)], alphas=np.atleast_1d(model.alphas_), alpha=[model.alpha_])
            if j <= 10:
                values.update(path=model.coef_path_, active=np.asarray(model.active_)+1,
                              cv_alphas=model.cv_alphas_, mse=model.mse_path_)
            else:
                values.update(criterion=model.criterion_, noise=[model.noise_variance_])
            expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name, actual[name].shape, want.shape)
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True,
                               reference="manual single-alpha CV" if int(name.split("_")[0][4:]) == 10
                               else "sklearn class"))
    (RESULTS / "lars_selection.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} least-angle model-selection result groups passed")


if __name__ == "__main__":
    main()
