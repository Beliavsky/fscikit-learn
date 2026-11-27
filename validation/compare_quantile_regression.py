"""Dense QuantileRegressor solutions and LP certificates versus sklearn HiGHS."""
from pathlib import Path
import argparse
import json
import subprocess
import numpy as np
import sklearn
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import r2_score
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"
FLAGS = "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, help="Use an already compiled reference driver")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable:
        executable = args.executable.resolve()
    else:
        resolved = subprocess.run(["fpm", "run", "--flag", FLAGS, "--target", "reference_quantile_driver", "--runner"],
                                  cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE)
        candidates = [ROOT / line.strip().strip('"') for line in resolved.stdout.splitlines()
                      if line.strip().strip('"').endswith(("reference_quantile_driver", "reference_quantile_driver.exe"))]
        candidates = [path for path in candidates if path.is_file()]
        assert len(candidates) == 1, resolved.stdout
        executable = candidates[0].resolve()
    checks = []; records = []
    for seed, n, p in ((7, 57, 3), (42, 71, 4), (2026, 43, 5), (19, 9, 15), (321, 31, 30)):
        rng = np.random.default_rng(seed)
        original = rng.normal(size=(n, p))
        target = 1.7 + original @ rng.normal(size=p) + rng.exponential(size=n) - .5
        weights = rng.uniform(.2, 2, n); weights[0] = 0
        for case in range(24):
            x, y, w = original.copy(), target.copy(), weights.copy()
            quantile = (.1, .25, .5, .75, .9)[case % 5]
            alpha = (0., .02, .15, 1.)[(case // 5) % 4]
            intercept = case % 3 != 0
            nonunique = n < p and alpha == 0
            if case == 4: w[:] = 1
            if case == 7: x[:, -1] = 0
            if case == 8: x[:, -1] = x[:, 0]; nonunique = True
            if case == 9: x = np.round(x); y = np.round(y); nonunique = True
            if case == 10: x *= 1000
            if case == 11: y *= 1000
            if case == 12: y[:] = 4
            if case == 13: y[0] = 1.e6
            if case == 14: w *= 1.e-12
            if case == 15: w[1:] = 0; w[0] = 1; nonunique = True
            if case == 16: x[:] = 0; nonunique = True
            if case == 17: x[:, 0] = 1; nonunique = True
            if case == 18: w[1] = 1.e-14
            if case == 19: x *= .001
            if case == 20: quantile = .01; alpha = 0.; nonunique = n < p
            if case == 21: quantile = .99; alpha = 0.; intercept = True; nonunique = n < p
            if case == 22: alpha = 1000.
            if case == 23: alpha = 0.; y = 2 + x @ (np.arange(p)/p); nonunique = n < p
            if alpha == 0 and np.count_nonzero(w) < p + int(intercept): nonunique = True
            data = RESULTS / "quantile_input.txt"
            output = RESULTS / "quantile_actual.txt"
            with data.open("w") as stream:
                stream.write(f"{n} {p} {int(intercept)}\n{quantile} {alpha} 1e-10 10000\n")
                np.savetxt(stream, np.column_stack([x, y, w]), fmt="%.17g")
            subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, check=True)
            actual = {}
            for line in output.read_text().splitlines():
                name, *values = line.split(); actual[name] = np.array(values, dtype=float)
            model = QuantileRegressor(quantile=quantile, alpha=alpha, fit_intercept=intercept,
                                      solver="highs").fit(x, y, sample_weight=w/w.sum() if case == 14 else w)
            norm = w / w.sum()
            def objective(prediction, coef):
                residual = y - prediction
                return np.dot(norm, np.maximum(quantile * residual, (quantile-1) * residual)) + alpha * np.abs(coef).sum()
            expected_objective = objective(model.predict(x), model.coef_)
            native_objective = objective(actual["predict"], actual["coef"])
            scale = max(1., abs(expected_objective), np.max(np.abs(y[w > 0])))
            def check(name, got, want, atol=2e-7, rtol=2e-7):
                np.testing.assert_allclose(got, want, atol=atol, rtol=rtol,
                                           err_msg=f"seed={seed} case={case} {name}")
                checks.append(dict(seed=seed, case=case, name=name, passed=True))
            check("converged", actual["state"][2], 1, 0, 0)
            check("features", actual["state"][1], p, 0, 0)
            check("objective", native_objective, expected_objective, 2e-8*scale)
            check("reported_objective", actual["diagnostics"][0], native_objective, 2e-8*scale)
            dual = actual["dual"]
            check("dual_objective", np.dot(y, dual), native_objective, 2e-8*scale)
            check("dual_gap", actual["diagnostics"][1], 0, 2e-8*scale)
            check("reduced_cost", actual["diagnostics"][2], 0, 1.01e-10)
            check("dual_upper", np.maximum(dual-quantile*norm, 0), 0, 2e-8)
            check("dual_lower", np.maximum((quantile-1)*norm-dual, 0), 0, 2e-8)
            check("dual_features", np.maximum(np.abs(x.T @ dual)-alpha, 0), 0, 2e-8*max(1., np.max(np.abs(x))))
            check("dual_zero_weights", dual[w == 0], 0, 0, 0)
            if intercept: check("dual_intercept", dual.sum(), 0, 2e-8)
            check("prediction_identity", actual["predict"], x @ actual["coef"]+actual["intercept"][0])
            check("query_identity", actual["query"], (x+.137) @ actual["coef"]+actual["intercept"][0])
            check("score_identity", actual["score"], r2_score(y, actual["predict"]), 2e-7*scale)
            if not nonunique:
                check("coef", actual["coef"], model.coef_, 2e-7*scale)
                check("intercept", actual["intercept"], model.intercept_, 2e-7*scale)
                check("prediction", actual["predict"], model.predict(x), 2e-7*scale)
                check("query", actual["query"], model.predict(x+.137), 2e-7*scale)
            records.append(dict(seed=seed,case=case,input=data.read_text(),source=dict(coef=model.coef_.tolist(),intercept=float(model.intercept_),predict=model.predict(x).tolist(),query=model.predict(x+.137).tolist(),objective=float(expected_objective))))
    (args.report or RESULTS / "quantile_regression.json").write_text(json.dumps(
        dict(sklearn=sklearn.__version__, checks=checks,records=records), indent=2)+"\n")
    print(f"{len(checks)} QuantileRegressor result groups passed")


if __name__ == "__main__":
    main()
