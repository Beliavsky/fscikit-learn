"""Least-angle estimators and every homotopy knot against local sklearn."""
from pathlib import Path
import argparse
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.linear_model import Lars, LassoLars, lars_path
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    checks = []
    dropout_count = 0
    for seed, n, p in ((7, 47, 8), (42, 21, 11), (2026, 9, 14), (91, 20, 10), (0, 12, 10)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))*np.linspace(.7, 1.4, p)+np.linspace(-.5, .3, p)
        if seed in (42, 91):
            x[:, 2:] += .9*x[:, :1]
        if seed == 91:
            x[:, 1:] = x[:, :1] + .15*rng.normal(size=(n, p-1))
        weights = np.zeros(p)
        weights[[0, 2, 4]] = [1.8, -2.1, 1.3]
        y = 1.7+x@weights+rng.normal(size=n)*.7
        if seed == 0:
            # Deliberately exercise four coefficient dropouts and re-entry events.
            rng = np.random.default_rng(0)
            x = rng.normal(size=(n, p)); x[:, 1:] += .8*x[:, :1]
            y = rng.normal(size=n)
        data = RESULTS / f"least_angle_{seed}.txt"
        output = RESULTS / f"least_angle_actual_{seed}.txt"
        np.savetxt(data, np.column_stack((x, y)), header=f"{n} {p}", comments="", fmt="%.17g")
        if args.executable:
            subprocess.run([str(args.executable.resolve()), str(data), str(output)], cwd=ROOT, check=True)
        else:
            subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                            "--target", "reference_least_angle_driver", "--", str(data), str(output)],
                           cwd=ROOT, check=True)
        actual, expected = {}, {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        for j in range(1, 21):
            if j <= 5:
                model = Lars(n_nonzero_coefs={2: 1, 3: 3, 5: 1000}.get(j, 500),
                             fit_intercept=j != 3, fit_path=j != 4)
            elif j <= 14:
                model = LassoLars(alpha={6: .1, 7: .01, 8: 0, 9: .03, 10: 0, 11: 1000,
                                        12: 0, 13: .02, 14: .05}[j],
                                  positive=j in (9, 10), max_iter=2 if j == 12 else 500,
                                  fit_intercept=j != 14, fit_path=j != 13)
            else:
                options = dict(method="lar" if j in (15, 19) else "lasso", return_n_iter=True,
                               max_iter={19: 0, 20: 1}.get(j, 500), alpha_min={17: .03, 20: .001}.get(j, 0),
                               positive=j == 18)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    alphas, active, coefs, iterations = lars_path(x, y, **options)
                values = dict(alphas=alphas, active=np.asarray(active)+1, path=coefs, iterations=[iterations])
                if j == 16:
                    dropouts = (np.abs(coefs[:, :-1]) > 1.e-8) & (np.abs(coefs[:, 1:]) < 1.e-10)
                    dropout_count += np.count_nonzero(dropouts)
                expected.update({f"case{j}_{name}": value for name, value in values.items()})
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model.fit(x, y)
            values = dict(coef=model.coef_, intercept=[model.intercept_],
                          shape=[model.n_features_in_, model.n_iter_],
                          predict=model.predict(x), query=model.predict(x+.137), score=[model.score(x, y)],
                          alphas=np.atleast_1d(model.alphas_))
            if model.fit_path:
                values.update(active=np.asarray(model.active_)+1, path=model.coef_path_)
            expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name, actual[name].shape, want.shape)
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    assert dropout_count > 0, "reference fixtures must exercise coefficient dropouts"
    (RESULTS / "least_angle.json").write_text(json.dumps(dict(sklearn=sklearn.__version__,
        dropout_events=int(dropout_count), checks=checks), indent=2)+"\n")
    print(f"{len(checks)} least-angle result groups passed; {dropout_count} dropout events")


if __name__ == "__main__":
    main()
