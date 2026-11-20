"""Ordered binary/OVR streaming SGD classification against sklearn 1.6.1."""
from pathlib import Path
import json
import subprocess
import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"

def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 43, 4), (42, 57, 3), (2026, 29, 10)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))*.5
        binary = np.where(x[:, 0]+.4*x[:, 1]+.3*rng.normal(size=n)>0, 11, -5)
        multi = np.array([-4, 7, 21])[np.argmax(np.column_stack([x[:, 0], x[:, 1], -x[:, 0]-x[:, 1]])+.3*rng.normal(size=(n, 3)), axis=1)]
        multi[:9] = 7
        multi[9:12] = [-4, 7, 21]
        w = rng.uniform(.2, 2, n); w[0] = 0
        data, output = RESULTS/f"sgdc_{seed}.txt", RESULTS/f"sgdc_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, w, binary, multi]), fmt="%.17g")
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_sgd_classifier_driver", "--", str(data), str(output)], cwd=ROOT, check=True)
        actual, expected = {}, {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        for j in range(1, 13):
            options = dict(max_iter=7, tol=None, eta0=.03, alpha=.01, shuffle=False, learning_rate="invscaling", power_t=.25)
            y = binary
            if j == 2: y = multi
            if j == 3: options["loss"] = "log_loss"
            if j == 4: options.update(loss="log_loss", class_weight={-4: .5, 7: 2., 21: 1.5}); y = multi
            if j == 5: options.update(loss="modified_huber", average=True)
            if j == 6: options.update(loss="modified_huber", average=10); y = multi
            if j == 7: options.update(loss="squared_hinge", class_weight="balanced"); y = multi
            if j == 8: options.update(loss="perceptron", penalty=None, learning_rate="constant")
            if j == 9: options.update(loss="log_loss", average=True, fit_intercept=False); y = multi
            if j == 10: options.update(fit_intercept=False, class_weight={-5: .7, 11: 2.})
            if j == 11: options.update(max_iter=300, tol=1e-3); y = multi
            if j == 12: options.update(loss="log_loss", average=10, fit_intercept=False)
            model = SGDClassifier(**options)
            if j == 9:
                model.partial_fit(x[:9], y[:9], classes=[-4, 7, 21], sample_weight=w[:9])
                model.partial_fit(x[9:10], y[9:10], sample_weight=w[9:10])
                model.partial_fit(x[10:], y[10:], sample_weight=w[10:])
            else:
                model.fit(x, y, sample_weight=w)
                if j == 12: model.partial_fit(x[:9], y[:9], sample_weight=w[:9])
            values = dict(coef=model.coef_, intercept=model.intercept_, classes=model.classes_,
                          train=model.decision_function(x).reshape(n, -1),
                          query=model.decision_function(x+.137).reshape(n, -1), predict=model.predict(x+.137),
                          score=[model.score(x, y)], clock=[model.n_iter_, model.t_])
            if options.get("loss") in ("log_loss", "modified_huber"): values["proba"] = model.predict_proba(x+.137)
            expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name)
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    (RESULTS/"sgd_classifier.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} SGD classifier result groups passed")

if __name__ == "__main__":
    main()
