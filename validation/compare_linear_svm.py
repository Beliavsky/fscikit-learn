"""Primal L2 squared-loss linear SVM reference comparisons."""
from pathlib import Path
import json
import subprocess
import numpy as np
import sklearn
from sklearn.svm import LinearSVC, LinearSVR
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"

def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 60, 3), (42, 45, 4), (2026, 9, 14)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))
        binary = np.where(x[:, 0]+.3*x[:, 1] > .2, 11, -5)
        scores = np.column_stack([x[:, 0], -.6*x[:, 0]+x[:, 1], -x[:, 1]])
        multi = np.array([-4, 7, 21])[scores.argmax(axis=1)]
        if n == 9:
            binary = np.array([-5]*4+[11]*5); multi = np.tile([-4, 7, 21], 3)
        assert len(np.unique(multi)) == 3 and len(np.unique(binary)) == 2
        target = x[:, 0]*1.3-x[:, 1]*.7+.8+rng.normal(scale=.3, size=n)
        w = rng.uniform(.2, 2, n); w[0] = 0
        data, output = RESULTS/f"linearsvm_{seed}.txt", RESULTS/f"linearsvm_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, w, target, binary, multi]), fmt="%.17g")
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_linear_svm_driver", "--", str(data), str(output)], cwd=ROOT, check=True)
        actual, expected = {}, {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        for j in range(1, 11):
            train, labels, weight = x.copy(), binary, np.ones(n)
            options = dict(dual=False, tol=1e-12, max_iter=10000)
            if j == 2: labels = multi
            if j == 3: options["fit_intercept"] = False
            if j == 4: options.update(C=.05, intercept_scaling=2)
            if j in (5, 6, 7): labels, weight = multi, w
            if j == 6: options["class_weight"] = "balanced"
            if j == 7: options["class_weight"] = {-4: .5, 7: 2., 21: 1.5}
            if j == 8: weight = w; options["class_weight"] = {-5: .5, 11: 2.}
            if j == 9: train[:, -1] = train[:, 0]; labels = multi
            if j == 10: options.update(C=10, intercept_scaling=4)
            model = LinearSVC(**options).fit(train, labels, sample_weight=weight)
            decisions = model.decision_function(train).reshape(n, -1)
            regularizer = .5*np.sum(model.coef_**2, axis=1)
            if model.fit_intercept: regularizer += .5*(model.intercept_/model.intercept_scaling)**2
            class_weights = np.ones(len(model.classes_))
            if options.get("class_weight") == "balanced":
                class_weights = np.array([n/(len(model.classes_)*np.sum(labels == c)) for c in model.classes_])
            elif isinstance(options.get("class_weight"), dict):
                class_weights = np.array([options["class_weight"][c] for c in model.classes_])
            objectives = []
            native_coefs = actual[f"svc{j}_coef"].reshape(model.coef_.shape, order="F")
            augmented = (np.column_stack([train, np.full(n, model.intercept_scaling)])
                         if model.fit_intercept else train)
            for k in range(decisions.shape[1]):
                positive = model.classes_[1] if len(model.classes_) == 2 else model.classes_[k]
                signed = np.where(labels == positive, 1., -1.)
                costs = model.C*weight
                if len(model.classes_) == 2:
                    costs = costs*class_weights[np.searchsorted(model.classes_, labels)]
                else:
                    costs = costs*np.where(labels == positive, class_weights[k], 1.)
                objectives.append(regularizer[k]+np.sum(costs*np.maximum(0, 1-signed*decisions[:, k])**2))
                native_w = native_coefs[k]
                if model.fit_intercept:
                    native_w = np.r_[native_w, actual[f"svc{j}_intercept"][k]/model.intercept_scaling]
                native_loss = np.maximum(0, 1-signed*(augmented@native_w))
                gradient = native_w-2*augmented.T@(costs*signed*native_loss)
                assert np.linalg.norm(gradient) <= 1e-8*max(1, np.linalg.norm(native_w)), (seed, j, k, gradient)
                native_objective = .5*native_w@native_w+np.sum(costs*native_loss**2)
                np.testing.assert_allclose(actual[f"svc{j}_objective"][k], native_objective, atol=1e-10, rtol=1e-10)
            values = dict(coef=model.coef_, intercept=np.broadcast_to(model.intercept_, (model.coef_.shape[0],)),
                          classes=model.classes_, train=decisions,
                          query=model.decision_function(train+.137).reshape(n, -1),
                          predict=model.predict(train+.137), score=[model.score(train, labels)], objective=objectives)
            expected.update({f"svc{j}_{name}": value for name, value in values.items()})
        for j in range(1, 9):
            train, weight = x.copy(), np.ones(n)
            options = dict(dual=False, loss="squared_epsilon_insensitive", tol=1e-12, max_iter=10000)
            if j == 2: options["epsilon"] = .2
            if j == 3: options.update(fit_intercept=False, epsilon=.1)
            if j == 4: options.update(C=.05, intercept_scaling=2, epsilon=.3)
            if j == 5: weight = w; options["epsilon"] = .2
            if j == 6: weight = w; options.update(C=10, intercept_scaling=4, epsilon=.5)
            if j == 7: train[:, -1] = train[:, 0]; options["epsilon"] = .15
            if j == 8: options["epsilon"] = 100
            model = LinearSVR(**options).fit(train, target, sample_weight=weight)
            prediction = model.predict(train)
            objective = .5*np.sum(model.coef_**2)
            if model.fit_intercept: objective += .5*(model.intercept_[0]/model.intercept_scaling)**2
            objective += model.C*np.sum(weight*np.maximum(0, np.abs(target-prediction)-model.epsilon)**2)
            augmented = (np.column_stack([train, np.full(n, model.intercept_scaling)])
                         if model.fit_intercept else train)
            native_w = actual[f"svr{j}_coef"]
            if model.fit_intercept:
                native_w = np.r_[native_w, actual[f"svr{j}_intercept"][0]/model.intercept_scaling]
            residual = augmented@native_w-target
            excess = np.sign(residual)*np.maximum(0, np.abs(residual)-model.epsilon)
            gradient = native_w+2*model.C*augmented.T@(weight*excess)
            assert np.linalg.norm(gradient) <= 1e-8*max(1, np.linalg.norm(native_w)), (seed, j, gradient)
            native_objective = .5*native_w@native_w+model.C*np.sum(weight*excess**2)
            np.testing.assert_allclose(actual[f"svr{j}_objective"][0], native_objective, atol=1e-10, rtol=1e-10)
            values = dict(coef=model.coef_, intercept=np.atleast_1d(model.intercept_), train=prediction,
                          query=model.predict(train+.137), score=[model.score(train, target)], objective=[objective])
            expected.update({f"svr{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name, actual[name].shape, want.shape)
            # TRON can stop on tiny objective reductions before its requested gradient tolerance.
            coefficient_output = name.startswith("svc") and name.rsplit("_", 1)[1] in ("coef", "intercept", "train", "query")
            tolerance = 1e-6 if coefficient_output else 3e-8
            np.testing.assert_allclose(actual[name], want, atol=tolerance, rtol=tolerance, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    (RESULTS/"linear_svm.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} linear SVM result groups passed")

if __name__ == "__main__":
    main()
