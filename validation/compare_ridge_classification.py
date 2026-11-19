"""Weighted ridge classification and default leave-one-out RidgeClassifierCV."""
from pathlib import Path
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"

def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 45, 3), (42, 54, 4), (2026, 9, 14)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))
        binary = np.where(x[:, 0]+.3*x[:, 1] > .2, 11, -5)
        if seed == 2026: binary = np.array([-5]*4+[11]*5)
        scores = np.column_stack([x[:, 0], -.6*x[:, 0]+x[:, 1], -x[:, 1]])
        multi = np.array([-4, 7, 21])[scores.argmax(axis=1)]
        if seed == 2026: multi = np.tile([-4, 7, 21], 3)
        assert len(np.unique(multi)) == 3 and len(np.unique(binary)) == 2
        w = rng.uniform(.2, 2, n); w[0] = 0
        data, output = RESULTS/f"ridgeclass_{seed}.txt", RESULTS/f"ridgeclass_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, w, binary, multi]), fmt="%.17g")
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_ridge_classification_driver", "--", str(data), str(output)],
                       cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        expected = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(1, 11):
                train, target, weight = x.copy(), multi, np.ones(n)
                options = dict(alpha=.7, solver="svd")
                if j == 2: target = binary
                if j == 3: options["fit_intercept"] = False
                if j in (4, 5, 6, 7): weight = w
                if j == 5: options["class_weight"] = "balanced"
                if j == 6: options["class_weight"] = {-4: .5, 7: 2., 21: 1.5}
                if j == 7: target = binary; options["fit_intercept"] = False
                if j == 8:
                    options["alpha"] = 0
                    if n <= p: options["fit_intercept"] = False
                if j == 9: train[:, -1] = train[:, 0]; options["alpha"] = .03
                if j == 10: target = binary; options.update(class_weight="balanced", alpha=10)
                model = RidgeClassifier(**options).fit(train, target, sample_weight=weight)
                values = dict(coef=model.coef_, intercept=np.broadcast_to(model.intercept_, (np.asarray(model.coef_).reshape(-1, p).shape[0],)), classes=model.classes_,
                              train=model.decision_function(train).reshape(n, -1),
                              query=model.decision_function(train+.137).reshape(n, -1),
                              predict=model.predict(train+.137), score=[model.score(train, target)])
                expected.update({f"ridge{j}_{name}": value for name, value in values.items()})
            for j in range(1, 8):
                target, weight = multi, np.ones(n)
                options = dict(alphas=[.03, .3, 3., 30.], store_cv_results=True)
                if j == 2: target = binary
                if j == 3: options["fit_intercept"] = False
                if j in (4, 5, 6, 7): weight = w
                if j == 5: options["class_weight"] = "balanced"
                if j == 6: options["class_weight"] = {-4: .5, 7: 2., 21: 1.5}
                if j == 7: target = binary; options["fit_intercept"] = False
                model = RidgeClassifierCV(**options).fit(x, target, sample_weight=weight)
                results = np.asarray(model.cv_results_).reshape((n, -1, 4))
                values = dict(coef=np.asarray(model.coef_).reshape(-1, p), intercept=np.broadcast_to(model.intercept_, (np.asarray(model.coef_).reshape(-1, p).shape[0],)),
                              classes=model.classes_, train=model.decision_function(x).reshape(n, -1),
                              query=model.decision_function(x+.137).reshape(n, -1), predict=model.predict(x+.137),
                              score=[model.score(x, target)], selected=[model.alpha_, model.best_score_],
                              cv_results=results, cv_scores=-results.mean(axis=(0, 1)))
                expected.update({f"ridgecv{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F"); got = actual[name]
            assert got.shape == want.shape, (seed, name, got.shape, want.shape)
            np.testing.assert_allclose(got, want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    (RESULTS/"ridge_classification.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} ridge classification result groups passed")

if __name__ == "__main__":
    main()
