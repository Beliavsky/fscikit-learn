"""RFE/RFECV elimination, fitted models and complete CV paths against sklearn."""
from pathlib import Path
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso, LogisticRegression
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"


def stable_coefficient_order(estimator):
    # RFE squares importance_getter output. Ranks preserve squared-coefficient
    # ordering and resolve exact ties by input order without changing fitted models.
    importance = estimator.coef_**2
    if importance.ndim == 2:
        importance = importance.sum(axis=0)
    ranks = np.empty(len(importance), dtype=float)
    ranks[np.argsort(importance, kind="stable")] = np.arange(1, len(importance)+1)
    return ranks


def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    fixtures = ((7, 53, 7), (42, 41, 6), (2026, 31, 5), (91, 35, 6))
    for seed, n, p in fixtures:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))*np.linspace(.7, 1.3, p) + np.linspace(-.4, .5, p)
        x[:, -1] += .25*x[:, 0]
        y = 2.1*x[:, 1] - 1.6*x[:, 3] + .25*x[:, 0] + rng.normal(size=n)*.5
        linear = 1.8*x[:, 1] - 1.5*x[:, 3] + rng.normal(size=n)*.5
        binary = np.where(linear > np.median(linear), 11, -7)
        logits = np.column_stack((1.5*x[:, 0]+x[:, 2], -1.8*x[:, 1]+.5*x[:, 3], x[:, 1]-1.3*x[:, 2]))
        multi = np.array([11, -7, 3])[np.argmax(logits+rng.normal(size=(n, 3))*.4, axis=1)]
        if seed == 91:
            # sklearn permits classes smaller than cv; all training folds still contain each class.
            binary = np.array([11]*2+[-7]*(n-2))[rng.permutation(n)]
            multi = np.array([11]*2+[-7]*7+[3]*(n-9))[rng.permutation(n)]
        data = RESULTS / f"recursive_{seed}.txt"
        output = RESULTS / f"recursive_actual_{seed}.txt"
        np.savetxt(data, np.column_stack((x, y, binary, multi)), header=f"{n} {p}", comments="", fmt="%.17g")
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_recursive_selection_driver", "--", str(data), str(output)],
                       cwd=ROOT, check=True)
        actual, expected = {}, {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        for j in range(1, 21):
            cv = 11 <= j <= 19
            classifier = j in (7, 8, 9, 10, 15, 16, 17, 18, 20)
            response = multi if j in (8, 10, 16, 18) else binary if classifier else y
            estimator = LinearRegression(fit_intercept=j != 19)
            if j in (3, 12):
                estimator = Ridge(alpha=1.3, solver="svd")
            elif j in (4, 13):
                estimator = ElasticNet(alpha=.05, l1_ratio=.4, tol=1.e-11, max_iter=10000)
            elif j in (5, 14):
                estimator = Lasso(alpha=.08, tol=1.e-11, max_iter=10000)
            elif classifier:
                estimator = LogisticRegression(C=.8, solver="newton-cg", tol=1.e-11,
                                               max_iter=200, fit_intercept=j != 20)
            if not cv:
                count = {1: None, 2: 1, 3: p-1, 4: 2, 5: 2, 6: p+2, 7: 2, 8: 3,
                         9: p, 10: None, 20: 1}[j]
                step = {2: 2, 3: .35, 5: 2, 6: 100, 8: .4, 9: 2, 20: .2}.get(j, 1)
                model = RFE(estimator, n_features_to_select=count, step=step)
            else:
                minimum = {12: 2, 14: 3, 16: 2, 17: p+2}.get(j, 1)
                step = {12: 2, 13: .35, 16: 2, 17: .3, 18: .4, 19: 100}.get(j, 1)
                model = RFECV(estimator, min_features_to_select=minimum, step=step, cv=4 if j in (12, 18) else 3)
            if j in (4, 5, 13, 14):
                model.set_params(importance_getter=stable_coefficient_order)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model.fit(x, response)
            z = model.transform(x)
            values = dict(support=model.get_support().astype(int), ranking=model.ranking_,
                          shape=[model.n_features_in_, model.n_features_], train=z,
                          query=model.transform(x+.137), inverse=model.inverse_transform(z),
                          predict=model.predict(x), predict_query=model.predict(x+.137),
                          score=[model.score(x, response)], coef=model.estimator_.coef_,
                          intercept=np.atleast_1d(model.estimator_.intercept_))
            if classifier:
                values.update(classes=model.estimator_.classes_, proba=model.predict_proba(x+.137),
                              decision=model.decision_function(x+.137))
            if cv:
                results = model.cv_results_
                values.update(cv_counts=results["n_features"], cv_mean=results["mean_test_score"],
                              cv_std=results["std_test_score"],
                              cv_splits=np.column_stack([results[f"split{i}_test_score"] for i in range(model.cv)]))
            if j in (8, 10, 16, 18):
                # Multinomial intercept/logit common shifts are unidentifiable.
                # Compare their zero-mean gauge; probabilities remain compared directly.
                values["intercept"] = values["intercept"] - np.mean(values["intercept"])
                values["decision"] = values["decision"] - np.mean(values["decision"], axis=1, keepdims=True)
                key = f"case{j}_intercept"
                actual[key] -= actual[key].mean()
                key = f"case{j}_decision"
                matrix = actual[key].reshape((n, len(values["classes"])), order="F")
                actual[key] = (matrix-matrix.mean(axis=1, keepdims=True)).ravel(order="F")
            expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name, actual[name].shape, want.shape)
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True,
                               reference="stable coefficient ties" if int(name.split("_")[0][4:]) in (4, 5, 13, 14)
                               else "sklearn default"))
    (RESULTS / "recursive_selection.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} RFE/RFECV result groups passed")


if __name__ == "__main__":
    main()
