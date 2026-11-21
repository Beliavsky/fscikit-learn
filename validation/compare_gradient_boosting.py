"""GradientBoostingClassifier stages, probabilities, losses and terminal predictions."""
from pathlib import Path
import json
import argparse
import subprocess
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--executable",type=Path)
    args=parser.parse_args();exe=args.executable.resolve() if args.executable else None
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n in ((7, 45), (42, 57), (2026, 69)):
        rng = np.random.default_rng(seed)
        x = np.sort(rng.normal(size=n))[:, None]
        binary = np.where(np.arange(n) < n//3, -5, 11)
        multi = np.select([np.arange(n) < n//5, np.arange(n) < n//5+n//3], [-4, 7], default=21)
        # A few isolated observations exercise non-pure leaves and deeper trees.
        if seed == 2026:
            binary[[3, n//3+5, n-4]] = [11, -5, -5]
        query = x+.137
        data, output = RESULTS/f"boost_{seed}.txt", RESULTS/f"boost_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} 1\n"); np.savetxt(stream, np.column_stack([x, binary, multi]), fmt="%.17g")
        subprocess.run(([str(exe),str(data),str(output)] if exe else
                       ["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_gradient_boosting_driver", "--", str(data), str(output)]),
                       cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        expected = {}
        for j in range(1, 11):
            options = dict(n_estimators=8, max_depth=1, random_state=0)
            target = binary
            if j == 2: options.update(max_depth=2, n_estimators=9, learning_rate=.2, init="zero")
            if j == 3: options.update(max_depth=3, n_estimators=6, learning_rate=.3, min_samples_leaf=3, min_samples_split=6)
            if j == 4: options.update(min_impurity_decrease=1, n_estimators=4)
            if j in (5, 6, 7, 8): target = multi
            if j == 6: options.update(max_depth=2, learning_rate=.15, init="zero")
            if j == 7: options.update(max_depth=3, n_estimators=6, learning_rate=.2, min_samples_leaf=3)
            if j == 8: options.update(min_impurity_decrease=1, n_estimators=4)
            if j == 9: options["loss"] = "exponential"
            if j == 10: options.update(loss="exponential", max_depth=2, n_estimators=6, learning_rate=.2, init="zero")
            model = GradientBoostingClassifier(**options).fit(x, target)
            staged_raw = np.stack([v.reshape(n, -1) for v in model.staged_decision_function(query)], axis=2)
            values = dict(classes=model.classes_, prior=np.array([(target == c).mean() for c in model.classes_]),
                          init=model._raw_predict_init(x)[0], loss=model.train_score_, importance=model.feature_importances_,
                          metadata=[model.n_estimators_, model.n_trees_per_iteration_, 1],
                          train=model.decision_function(x).reshape(n, -1), query=model.decision_function(query).reshape(n, -1),
                          proba=model.predict_proba(query), logproba=model.predict_log_proba(query),
                          predict=model.predict(query), score=[model.score(x, target)], staged_raw=staged_raw,
                          staged_proba=np.stack(list(model.staged_predict_proba(query)), axis=2),
                          staged_predict=np.stack(list(model.staged_predict(query)), axis=1), leaves=model.apply(query)+1,
                          tree_values=np.stack([np.column_stack([tree.predict(query) for tree in row])
                                                for row in model.estimators_], axis=1))
            expected.update({f"boost{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F"); got = actual[name]
            assert got.shape == want.shape, (seed, name, got.shape, want.shape)
            if name.endswith("_leaves"):
                # sklearn may subdivide constant-prediction regions redundantly.
                # Native leaves must map to constant reference tree predictions.
                assert np.all(got >= 1) and np.all(got == got.astype(int))
                got = got.reshape((n, -1), order="F")
                tree_values = np.asarray(expected[name.replace("_leaves", "_tree_values")]).reshape((n, -1), order="F")
                for column in range(got.shape[1]):
                    for leaf in np.unique(got[:, column]):
                        values = tree_values[got[:, column] == leaf, column]
                        np.testing.assert_allclose(values, values[0], atol=3e-8, rtol=3e-8,
                                                   err_msg=f"{seed} {name} tree {column}")
            else:
                np.testing.assert_allclose(got, want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    (RESULTS/"gradient_boosting.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} GradientBoostingClassifier result groups passed")

if __name__ == "__main__":
    main()
