"""IsolationForest streams, weighted random trees, warm starts and path scores."""
from pathlib import Path
import argparse
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"
FLAGS = "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--executable", type=Path); args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable:
        executable = args.executable.resolve()
    else:
        result = subprocess.run(["fpm", "run", "--flag", FLAGS, "--target", "reference_isolation_driver", "--runner"],
                                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(result.stdout)
        paths = [ROOT / s.strip().strip('"') for s in result.stdout.splitlines()
                 if s.strip().strip('"').endswith(("reference_isolation_driver", "reference_isolation_driver.exe"))]
        paths = [p for p in paths if p.is_file()]; assert len(paths) == 1, result.stdout
        executable = paths[0].resolve()
    checks = []
    for seed, n, p in ((7, 57, 3), (42, 72, 4), (2026, 45, 5), (0, 301, 6)):
        rng = np.random.default_rng(seed)
        original = rng.normal(size=(n, p)); weights = rng.uniform(.2, 2, n); weights[0] = 0
        for case in range(21):
            x, w = original.copy(), np.ones(n)
            options = dict(n_estimators=7, max_samples="auto", max_features=1., contamination="auto",
                           bootstrap=False, random_state=seed)
            warm = False
            if case == 1: options["max_samples"] = min(23, n)
            if case == 2: options["max_samples"] = .45
            if case == 3: options["bootstrap"] = True
            if case == 4: options.update(max_features=2, max_samples=.6)
            if case == 5: options.update(max_features=.5, contamination=.15)
            if case == 6: w = weights.copy()
            if case == 7: options.update(bootstrap=True, max_samples=19); w = weights.copy()
            if case == 8: warm = True; options["max_samples"] = .7
            if case == 9: options["max_samples"] = 1
            if case == 10: x[:, 1:] = 0
            if case == 11: x[:] = 2
            if case == 12: options.update(max_features=1, contamination=.5)
            if case == 13: options["max_samples"] = n+10
            if case == 14: w[:] = 0
            if case == 15: w[:] = 0; w[-1] = 1; options["max_samples"] = .2
            if case == 16: options.update(max_samples=.001, bootstrap=True)
            if case == 17: options["max_samples"] = .995
            if case == 18: options.update(max_samples=2, bootstrap=True)
            if case == 19: options.update(max_samples=1, max_features=.01, bootstrap=False)
            if case == 20: options["n_estimators"] = 100
            query = np.vstack([x+.137, np.zeros((1, p)), np.full((1, p), 8.)])
            ms, mf = options["max_samples"], options["max_features"]
            data, output = RESULTS / "isolation_input.txt", RESULTS / "isolation_actual.txt"
            with data.open("w") as stream:
                stream.write(f"{n} {p} {len(query)} {int(options['bootstrap'])} {int(warm)}\n")
                stream.write(f"{options['n_estimators']} {ms if isinstance(ms, int) else 0} {mf if isinstance(mf, int) else 0} {seed}\n")
                stream.write(f"{ms if isinstance(ms, float) else 0} {mf if isinstance(mf, float) else 0} "
                             f"{options['contamination'] if options['contamination'] != 'auto' else -1}\n")
                np.savetxt(stream, np.column_stack([x, w]), fmt="%.17g"); np.savetxt(stream, query, fmt="%.17g")
            subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, check=True)
            actual = {}
            for line in output.read_text().splitlines():
                name, *values = line.split(); actual[name] = np.array(values, dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))
                model = IsolationForest(**options)
                old_samples = []
                if warm:
                    model.n_estimators = 3; model.fit(x, sample_weight=w)
                    old_samples = [v.copy() for v in model.estimators_samples_]
                    model.warm_start = True; model.n_estimators = 7
                model.fit(x, sample_weight=w)
                samples = old_samples + model.estimators_samples_
                expected = dict(scores=model.score_samples(query), decision=model.decision_function(query),
                                predict=model.predict(query), train=model.score_samples(x),
                                meta=[model.offset_, len(model.estimators_), model.max_samples_,
                                      model._max_features, model.n_features_in_])
                for k, (tree, features, indices) in enumerate(zip(model.estimators_, model.estimators_features_, samples), 1):
                    t = tree.tree_
                    empty = t.n_node_samples == 0
                    values = dict(features=features+1, samples=indices+1,
                                  meta=[tree.random_state, tree.get_depth(), tree.get_n_leaves()],
                                  threshold=np.where(t.children_left < 0, 0, t.threshold),
                                  feature=np.where(t.children_left < 0, 0, t.feature+1), count=t.n_node_samples,
                                  value=np.where(empty, 0, t.value[:, 0, 0]),
                                  impurity=np.where(empty, 0, t.impurity),
                                  importance=np.nan_to_num(tree.feature_importances_))
                    expected.update({f"tree{k}_{name}": value for name, value in values.items()})
            assert actual.keys() == expected.keys(), (seed, case, actual.keys() ^ expected.keys())
            for name, value in expected.items():
                tolerance = 0 if name.endswith(("_features", "_samples", "_feature", "_count", "_meta")) or name == "predict" else 3e-10
                np.testing.assert_allclose(actual[name], np.asarray(value).ravel(), atol=tolerance, rtol=tolerance,
                                           err_msg=f"seed={seed} case={case} {name}")
                checks.append(dict(seed=seed, case=case, name=name, passed=True))
    (RESULTS / "isolation_forest.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} IsolationForest result groups passed")


if __name__ == "__main__":
    main()
