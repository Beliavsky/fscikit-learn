"""Weighted and streaming categorical Naive Bayes reference checks."""
from pathlib import Path
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.naive_bayes import CategoricalNB
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"

def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 48, 3), (42, 72, 4), (2026, 24, 2)):
        rng = np.random.default_rng(seed)
        x = np.column_stack([rng.integers(0, j+2, n) for j in range(p)])
        for start in (0, n//3, 2*n//3): x[start, :] = np.arange(p)+1
        y = np.repeat([-4, 7, 21], n//3)
        w = rng.uniform(.2, 2, n); w[1] = 0
        data, output = RESULTS/f"categorical_{seed}.txt", RESULTS/f"categorical_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, w, y]), fmt="%.17g")
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_categorical_nb_driver", "--", str(data), str(output)],
                       cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        expected, label_likelihoods = {}, {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(1, 11):
                options, target, weight = {}, y, None
                if j == 2: options["alpha"] = .1
                if j == 3: options["fit_prior"] = False
                if j in (4, 5, 8): weight = w
                if j == 5: options["class_prior"] = [.2, .3, .5]
                if j == 6: options["min_categories"] = 8
                if j == 7: options["min_categories"] = np.arange(p)+5
                if j == 9: options["class_prior"] = [1., 0., 0.]
                if j == 10: target = np.full(n, 5)
                model = CategoricalNB(**options)
                if j == 8:
                    for start in (0, n//3, 2*n//3):
                        stop = start+n//3
                        model.partial_fit(x[start:stop], target[start:stop], classes=[-4, 7, 21],
                                          sample_weight=weight[start:stop])
                else:
                    model.fit(x, target, sample_weight=weight)
                query = (x+1) % model.n_categories_
                values = dict(classes=model.classes_, counts=model.class_count_, prior=model.class_log_prior_,
                              categories=model.n_categories_, joint=model._joint_log_likelihood(query),
                              logp=model.predict_log_proba(query), prob=model.predict_proba(query),
                              predict=model.predict(query), train_predict=model.predict(x),
                              score=[np.mean(actual[f"case{j}_train_predict"] == target)])
                label_likelihoods[f"case{j}_predict"] = (model.classes_, model._joint_log_likelihood(query))
                label_likelihoods[f"case{j}_train_predict"] = (model.classes_, model._joint_log_likelihood(x))
                for k in range(p):
                    values[f"category{k+1}"] = model.category_count_[k]
                    values[f"feature{k+1}"] = model.feature_log_prob_[k]
                expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name, actual[name].shape, want.shape)
            if name in label_likelihoods:
                classes, joint = label_likelihoods[name]
                assert np.all(np.isin(actual[name], classes)), (seed, name)
                selected = np.searchsorted(classes, actual[name])
                selected_scores = joint[np.arange(n), selected]
                assert np.all(selected_scores >= joint.max(axis=1)-1e-12), (seed, name)
            else:
                np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    (RESULTS/"categorical_nb.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} CategoricalNB result groups passed")

if __name__ == "__main__":
    main()
