"""Dense BIRCH tree, streaming, Ward partitions and distance reference checks."""
from pathlib import Path
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.cluster import Birch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"

def canonical(values, reference):
    mapping = {}
    for value in reference:
        if int(value) not in mapping:
            mapping[int(value)] = len(mapping)
    return np.array([mapping[int(value)] for value in values])

def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, p in ((7, 2), (42, 3), (2026, 4)):
        rng = np.random.default_rng(seed)
        x = rng.normal(scale=.24, size=(63, p)) + np.repeat([-2., 0., 3.], 21)[:, None]
        rng.shuffle(x)
        x[1] = x[0]
        query = x+.137
        data, output = RESULTS/f"birch_{seed}.txt", RESULTS/f"birch_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{len(x)} {p}\n")
            np.savetxt(stream, x, fmt="%.17g")
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_birch_driver", "--", str(data), str(output)], cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(1, 11):
                options = dict(threshold=.18, branching_factor=3, n_clusters=None)
                if j in (2, 7): options.update(threshold=.04, branching_factor=2)
                if j == 3: options["branching_factor"] = 50
                if j == 4: options["n_clusters"] = 3
                if j == 5: options.update(threshold=100, n_clusters=8)
                if j == 6: options["compute_labels"] = False
                if j == 8: options["n_clusters"] = 4
                if j == 9: options.update(threshold=.4, branching_factor=2)
                if j == 10: options.update(threshold=.001, branching_factor=2)
                model = Birch(**options)
                if j in (7, 8):
                    for i in range(0, len(x), 7): model.partial_fit(x[i:i+7])
                    if j == 8: model.n_clusters = 2; model.partial_fit()
                else: model.fit(x)
                values = dict(centers=model.subcluster_centers_, sub_labels=model.subcluster_labels_,
                              train=model.predict(x), query=model.predict(query),
                              distances=model.transform(x), query_distances=model.transform(query),
                              metadata=[len(x), p, int(model.n_clusters is not None and
                                                       len(model.subcluster_centers_) < model.n_clusters)])
                if hasattr(model, "labels_"): values["labels"] = model.labels_
                prefix = f"birch{j}_"
                # Farthest-pair orientation is sensitive to last-bit dot-product ties.
                # Align all centroid-indexed outputs once, then require a shared label mapping.
                centers = actual[prefix+"centers"].reshape((-1, p), order="F")
                want_centers = model.subcluster_centers_
                assert centers.shape == want_centers.shape, (seed, j, "centroid count")
                rows, columns = linear_sum_assignment(np.sum((want_centers[:, None]-centers[None, :])**2, axis=2))
                assert np.array_equal(rows, np.arange(len(want_centers)))
                actual[prefix+"centers"] = centers[columns].ravel(order="F")
                actual[prefix+"sub_labels"] = actual[prefix+"sub_labels"][columns]
                for suffix in ("distances", "query_distances"):
                    matrix = actual[prefix+suffix].reshape((len(x), -1), order="F")
                    actual[prefix+suffix] = matrix[:, columns].ravel(order="F")
                for name, value in values.items():
                    want = np.asarray(value).ravel(order="F")
                    got = actual[prefix+name]
                    assert got.shape == want.shape, (seed, j, name, got.shape, want.shape)
                    if name in ("labels", "sub_labels", "train", "query"):
                        # Cluster integer names may differ; compare the complete shared mapping.
                        got = canonical(got, got if j == 8 and name == "labels" else actual[prefix+"sub_labels"])
                        want = canonical(want, want if j == 8 and name == "labels" else model.subcluster_labels_)
                        np.testing.assert_array_equal(got, want, err_msg=f"{seed} {j} {name}")
                    elif name == "metadata":
                        np.testing.assert_array_equal(got, want)
                    else:
                        tol = 2e-7 if "distances" in name else 2e-12
                        np.testing.assert_allclose(got, want, atol=tol, rtol=2e-12, err_msg=f"{seed} {j} {name}")
                    expected[prefix+name] = value
                    checks.append(dict(seed=seed, name=prefix+name, passed=True))
        assert actual.keys() == expected.keys()
    (RESULTS/"birch.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} BIRCH result groups passed")

if __name__ == "__main__":
    main()
