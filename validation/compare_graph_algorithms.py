"""Dense sklearn reference checks for neighbor, graph and outlier estimators."""
from pathlib import Path
import json
import os
import subprocess
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
import numpy as np
import sklearn
from sklearn.neighbors import (RadiusNeighborsClassifier, RadiusNeighborsRegressor,
                               NearestCentroid, LocalOutlierFactor)
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics.pairwise import rbf_kernel

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def reference(x, target, labels):
    query = x+.013
    expected = {}
    for weights in ("uniform", "distance"):
        model = RadiusNeighborsClassifier(radius=1.3, weights=weights).fit(x, labels)
        expected[f"radius_class_{weights}"] = model.predict_proba(query)
        model = RadiusNeighborsRegressor(radius=1.3, weights=weights).fit(x, target)
        expected[f"radius_reg_{weights}"] = model.predict(query)
    model = NearestCentroid().fit(x, labels)
    expected["centroid_mean"] = model.centroids_
    expected["centroid_predict"] = model.predict(query)
    expected["centroid_std"] = model.within_class_std_dev_
    expected["centroid_deviations"] = model.deviations_
    expected["centroid_shrink"] = NearestCentroid(shrink_threshold=.3).fit(x, labels).centroids_
    model = NearestCentroid(metric="manhattan").fit(x, labels)
    expected["centroid_median"] = model.centroids_
    expected["centroid_manhattan"] = model.predict(query)
    model = LocalOutlierFactor(n_neighbors=7).fit(x)
    expected["lof_train"] = model.negative_outlier_factor_
    model = LocalOutlierFactor(n_neighbors=7, contamination=.15, novelty=True).fit(x)
    expected["lof_offset"] = [model.offset_]
    expected["lof_query"] = model.score_samples(query)
    expected["lof_decision"] = model.decision_function(query)
    expected["lof_predict"] = model.predict(query)
    model = KernelPCA(n_components=3, kernel="rbf", gamma=.2, eigen_solver="dense").fit(x)
    expected["kpca_rbf_fit"] = model.transform(x)
    expected["kpca_rbf_query"] = model.transform(query)
    expected["kpca_rbf_values"] = model.eigenvalues_
    model = KernelPCA(n_components=3, kernel="precomputed", eigen_solver="dense").fit(rbf_kernel(x, gamma=.2))
    expected["kpca_precomputed"] = model.transform(rbf_kernel(query, x, gamma=.2))
    for kernel in ("linear", "poly", "cosine"):
        model = KernelPCA(n_components=3, kernel=kernel, gamma=.2, degree=2, eigen_solver="dense").fit(x)
        expected[f"kpca_{kernel}"] = model.transform(query)
    model = Isomap(n_neighbors=8, n_components=2, eigen_solver="dense", path_method="FW").fit(x)
    expected["isomap_distances"] = model.dist_matrix_
    expected["isomap_fit"] = model.embedding_
    expected["isomap_query"] = model.transform(query)
    expected["isomap_error"] = [model.reconstruction_error()]
    # This fixture intentionally has disconnected components. sklearn emits
    # many sparse-matrix efficiency warnings while adding completion edges;
    # capture them and assert that the intended completion path was exercised.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = Isomap(n_neighbors=1, n_components=2, eigen_solver="dense", path_method="FW").fit(x)
    assert any("connected components" in str(w.message) for w in caught)
    expected["isomap_completed_distances"] = model.dist_matrix_
    semi = np.full(len(labels), -1, dtype=int)
    semi[::4] = labels[::4]
    for label in np.unique(labels):
        i = np.flatnonzero(labels == label)[0]
        semi[i] = label
    for kind, cls in (("propagation", LabelPropagation), ("spreading", LabelSpreading)):
        for kernel in ("rbf", "knn"):
            model = cls(kernel=kernel, gamma=.2, n_neighbors=7, max_iter=3000, tol=1e-10).fit(x, semi)
            expected[f"{kind}_{kernel}_train"] = model.label_distributions_
            expected[f"{kind}_{kernel}_query"] = model.predict_proba(query)
            expected[f"{kind}_{kernel}_status"] = [model.n_iter_, int(model.n_iter_ < model.max_iter)]
    return expected


def main():
    RESULTS.mkdir(exist_ok=True)
    report = {"sklearn_version": sklearn.__version__, "checks": []}
    for seed, classes in ((7, 2), (42, 3), (2026, 4)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(60, 4)) @ rng.normal(size=(4, 4))
        target = x @ rng.normal(size=4)+rng.normal(size=60)*.1
        labels = np.argmax(x @ rng.normal(size=(4, classes))+rng.normal(size=(60, classes)), axis=1)*3-2
        data, output = RESULTS/f"graph_case_{seed}.txt", RESULTS/f"graph_actual_{seed}.txt"
        with data.open("w") as f:
            f.write(f"{len(x)} {x.shape[1]}\n")
            np.savetxt(f, np.column_stack([x, target, labels]), fmt="%.17g")
        subprocess.run(["fpm", "run", "--target", "reference_graph_driver", "--", str(data), str(output)],
                       cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = reference(x, target, labels)
        assert actual.keys() == expected.keys(), actual.keys() ^ expected.keys()
        for name, values in expected.items():
            want = np.asarray(values).ravel(order="F")
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8,
                                       err_msg=f"seed={seed} {name}")
            report["checks"].append({"seed": seed, "api": name, "passed": True})
            print(f"PASS seed={seed} {name}")
    (RESULTS/"graph_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"{len(report['checks'])} graph/neighbor reference comparisons passed against sklearn {sklearn.__version__}")


if __name__ == "__main__":
    main()
