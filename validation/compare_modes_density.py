"""Dense sklearn references for modes, spectral methods and kernel density."""
from pathlib import Path
import json
import os
import subprocess
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
import numpy as np
import sklearn
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MeanShift, AffinityPropagation, SpectralClustering, estimate_bandwidth
from sklearn.manifold import SpectralEmbedding, spectral_embedding
from sklearn.neighbors import KernelDensity
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def reference(x, continuous):
    expected = {}
    query = x+.017
    for prefix, options in (("mean", dict(bandwidth=.8)), ("mean_auto", {}),
                            ("mean_capped", dict(bandwidth=.8, max_iter=0))):
        m = MeanShift(**options).fit(x)
        expected[prefix+"_centers"] = m.cluster_centers_
        expected[prefix+"_labels"] = m.labels_
        expected[prefix+"_iterations"] = [m.n_iter_]
        if prefix == "mean":
            expected["mean_predict"] = m.predict(query)
        if prefix == "mean_auto":
            expected["mean_auto_bandwidth"] = [estimate_bandwidth(x)]
    expected["bandwidth_quantiles"] = [estimate_bandwidth(x, quantile=q) for q in (.2, .3, .5)]
    m = MeanShift(bandwidth=.8, cluster_all=False, seeds=x[[0, len(x)//3]]).fit(x)
    expected["mean_orphan_centers"] = m.cluster_centers_
    expected["mean_orphan_labels"] = m.labels_
    expected["mean_orphan_predict"] = m.predict(query)
    m = AffinityPropagation(damping=.7, max_iter=500, random_state=21).fit(x)
    expected["ap_indices"] = m.cluster_centers_indices_+1
    expected["ap_centers"] = m.cluster_centers_
    expected["ap_labels"] = m.labels_
    expected["ap_predict"] = m.predict(query)
    expected["ap_affinity"] = m.affinity_matrix_
    expected["ap_status"] = [m.n_iter_, int(m.n_iter_<m.max_iter)]
    m = AffinityPropagation(damping=.7, max_iter=500, random_state=21, affinity="precomputed", preference=-2.5)
    m.fit(-euclidean_distances(x, squared=True))
    expected["ap_pre_indices"] = m.cluster_centers_indices_+1
    expected["ap_pre_labels"] = m.labels_
    expected["ap_pre_affinity"] = m.affinity_matrix_
    expected["ap_pre_status"] = [m.n_iter_, int(m.n_iter_<m.max_iter)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Affinity propagation did not converge")
        warnings.filterwarnings("ignore", message="This model does not have any cluster centers")
        m = AffinityPropagation(damping=.7, max_iter=1, preference=-1e8, random_state=21).fit(x)
        expected["ap_capped_labels"] = m.labels_
        expected["ap_capped_predict"] = m.predict(query)
    expected["ap_capped_status"] = [m.n_iter_, 0]
    for affinity in ("rbf", "nearest_neighbors", "precomputed"):
        m = SpectralEmbedding(n_components=3, affinity=affinity, gamma=.4, n_neighbors=9, random_state=21)
        m.fit(rbf_kernel(continuous, gamma=.4) if affinity == "precomputed" else continuous)
        a = m.affinity_matrix_
        a = a.toarray() if hasattr(a, "toarray") else a
        prefix = "embedding_"+affinity
        expected[prefix+"_affinity"] = a
        expected[prefix+"_gram"] = m.embedding_ @ m.embedding_.T
        expected[prefix+"_components"] = [connected_components(a, directed=False, return_labels=False)]
    for i, (affinity, assignment) in enumerate((("rbf", "cluster_qr"), ("rbf", "kmeans"),
                                               ("nearest_neighbors", "cluster_qr"), ("precomputed", "cluster_qr")), 1):
        m = SpectralClustering(n_clusters=3, affinity=affinity, gamma=.3, assign_labels=assignment, random_state=21)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Graph is not fully connected")
            m.fit(rbf_kernel(x, gamma=.3) if affinity == "precomputed" else x)
            a = m.affinity_matrix_
            a = a.toarray() if hasattr(a, "toarray") else a
            z = spectral_embedding(a, n_components=3, drop_first=False, random_state=21)
        prefix = f"spectral_{i}"
        expected[prefix+"_labels"] = m.labels_
        expected[prefix+"_affinity"] = a
        expected[prefix+"_gram"] = z @ z.T
        expected[prefix+"_components"] = [connected_components(a, directed=False, return_labels=False)]
    query = continuous+.017
    for kernel in ("gaussian", "tophat", "epanechnikov", "exponential", "linear"):
        for prefix, weights in (("density", None), ("weighted", np.arange(len(x))%4)):
            m = KernelDensity(kernel=kernel, bandwidth=.9).fit(continuous, sample_weight=weights)
            supported_query = query.copy()
            if weights is not None:
                supported_query[weights==0] = (np.roll(continuous, -1, axis=0)+.017)[weights==0]
            expected[prefix+"_"+kernel] = m.score_samples(supported_query)
            expected[prefix+"_score_"+kernel] = [m.score(supported_query)]
    for rule in ("scott", "silverman"):
        m = KernelDensity(bandwidth=rule).fit(continuous)
        expected["density_"+rule] = m.score_samples(query)
        expected["bandwidth_"+rule] = [m.bandwidth_]
    return expected


def main():
    RESULTS.mkdir(exist_ok=True)
    report = {"sklearn_version": sklearn.__version__, "checks": []}
    for seed, n, p in ((7, 60, 2), (42, 72, 3), (2026, 84, 4)):
        rng = np.random.default_rng(seed)
        truth = np.repeat(np.arange(3), n//3)
        centers = np.zeros((3, p)); centers[:, :2] = [[-3, -3], [0, 3], [3, -3]]
        x = centers[truth] + rng.normal(size=(n, p))*.08
        continuous = rng.normal(size=(n, p))
        data, output = RESULTS/f"modes_case_{seed}.txt", RESULTS/f"modes_actual_{seed}.txt"
        with data.open("w") as f:
            f.write(f"{n} {p}\n")
            np.savetxt(f, np.column_stack([x, continuous]), fmt="%.17g")
        subprocess.run(["fpm", "run", "--target", "reference_modes_density_driver", "--", str(data), str(output)],
                       cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        expected = reference(x, continuous)
        assert actual.keys() == expected.keys(), actual.keys() ^ expected.keys()
        for name, values in expected.items():
            want = np.asarray(values).ravel(order="F")
            if name.startswith("spectral_") and name.endswith("_labels"):
                assert adjusted_rand_score(actual[name], want) == 1, (seed, name)
                assert adjusted_rand_score(actual[name], truth) == 1, (seed, name, "known clusters")
            else:
                np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"seed={seed} {name}")
            report["checks"].append({"seed": seed, "api": name, "passed": True})
            print(f"PASS seed={seed} {name}", flush=True)
    (RESULTS/"modes_density_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"{len(report['checks'])} mode/spectral/density reference comparisons passed against sklearn {sklearn.__version__}")


if __name__ == "__main__":
    main()
