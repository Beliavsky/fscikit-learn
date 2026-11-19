"""Compare dense OPTICS graphs, Xi hierarchies and re-extraction with sklearn."""
from pathlib import Path
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.cluster import OPTICS, cluster_optics_dbscan, cluster_optics_xi
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"

def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed in (7, 42, 2026):
        rng = np.random.default_rng(seed)
        x = np.vstack([rng.integers(-12, 13, (18, 2))/128,
                       rng.integers(-40, 41, (18, 2))/128 + [2, 0],
                       rng.integers(-80, 81, (12, 2))/128 + [0, 3]])
        x[1] = x[0]  # zero distances and duplicate-point plateaus
        d = pairwise_distances(x, metric="euclidean")
        np.fill_diagonal(d, 0)
        data, output = RESULTS/f"optics_{seed}.txt", RESULTS/f"optics_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{len(x)} 2\n")
            np.savetxt(stream, x, fmt="%.17g")
            np.savetxt(stream, d, fmt="%.17g")
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_optics_driver", "--", str(data), str(output)], cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(1, 13):
                opts = dict(min_samples=4, min_cluster_size=5, metric="precomputed")
                train = d
                if j == 2: opts["predecessor_correction"] = False
                if j == 3: opts["xi"] = .15
                if j == 4: opts["max_eps"] = .7
                if j == 5: opts.update(min_samples=.12, min_cluster_size=.18)
                if j == 6: opts["min_cluster_size"] = None
                if j == 7: opts.update(cluster_method="dbscan", eps=.3)
                if j == 8: opts.update(cluster_method="dbscan", eps=.8, max_eps=1)
                if j == 9: opts["metric"] = "manhattan"; train = x
                if j == 10: opts.update(metric="minkowski", p=3); train = x
                if j == 11: opts["max_eps"] = 0
                if j == 12: opts.update(min_samples=2, min_cluster_size=2)
                m = OPTICS(**opts).fit(train)
                ns = max(2, int(.12*len(x))) if j == 5 else opts["min_samples"]
                nc = max(2, int(.18*len(x))) if j == 5 else opts["min_cluster_size"] or ns
                values = dict(order=m.ordering_+1, predecessor=np.where(m.predecessor_ < 0, -1, m.predecessor_+1),
                              core=m.core_distances_, reach=m.reachability_, labels=m.labels_,
                              hierarchy=m.cluster_hierarchy_+1 if opts.get("cluster_method") != "dbscan" else [],
                              sizes=[ns, nc])
                for k in range(1, 4):
                    values[f"dbscan{k}"] = cluster_optics_dbscan(reachability=m.reachability_,
                        core_distances=m.core_distances_, ordering=m.ordering_, eps=k*.2)
                labels, hierarchy = cluster_optics_xi(reachability=m.reachability_, predecessor=m.predecessor_,
                    ordering=m.ordering_, min_samples=ns, xi=.1, min_cluster_size=3, predecessor_correction=False)
                values.update(rexi=labels, rehierarchy=hierarchy+1)
                expected.update({f"optics{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            got = actual[name]
            assert got.shape == want.shape, (seed, name, got.shape, want.shape)
            if name.endswith(("_core", "_reach")):
                np.testing.assert_allclose(got, want, atol=3e-14, rtol=3e-14, err_msg=f"{seed} {name}")
            else:
                np.testing.assert_array_equal(got, want, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    (RESULTS/"optics.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} OPTICS result groups passed")

if __name__ == "__main__":
    main()
