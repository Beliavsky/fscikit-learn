"""Dense MDS/SMACOF and LLE/LTSA reference checks against sklearn 1.6.1."""
from pathlib import Path
import json
import argparse
import os
import subprocess
import warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import sklearn
from sklearn.manifold import MDS, LocallyLinearEmbedding
from sklearn.manifold._locally_linear import barycenter_weights
from sklearn.metrics import pairwise_distances
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def reference(x, init, actual):
    expected = {}
    n = len(x)
    d = pairwise_distances(x)
    missing = d**.7
    missing[0, 1] = missing[1, 0] = missing[2, 3] = missing[3, 2] = 0
    for j in range(1, 9):
        options = dict(metric=j <= 3 or j == 8, normalized_stress="auto", eps=1e-7, max_iter=300,
                       n_init=1, n_jobs=1, random_state=42,
                       dissimilarity="euclidean" if j in (1, 6, 8) else "precomputed")
        if j == 3:
            options["max_iter"] = 1
        if j == 5:
            options["normalized_stress"] = False
        if j == 7:
            options.update(normalized_stress=True, max_iter=2)
        train = x if j in (1, 6, 8) else (d if j in (2, 3) else missing)
        if j == 8:
            generator=np.random.RandomState(42)
            for k in (1,2,3):expected[f'start{k}']=generator.uniform(size=(n,2))
            options['n_init']=3
            m=MDS(**options).fit(train)
        else:
            start = np.column_stack([init, np.sin(.3*np.arange(1, n+1))]) if j == 3 else init
            m = MDS(**options).fit(train, init=start)
        expected.update({f"mds{j}_embedding": m.embedding_, f"mds{j}_dissimilarity": m.dissimilarity_matrix_,
                         f"mds{j}_parameters": [m.stress_, m.n_iter_]})
    for j in range(1, 5):
        options = dict(method="standard" if j <= 2 else "ltsa", n_neighbors={1: 8, 2: 10, 3: 16, 4: 18}[j],
                       n_components=1 if j in (2, 4) else 2, reg=.05 if j == 2 else .001,
                       eigen_solver="dense", neighbors_algorithm="brute", n_jobs=1)
        # Larger LTSA neighborhoods ensure every observation participates and
        # the skipped constant eigenvector is the only zero mode. Extra null
        # directions otherwise make truncated embedding geometry nonunique.
        m = LocallyLinearEmbedding(**options).fit(x)
        z, query = m.embedding_, m.transform(x+.017)
        neighbors = m.nbrs_.kneighbors(x, n_neighbors=options["n_neighbors"]+1, return_distance=False)[:, 1:]
        expected.update({f"lle{j}_gram": z @ z.T, f"lle{j}_query_gram": query @ query.T,
                         f"lle{j}_cross_gram": query @ z.T, f"lle{j}_neighbors": neighbors+1,
                         f"lle{j}_error": [m.reconstruction_error_]})
        if j <= 2:
            expected[f"lle{j}_weights"] = barycenter_weights(x, x, neighbors, reg=options["reg"])
    return expected


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--executable",type=Path)
    args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 24, 3), (42, 30, 4), (2026, 36, 5)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))*np.linspace(.7, 1.3, p)
        init = rng.normal(size=(n, 2))*.4
        data, output = RESULTS/f"manifold_{seed}.txt", RESULTS/f"manifold_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, init]), fmt="%.17g")
        command=([str(args.executable.resolve()),str(data),str(output)] if args.executable else
                 ["fpm","run","--target","reference_manifold_driver","--",str(data),str(output)])
        subprocess.run(command,cwd=ROOT,check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = reference(x, init, actual)
        assert actual.keys() == expected.keys()
        for name, want in expected.items():
            np.testing.assert_allclose(actual[name], np.asarray(want).ravel(order="F"), atol=2e-7, rtol=2e-7,
                                       err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, api=name, passed=True))
        print(f"PASS manifold methods seed={seed}", flush=True)
    (RESULTS/"manifold_report.json").write_text(json.dumps(dict(sklearn_version=sklearn.__version__, checks=checks),
                                                         indent=2)+"\n")
    print(f"{len(checks)} manifold comparisons passed against sklearn {sklearn.__version__}")

if __name__ == "__main__":
    main()
