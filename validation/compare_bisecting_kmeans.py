"""BisectingKMeans seeded hierarchy, weighted centers and query semantics."""
from pathlib import Path
import argparse
import json
import os
import subprocess
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import sklearn
from sklearn.cluster import BisectingKMeans
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"
FLAGS = "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"


def tree_arrays(model):
    nodes = [model._bisecting_tree]; frontier = [nodes[0]]
    children = np.zeros((2*model.n_clusters-1, 2), dtype=int)
    for _ in range(model.n_clusters-1):
        selected = max(frontier, key=lambda node: node.score)
        assert selected.left is not None
        position = next(i for i, node in enumerate(nodes) if node is selected)
        children[position] = [len(nodes)+1, len(nodes)+2]
        nodes += [selected.left, selected.right]
        at = next(i for i, node in enumerate(frontier) if node is selected)
        frontier[at:at+1] = [selected.left, selected.right]
    def descendants(node):
        if node.left is None: return [node.label]
        return descendants(node.left)+descendants(node.right)
    return dict(tree_centers=np.array([node.center+model._X_mean for node in nodes]), children=children,
                tree_labels=[node.label if node.left is None else -1 for node in nodes],
                tree_counts=[np.count_nonzero(np.isin(model.labels_, descendants(node))) for node in nodes],
                tree_scores=[node.score for node in nodes])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--executable", type=Path); args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable:
        executable = args.executable.resolve()
    else:
        result = subprocess.run(["fpm", "run", "--flag", FLAGS, "--target", "reference_bisecting_driver", "--runner"],
                                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(result.stdout)
        paths = [ROOT / s.strip().strip('"') for s in result.stdout.splitlines()
                 if s.strip().strip('"').endswith(("reference_bisecting_driver", "reference_bisecting_driver.exe"))]
        paths = [p for p in paths if p.is_file()]; assert len(paths) == 1, result.stdout
        executable = paths[0].resolve()
    checks = []
    for seed, n, p in ((7, 57, 3), (42, 72, 4), (2026, 45, 5), (0, 301, 2)):
        rng = np.random.default_rng(seed)
        original = rng.normal(size=(n,p)); w0 = rng.uniform(.2, 2, n); w0[0] = 0
        for case in range(22):
            x, w = original.copy(), np.ones(n)
            options = dict(n_clusters=8, n_init=1, max_iter=300, random_state=seed, tol=1e-4,
                           init="random", bisecting_strategy="biggest_inertia", algorithm="lloyd")
            operation = 1
            if case == 1: options["init"] = "k-means++"
            if case == 2: options["bisecting_strategy"] = "largest_cluster"
            if case == 3: options["algorithm"] = "elkan"
            if case == 4: options.update(init="k-means++", algorithm="elkan", n_init=4)
            if case == 5: w = w0.copy()
            if case == 6: w = w0.copy(); options.update(init="k-means++", n_init=3, bisecting_strategy="largest_cluster")
            if case == 7: options.update(max_iter=1, init="k-means++")
            if case == 8: options.update(tol=100., algorithm="elkan", init="k-means++")
            if case == 9: options["n_clusters"] = 1; w = w0.copy()
            if case == 10: options.update(n_clusters=2, n_init=5); operation = 2
            if case == 11: options.update(n_clusters=4, init="k-means++"); operation = 3
            if case == 12: x[:] = 2
            if case == 13: x[:, 1:] = 0; options.update(init="k-means++", n_init=2)
            if case == 14: options.update(n_clusters=3, algorithm="elkan", bisecting_strategy="largest_cluster"); w = w0.copy()
            if case == 15: options.update(n_clusters=1); w[:] = 0
            if case == 16: x += 100
            if case == 17: options.update(max_iter=1, algorithm="elkan")
            if case == 18: options.update(n_clusters=2, init="k-means++"); w[:] = 0; w[-1] = 1
            if case == 19: options.update(init="k-means++", algorithm="elkan", tol=0.)
            if case == 20: x[:] = 2; options.update(init="k-means++", algorithm="elkan")
            if case == 21: options["n_clusters"] = n
            query = np.vstack([x+.137, rng.normal(size=(13,p))])
            qw = rng.uniform(.2, 2, len(query))
            data, output = RESULTS / "bisecting_input.txt", RESULTS / "bisecting_actual.txt"
            with data.open("w") as stream:
                stream.write(f"{n} {p} {len(query)} {operation}\n")
                stream.write(f"{options['n_clusters']} {options['n_init']} {options['max_iter']} {seed} {options['tol']}\n")
                stream.write(f"{options['init']} {options['bisecting_strategy']} {options['algorithm']}\n")
                np.savetxt(stream, np.column_stack([x,w]), fmt="%.17g")
                np.savetxt(stream, np.column_stack([query,qw]), fmt="%.17g")
            subprocess.run([str(executable),str(data),str(output)], cwd=ROOT, check=True)
            actual = {}
            for line in output.read_text().splitlines():
                name, *values = line.split(); actual[name] = np.array(values,dtype=float)
            model = BisectingKMeans(**options)
            extra = {}
            if operation == 2: extra["fit_predict"] = model.fit_predict(x, sample_weight=w)
            elif operation == 3: extra["fit_transform"] = model.fit_transform(x, sample_weight=w)
            else: model.fit(x, sample_weight=w)
            expected = dict(centers=model.cluster_centers_, labels=model.labels_, inertia=[model.inertia_],
                            predict=model.predict(query), train_predict=model.predict(x), transform=model.transform(query),
                            score=[model.score(x),model.score(x,sample_weight=w),model.score(query,sample_weight=qw)],
                            meta=[model.n_features_in_,2*options['n_clusters']-1], **tree_arrays(model), **extra)
            assert actual.keys() == expected.keys(),(seed,case,actual.keys() ^ expected.keys())
            for name,value in expected.items():
                tol = 0 if name in ("labels","predict","train_predict","children","tree_labels","tree_counts","meta","fit_predict") else 3e-8
                np.testing.assert_allclose(actual[name],np.asarray(value).ravel(order="F"),atol=tol,rtol=tol,
                                           err_msg=f"seed={seed} case={case} {name}")
                checks.append(dict(seed=seed,case=case,name=name,passed=True))
    (RESULTS/"bisecting_kmeans.json").write_text(json.dumps(dict(sklearn=sklearn.__version__,checks=checks),indent=2)+"\n")
    print(f"{len(checks)} BisectingKMeans result groups passed")


if __name__ == "__main__":
    main()
