"""Exact index parity with pinned scikit-learn, plus group isolation tests."""
from pathlib import Path
import json
import subprocess
import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold, LeaveOneOut, LeaveOneGroupOut, PredefinedSplit
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"


def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, counts in ((7, [1, 2, 3, 4, 5, 6]), (42, [2, 4, 6, 8]), (2026, [1, 1, 1, 1])):
        rng = np.random.default_rng(seed)
        groups = rng.permutation(np.repeat(np.arange(len(counts))*17-32, counts))
        n = len(groups)
        assignment = rng.choice([-3, -1, 0, 8], size=n)
        for k in (2, len(counts)):
            data = RESULTS / f"group_splitters_{seed}_{k}.txt"
            output = RESULTS / f"group_splitters_actual_{seed}_{k}.txt"
            data.write_text(f"{n} {k}\n" + " ".join(map(str, groups)) + "\n" +
                            " ".join(map(str, assignment)) + "\n")
            subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                            "--target", "reference_group_splitters_driver", "--", str(data), str(output)],
                           cwd=ROOT, check=True)
            actual = {}
            for line in output.read_text().splitlines():
                name, *values = line.split()
                actual[name] = np.array(values, dtype=int)
            expected = {}
            configs = dict(group=GroupKFold(k).split(np.zeros(n), groups=groups),
                           loo=LeaveOneOut().split(np.zeros(n)),
                           logo=LeaveOneGroupOut().split(np.zeros(n), groups=groups),
                           predefined=PredefinedSplit(assignment).split(),
                           all_train=PredefinedSplit([-1, -1]).split(), empty=PredefinedSplit([]).split(),
                           all_test=PredefinedSplit([8, 8]).split())
            for name, iterator in configs.items():
                folds = list(iterator)
                expected[name+"_count"] = [len(folds)]
                for i, (train, test) in enumerate(folds, 1):
                    expected[f"{name}_{i}_train"] = train+1
                    expected[f"{name}_{i}_test"] = test+1
                    if name == "group":
                        assert not set(groups[train]) & set(groups[test])
            assert actual.keys() == expected.keys()
            for name, want in expected.items():
                np.testing.assert_array_equal(actual[name], want, err_msg=f"{seed} {k} {name}")
                checks.append(dict(seed=seed, n_splits=k, name=name, passed=True))
    (RESULTS / "group_splitters.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} splitter result groups passed")


if __name__ == "__main__":
    main()
