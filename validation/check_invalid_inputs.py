"""Check public input diagnostics in isolated processes (expected error stops)."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "unfitted": "not fitted", "nan": "NaN or infinity", "features": "sample counts differ",
    "target_size": "sample counts differ", "single_class": "at least two classes",
    "neighbors": "exceeds training size", "clusters": "number of kmeans clusters",
    "components": "PCA n_components", "svm_c": "SVC controls", "tree_depth": "tree size controls",
    "mixture_reg": "mixture controls", "gp_length": "Gaussian process kernel parameters",
    "variance": "no feature", "missing": "all-missing feature", "empty_text": "empty text vocabulary",
    "folds": "KFold n_splits",
    "radius_empty": "no neighbors within radius", "centroid_metric": "unknown centroid metric",
    "lof_mode": "novelty=true", "lof_contamination": "LOF contamination",
    "kpca_symmetry": "symmetric training kernel", "kpca_indefinite": "negative eigenvalues",
    "kpca_width": "query width mismatch", "isomap_neighbors": "n_neighbors < n_samples",
    "unlabeled": "at least one labeled sample", "spreading_alpha": "alpha must lie in (0,1)",
    "bayes_prior": "hyperpriors must be positive", "bayes_unfitted": "not fitted",
    "ard_threshold": "invalid ARD parameters", "omp_sparsity": "invalid OMP sparsity",
    "factor_components": "invalid FactorAnalysis n_components", "factor_noise": "noise variance must be positive",
    "projection_auto": "automatic projection dimension", "projection_density": "projection density",
    "projection_unfitted": "not fitted",
}


def main():
    subprocess.run(["fpm", "build", "--flag", "-fcheck=all -fbacktrace"], cwd=ROOT, check=True)
    results = []
    for case, message in CASES.items():
        proc = subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace", "--target",
                               "invalid_input_driver", "--", case], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.returncode != 0, (case, "invalid input accepted", proc.stdout)
        assert message in proc.stdout, (case, "wrong diagnostic", proc.stdout)
        assert "Index '" not in proc.stdout and "Segmentation fault" not in proc.stdout, (case, proc.stdout)
        results.append({"case": case, "passed": True})
        print(f"PASS rejected {case}: {message}")
    destination = ROOT / "validation" / "results"
    destination.mkdir(exist_ok=True)
    (destination / "invalid_inputs.json").write_text(json.dumps(results, indent=2)+"\n")
    print(f"{len(results)} invalid-input cases passed")


if __name__ == "__main__":
    main()
