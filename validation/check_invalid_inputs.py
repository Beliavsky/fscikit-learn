"""Check public input diagnostics in isolated processes (expected error stops)."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "ica_unfitted": "not fitted", "ica_algorithm": "invalid FastICA algorithm",
    "ica_fun": "invalid FastICA contrast", "ica_whiten": "invalid FastICA whitening",
    "ica_alpha": "invalid FastICA alpha", "ica_iterations": "invalid FastICA max_iter",
    "ica_tolerance": "invalid FastICA tolerance", "ica_rank": "retained full rank",
    "ica_init_shape": "w_init shape mismatch", "ica_init_singular": "singular fixed-point matrix",
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
    "huber_epsilon": "invalid HuberRegressor controls", "ransac_samples": "invalid RANSAC sample/trial controls",
    "ransac_threshold": "invalid RANSAC residual threshold", "ransac_loss": "invalid RANSAC loss",
    "ransac_probability": "invalid RANSAC stopping controls", "poisson_targets": "targets must be nonnegative",
    "gamma_targets": "Gamma targets must be positive", "tweedie_power": "supported Tweedie powers",
    "tweedie_link": "supported GLM links", "glm_alpha": "invalid GLM controls",
    "glm_unfitted": "not fitted", "glm_constant_score": "D2 requires a nonconstant target",
    "meanshift_bandwidth": "MeanShift bandwidth must be positive", "meanshift_seeds": "no point within MeanShift bandwidth",
    "affinity_damping": "invalid AffinityPropagation controls", "affinity_square": "similarities must be square",
    "affinity_predict": "prediction requires feature-input fitting", "spectral_clusters": "invalid spectral n_clusters",
    "spectral_neighbors": "invalid spectral n_neighbors", "spectral_symmetry": "affinity must be symmetric",
    "spectral_negative": "affinity must be nonnegative", "spectral_components": "invalid spectral n_components",
    "density_bandwidth": "density bandwidth must be positive", "density_weights": "invalid density sample weights",
    "density_kernel": "unknown density kernel", "density_unfitted": "not fitted",
    "density_sampling": "density sampling supports gaussian/tophat only",
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
