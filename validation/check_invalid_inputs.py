"""Check public input diagnostics in isolated processes (expected error stops)."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "power_method": "invalid PowerTransformer method", "power_positive": "strictly positive",
    "power_constant": "nonconstant data", "power_missing": "all-missing feature",
    "power_unfitted": "not fitted", "power_inverse_unfitted": "not fitted",
    "power_features": "feature count mismatch", "power_inverse_features": "feature count mismatch",
    "power_domain": "inverse outside power domain", "power_query_positive": "strictly positive",
    "power_empty": "nonempty matrix",
    "ksvm_c": "C must be positive", "svr_epsilon": "invalid SVR epsilon",
    "ksvm_tolerance": "invalid kernel SVM tolerance", "ksvm_iterations": "max_iter must be positive",
    "ksvm_kernel": "unsupported kernel SVM kernel", "ksvm_gamma": "invalid kernel SVM parameters",
    "ksvm_gamma_rule": "invalid kernel SVM gamma rule", "ksvm_poly": "invalid kernel SVM parameters",
    "ksvm_square": "precomputed matrix must be square", "ksvm_symmetry": "precomputed matrix must be symmetric",
    "ksvm_indefinite": "positive semidefinite", "ksvm_features": "sample counts differ",
    "ksvm_unfitted": "not fitted", "ksvm_targets": "sample counts differ",
    "ksvm_weights": "invalid kernel SVM sample weights", "ksvm_zero_weights": "invalid kernel SVM sample weights",
    "ksvm_weight_shape": "weight count mismatch", "nusvr_nu0": "invalid NuSVR nu",
    "nusvr_nu_high": "invalid NuSVR nu", "oneclass_nu1": "requires 0 < nu < 1",
    "oneclass_nu0": "requires 0 < nu < 1", "oneclass_unfitted": "not fitted",
    "oneclass_weight_shape": "weight count mismatch", "ksvm_nan_c": "C must be positive",
    "mds_components": "invalid MDS iteration/dimension controls",
    "mds_restarts": "invalid MDS iteration/dimension controls",
    "mds_iterations": "invalid MDS iteration/dimension controls", "mds_tolerance": "invalid MDS tolerance",
    "mds_stress": "invalid MDS normalized_stress", "mds_metric_stress": "requires nonmetric MDS",
    "mds_metric": "unsupported MDS dissimilarity", "mds_square": "dissimilarities must be square",
    "mds_symmetry": "dissimilarities must be symmetric", "mds_negative": "dissimilarities must be nonnegative",
    "mds_diagonal": "zero diagonal", "mds_init": "initial sample count mismatch",
    "mds_collapsed": "collapsed initial configuration", "mds_nonmetric_zero": "needs positive dissimilarities",
    "lle_neighbors": "0 < n_neighbors < n_samples", "lle_size": "0 < n_neighbors < n_samples",
    "lle_components": "invalid LLE n_components", "lle_reg": "invalid LLE regularization",
    "lle_method": "unsupported LLE method", "lle_ltsa_neighbors": "n_neighbors > n_components",
    "lle_unfitted": "not fitted", "lle_features": "sample counts differ", "lle_singular": "singular linear system",
    "cross_components": "invalid cross-decomposition n_components",
    "canonical_components": "invalid cross-decomposition n_components",
    "cca_components": "invalid cross-decomposition n_components",
    "svd_components": "invalid cross-decomposition n_components",
    "cross_iterations": "invalid cross-decomposition iteration controls",
    "cross_tolerance": "invalid cross-decomposition iteration controls",
    "cross_algorithm": "unsupported cross-decomposition algorithm",
    "cca_algorithm": "unsupported cross-decomposition algorithm",
    "cross_counts": "sample counts differ", "cross_samples": "at least two samples",
    "cross_unfitted": "not fitted", "cross_features": "sample counts differ",
    "cross_targets": "sample counts differ", "cross_inverse": "sample counts differ",
    "cross_score": "target width mismatch", "svd_rank": "retained full cross-covariance rank",
    "cross_degenerate": "degenerate predictor scores",
    "iso_unfitted": "not fitted", "iso_bounds": "invalid isotonic bounds",
    "iso_weight_count": "weight count mismatch", "iso_weights": "invalid isotonic weights",
    "iso_zero_weights": "invalid isotonic weights", "iso_length": "sample counts differ",
    "iso_out_of_bounds": "invalid isotonic out_of_bounds", "iso_query": "outside fitted bounds",
    "iso_pava_weights": "weights must be positive", "iso_nan": "NaN or infinity",
    "fourier_components": "RBF sampler n_components", "fourier_gamma": "invalid RBF sampler gamma",
    "fourier_unfitted": "not fitted", "fourier_features": "sample counts differ",
    "landmark_components": "Nystroem n_components", "landmark_kernel": "unsupported Nystroem kernel",
    "landmark_gamma": "invalid Nystroem gamma", "landmark_poly": "invalid Nystroem polynomial parameters",
    "landmark_unfitted": "not fitted", "landmark_features": "sample counts differ",
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
    # Ask fpm for the selected executable instead of rediscovering its build
    # directory or starting a complete fpm process for every rejection case.
    resolved = subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace", "--target",
                               "invalid_input_driver", "--runner"], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    candidates = []
    for line in resolved.stdout.splitlines():
        name = line.strip().strip('"')
        if name.endswith(("invalid_input_driver", "invalid_input_driver.exe")):
            path = ROOT / name
            if path.is_file():
                candidates.append(path.resolve())
    assert len(candidates) == 1, ("could not resolve unique fpm executable", resolved.stdout)
    results = []
    for case, message in CASES.items():
        proc = subprocess.run([str(candidates[0]), case], cwd=ROOT, text=True,
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
