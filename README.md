# fscikit-learn

A native Fortran fpm package translating a practical array subset of
scikit-learn: **207 estimator, transformer and splitter types**, metrics,
kernels, text and image features, and synthetic dataset generators. This is not a complete
or drop-in translation. Read [coverage and limits](docs/COVERAGE.md) before use.
OpenAI Codex generated the code.

Scikit-learn algorithms are the core scope. Selective public extensions use
separate `fsklearn_extra_*` modules and are listed in the
[extension inventory](docs/EXTENSIONS.md); none are currently implemented.
See the [scope policy](docs/SCOPE.md) for admission and attribution rules.

## Build and test

```
fpm build
fpm test
fpm run --example histogram_boosting
fpm run --example image_patches
fpm run --example neighborhood_components
fpm run --example chained_imputation
fpm run --example neighbor_graphs
fpm run --example random_feature_maps
fpm run --example threshold_tuning
fpm run --example logistic_cv
fpm run --example graphical_cv
fpm run --example ordinal_encoding
fpm run --example target_encoding
fpm run --example stacking_ensembles
fpm run --example bagging_ensembles
fpm run --example weighted_extra_trees
fpm run --example classification
fpm run --example polynomial_kernel_features
fpm run --example label_encoding
fpm run --example nonlinear_feature_scoring
fpm run --example clustering_agreement
fpm run --example resampling
fpm run --example nonlinear_embedding
fpm run --example lle_variants
fpm run --example partial_dependence
fpm run --example permutation_importance
fpm run --example voting_ensembles
fpm run --example multiclass_reductions
fpm run --example classifier_chains
fpm run --example regressor_chains
fpm run --example multioutput
fpm run --example weighted_streaming_classifiers
fpm run --example calibrated_probabilities
fpm run --example self_training
fpm run --example ranking_curves
python validation/compare_graphical_cv.py
python validation/compare_ordinal_encoding.py
python validation/compare_target_encoding.py
python validation/compare_stacking.py
python validation/compare_bagging.py
python validation/compare_weighted_trees.py
python validation/compare_cluster_information.py
python validation/compare_resampling.py
python validation/compare_tsne.py
python validation/compare_lle_variants.py
python validation/compare_partial_dependence.py
python validation/compare_inspection.py
python validation/compare_voting.py
python validation/compare_multiclass.py
python validation/compare_classifier_chains.py
python validation/compare_chains.py
python validation/compare_multioutput.py
python validation/compare_weighted_classifiers.py
python validation/compare_calibrated.py
python validation/compare_self_training.py
python validation/compare_ranking.py
python validation/compare_mutual_info.py
python validation/compare_label_binarization.py
python validation/compare_polynomial_count_sketch.py
python validation/compare_univariate_selection.py
python validation/compare_omp_cv.py
python validation/compare_multitask_cv.py
python validation/compare_minibatch_nmf.py
python validation/compare_minibatch_dictionary.py
python validation/compare_lda.py
python validation/compare_rbm.py
python validation/compare_hdbscan.py
python validation/compare_bayesian_mixture.py
python validation/compare_bicluster.py
python validation/compare_minibatch_kmeans.py
python validation/compare_bisecting_kmeans.py
python validation/compare_robust_covariance.py
python validation/compare_isolation_forest.py
python validation/compare_sgd_one_class.py
python validation/compare_nusvc.py
python validation/compare_quantile_regression.py
python validation/compare_adaboost_regression.py
python validation/compare_dictionary_learning.py
python validation/compare_lars_selection.py
python validation/compare_least_angle.py
python validation/compare_recursive_selection.py
python validation/compare_sequential_selection.py
python validation/compare_group_splitters.py
python validation/compare_sklearn.py
python validation/compare_graph_algorithms.py
python validation/compare_regression_projection.py
python validation/compare_robust_glm.py
python validation/compare_modes_density.py
python validation/compare_ica.py
python validation/compare_isotonic_kernel.py
python validation/compare_cross_decomposition.py
python validation/compare_manifold.py
python validation/compare_kernel_svm.py
python validation/compare_power_transform.py
python validation/compare_discretization.py
python validation/compare_graphical_lasso.py
python validation/compare_theil_sen.py
python validation/compare_passive_aggressive.py
python validation/compare_optics.py
python validation/compare_birch.py
python validation/compare_quantile_transform.py
python validation/compare_chi2_samplers.py
python validation/compare_gradient_boosting.py
python validation/compare_ridge_classification.py
python validation/compare_multitask.py
python validation/compare_categorical_nb.py
python validation/compare_knn_impute.py
python validation/compare_gp_classification.py
python validation/compare_spline.py
python validation/compare_linear_svm.py
python validation/compare_incremental_pca.py
python validation/compare_feature_agglomeration.py
python validation/compare_coordinate_cv.py
python validation/compare_adaboost.py
python validation/compare_sgd.py
python validation/compare_sgd_classifier.py
python validation/check_invalid_inputs.py
```

Arrays use `real(real64)` and shape `(n_samples, n_features)`, matching the
mathematical layout used by scikit-learn. Classification targets are integers.
Fitted estimators use type-bound `fit`, `predict`, and/or `transform` methods.
Invalid inputs fail with a diagnostic using `error stop`.

Import a specific module such as `sklearn_linear_model`, or use the convenience
facade `use sklearn`. See [API conventions and examples](docs/API.md).

```fortran
use sklearn
implicit none
type(standard_scaler) :: scaler
type(logistic_regression) :: model
! x is real(dp), shape (n_samples,n_features); y is integer.
call scaler%fit(x)
call model%fit(scaler%transform(x),y)
prediction = model%predict(scaler%transform(x_new))
```

Fit preprocessing only on training data and use its stored transform for the
test data. The example above illustrates API calls; see `example/classification.f90`
for a complete compilable program.

Reference: locally installed scikit-learn 1.6.1 (BSD-3-Clause). The Fortran
implementations are adapted to native arrays and do not invoke Python at runtime.
Python is used only for optional reference validation. See `docs/COVERAGE.md`.

SciFort at `../scifort` is available as a future numerical backend. NCA uses
a vendored, statically linked L-BFGS-B kernel under BSD-3-Clause; its pinned
source and license are in `vendor/lbfgsb`. No network fetch is needed to build.

Original project contributions use the [MIT License](LICENSE).
Scikit-learn-, LIBSVM- and L-BFGS-B-derived portions retain their BSD-3-Clause notices; see
[third-party notices](THIRD_PARTY_NOTICES.md). Recommended external Fortran code
and its attribution requirements are reviewed in [UPSTREAM_REUSE.md](docs/UPSTREAM_REUSE.md).
