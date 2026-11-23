# fscikit-learn

A native Fortran fpm package translating a practical array subset of
scikit-learn: **217 estimator, transformer and splitter types**, metrics,
kernels, structured, text and image features, and synthetic dataset generators. This is not a complete
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
fpm run --example coordinate_descent_nmf
fpm run --example beta_divergence_nmf
fpm run --example positive_linear_models
fpm run --example image_patches
fpm run --example neighborhood_components
fpm run --example gaussian_process_classification
fpm run --example gaussian_process_regression
fpm run --example gaussian_process_kernels
fpm run --example classification_sample_generators
fpm run --example synthetic_sample_generators
fpm run --example model_learning_curves
fpm run --example out_of_fold_predictions
fpm run --example parameter_search
fpm run --example model_based_selection
fpm run --example sequential_pipeline
fpm run --example feature_composition
fpm run --example transformed_targets
fpm run --example structured_features
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
python validation/compare_isomap_graphs.py --executable build/<profile>/app/reference_isomap_graphs_driver.exe --profile checked --report validation/results/isomap_graphs_checked.json
python validation/compare_regression_projection.py
python validation/compare_robust_glm.py
python validation/compare_modes_density.py
python validation/compare_density_extended.py --executable build/<profile>/app/reference_density_extended_driver.exe --profile checked --report validation/results/density_extended_checked.json
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
python validation/compare_adaboost_cart.py --executable build/<profile>/app/reference_adaboost_cart_driver.exe --profile checked --report validation/results/adaboost_cart_checked.json
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

Learning/validation curves and permutation significance tests are available through
`sklearn_validation` and the facade; see [API details](docs/API.md) and
[the example](example/model_learning_curves.f90).

Gaussian process classification supports independent one-vs-one pair fits and
confidence-based voting; see [the example](example/gaussian_process_ovo.f90).

Gaussian process regression supports seeded prior/posterior sampling, including
independent multiple targets; see [the example](example/gaussian_process_sampling.f90).

Factor analysis supports randomized SVD and varimax/quartimax rotation; see
[the example](example/factor_analysis_rotation.f90).

PCA supports variance-fraction and Minka dimension selection, covariance,
precision and Gaussian likelihoods; see [the example](example/probabilistic_pca.f90).

PCA offers full, randomized, covariance and automatic solver selection; see
[the example](example/pca_solver_selection.f90).

Randomized TruncatedSVD accepts dense and CSR input without densifying the sparse
matrix; see [the latent semantic example](example/sparse_latent_semantics.f90).

KernelPCA can learn an approximate inverse using kernel ridge regression; see
[the reconstruction example](example/kernel_preimage.f90).

LinearDiscriminantAnalysis offers SVD classification and discriminant projection
for wide or rank-deficient data; see [the example](example/discriminant_projection.f90).

LDA also supports the generalized eigen solver with scalar shrinkage; see
[the regularized projection example](example/shrunk_discriminant_projection.f90).

LDA supports automatic standardized Ledoit-Wolf shrinkage; see
[the example](example/automatic_discriminant_shrinkage.f90).

LDA and QDA accept custom class priors, including zero-weight classes; see
[the example](example/prior_weighted_discriminants.f90).

Multinomial, Bernoulli and Complement Naive Bayes support weighted streaming; see
[the example](example/streaming_document_nb.f90).

CategoricalNB supports zero smoothing and `force_alpha`; see
[the example](example/unsmoothed_categories.f90).

NearestCentroid supports empirical/custom priors and Euclidean probabilities; see
[the example](example/prior_weighted_centroids.f90).

Discrete Naive Bayes supports zero smoothing and `force_alpha`; Multinomial and
Complement NB also support feature-wise smoothing. See
[the example](example/feature_smoothed_nb.f90).

Multinomial, Bernoulli and Complement Naive Bayes accept CSR input, including
weighted streaming and mixed dense/CSR updates. See
[the sparse document example](example/sparse_document_nb.f90).

NMF supports dense coordinate descent with L1/L2 regularization, seeded shuffling
and random/custom/NNDSVD initialization. Select `solver='cd'`; see
[the example](example/coordinate_descent_nmf.f90) and
[the API contract](docs/API.md#coordinate-descent-nmf).

NMF's multiplicative-update solver also supports KL, Itakura-Saito and general
beta divergences with regularization and shared initialization controls. See
[the KL example](example/beta_divergence_nmf.f90) and
[the MU contract](docs/API.md#beta-divergence-nmf-multiplicative-updates).

LinearRegression, Ridge, ElasticNet/Lasso and their native coordinate-CV wrappers
support nonnegative coefficients with `positive=.true.`. Supported composition
and weighted OLS/Ridge adapters preserve the option. See
[the example](example/positive_linear_models.f90) and
[the contract and numerical differences](docs/API.md#positive-linear-model-coefficients).

The free `non_negative_factorization` API fits both factors or solves for W
with H fixed, returning both factors and the iteration count. See
[the fixed-component example](example/fixed_component_factorization.f90) and
[the function contract](docs/API.md#free-nonnegative-matrix-factorization).

`HashingVectorizer` supplies stateless word/character n-gram features as CSR,
with stop words, signed collisions, binary mode and L1/L2 normalization. Text
analysis currently accepts ASCII; see [the API contract](docs/API.md#hashingvectorizer-text-features)
and [the CSR classification example](example/hashed_text_documents.f90).

`Normalizer` and `Binarizer` provide dense/CSR estimator APIs and integrate with
dense pipelines and feature compositions. Their free functions also support
CSR input, and normalization supports both axes and returned norms. See
[the contract](docs/API.md#normalizer-binarizer-and-sparse-preprocessing-functions)
and [the weighted classification example](example/stateless_feature_preprocessing.f90).

`PolynomialFeatures` now provides an estimator API with degree ranges, lazy
powers/names, CSR expansion and dense pipeline support. The existing
`polynomial_features(x, ...)` call remains available and its degree-12 cap is
removed. See [the contract and source corrections](docs/API.md#polynomialfeatures-estimator-and-retained-array-api)
and [the quadratic pipeline example](example/polynomial_feature_pipeline.f90).

Image/grid graph builders now provide sparse axis connectivity and pixel/voxel
contrast weights, including masks and explicit zero edges. See
[the graph API](docs/API.md#image-and-grid-graph-builders) and
[the example](example/image_graphs.f90).

`tsne` now defaults to Barnes-Hut, with one through three embedding dimensions,
an adjustable tree angle and CSR precomputed distances. The dense exact solver
remains available with `method='exact'`. See [the example](example/barnes_hut_tsne.f90)
and [the API contract](docs/API.md#exact-and-barnes-hut-t-sne-and-trustworthiness).

SGD regression and classification now support L1/elastic-net penalties,
optimal/adaptive learning rates, source-compatible lazy averaging and seeded
shuffling. See [the example](example/regularized_sgd.f90) and
[the SGD controls](docs/API.md#streaming-sgd-classification).

SGD also supports warm fits, explicit coefficient/intercept initialization,
all nine classifier losses and log probabilities. See
[the warm-start example](example/warm_started_sgd.f90).


SGD regression and classification also accept CSR training, streaming batches
and queries, including unsorted/duplicate storage and source sparse intercept
updates. See [the sparse text SGD example](example/sparse_sgd_text.f90) and
[the CSR contract](docs/API.md#csr-sgd-input-and-queries).

MLP classification and regression use seeded minibatch Adam or SGD, with
momentum/Nesterov updates and constant, inverse-scaling or adaptive SGD rates.
See [the XOR example](example/stochastic_mlp.f90) and
[the MLP controls](docs/API.md#stochastic-multilayer-perceptrons).


MLPs also support full-data `solver='lbfgs'`, with iteration/evaluation limits
and optimizer diagnostics. See [the L-BFGS XOR example](example/lbfgs_mlp.f90)
and [the solver contract](docs/API.md#l-bfgs-multilayer-perceptrons), including
long-fit numerical differences from SciPy.


MLP `warm_start` reuses fitted parameters and accumulates training history across
fits, with fresh stochastic optimizer state per call. See
[the warm-start MLP example](example/warm_started_mlp.f90) and
[the warm-start contract](docs/API.md#warm-start-multilayer-perceptrons).


MLPs now support joint multioutput regression and multilabel indicator targets.
Use the matrix prediction methods and target-rank score overloads shown in
[the example](example/mlp_matrix_targets.f90) and
[the API recipe](docs/API.md#mlp-matrix-targets).


MLP `partial_fit` now retains Adam moments or SGD velocities across one-epoch
calls, with declared-class streaming and matrix targets. See
[the streaming example](example/incremental_mlp.f90) and
[the incremental contract](docs/API.md#incremental-multilayer-perceptrons).

MLP `early_stopping=.true.` holds out a seeded validation partition, records
validation scores and restores the best checkpoint. Binary targets use a
stratified holdout. See [the example](example/early_stopping_mlp.f90) and
[the early-stopping contract](docs/API.md#mlp-validation-set-early-stopping).

MLP fits, partial fits, predictions and scores also accept the native
`csr_matrix`, including warm transitions between dense and CSR input. The
first layer uses sparse products; parameters and later activations remain
dense. See [the CSR example](example/sparse_mlp.f90) and
[the input contract](docs/API.md#csr-multilayer-perceptrons).

MLP classifiers also support a single observed or declared class, with dense
or CSR input, including warm/incremental fits and validation stopping. See
[the example](example/single_class_mlp.f90) and
[the single-class contract](docs/API.md#single-class-mlp-classifiers).
