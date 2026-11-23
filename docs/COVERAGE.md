# Translation coverage

Reference baseline: scikit-learn 1.6.1 installed in
`c:/Programs/Python313/Lib/site-packages/sklearn`.

There are 221 public estimator, transformer, and splitter types. Each is a
**subset** of the corresponding Python API, not a feature-complete class port.
All types have native tests. One hundred and fifty reference suites additionally check 1,594,588
result groups across seeded datasets using installed scikit-learn 1.6.1 and
independent numerical identities and optimality certificates.

This document counts core scikit-learn counterparts. Public extensions are
tracked separately in [EXTENSIONS.md](EXTENSIONS.md) and excluded from the
`sklearn` facade and core coverage counts. The [scope policy](SCOPE.md) allows
selective, clearly identified extensions while prioritizing core correctness.

Global scope: dense real64 CPU arrays, integer classification labels, generally one target
per sample; cross-decomposition, multi-task regression, joint MLP regression and independent multioutput wrappers explicitly accept multiple response columns. General sparse estimator inputs, pandas integration, Python estimator cloning,
metadata routing, parallel backends and binary model persistence are outside the
initial translation. The neighbor graph transformers provide CSR graph outputs and
accept CSR precomputed distances, as do Isomap and Barnes-Hut TSNE. Randomized TruncatedSVD and direct Multinomial/
Bernoulli/Complement Naive Bayes, SGD regression/classification, Normalizer/Binarizer and PolynomialFeatures also accept CSR matrices. Other estimator and
meta-estimator inputs remain dense.
Fortran identifiers use snake_case rather than CamelCase.

## Implemented domains

| Module (`sklearn_` prefix) | Public types and functions | Scope and numerical choices |
| --- | --- | --- |
| pipeline | `pipeline`; `pipeline_step` storage helper | Sequential dense transforms, supervised selectors and cross-fitted target encoding, supported native estimator tails, nested graphs, weighted fitting/scoring, inverse transforms, names and contiguous slices |
| feature_composition / pipeline | `column_transformer`, `feature_union` | Dense parallel transformer graphs, supported dense child types, recursive composition, positional/mask/name/callback column selection, remainder transforms, branch weights, feature names and output indices |
| function_transformer / compose | `function_transformer`, `transformed_target_regressor` | Dense matrix callbacks, feature names and inverse diagnostics; cloned target transforms, supported native bases, weighted linear fits and independent multiple targets; joint multi-target trees excluded |
| neighbor_graph / sparse | `kneighbors_transformer`, `radius_neighbors_transformer`; `csr_matrix` storage helper | Brute dense-feature and dense/CSR precomputed distances; connectivity/distance CSR outputs with explicit zeros |
| threshold | `fixed_threshold_classifier`, `tuned_threshold_classifier_cv` | Binary probability/margin cutoffs; weighted fold score curves, interpolation and first-best threshold selection; holdout, custom folds and prefit validation |
| tree_embedding | `random_trees_embedding` (also exported by `ensemble`) | Seeded totally random regression trees, dense one-hot leaf codes, warm starts, weights, leaf assignments and dense decision paths |
| multiclass | `one_vs_rest_classifier`, `one_vs_one_classifier`, `output_code_classifier` | Dense independent binary reductions; supported native classifier and regressor bases; constant heads; weighted fitting where the base supports it; Gaussian NB/SGD classification streaming for OVR/OVO; seeded legacy NumPy codebooks |
| inspection | `permutation_importance`, `permutation_importance_multi`, `partial_dependence` | Dense callback-based permutation importance and brute partial dependence/ICE; seeded subsampling/shuffles; weighted averages; categorical and Cunnane percentile grids; numeric multiple-output responses |
| multioutput | `multi_output_regressor`, `multi_output_classifier`, `regressor_chain`, `classifier_chain` | Independent fitted base models per response column; supported native linear/SGD, NB, neighbors and tree families; ragged class probabilities; SGD/Gaussian NB streaming; native base weighting limits apply |
| preprocessing | `kernel_centerer`, `ordinal_encoder`, `target_encoder`, `standard_scaler`, `minmax_scaler`, `maxabs_scaler`, `robust_scaler`, `simple_imputer`, `iterative_imputer`, `knn_imputer`, `missing_indicator`, `spline_transformer`, `label_encoder`, `one_hot_encoder`, `label_binarizer`, `multi_label_binarizer`, `power_transformer`, `kbins_discretizer`, `quantile_transformer`, `normalizer`, `binarizer`, `polynomial_features`; `normalize`, `binarize`, retained `polynomial_features(x,...)` | Ordinal category codes and infrequent grouping; cross-fitted target means; population variance; linear percentile interpolation; NaN imputation with mean/median/most-frequent/constant; numeric categories; empirical uniform/normal quantile transforms; dense/CSR polynomial estimator and retained array call with degree ranges, interactions and lazy powers/names; B-spline features and extrapolation; dense/CSR row normalization and thresholding, both-axis free normalization with returned norms |
| linear_model | `linear_regression`, `lars`, `lasso_lars`, `lars_cv`, `lasso_lars_cv`, `lasso_lars_ic`, `ridge`, `ridge_classifier`, `ridge_classifier_cv`, `multi_task_elastic_net`, `multi_task_lasso`, `multi_task_elastic_net_cv`, `multi_task_lasso_cv`, `lasso`, `elastic_net`, `lasso_cv`, `elastic_net_cv`, `logistic_regression`, `logistic_regression_cv`, `perceptron`, `passive_aggressive_classifier`, `passive_aggressive_regressor`, `sgd_regressor`, `sgd_classifier`, `sgd_one_class_svm`, `bayesian_ridge`, `ard_regression`, `orthogonal_matching_pursuit`, `orthogonal_matching_pursuit_cv`, `huber_regressor`, `quantile_regressor`, `ransac_regressor`, `theil_sen_regressor`, `poisson_regressor`, `gamma_regressor`, `tweedie_regressor` | SVD least squares/ridge; least-angle/Lasso homotopy paths with CV and AIC/BIC selection; coordinate descent with K-fold alpha/l1-ratio selection; weighted logistic Newton solver with raw-score prediction; perceptron; PA-I/II classification/regression with averaging; streaming stochastic-gradient regression/classification and one-class detection; Bayesian precision updates/uncertainty; ARD pruning; OMP; joint Huber coefficient/scale fitting; weighted L1-regularized quantile regression; RANSAC linear consensus; Theil-Sen subset/spatial-median fits; Newton GLMs |
| naive_bayes | `gaussian_nb`, `multinomial_nb`, `bernoulli_nb`, `complement_nb`, `categorical_nb` | Weighted/streaming Gaussian moments, empirical/fixed priors and variance smoothing; dense/CSR weighted and streaming discrete event counts and fixed/empirical priors; nonnegative scalar alpha and force_alpha flooring, plus feature-wise Multinomial/Complement smoothing; Bernoulli threshold; optional complement normalization |
| neighbors | `neighborhood_components_analysis`, `nearest_neighbors`, `kneighbors_classifier`, `kneighbors_regressor`, `radius_neighbors_classifier`, `radius_neighbors_regressor`, `nearest_centroid`; re-exports `local_outlier_factor`, `kernel_density` | Supervised NCA metric learning; brute-force search; uniform/distance weights; radius queries/classification/regression; mean/median nearest centroids with shrinkage, uniform/empirical/custom priors and Euclidean discriminant probabilities; tree-based kernel density |
| cluster | `kmeans`, `bisecting_kmeans`, `mini_batch_kmeans`, `spectral_coclustering`, `spectral_biclustering`, `hdbscan`, `dbscan`, `agglomerative_clustering`, `feature_agglomeration`, `mean_shift`, `affinity_propagation`, `spectral_clustering`, `optics`, `birch`; `estimate_bandwidth`, `cluster_optics_dbscan`, `cluster_optics_xi` | Lloyd/k-means++; paired row/column spectral partitions and checkerboard biclustering; weighted streaming mini-batch centroids; bisecting centroid hierarchy with Lloyd/Elkan splits; density expansion; single/complete/average/Ward merge trees and feature pooling; flat-kernel mode seeking; damped exemplar messages; normalized-Laplacian clustering with k-means or pivoted QR assignment; OPTICS reachability ordering, Xi hierarchy and DBSCAN extraction; BIRCH streaming CF tree and optional Ward clustering |
| decomposition | `pca`, `incremental_pca`, `truncated_svd`, `nmf`, `mini_batch_nmf`, `sparse_pca`, `mini_batch_sparse_pca`, `dictionary_learning`, `mini_batch_dictionary_learning`, `sparse_coder`, `latent_dirichlet_allocation`; `sparse_encode`, `non_negative_factorization`; re-exports `kernel_pca`, `factor_analysis`, `fast_ica` | Full/randomized/covariance PCA with automatic routing; PCA whitening/inversion, variance/Minka dimension selection and probabilistic scores; streaming stacked-SVD IncrementalPCA; exact dense and randomized dense/CSR TruncatedSVD; regularized NMF coordinate descent and beta-divergence multiplicative updates, with a free function supporting fixed-H solves; sparse dictionary factorization and SparsePCA; five sparse coding methods; nonlinear kernel PCA; full/randomized-SVD factor analysis with orthogonal rotations; parallel/deflation ICA |
| cross_decomposition | `pls_regression`, `pls_canonical`, `cca`, `pls_svd` | Paired-matrix latent components: NIPALS regression/canonical deflation, CCA pseudoinverse mode, canonical direct SVD and PLSSVD cross-covariance projection; multiple response columns |
| isotonic | `isotonic_regression`; `isotonic_fit`, `check_increasing` | Weighted pool-adjacent-violators, automatic Spearman direction, bounds, repeated-input aggregation and piecewise linear interpolation |
| kernel_approximation | `rbf_sampler`, `nystroem`, `additive_chi2_sampler`, `skewed_chi2_sampler`, `polynomial_count_sketch` | Random Fourier RBF maps; sampled-landmark inverse-square-root maps for RBF/linear/polynomial/cosine kernels; additive and skewed chi2 maps; seeded polynomial TensorSketch with arbitrary-length FFTs |
| random_projection | `gaussian_random_projection`, `sparse_random_projection`; `johnson_lindenstrauss_min_dim` | Gaussian/three-point random matrices, dense storage, automatic JL dimension, cached or on-demand pseudoinverse |
| kernel_pca | `kernel_pca` | Centered linear/RBF/polynomial/sigmoid/cosine/precomputed Gram matrices; dense Jacobi eigensolver; out-of-sample transform and learned kernel-ridge pre-images |
| manifold | `isomap`, `tsne`; `trustworthiness`; re-exports `spectral_embedding`, `mds`, `locally_linear_embedding` | Isomap radius/fixed-count graphs, dense/CSR precomputed distances and FW/Dijkstra paths; exact and Barnes-Hut t-SNE with sparse affinities, CSR distances, adaptive gains and two-stage momentum; neighborhood rank trustworthiness; spectral embedding; metric/nonmetric SMACOF; dense standard/Hessian/modified LLE and LTSA with barycentric query transforms |
| spectral | `spectral_embedding` | Dense symmetric eigensolver; RBF, symmetrized nearest-neighbor or precomputed affinity; degree normalization; connected-component count |
| density | `kernel_density` | KD/Ball-tree log densities with six kernels; Minkowski-family metrics and feature weights; query tolerances/traversal controls; sample weights and bandwidth rules; NumPy-compatible Gaussian/tophat draws |
| outlier | `local_outlier_factor` | Training outlier scores and novelty detection, automatic/percentile threshold, Minkowski and other supported pairwise metrics |
| semi_supervised | `label_propagation`, `label_spreading`, `self_training_classifier` | Dense RBF or directed k-neighbor graph, hard/soft clamping, transductive labels and inductive probabilities; threshold/k-best self-training with logistic, Gaussian NB or KNN bases |
| tree | `decision_tree_classifier`, `decision_tree_regressor`, `extra_tree_classifier`, `extra_tree_regressor` | Weighted CART and standalone randomized trees; Gini/entropy/log-loss classification, squared-error regression; depth/leaf/split/feature controls, normalized impurity importance, leaf indices |
| ensemble | `hist_gradient_boosting_regressor`, `hist_gradient_boosting_classifier`, `stacking_classifier`, `stacking_regressor`, `bagging_classifier`, `bagging_regressor`, `random_forest_classifier`, `random_forest_regressor`, `extra_trees_classifier`, `extra_trees_regressor`, `gradient_boosting_regressor`, `gradient_boosting_classifier`, `adaboost_classifier`, `adaboost_regressor`, `isolation_forest`, `voting_classifier`, `voting_regressor` | Histogram Newton boosting with missing-value routing and five regression losses; heterogeneous hard/soft voting and averaged regression; bootstrap forests, randomized feature subsets, random-threshold Extra Trees, squared-error residual boosting; binary/multiclass classification boosting with Newton leaf updates; seeded SAMME with configurable weighted CART; AdaBoost.R2 with depth-controlled CART; random isolation trees and path-length anomaly scores |
| validation | `cross_validate`, general `cross_val_score`, `cross_val_predict`, typed `cross_val_predict_class` adapter; `learning_curve`, `validation_curve`, `permutation_test_score` | Dense vector targets, multiple/scalar scorers, fit weights, returned fold models/indices, no-target transformer callbacks, partitioned out-of-fold responses, class encoding and missing-class columns |
| search | `grid_search_cv`, `randomized_search_cv`; typed parameter grids and samplers | Serial dense candidate/fold evaluation, default or explicit folds, single/multiple scorers, callbacks, weighted fits, ranking and refitting; native scalar parameter adapters and nested Pipeline paths |
| model_selection | `kfold`, `stratified_kfold`, `time_series_split`, `group_kfold`, `leave_one_out`, `leave_one_group_out`, `predefined_split`, `shuffle_split`, `stratified_shuffle_split`, `group_shuffle_split`, `repeated_kfold`, `repeated_stratified_kfold`, `leave_p_out`, `leave_p_groups_out`, `ridge_cv`; `train_test_split`, `shuffle_indices`, `cross_val_score` | Index-based splits; seeded random/group/stratified holdouts; repeated shuffled folds; exhaustive sample/group combinations; fraction/count size controls; optional stratification; time gap/window; CV for linear-regression family and logistic regression; ridge CV selects negative mean squared error |
| feature_selection | `select_from_model`, `variance_threshold`, `select_k_best`, `select_percentile`, `select_fpr`, `select_fdr`, `select_fwe`, `generic_univariate_select`, `sequential_feature_selector`, `rfe`, `rfecv`; `f_classif`, `f_regression`, `chi2`, `r_regression`, `mutual_info_classif`, `mutual_info_regression` | Model coefficient/tree-importance thresholds and stable caps; dense scores and masks; greedy forward/backward CV regression selection; coefficient-based recursive elimination and CV for regression/logistic classification; inverse transform fills dropped columns with zero; new statistical selectors include p-values and multiplicity control |
| image / image_graph | `patch_extractor`; `extract_patches_2d`, `reconstruct_from_patches_2d`, `grid_to_graph`, `img_to_graph` | Exhaustive or seeded with-replacement image patches; grayscale and multichannel batches; source overlap averaging; masked real64 CSR grid connectivity and pixel/voxel gradients |
| feature_extraction | `dict_vectorizer`, `feature_hasher` | Named numeric/categorical mappings; string lists; learned or hashed columns; real64 CSR and dense outputs |
| feature_extraction_text | `count_vectorizer`, `tfidf_transformer`, `tfidf_vectorizer`, `hashing_vectorizer` | Dense vocabulary/TF-IDF unigram counts with ASCII tokens of length 2-128; stateless CSR signed/binary hashed word, character and within-word character n-grams with dynamic token storage, ASCII analysis and English/custom stop words; l1/l2/no normalization |
| dummy | `dummy_classifier`, `dummy_regressor` | Prior/most-frequent/stratified/uniform/constant classification; mean/median/quantile/constant regression |
| kernel_ridge | `kernel_ridge` | Positive alpha; linear/RBF/polynomial/sigmoid kernels; no intercept |
| discriminant_analysis | `linear_discriminant_analysis`, `quadratic_discriminant_analysis` | LDA offers LSQR, SVD and generalized-eigen classification/projection with fixed/automatic shrinkage and custom priors; QDA full/thin regularization and custom priors; solver-specific rank contracts are documented below |
| svm | `svc`, `nu_svc`, `linear_svc`, `linear_svr`; re-exports `svr`, `nu_svr`, `one_class_svm` | Primal L2 squared-loss linear classification/regression; OVO SVC/NuSVC; epsilon/nu support-vector regression and one-class detection; native pairwise dual optimization and weighted capacities |
| mixture | `gaussian_mixture`, `bayesian_gaussian_mixture` | EM Gaussian mixtures and variational Bayesian mixtures with finite Dirichlet or truncated stick-breaking weights; full/tied/diag/spherical covariances; responsibilities and likelihood scores; GaussianMixture BIC/AIC |
| neural_network | `mlp_classifier`, `mlp_regressor`, `bernoulli_rbm` | Arbitrary dense hidden layers, relu/tanh/logistic/identity, seeded minibatch Adam/SGD or full-data L-BFGS, momentum/Nesterov and SGD schedules, binary/multilabel logistic or multiclass softmax classification, scalar/joint squared-error regression, L2 regularization |
| gaussian_process | `gaussian_process_regressor`, `gaussian_process_classifier` | Configurable covariance expressions or legacy RBF controls; anisotropy, Matern/RQ/periodic/dot/constant/white kernels and sums/products/powers; scalar/vector-alpha Cholesky regression, independent multiple targets, likelihood gradients and opt-in bounded kernel fitting; Laplace binary/OVR classification with optional independent kernel fitting, binary likelihood gradients and warm starts |
| covariance | `empirical_covariance`, `shrunk_covariance`, `ledoit_wolf`, `oas`, `graphical_lasso`, `graphical_lasso_cv`, `min_cov_det`, `elliptic_envelope`; `fast_mcd` | Population covariance, pseudoinverse precision, squared Mahalanobis distance, likelihood, Frobenius/spectral covariance error; L1-penalized sparse precision via block coordinate descent and refined cross-validation; FastMCD robust covariance and elliptic outlier detection |
| calibration | `calibrated_classifier_cv`; `calibration_curve` | Sigmoid/isotonic CV ensembles, out-of-fold calibration and prefit calibration with logistic, Gaussian NB or KNN bases; uniform/quantile reliability bins |
| datasets | `make_classification`, `make_hastie_10_2`, `make_blobs`, `make_regression`, `make_moons`, `make_circles`, `make_friedman1/2/3`, `make_sparse_uncorrelated`, `make_s_curve`, `make_swiss_roll`, `make_gaussian_quantiles` | Native synthetic generators; new regression/manifold/quantile generators match seeded legacy NumPy draws; no bundled real-world datasets |

## Metrics and kernels

Discrete clustering mutual information: `mutual_info_score` returns nats from
integer label pairs, with source contingency and zero-entropy contracts.
Normalized/adjusted MI, expected MI, Fowlkes-Mallows, entropy, dense contingency
tables, int64 ordered pair confusion and weighted homogeneity/completeness/
V-measure are also available. The three individual H/C/V scorers now share the
source-compatible implementation, including empty inputs and V-measure beta.

Regression: mean/root-mean squared error, mean/median/max absolute error,
mean absolute percentage error, mean squared log error, R2, explained variance,
Poisson and Gamma deviances.

Classification: accuracy, confusion matrix, precision, recall, F1, Jaccard,
Hamming loss, balanced accuracy, Matthews correlation, Cohen kappa, log loss,
binary ROC AUC and average precision with sample weights; standardized partial ROC AUC;
ROC, precision-recall and DET threshold curves; generic trapezoidal AUC; binary
Brier probability loss. Calibration curves use uniform or quantile bins. Precision/recall/F1/Jaccard support binary,
micro, macro and support-weighted averaging. Zero denominators yield zero.

Clustering: silhouette samples/score, Calinski-Harabasz, Davies-Bouldin,
Rand/adjusted Rand, homogeneity, completeness and V measure.

Pairwise: Euclidean/squared Euclidean, Manhattan, Chebyshev, Minkowski and cosine
distances; cosine similarity; linear, RBF, polynomial and sigmoid kernels.

## Intentional differences and limits

- MeanShift uses all training points as seeds, or explicit seeds passed to
  `fit(x,seeds=...)`. There is no bin seeding or subsampling/parallel bandwidth
  estimation. `bandwidth=-1` estimates the full-data 0.3-quantile bandwidth;
  automatically estimated zero bandwidth is allowed for degenerate/small data.
  `cluster_all=false` gives orphan training labels -1; `predict` still returns
  the nearest center for every query, matching sklearn. Exact seed endpoints
  are deduplicated with the last seed's intensity before mode suppression.
- AffinityPropagation supports Euclidean negative squared distance or a square
  precomputed similarity matrix and a scalar preference. `preference=huge(1._dp)`
  selects the median similarity. It uses tiny Fortran-RNG perturbations rather
  than NumPy's stream, so degenerate exemplars/iterations can differ. Inspect
  `converged_`, `n_iter_` and `cluster_centers_indices_`; without exemplars the
  labels/predictions are -1. Precomputed fitting has no prediction method.
- SpectralEmbedding and SpectralClustering use dense normalized-Laplacian
  eigendecomposition, not ARPACK/LOBPCG/AMG. Affinities are RBF, symmetric
  k-neighbor connectivity including self, or a nonnegative symmetric square
  precomputed matrix. Nonsymmetric/negative affinities are rejected. Self loops
  are excluded before degree summation, including large diagonals. Disconnected
  graphs are allowed and counted in `n_connected_components_` without a warning;
  selected eigenvectors in repeated eigenspaces can differ from Python.
  Embedding drops the first eigenvector; clustering retains it and embeds into
  exactly `n_clusters` dimensions. Cluster assignment is k-means or `cluster_qr`,
  not discretization; there are no callable kernels, precomputed-distance graph
  mode, sparse storage, radius affinity, or out-of-sample spectral transform.
- KernelDensity supports all six source kernels and native KD/Ball trees,
  exact or tolerance-bounded queries, Minkowski-family metrics and weighted
  Minkowski distances. Kernel volume factors follow the source Euclidean formulas;
  non-Euclidean distances generally do not yield normalized densities. Gaussian/tophat sampling uses
  NumPy-compatible seeded draws and the source incomplete-gamma ball mapping.
  Source zero-weight and cosine-normalization nonfinite outcomes are retained.
  Other BallTree metrics, sparse features and parallel execution are absent.
  Detailed backend behavior and numerical comparison limits appear below.

- HuberRegressor uses bounded full-memory BFGS for coefficients, intercept and
  scale, with the sklearn Huber objective and L2 penalty. It has no warm start,
  sample weighting or sparse inputs. Its projected-gradient stopping rule and
  iterations can differ from sklearn's L-BFGS-B relative-loss stopping; inspect
  `converged_` and `n_iter_`. Full BFGS uses quadratic storage in feature count.
- RANSAC supports a built-in ordinary least-squares estimator only. It accepts
  an integer `min_samples` (zero means `n_features+1`), absolute/squared residual
  loss, a scalar threshold (negative-one means target MAD), and probability,
  inlier-count and score stopping controls. There are no estimator/callback
  hooks, fractional sample counts, sample weights or maximum-skip control.
  Seeded trials use the Fortran RNG; consensus sets can differ from NumPy's.
- Poisson, Gamma and Tweedie use damped Newton updates with a symmetric-eigen
  pseudoinverse Hessian, supporting singular unpenalized designs. Tweedie
  powers are limited to zero (identity link) or `[1,2]` (log link). Other powers,
  log-link normal regression, identity-link positive-power GLMs, alternative
  solvers, sample weights, sparse arrays and warm starts are not supported.
  Positive-power targets must be nonnegative, Gamma targets strictly positive,
  and log-link fitting requires positive target mean. Extremely large raw log
  predictions are rejected during prediction or backtracked during fitting.
  `score` is D2, not R2 except at power zero; constant scoring targets are
  diagnosed instead of returning undefined NaN/infinity. The fitted objective
  is mean half-deviance plus `alpha*sum(coef**2)/2`, so power-zero regularization
  matches ridge with `ridge%alpha=n_samples*glm%alpha`.

- BayesianRidge and ARD support one real target, positive hyperpriors and
  predictive standard deviations, without marginal-likelihood score history,
  sample weights or warm starts. ARD uses a Cholesky inverse of the active
  feature precision even for wide input; sklearn switches to a Woodbury/SVD
  inverse there, so numerical convergence paths can differ. BayesianRidge
  follows sklearn's thin-SVD covariance convention for wide matrices. When
  such a matrix is rank deficient (including centered wide data), the retained
  null-space basis is not unique and covariance entries/predicted uncertainties
  outside the data row space can differ between SVD implementations.
- OMP supports a sparsity count or residual tolerance, with a direct SVD solve
  on the active dictionary. There is no precomputed Gram input, multi-target
  output or coefficient path. A separate OMP CV estimator is documented below.
- FactorAnalysis supports full and randomized SVD and varimax/quartimax
  orthogonal rotations. Native full SVD remains the default. Requested
  components cannot exceed `min(n_samples,n_features)`. Exact zero singular
  values are floored only in the training likelihood to avoid log-zero traps;
  singular-model likelihood history can therefore differ from Python's infinities.
- Random projections use Gaussian or independent three-point entries with
  the sklearn variance scaling, but no sparse matrix storage or NumPy seed
  equivalence. Automatic JL dimensions exceeding the input width are rejected;
  explicitly requested dimensions may exceed it without a warning. These are
  standard sklearn counterparts, not project extensions.

- Inputs are finite dense `real(dp)` arrays, except NaNs specifically accepted by
  `simple_imputer`. Strings are accepted only by text vectorizers. Empty feature
  matrices are generally rejected. No sample weights, sparse storage, multioutput
  targets, partial fitting, GPU, joblib, model persistence, or Python pipelines.
- Invalid input terminates with `error stop`; recoverable status objects are not
  yet implemented. Numerical failures are diagnosed rather than silently hidden.
- Fortran sample/feature/neighbor/leaf indices are **one-based**. Encoded labels
  and cluster IDs are **zero-based**. DBSCAN noise is `-1`. Integer target labels
  may be arbitrary, negative, or noncontiguous; class arrays are sorted.
- Seeded runs are repeatable for a given Fortran compiler/runtime. They do not
  reproduce NumPy random streams. Seeding currently uses the process-global
  Fortran RNG. Concurrent stochastic fits therefore need external coordination.
- `pca%n_components=0` means all available components, unlike Python's integer
  zero. Both PCA/SVD store components as `(n_components,n_features)`.
- The pure Fortran SVD/eigensolvers are designed for small/medium dense problems;
  they do not provide LAPACK-scale performance. Huge magnitudes can overflow
  intermediate squares; scale data before fitting. Pairwise algorithms and
  kernel estimators use quadratic memory. Trees use a straightforward exhaustive
  split search and are slower than sklearn's incremental Cython implementation.
- Tree split ties follow feature/threshold traversal, which differs from
  sklearn's random feature traversal; predictions and feature attribution can
  differ on ties. Leaf indices are native node indices, not sklearn node IDs.
  Extra Trees fits use no bootstrap. No forest OOB scores, pruning, monotonic
  constraints, class weights, warm starts, or boosting classification.
- K-means++ uses one draw per center, not sklearn's greedy local trials. Empty
  cluster relocation may differ. DBSCAN/agglomerative tie ordering may differ.
- LDA has no SVD solver or discriminant transform. QDA requires positive-definite
  (possibly regularized) class covariance. Gaussian NB retains zero-variance
  statistics and source-compatible undefined posteriors, as described below.
- SVC exposes one decision column per class pair, positive toward the pair's
  second class. This differs from sklearn's multiclass decision conventions.
  SVR, NuSVC and one-class SVM are separate native types. Platt probability
  scaling and LIBSVM model-file compatibility are not provided.
- MLP supports seeded minibatch Adam/SGD and full-data L-BFGS. Validation-set
  early stopping, partial_fit and sparse input remain outside its interface. Floating-point rounding
  can amplify in sensitive trajectories; see the stochastic MLP increment below.
- Gaussian-process kernel hyperparameters are fixed; no optimizer or kernel
  algebra interface. Only one real target and isotropic RBF are supported.
- Text uses ASCII unigrams only, without Unicode word rules, custom tokenizers,
  n-grams, stop words, hashing, fixed vocabulary, or sparse output.
- Imputation rejects all-missing columns unless strategy is constant; it does
  not silently drop columns. One-hot encoding has no category dropping/grouping.
- Stratified folds currently require at least `n_splits` samples in each class.
  Split randomness and stratified rounding need not match sklearn. `ridge_cv`
  uses ordinary K-fold CV, not generalized leave-one-out CV.
- Radius classification raises an error for an empty neighborhood unless
  `allow_outliers` is enabled with a numeric `outlier_label`. If that label is
  not a fitted class, its probability row is zero. Radius regression returns
  NaN for an empty neighborhood, matching sklearn (without its warning).
- Nearest centroid supports Euclidean means or Manhattan medians, scalar
  shrinkage and uniform/empirical/custom priors. Euclidean decisions and
  probabilities follow sklearn 1.6.1 standardized discriminants. Zero-scale
  diagnostics retain source NaN/Inf; undefined standardized queries are rejected.
  See the prior-aware NearestCentroid section below.
- LOF uses `contamination=-1` for sklearn's `'auto'`, otherwise a fraction in
  `(0,0.5]`. `fit_predict` requires `novelty=false`; query scoring/prediction
  requires `novelty=true`. Training scores exclude each point itself; novelty
  queries include matching training points. Neighbor-count clamping is reported
  in `n_neighbors_`; warnings are not emitted. Duplicate/tied-neighbor ordering
  can differ from sklearn's search backend.
- Kernel PCA has no callable kernels or sparse/randomized/
  ARPACK solver; optional fitted inverse transforms are supported. `n_components=0` means all nonzero components; other counts are
  clipped to sample count. Significant selected negative eigenvalues are
  rejected. Repeated eigenvalues can yield different valid bases.
- Isomap supports fixed-count or radius neighborhoods, dense features and
  dense/CSR precomputed distances, and FW/Dijkstra geodesics. Full distances
  permit disconnected-graph completion; disconnected CSR graphs are rejected.
  Distances and eigendecomposition remain dense. Native stable neighbor tie
  ordering can differ from sklearn's search backends. See the detailed section
  below for parameter sentinels, fitted state and numerical scope.
- Semi-supervised labels reserve `-1` for unlabeled samples and require at least
  one labeled sample. RBF and KNN kernels are supported, not callable kernels.
  Zero-evidence inductive queries are rejected instead of returning NaN
  probabilities. Unreachable training rows remain zero and transduce to the
  first class, as in sklearn. Inspect `converged_` and `n_iter_`; reaching the
  iteration cap does not emit a Python-style warning.

## Remaining major scikit-learn areas

Not translated: sparse data paths, general Python estimator base/clone/parameter
reflection, sparse/parallel/recoverable-error search paths, sparse/cached/metadata-routed pipeline paths, broader SVM/GLM/SGD options,
broader histogram boosting, AdaBoost and bagging APIs,
outlier detection beyond LOF/OneClassSVM/SGDOneClassSVM/IsolationForest/EllipticEnvelope,
calibration with further base estimators and general metadata routing,
additional multiclass base estimators and sparse reductions,
additional inspection utilities, further feature-selection wrappers,
real dataset loaders and the full plotting/display API.

SciFort is available at `../scifort` but not linked by this initial package.
An optional SciFort/LAPACK backend and recoverable errors would be useful next
steps before scaling this package to larger production workloads.

## Independent component analysis

`fast_ica` (`sklearn_decomposition`, also the `sklearn` facade) translates
`sklearn.decomposition.FastICA`: parallel symmetric and sequential deflation
solvers, logcosh/exp/cube contrasts, dense SVD whitening, unit-variance and
arbitrary-variance sources, and already-whitened input (`whiten='none'`).
Methods: `fit`, `fit_transform`, `transform`, `inverse_transform`.
Fitted arrays: `components_`, `mixing_`, `mean_`, and `whitening_` (only with
whitening); convergence is exposed through `n_iter_` and `converged_`.

`n_components=0` selects min(samples, features); oversized requests are clipped.
Unwhitened fitting ignores the requested dimension, uses every feature, does
not center, and requires samples >= features. Whitening requires full rank
in the retained subspace; singular initial/fixed-point matrices fail with a
diagnostic rather than propagating invalid arithmetic. No eigh whitening
solver, callable contrast or Python warning mechanism is implemented.
The `alpha` parameter (1 to 2) controls logcosh. Optional `w_init` is a square
finite initial matrix. Random starts use the package's Fortran RNG.
Numerical scaling and ill-conditioned fixed points can still limit convergence.

## Isotonic regression and kernel approximations

- `isotonic_regression` takes rank-one `x` and `y` vectors. Its `fit`,
  `fit_transform`, `predict`, `transform`, and unweighted R2 `score` methods
  support nonnegative sample weights with positive total mass, increasing or
  decreasing fits, and optional automatic direction (`auto_increasing`).
  Automatic direction uses average ranks for ties and the Spearman sign; an
  undefined coefficient selects decreasing, matching sklearn, without warnings.
  Zero-weight samples are removed after direction determination. Sorted inputs
  less than 1e-15 apart are grouped, as in sklearn's float64 implementation.
  Bounds default to finite `+/-huge`; `y_min <= y_max` is required.
  `out_of_bounds` accepts `nan` (default), `clip`, or `raise`. A one-threshold
  fit predicts a constant outside its domain, matching sklearn 1.6.1.
  `x_thresholds_` and `y_thresholds_` omit redundant plateau interiors.
  The standalone Python `isotonic_regression` function is named `isotonic_fit`
  in Fortran to avoid colliding with the estimator type; it requires strictly
  positive weights. `check_increasing` is also available. No rank-two input
  overload, weighted score, warning machinery or inverse transform is supplied.
- `rbf_sampler` supplies `fit`, `transform`, and `fit_transform` with numeric
  nonnegative `gamma` (default 1) or `gamma_scale=.true.`. Scaling uses variance
  across all training entries and falls back to gamma=1 for constant data.
  Fitted `random_weights_` are (features, components), `random_offset_` contains
  uniform phases, and `gamma_` records the effective kernel parameter.
- `nystroem` supplies the same three methods, with kernels `rbf` (default),
  `linear`, `poly`/`polynomial`, or `cosine`. Numeric `gamma=-1` requests 1/features.
  Nonnegative polynomial coefficients and integer degrees ensure a PSD kernel.
  Landmarks are sampled without replacement; requests larger than the sample
  count are capped. `components_`, one-based `component_indices_`,
  `normalization_`, and `gamma_` expose fitted state. A native symmetric
  eigensolver constructs the PSD inverse square root with a 1e-12 floor.
  With rank-deficient bases, compare reconstructed kernels rather than the
  null-space entries of normalization against an SVD implementation.
  Callable, precomputed, indefinite, sparse and other kernels are unsupported.

Both randomized maps use the package's Fortran RNG and `random_state`; draws
differ from NumPy. No sparse input/output path or threaded backend is included.

## Cross-decomposition

`sklearn_cross_decomposition` exports `pls_regression`, `pls_canonical`, `cca`,
and `pls_svd`, also available from `sklearn`. All take rank-two `x(samples,
features)` and `y(samples, targets)` matrices. A single response is a matrix
with one column. Defaults are two components and `scale=.true.`; both matrices
are centered and scaled with sample standard deviations (ddof=1), with zero
standard deviations replaced by one.

All four support `fit`, `fit_transform` (returns X scores), `transform`, and
`transform_y`. The latter provides the response scores separately from the
Fortran function result for X. `x_mean_`, `y_mean_`, `x_std_`, `y_std_`,
`x_weights_`, `y_weights_`, `x_rotations_` and `y_rotations_` expose fitted state.
PLSSVD uses the singular vectors of the full cross-covariance, without
component-wise deflation. Selected singular values must have numerical full
rank; arbitrary null-space singular vectors are not returned.

PLSRegression, PLSCanonical and CCA also support `predict`, uniform-average
response R2 `score`, `inverse_transform`, and `inverse_transform_y`. They expose
`x_loadings_`, `y_loadings_`, `x_scores_`, `y_scores_`, `coef_` (targets, features)
and `intercept_`. Predictions follow sklearn's centered convention:
`(x - x_mean_) @ transpose(coef_) + intercept_`; `intercept_` is the response
mean. Inverse transforms reconstruct the retained component subspace. Training
deflation scores and rotated response transforms can differ for PLSRegression.

- PLSRegression uses NIPALS mode A with regression response deflation.
  Its component bound is min(samples, features), independently of targets.
- PLSCanonical uses NIPALS (default) or `algorithm='svd'` with canonical
  deflation of both matrices.
- CCA uses NIPALS mode B with native SVD pseudoinverses and sklearn's historical
  relative rank cutoff of 1e6*epsilon.
- The latter two and PLSSVD bound components by min(samples, features, targets).

`max_iter` and `tol` control NIPALS; `n_iter_` stores counts for components
actually extracted (empty for direct SVD), `converged_` reports iteration
convergence, and `n_components_fitted_` reports the number extracted. A constant
response residual stops NIPALS early, leaving unused component arrays zero;
a wholly constant response yields mean-only predictions. Degenerate zero
predictor/response scores fail with diagnostics. Rank-deficient CCA can depend
on pseudoinverse tolerance; no ridge regularization is added.

Sparse inputs, rank-one overloads, in-place mutation/copy controls, Python tuple
returns, sample weights, cross-validation wrappers and Python warnings are
not implemented.

## Multidimensional scaling and locally linear embedding

`mds` and `locally_linear_embedding` are re-exported by `sklearn_manifold` and
the `sklearn` facade. Their implementation modules are `sklearn_mds` and
`sklearn_locally_linear`.

MDS implements dense metric and nonmetric SMACOF with `fit(x,init)` and
`fit_transform(x,init)`; `init` is optional. `dissimilarity` is `euclidean`
(default) or `precomputed`. Precomputed matrices must be finite, square,
nonnegative, symmetric and zero on the diagonal. There must be two or more
observations. Nonmetric MDS treats zero upper-triangle dissimilarities as
missing and requires at least one positive observation. `n_components=2`,
`n_init=4`, `max_iter=300` and `eps=1e-3` follow the sklearn 1.6.1 defaults.
`normalized_stress` is a string: `auto` (default, normalized only for nonmetric),
`true`, or `false`. Normalized metric stress is unsupported in this baseline.

Explicit initial coordinates determine the output dimension and reduce the
run count to one. Otherwise, seeded uniform random starts are tried and the
lowest-stress result is retained. Fitted fields are `embedding_`,
`dissimilarity_matrix_`, `stress_`, `n_iter_` and `converged_` (for the chosen
run). The implementation preserves sklearn 1.6.1's stress/convergence convention:
stress is evaluated before the final Guttman coordinate update, so it can
differ from stress recomputed from the returned embedding. Nonmetric updates
also preserve that version's upper-triangle isotonic update and full-matrix
normalization. Newer sklearn versions can use different SMACOF semantics.
All-zero metric dissimilarities return a zero embedding and zero stress; a
collapsed initialization with positive dissimilarities fails diagnostically.
There is no new-observation transform, inverse transform or threaded backend.

LLE supports `method='standard'` (default), `hessian`, `modified` and `ltsa`, with a native
dense eigensolver. `fit`, `fit_transform` and `transform` use Euclidean nearest
neighbors, local regularized barycentric weights, and the lowest nonconstant
alignment modes. The known constant mode is removed analytically with an
orthonormal Helmert basis, avoiding mixing with near-zero manifold modes.
`n_neighbors=5`, `n_components=2`, and `reg=1e-3` are the
defaults. Require 0 < neighbors < samples and 0 < components <= input features,
with components < samples; LTSA also requires neighbors > components.
Regularization is nonnegative; a singular local solve (possible with zero
regularization) fails with a diagnostic.

Fitted fields include `x_fit_`, `embedding_`, `reconstruction_error_` (sum of
selected eigenvalues, floored at zero for roundoff), and one-based `neighbors_`.
`weights_` stores (samples, neighbors) barycentric fitting weights for standard
LLE and is zero for the three nonstandard methods. LTSA and Hessian use local
projectors; modified LLE uses multiple weights and its fixed source fit
regularizer. For nonstandard methods, the public reg control applies only
to query transforms. Queries use the fitted neighbors
count and barycentric projection of training coordinates; a query equal to a
training point can have a different neighbor set from the training fit.
Neighbor ties, disconnected graphs, unreferenced LTSA observations, or repeated
alignment eigenvalues can make truncated embeddings nonunique across solvers.
Hessian/modified details and tolerances are described below. ARPACK, sparse
graph paths and inverse transforms remain unimplemented.

## Support-vector regression and one-class detection

`svr`, `nu_svr`, and `one_class_svm` are available through `sklearn_svm` and
the facade; their implementation module is `sklearn_kernel_svm`. They use native
dense dual SMO with a most-violating first coordinate and gain-based second
coordinate, independently of the existing SVC
solver. All support `fit` with optional nonnegative `sample_weight`, requiring
at least one positive weight. SVR/NuSVR take one finite continuous target vector
and provide `predict` and R2 `score`. OneClassSVM takes only X and optional
weights, with `predict`/`fit_predict` labels +1 for positive decision values and
-1 otherwise; `score_samples` returns the unshifted kernel score.

The added models support `linear`, `rbf` (default), `poly`/`polynomial`, and
`precomputed` kernels. Precomputed training matrices must be square, symmetric
and PSD within the numerical tolerance; queries have one column per original
training sample. Polynomial degree and coef0 must be nonnegative. Callable,
sigmoid and other indefinite kernels are unsupported. These constraints keep
the dual problem convex. Inputs and Gram matrices are dense real64.

`gamma=-1` (default) selects the string `gamma_rule`: `scale` (default) is
1/(features * global training-entry variance), with a constant-data fallback of
1; `auto` is 1/features. Numeric gamma >= 0 overrides the rule. Gamma scaling
uses all input rows, including zero-weight rows, as in sklearn. `gamma_` stores
the effective value (zero for precomputed kernels). The default C is 1 for
both regression types, epsilon is 0.1 for SVR, and nu is 0.5 for NuSVR and
OneClassSVM. NuSVR requires 0 < nu <= 1 and exposes its learned `epsilon_`.
OneClassSVM requires 0 < nu < 1: the fully saturated endpoint with a nonunique
threshold is not supported.

Fitted state includes one-based original-row `support_`, `support_vectors_`,
rank-one `dual_coef_`, scalar `intercept_`, and integer `n_support_`. Linear
kernels also allocate the feature vector `coef_`; it is unallocated for other
kernels. Precomputed kernels store an empty (0,0) support-vector matrix.
Zero-weight observations keep their original indexing in this package;
sklearn 1.6.1's dense wrapper reports support indices after dropping such rows.
The reference harness maps those indices back before comparison.

OneClassSVM's `offset_=-intercept_` and
`decision_function(x)=score_samples(x)-offset_`. Its dual coefficients sum to
nu times total sample weight, preserving LIBSVM's score normalization.
SVR coefficients are differences of positive/negative dual variables, with
zero total sum and per-row bounds +/- C*weight. NuSVR also constrains each
signed dual group's mass to nu*sum(C*weight)/2. Large epsilon can leave an
SVR with no support vectors; this produces a constant prediction.

`tol=1e-3` and `max_iter=100000` control the optimizer. `n_iter_` counts native
pair updates; it need not equal LIBSVM's iterations. `kkt_violation_` reports
the final maximum feasible pair gradient gap. `converged_` and `fit_status_`
(0 converged, 1 otherwise) report whether it meets tol; capped fits return an
approximate model. Gradients are refreshed before reporting convergence. OneClassSVM starts with
alpha=nu*weight, a feasible interior initialization that preserves symmetry.
No shrinking, kernel cache, sparse path, unlimited-iteration sentinel, Python
warning or probability-calibration APIs are supplied. Dual representations can
be nonunique for linear/polynomial/rank-deficient kernels even when decision
functions agree. A fully bounded one-class dual can also have a nonunique
threshold; this implementation chooses the midpoint of the admissible interval.
Parameters should be chosen for well-scaled finite arithmetic.

## PowerTransformer

`power_transformer` (implemented in `sklearn_power_transform` and re-exported
by `sklearn_preprocessing`) adds Yeo-Johnson and Box-Cox maximum-likelihood
power selection for each feature. Standardization defaults to population mean
zero and variance one. `fit`, `transform`, `fit_transform` and
`inverse_transform` operate on dense rank-two arrays; `lambdas_`, `mean_` and
`scale_` expose fitted parameters. NaNs are excluded during fitting and
preserved by transformations; infinity and all-missing features are rejected.
Constant Yeo-Johnson columns use lambda=1, including partially missing constant
columns. Box-Cox requires strictly positive values and rejects constant columns.

The native optimizer brackets the likelihood minimum from (-2,2), then uses
relative-tolerance golden-section search, rather than SciPy Brent iteration.
Log-magnitude scaling avoids overflow in trial likelihood powers. This changes
iteration paths and last-digit fitted lambdas. Arithmetic for fitted moments,
actual transforms and inverses still requires suitably scaled inputs. Powers
that overflow and inverse values outside the real power domain produce explicit
diagnostics. SciPy's Box-Cox maximum-output constraint, confidence intervals,
sparse arrays, in-place copy controls and Python warnings are not implemented.
The fitted method and standardization setting are retained until the next fit.
This batch brings the package to 94 core types and 39 source modules.

## KBinsDiscretizer

`kbins_discretizer` lives in `sklearn_discretization` and is re-exported through
`sklearn_preprocessing`. Its strategies are `uniform`, `quantile` (default) and
`kmeans`; encodings are `ordinal` and `onehot-dense` (default). All arrays are
dense real64. `n_bins=5` sets a scalar requested count; optional `n_bins(:)` in
`fit`/`fit_transform` overrides it per feature. Each requested count must be at
least two. Fitted `n_bins_` contains the counts after removing narrow bins;
`bin_edges_(j)%values` stores each feature's edge vector. `n_features_out_`
records transformed width. Ordinal codes are real64, zero-based integers.

Quantiles use linear interpolation without weights and lower inverse-CDF
percentiles with weights, following sklearn 1.6.1. Quantile and k-means fits
accept optional nonnegative finite `sample_weight(:)` with positive total;
uniform fits reject weights. K-means uses weighted one-dimensional Lloyd
iterations, deterministic uniform-midpoint starts, at most 300 updates and
variance-scaled tolerance 1e-4. `n_iter_(:)` is zero outside k-means;
`converged_` reports whether all fitted k-means columns stopped within that cap.
Empty clusters are relocated to farthest observations; tied relocations can
differ from NumPy's partition ordering. Full data are always used (the Python
reference sets `subsample=None`). No random sampling, sparse encoding, dtype
controls, Python warnings or feature-name APIs are provided.

Transform ignores the outer endpoints, clips out-of-range observations into
extreme bins and assigns values exactly at an interior edge to its upper bin.
For quantile/k-means, original adjacent widths <=1e-8 are removed, as in sklearn;
if no bins survive, fitting fails explicitly. Constant columns become one bin
with edges (-infinity,+infinity). Its code is zero and its inverse is NaN,
matching sklearn 1.6.1's midpoint convention. Other inverses return bin
midpoints, so they approximate input values. Inverse ordinal codes must be
integers within range; dense inputs must contain exactly one active 1 per
feature and otherwise zeros. These inverse checks are stricter than Python's
integer casting and argmax behavior. Inputs reject NaN/infinity. Fitted encoding
is retained until refit. Choose inputs/weights for finite moment and distance
arithmetic. The package now contains 95 core types in 40 source modules.

## GraphicalLasso

`graphical_lasso` extends the covariance family with an L1-penalized sparse
precision estimator. It inherits `assume_centered`, `location_`, `covariance_`,
`precision_`, `mahalanobis`, `score` and `error_norm`. Default `alpha=0.01`
penalizes off-diagonal precision entries only. `covariance='empirical'`
(default) computes a population covariance from training samples;
`covariance='precomputed'` accepts a square, symmetric covariance with scaled
entrywise symmetry tolerance 1e-10 and relative PSD tolerance 1e-10. It uses zero
location. Training arrays need at least two rows and two features, with
positive diagonal variances; all values must be finite. A positive alpha can
regularize a singular sample covariance. Alpha zero requires a positive-definite
covariance and directly inverts it with Cholesky (zero outer iterations).

The solver implements sklearn 1.6.1's `mode='cd'` cyclic block updates and
Gram coordinate descent, including its inner dual-gap stopping rule. It uses
native Cholesky solves instead of the starting `pinvh`. Controls are
`max_iter=100`, `tol=1e-4`, `enet_tol=1e-4`, and `eps=epsilon(1._dp)`.
Both inner and outer loops use max_iter, following sklearn; outer convergence
requires abs(dual_gap_) < tol. Capped fits return their latest approximation
with `converged_=.false.` and `n_iter_` at the cap. Tolerances/eps must be
positive and max_iter must be positive; Python's zero-iteration edge case is
unsupported. Only mode `cd` is accepted; no LARS, GraphicalLassoCV, public
functional solver, covariance warm start, sparse matrix or warning API is added.

`costs_(iteration,1:2)` stores the shifted objective and dual gap after every
outer sweep; `objective_` and `dual_gap_` hold final values. For alpha zero it
contains one diagnostic row despite n_iter_=0. The objective includes sklearn's
Gaussian additive constant and should not be confused with unshifted negative
log likelihood. Ill-conditioned fits with a nonpositive Schur complement or a
non-SPD precision terminate with a diagnostic; the native solver checks SPD at
every cost evaluation, including the first sweep. Use appropriately scaled
inputs and check convergence before interpreting sparse edges. This addition
brings the package to 96 core types in 40 source modules.

## TheilSenRegressor

`theil_sen_regressor` adds subset-based robust regression to `sklearn_linear_model`.
It inherits `fit_intercept`, `coef_`, `intercept_`, `predict` and R2 `score`.
Each row subset is fitted by minimum-norm dense SVD with the intercept column
first; coefficients are combined using sklearn 1.6.1's modified Weiszfeld spatial
median. With a single fitted coefficient (one feature and no intercept), an
ordinary scalar median is used. These are dense real64, single-target fits;
there are no sample weights or joblib/parallel controls.

`n_subsamples=0` means automatic: min(features + intercept, samples). An explicit
positive count must not exceed sample count. When samples >= features it must
cover the full design dimension, including an intercept if fitted; when samples
< features it must equal sample count, following sklearn's parameter rules.
`n_subsamples_` records the resolved value. `max_subpopulation=10000` caps the
number of least-squares fits. All subsets are enumerated lexicographically when
the combination count fits that cap (`exhaustive_=.true.`); otherwise each fit
uses a random subset without replacement within that subset. Subsets may repeat
across fits, as in sklearn. A bounded binomial calculation avoids integer
overflow when deciding between the two paths. `subsets_(population,row_in_subset)`
retains one-based original row indices; `n_subpopulation_` stores population size.

`random_state=0` seeds the native Fortran stream for sampling. Results are
repeatable with this compiler but selected subsets differ from NumPy. Exhaustive
fits do not draw random numbers. `breakdown_` is sklearn's approximate breakdown
formula, not a count of detected outliers. Default `tol=1e-3` and `max_iter=300`
control spatial-median updates; `converged_` reports the stopping test. `n_iter_`
matches sklearn 1.6.1's zero-based last Weiszfeld iteration, so first-step
convergence reports zero and a capped fit reports max_iter-1. The scalar-median
shortcut reports one. A capped fit returns its latest coefficients. Tolerance
and iteration/population limits must be positive. The native zero-quotient
Weiszfeld convention follows the referenced sklearn version.

Subset pseudoinverses use the common SVD threshold
machine_epsilon * max(subset_rows, design_columns) * largest_singular_value.
This can differ from LAPACK GELSS on nearly singular problems; minimum-norm
fits with exact null directions are supported. Input scaling affects the spatial
median's coefficient-space geometry, as in sklearn. Use finite inputs suitable
for dense arithmetic. Inherited `rank_` is left zero, as for the other robust
regressors. This addition brings the package to 97 core types in 40 modules.

## Passive-aggressive linear models

`passive_aggressive_classifier` and `passive_aggressive_regressor` provide
PA-I and PA-II updates in `sklearn_linear_model`. Classifier `loss='hinge'`
(default) chooses PA-I; `squared_hinge` chooses PA-II. Regression uses
`epsilon_insensitive` (default) or `squared_epsilon_insensitive`, with default
epsilon=0.1. As in sklearn 1.6.1's wrapper, the squared names select the PA-II
step denominator; the update numerator and stopping loss still use the linear
hinge/epsilon-insensitive loss. Only off-margin violations trigger updates.
The feature squared norm excludes the intercept. PA-I caps the step at C;
PA-II uses norm + 0.5/C without a cap. Default C is 1.

Dense real64 features and integer class labels are supported. Binary fits use
one plane with the larger sorted class positive; `decision_function` always
returns rank two (n_samples,1) in this case. A zero margin predicts the lower
class. Multiclass fits use independent one-versus-rest planes and first-class
argmax tie handling. `class_weight_balanced=.true.` uses sklearn's binary
balanced positive/negative weights or multiclass balanced positive weights with
unit negative weights. Explicit class-weight dictionaries are unsupported.
The regressor has rank-one coefficients and inherits linear prediction and R2
score; the classifier supplies decision_function, predict and accuracy score.

Defaults are `max_iter=1000`, `tol=1e-3`, `n_iter_no_change=5`, `shuffle=.true.`,
`random_state=0`, `fit_intercept=.true.` and integer `average=0`. Averaging is
disabled at zero; a positive value starts averaging post-update coefficients
at that sample counter (one enables it from the first update). If its threshold
is not reached the standard coefficients are used. The native average follows
sklearn's skipped-zero-row counter and multiclass selection conventions.

Stopping compares unweighted online loss measured before each update. An epoch
without an improvement of tol per training row increments the patience counter;
the comparison is strictly greater, following sklearn. `tol=-1` disables this
stopping test (Python's `tol=None`); other negative values are rejected. Zero
tolerance is allowed. `converged_` reports whether patience stopped every fit;
`n_iter_` counts epochs, taking the maximum over multiclass heads.
`n_iter_per_class_` gives individual classifier counts. `t_=1+n_iter_*n_samples`
matches sklearn's public counter. `loss_` stores last-epoch mean pre-update loss
(a vector per classifier plane), not the loss evaluated at final coefficients.

PA-I skips zero-feature rows before intercept updates, averaging and its internal
sample counter, though their loss and public epoch counts are still included.
PA-II can update an intercept for such rows. These version-specific details are
preserved. Shuffling uses native Fortran permutations and is reproducible with
this compiler; it does not match NumPy's stream. Reference tests set shuffle
false. Each fit starts at zero. Partial-fit/streaming APIs, warm starts, custom
initial coefficients, sample weights, validation early stopping, sparse inputs
and parallel training are not implemented. Controls must be finite, with C>0,
epsilon>=0, positive epoch/patience limits and average>=0. Use appropriately
scaled finite inputs. The package now contains 99 core types in 40 source modules.


## OPTICS density clustering

`optics` adds the dense OPTICS graph, Xi hierarchy extraction with optional
predecessor correction, and DBSCAN extraction. Public helpers can re-extract
labels without rebuilding the graph. Ordering and predecessor row IDs are
1-based (root predecessor -1); hierarchy endpoints are inclusive 1-based
positions in `ordering_`. Cluster labels start at zero and noise is -1.
Reachability and core distances retain IEEE positive infinity for roots and
unreachable neighborhoods. `min_samples_fraction` and `min_cluster_fraction`
use sklearn's floor-and-clamp-to-two rule; -1 disables each fraction. A zero
integer `min_cluster_size` selects the resolved `min_samples`. The unbounded
`max_eps` default uses `huge(real64)` as an input sentinel for infinity;
`eps=-1` selects `max_eps_` during DBSCAN extraction. Set a finite extraction
radius for useful DBSCAN labels; an infinite radius follows sklearn's
infinity comparisons and can return all noise.

Supported distances are Euclidean, squared Euclidean, Manhattan/cityblock,
Chebyshev, Minkowski (finite p >= 1), cosine, and precomputed. Precomputed
matrices must be finite, square, symmetric, nonnegative, with zero diagonal.
Xi is restricted to 0 < xi < 1. Size counts must resolve to 2..n. Dense pairwise
storage costs O(n^2); there are no sparse, tree, parallel, callable-distance,
sample-weight or incremental interfaces. Equal reachabilities select the
lowest original row ID. Geometric floating-point ties can differ from
sklearn's tree backends. This batch brings the package to 100 core types in
41 source modules; no extensions or dependencies were added.


## BIRCH streaming clustering

`birch` implements the clustering-feature tree: count/linear/squared statistics,
radius-threshold merges, recursive farthest-pair splits and ordered leaf
subcluster extraction. `fit` rebuilds the tree; `partial_fit(X)` inserts only
new rows. The resulting tree is independent of batch boundaries when the
sample order and tree controls are unchanged. Private nodes use owned
allocatable storage, so copying the model copies its tree.

`n_clusters=0` corresponds to Python's `None`; positive values run unweighted
Ward clustering on the leaf centroids using the package's existing
`agglomerative_clustering`. Integer cluster names can differ from sklearn;
partitions are compared up to a shared label permutation. Floating-point
farthest-pair ties can reverse split orientation and reorder leaf centroids;
reference checks explicitly align centroid-indexed outputs before comparison. If fewer subclusters
than requested remain, each receives its own label and
`insufficient_subclusters_` is true (no Python-style warning is emitted).
`predict` uses the closest leaf centroid, then its global label. `transform`
returns distances to every leaf centroid, even after global clustering.

`compute_labels=false` skips storing labels for inserted rows; `predict` still
works. `labels_` describes the most recent batch for which labels were
computed. `partial_fit()` with no X repeats only global clustering, retaining
those batch labels as sklearn does; call `predict` for current assignments.
`fit_predict` requires compute_labels=true. `fit_transform` is also provided.
The metadata `n_features_in_` and `n_samples_seen_` records input width and
inserted sample count. Threshold must be finite and positive, branching
factor at least two, and n_clusters nonnegative. Changing threshold or
branching factor requires a fresh `fit`; n_clusters may change before a
global-only update. Dense real64 input only: no sparse input, sample weights,
custom global estimator objects, exposed mutable tree nodes or feature names.
Tree storage grows with the subclusters; optional dense Ward clustering has
quadratic storage and cubic-time pair selection. The package now has 101 core
types in 41 source modules; no public extensions or dependencies were added.


## QuantileTransformer empirical distribution mapping

`quantile_transformer` estimates column-wise linear percentiles, interpolates
to a uniform distribution, and optionally maps through the normal inverse CDF.
Repeated quantiles use the mean of ascending/descending interpolation, as in
sklearn. `inverse_transform` maps either output distribution back through the
learned percentile table. Finite values outside the fitted range saturate;
normal outputs use sklearn's 1e-7 boundary tolerance and finite tail clipping.
The inverse normal uses monotone erfc inversion, with no SciPy runtime or
third-party approximation coefficients.

NaNs are ignored when fitting and preserved in transforms. An all-missing
feature retains NaN quantiles and produces NaNs even for finite queries.
Constant columns and single-quantile tables follow sklearn's boundary-order
conventions. Infinity is rejected in fit, transform and inverse_transform.
`n_quantiles_` is capped at the training sample count, including rows with NaN;
`references_` is the corresponding uniform grid. No Python warning is emitted
when capping. `quantiles_` has shape (n_quantiles_, n_features_in_).

`subsample=0` means Python `subsample=None`; the default is 10000. A positive
subsample must be at least the requested n_quantiles. Smaller subsamples select
shared rows without replacement using a seeded native random stream; repeated
fits are reproducible on the same compiler, but differ from NumPy selections.
Reference comparisons disable subsampling. Dense real64 arrays only; no sparse
implicit-zero controls, sample weights, feature names, copy/in-place control,
or incremental fitting. Percentile fitting sorts each sampled column; storage
includes a full row-index array, sampled columns and the quantile table. This addition brings
the package to 102 core types in 42 source modules, with no new dependencies or
public extensions.


## Chi-squared kernel feature maps

`additive_chi2_sampler` implements sklearn's regular-frequency approximation
to the positive additive kernel sum_j 2*x_j*y_j/(x_j+y_j), with a zero term
when both entries are zero. This differs from the negative chi-squared
distance returned by sklearn's pairwise `additive_chi2_kernel`. Input must be
finite and nonnegative. Output columns are the entire square-root block,
then each frequency's cosine block followed by its sine block; width is
n_features*(2*sample_steps-1). Zero inputs map to zero. Log-scaled harmonic
amplitudes avoid cosh overflow for large intervals.

`sample_interval=-1` selects sklearn's defaults: .8, .5 and .4 for one, two
and three sample steps. Other step counts require an explicit nonnegative
interval; zero gives an all-zero map. `transform` is stateless and can run
before `fit`. `fit` records n_features_in_ and subsequent transforms enforce
that feature count. Current sampling controls are checked on every call.

`skewed_chi2_sampler` samples hyperbolic-secant random weights and uniform
phase offsets, then maps log(X+skewedness) with scaled cosine features.
Its kernel is the product over features of
2*sqrt((x_j+c)*(y_j+c))/(x_j+y_j+2*c), where c is skewedness. Any finite
skewedness (including zero/negative) is supported when each transformed entry
is strictly greater than -skewedness. Fit validates finite data and draws the
map from its feature count; domain validation is performed by transform.
A stable positive log-sum avoids overflow in X+skewedness. A possible zero
uniform draw is clipped to the smallest positive normal value before taking
the inverse CDF. n_components must be positive; output width is n_components.

Both expose fit, transform and fit_transform. Fitted skewed-map arrays are
random_weights_(n_features,n_components) and random_offset_(n_components).
Native random draws repeat for a fixed seed/compiler and differ from NumPy;
reference checks supply identical native uniform draws to sklearn's full
fit, independently checking its sampling formula and feature algebra.
Dense real64 input only: no sparse arrays, NaN/infinity, feature names or
incremental updates. No public extensions or dependencies were added. The
package now has 104 core types in 42 source modules.


## GradientBoostingClassifier

`gradient_boosting_classifier` fits negative-gradient regression trees and
performs Newton leaf updates for binary/multiclass log loss and binary
exponential loss. Multiclass gradients are frozen at the start of each stage
and include sklearn's (K-1)/K leaf correction. Defaults use class-prior
initialization: log odds for binary log loss, centered log priors for multiclass,
and half log odds for exponential loss. `init_zero=true` starts at zero.
At least two distinct integer labels are required; exponential loss is binary.

`estimators_(stage,channel)` contains one native regression tree per stage for
binary tasks and one per class for multiclass tasks. Native trees use squared
error splits; these rank unweighted candidate splits like Friedman MSE.
The minimum-impurity control is rescaled to Friedman's SSE-reduction units.
Tie handling, float64 thresholds and random feature selection can differ from
sklearn's float32 tree backend. `max_features=0` uses all features; positive
values use the native seeded feature-selection stream. No RNG equality with
NumPy is claimed. Split-based importances sum unnormalized impurity reductions
across trees before normalization. `train_score_` preserves sklearn 1.6.1's
binary log-loss deviance factor of two; multiclass and exponential scores are
mean cross-entropy and mean exponential loss, respectively.

Methods include fit, predict, decision_function, predict_proba,
predict_log_proba, score, apply, and staged decision/probability/label outputs.
Decision matrices always have rank two: (n,1) for binary, (n,K) for multiclass.
Staged decisions have shape (n,n_trees_per_iteration_,n_stages), staged
probabilities (n,n_classes,n_stages), and staged labels (n,n_stages). Apply returns native 1-based leaf IDs with shape
(n,n_stages,n_trees_per_iteration_). Leaf numbering can differ from sklearn;
reference checks compare per-tree prediction functions and require each
native leaf to map to a constant reference prediction, allowing redundant
constant-valued subdivisions. Staged methods
materialize all stages rather than returning Python generators. Log
probabilities use stable direct formulas and remain finite for finite raw
scores even when probability output underflows to zero. Fitted learning rate
and loss are retained for prediction; refit to change model hyperparameters.

The initial dense real64 port uses full-data fitting only. Sample weights,
subsampling/OOB statistics, early stopping, warm starts, custom initializers,
pruning, alternate criteria, feature-name metadata and sparse input are not
implemented. Native tree count/depth/leaf/split/max-feature and minimum-impurity
controls are supported. Learning rate must be finite and positive. The package
now contains 105 core types in 42 source modules; no public extensions or
runtime dependencies were added.


## Ridge classification and leave-one-out selection

`ridge_classifier` solves weighted ridge regression on -1/+1 class-indicator
columns, using a shared dense SVD. Binary problems use one response column;
multiclass problems fit one-vs-rest columns together. Weighted feature/target
means give an unpenalized intercept. The scalar alpha must be finite and
nonnegative; zero uses the native SVD rank truncation/minimum-norm solution.
`coef_` has shape (channels,n_features), `intercept_` has one entry per channel,
and decision_function always returns (n,channels). Binary scores >0 select
the larger sorted label; ties choose the smaller label. Multiclass uses the
first maximum. At least two distinct integer labels are required.

Fit accepts optional sample_weight and class_weight vectors. Class weights
align with sorted classes_; class_weight_balanced=true instead uses
n_samples/(n_classes*class_count), based on unweighted label frequencies.
Class and row weights multiply. Weights must be finite and nonnegative, with
positive effective total. Balanced and explicit class weights are mutually
exclusive. Sample/class weighting affects both centering and ridge residuals;
alpha is not divided by the sample count. Zero-weight rows are supported.

`ridge_classifier_cv` implements sklearn's default cv=None, scoring=None
leave-one-out mode only. It shares the weighted SVD across positive candidate
alphas, using diagonal dual-resolvent identities for exact leave-one-out
errors. Full row-span designs use a cancellation-free residual calculation.
Selection maximizes negative mean squared error on binarized targets, with
squared errors multiplied by effective row weights and averaged over all
samples and response columns. It does not select by classification accuracy.
The selected alpha is refitted on all data. At least two positive-weight rows
are required. Default alphas are [.1,1,10]. cv_scores_ records all candidate
scores, alpha_ and best_score_ record the selected result, and optional
store_cv_results=true retains (n,channels,n_alphas) weighted squared errors.
Results are cleared on a later fit with storage disabled.

Both provide fit, decision_function, predict and score. Dense real64 input
only; sparse/multilabel data, positive-coefficient constraints, per-target
penalties, iterative solvers, probability calibration, arbitrary CV folds,
custom scoring and metadata routing are not implemented. A single class is
rejected in this port. n_features_in_ and native rank_ expose fitted dimensions.
Original code remains MIT with sklearn BSD notices retained. The package now
has 107 core types in 43 source modules, with no new runtime dependencies or
public extensions.


## Multi-task sparse regression

`multi_task_elastic_net` and `multi_task_lasso` fit a dense response matrix
(n_samples,n_targets), including a single response column. Their joint penalty
selects features across targets: the objective is ||Y-XW||_F^2/(2*n) +
alpha*l1_ratio*sum_j ||W(j,:)||_2 + alpha*(1-l1_ratio)*||W||_F^2/2.
The Lasso variant always uses l1_ratio=1, ignoring the inherited mixing field.
Block coordinate descent updates one feature's complete target vector using
group soft thresholding. Feature order is cyclic or seeded random with replacement.
Each intercept is an
unpenalized response mean minus its coefficient-weighted feature means.

Both provide fit, predict and uniform-average R2 score. coef_ has shape
(n_targets,n_features), intercept_ has n_targets entries, and predictions
have shape (n_samples,n_targets). Controls are alpha (default 1), fit_intercept
(true), max_iter (1000), tol (1e-4), and warm_start (false). ElasticNet's mixing
ratio defaults to .5. Finite alpha>=0 and 0<=l1_ratio<=1 are supported, including
unpenalized least squares and pure L2 boundary cases.
Tolerance must be finite and nonnegative. Targets and features must be finite.

Stopping first checks the largest coordinate update relative to the largest
coefficient, then checks the primal-dual gap against tol*||centered Y||_F^2.
The gap is also checked at the iteration cap. n_iter_, dual_gap_ (divided by
n_samples), eps_ (unscaled stopping threshold), n_features_in_, and native
converged_ expose the result. An unconverged fit returns its final iterate
with converged_=false. Warm starts reuse coefficients with matching target and
feature dimensions; otherwise a fit starts at zero. Input arrays are not
modified. Zero-norm centered features are skipped, matching the reference.

Sparse matrices, sample weights and positivity constraints are not implemented.
The CV variants, warm-started alpha paths and seeded random updates are described
below. The types are re-exported through sklearn_linear_model and sklearn.
MIT and sklearn BSD attribution are retained.


## Categorical Naive Bayes

`categorical_nb` fits a separate discrete probability table for each input
feature, with Laplace/Lidstone smoothing controlled by finite alpha>=0 (default
1). `force_alpha=true` retains zero/tiny smoothing; false floors it to 1e-10. Input is a dense real64 matrix containing nonnegative integer category
codes. Categories start at zero; table column one represents category zero.
This port rejects fractional codes rather than silently truncating them.
Classification labels are arbitrary integers, sorted in classes_. Single-class
fits are supported. Sample weights must be finite and nonnegative; all-zero
weights are supported. Zero-weight rows still determine the supported category range.

fit resets counters. partial_fit requires the full classes vector on its first
call; later calls may omit it or supply the same set. New categories expand
counters without discarding earlier data. Unknown class labels and changed
feature counts are rejected. `n_categories_` records the latest batch ranges,
matching sklearn 1.6.1. Count/probability tables retain their largest observed/
reserved widths, and queries may use any category stored in those tables.

min_categories=0 infers widths from data. A positive scalar reserves at least
that many categories for every feature; optional min_categories_per_feature
reserves positive per-feature minima. If both are supplied their maximum is
used. Codes missing inside the fitted/reserved range receive smoothed mass;
query codes outside it are rejected. Category tables use dense storage, so
codes should be compact consecutive integers rather than large identifiers.

fit_prior=true learns empirical weighted class priors; false uses uniform
priors. Optional class_prior supplies a nonnegative finite vector aligned to
classes_, without renormalization; this overrides fit_prior. Zero priors are
represented by negative infinity. Zero-mass empirical priors, unsmoothed
zero-mass category rows and impossible posterior normalizations retain source
NaNs explicitly. Predictions select the first NaN score, or the first sorted
label at finite ties. An unobserved class has zero empirical prior until observed.

Methods are fit, partial_fit, joint_log_likelihood, predict_log_proba,
predict_proba, predict and score (accuracy). Fitted fields include classes_,
class_count_, class_log_prior_, n_features_in_ and n_categories_. Ragged tables
category_count_(feature)%values and feature_log_prob_(feature)%values each have
shape (n_classes,retained_categories_for_feature), using the public categorical_table
storage type. This helper is not counted as an estimator. Log probabilities
use log-sum-exp normalization; class ties choose the first sorted label.

Sparse data, string categories, automatic encoding, vector smoothing alpha and
Python warning objects are not implemented. The type is re-exported
through sklearn_naive_bayes and sklearn. There are now 110 core types in
45 source modules. No dependency or public algorithm extension was added;
original contributions remain MIT with sklearn BSD attribution retained.


## Neighbor imputation and missingness indicators

`knn_imputer` implements dense KNNImputer with NaN missing values and the
nan-Euclidean metric: sqrt(n_features/n_shared * sum_shared (x-y)^2).
Distances use the original receiver's observed values throughout imputation;
newly filled cells never influence later neighbor choices. For each missing
feature, donors must have an observed value in that feature. Undefined
(no-overlap) distances are ignored. If no eligible donor has a defined
distance, the training mean of that feature is used.

n_neighbors defaults to 5 and must be positive. weights is 'uniform' or
'distance'. The latter averages only zero-distance donors when any are present;
otherwise it uses inverse distances, rescaled to avoid reciprocal overflow.
Neighbors are capped at the available eligible defined donors. Equal-distance
donors retain training-row order; sklearn's partition-based tie selection can
choose a different subset at an ambiguous neighbor cutoff.

fit stores a private training copy. transform and fit_transform return dense
real64 arrays and never modify the input. Columns entirely missing in training
are dropped by default. keep_empty_features=true retains them as zeros,
including when a query supplies an observed value in such a column. When all
training columns are empty, dropping them produces (n_samples,0). Optional
add_indicator=true appends numeric 0/1 missingness columns selected at fit.
Their values reflect original query missingness, before any cells are filled.
New missingness in a previously complete column is imputed but adds no new
indicator column. valid_mask_ records retained training features;
n_features_in_ and indicator_ expose fitted metadata.

`missing_indicator` provides fit, transform and fit_transform. Its output is
a dense logical matrix. features='missing-only' (default) selects columns
containing NaNs at fit; features='all' selects every column. features_ contains
1-based native input indices. With missing-only mode, error_on_new=true
(default) rejects missingness in previously complete features. Setting it
false ignores those new indicator columns. All-feature mode accepts new
missingness. Fitting a complete matrix in missing-only mode produces zero
indicator columns. The KNN imputer's fitted indicator_ always uses missing-only
mode and error_on_new=false, and is available even when add_indicator=false.

Both types accept NaNs but reject infinities, empty input dimensions and
changed feature counts. Only the NaN sentinel and dense numeric CPU arrays
are supported; callable/alternative metrics, sparse input, string columns,
feature-name APIs and partial_fit are not implemented. Neighbor search is
brute force with stable insertion sorting per missing cell, intended for
moderate dense datasets. The types are re-exported through
sklearn_preprocessing and sklearn. The package now has 112 core types in
46 source modules; no runtime dependency or public algorithm extension was
added. Original contributions remain MIT with sklearn BSD attribution.


## Gaussian process classification

`gaussian_process_classifier` implements fixed-kernel GaussianProcessClassifier
with a logistic link and Laplace posterior approximation. The kernel is
amplitude*exp(-||x-y||^2/(2*length_scale^2)); both finite parameters must be
positive and default to 1. Kernel hyperparameters are fixed (the reference
configuration uses optimizer=None). This also matches sklearn 1.6.1's fixed
unit-amplitude/unit-length kernel when kernel=None; optimization of supplied
tunable kernels is not supported here. No observation-noise or diagonal-alpha
parameter is exposed for this classifier. Duplicate observations are supported
because posterior mode
iterations factor I+sqrt(W)*K*sqrt(W), not the raw Gram matrix.

Binary targets fit one latent GP for the larger sorted label. Multiclass
problems fit independent one-vs-rest binary GPs and normalize their positive
probabilities to sum to one. This is sklearn's one_vs_rest strategy, rather
than a joint multiclass Laplace approximation. At least two integer class
labels are required. Features must be dense finite real64 values.

Newton mode updates follow GPML Algorithm 3.1, using dense Cholesky solves.
max_iter_predict defaults to 100 and must be positive. Iteration stops when
the approximate log evidence improvement is below 1e-10, including a decrease,
matching the reference criterion. Stable logistic/softplus expressions avoid
exponential overflow. As in sklearn 1.6.1, convergence retains pre-update
pi/W/Cholesky state and the previous evidence value, with the updated latent
mode. A capped fit returns its last iterate and converged_=false. Fits always
start from zero latent values; warm starts are not implemented.

Predictive probabilities use GPML Algorithm 3.2 with sklearn's weighted sum of
five error functions approximating the logistic-normal integral. The native expression
is simplified algebraically to avoid division by predictive variance near
zero. Negative variance from roundoff is clamped to zero, and probability
roundoff is clipped to [0,1]. An all-zero OVR probability sum falls back to
uniform probabilities. Binary labels use the sign of the predictive latent
mean (zero selects the smaller label); exact multiclass probability ties
select the last sorted label, following sklearn's OVR implementation.

Methods are fit, predict, predict_proba, score (accuracy), and
log_marginal_likelihood() returning fitted evidence. The latter does not accept
alternate hyperparameters or gradients. log_marginal_likelihood_ is the binary
evidence or mean OVR evidence; log_marginal_likelihoods_ contains per-channel
values. Other fitted fields include classes_, n_classes_, n_features_in_,
x_train_, posterior_mode_, pi_, w_sqrt_, l_ and alpha_ (=binary targets-pi).
Per-channel vectors n_iter_ and converged_ provide native diagnostics. Latent
arrays are (n_train,channels); Cholesky factors are (n_train,n_train,channels).
Fitted kernel parameters are retained privately for subsequent predictions.

Sparse inputs, kernel objects/alternative kernels, ARD length scales, kernel
optimization/restarts, gradients, sample weights, one-vs-one classification,
parallel fitting and warm starts are outside this port. Dense storage/solves
are intended for moderate training sizes. The type is re-exported through
sklearn_gaussian_process and sklearn. Inventory: 113 core types in 47 source
modules; no runtime dependency or public algorithm extension was added.
Original contributions remain MIT with sklearn BSD attribution retained.


## B-spline features

`spline_transformer` implements dense SplineTransformer with native Cox-de Boor
basis evaluation. Controls are n_knots=5, degree=3, knots='uniform',
extrapolation='constant' and include_bias=true. fit, transform and
fit_transform accept dense finite real64 arrays. Fit requires at least two
samples, at least two base knots and positive degree. Degree zero is outside
this initial port. Automatic knots use uniformly spaced extrema or linearly
interpolated empirical quantiles (knots='quantile'). Optional real knots(:,:)
in fit/fit_transform supplies explicit (n_base_knots,n_features) positions,
overriding automatic placement. Every feature's base knots must be strictly
increasing: constant features and repeated quantile knots are rejected.
Sample-weighted placement is not implemented.

Nonperiodic knot vectors extend degree positions on either side using the
first/last base-knot spacing. There are n_base_knots+degree-1 basis columns
per feature. Periodic vectors continue spacings across the period, folding
the final degree basis functions onto the first ones. They have
n_base_knots-1 columns per feature and require degree<n_base_knots.
include_bias=false drops the last basis separately for each feature;
this avoids the implicit intercept from a complete partition of unity.
Features are concatenated in input-column order. Dropping the only periodic
basis is allowed and produces a zero-width output.

All five extrapolation modes are supported: 'error' rejects points outside
the base-knot interval; 'constant' uses boundary basis values; 'linear' uses
boundary values and analytic derivatives; 'continue' continues the first/last
polynomial segment; 'periodic' wraps into the base interval. Linear and
polynomial continuation may produce negative basis values outside the range.
Endpoints are evaluated from the valid base interval. Periodic right endpoints
map to the left endpoint. Transform uses the fitted degree/extrapolation/bias
settings and preserves input arrays.

Fitted fields are n_features_in_, n_features_out_, base_knots_ and knots_.
The latter is the extended knot matrix; its row count is n_base_knots+2*degree.
No SciPy BSpline objects or coefficient matrices are required at runtime.
Sparse output, sample weights, degree zero, repeated knots, string/feature-name
APIs and inverse transforms are not implemented. Positive higher degrees are
supported, with integer output-dimension guards; very high degrees or distant
polynomial continuation can amplify floating-point error. This is a feature
transformer, not a fitted smoothing/regression model.

The type is re-exported through sklearn_preprocessing and sklearn. Inventory:
114 core types in 48 source modules. No dependency or public algorithm extension
was added. Original contributions remain MIT with sklearn BSD attribution.


## Primal linear support-vector models

`linear_svc` implements L2-regularized squared-hinge LinearSVC. `linear_svr`
implements L2-regularized squared-epsilon-insensitive LinearSVR. These are
fixed primal formulations, corresponding to sklearn dual=False, penalty='l2'
and the named squared losses. LinearSVR here defaults to the squared loss;
sklearn's default unsquared epsilon-insensitive loss is outside this port.
The optimizer is a native active-set Newton method with dense Cholesky solves
and Armijo backtracking, rather than liblinear's trust-region implementation.
No LIBLINEAR binary or C/C++ source is linked.

The objectives are .5*||w||^2 + C*sum_i sample_weight_i*loss_i, with losses
max(0,1-y_i*f_i)^2 for signed binary classification and
max(0,abs(y_i-f_i)-epsilon)^2 for regression. C is not divided by sample count.
An optional intercept is represented by a synthetic feature of value
intercept_scaling, so its contribution to the penalty is
.5*(intercept/intercept_scaling)^2. Both C and intercept_scaling default to 1
and must be finite and positive. fit_intercept defaults true. The regression
epsilon defaults to 0 and must be finite/nonnegative.

Classification supports binary labels and multiclass one-vs-rest. classes_ is
sorted. Binary decisions have shape (n,1), multiclass decisions (n,n_classes).
coef_ is (channels,n_features), intercept_ has one entry per channel. Binary
zero scores choose the smaller label; multiclass ties choose the first label.
At least two classes are required. Methods are fit, decision_function, predict
and accuracy score. Regression provides fit, predict and R2 score, with
coef_(n_features) and a scalar intercept_. Neither model provides probabilities
or support-vector arrays: the primal representation stores coefficients.

Both fits accept finite nonnegative sample_weight, requiring positive total.
Classifier fit also accepts positive finite class_weight aligned with classes_,
or class_weight_balanced=true using unweighted label frequencies. Explicit and
balanced class weights are mutually exclusive. Binary class weights multiply
all corresponding row costs. In liblinear's multiclass OVR convention, each
channel weights its positive class while all negative classes retain base C;
the native implementation preserves this distinction. Zero class weights are
rejected in this port, while zero sample weights are supported.

max_iter defaults to 1000; finite tol>0 defaults to 1e-4. Stopping uses the
augmented primal gradient norm <= tol*max(1,initial_gradient_norm), starting at
zero coefficients each fit. A capped fit returns the final iterate and
converged_=false if the criterion was not met. n_iter_ counts native Newton
updates and can differ from liblinear. objective_ and gradient_norm_ expose
native diagnostics; they are per-channel vectors for classification and
scalars for regression. Classifier n_iter_per_class_ and
converged_per_class_ describe each OVR model; n_iter_ is their maximum and
converged_ is their conjunction. n_features_in_ records fitted input width.

Only dense finite real64 arrays and one response per sample are supported.
Sparse/dual solvers, L1 penalties, unsquared losses, Crammer-Singer mode,
partial_fit, warm starts and probability calibration are not implemented.
Dense feature-space Hessians favor moderate feature counts. The types are
re-exported through sklearn_svm and sklearn. Inventory: 116 core types in
49 source modules. No runtime dependency or public algorithm extension was
added. Original contributions remain MIT with sklearn/LIBLINEAR BSD notices.


## Incremental principal components

`incremental_pca` follows sklearn 1.6.1 IncrementalPCA's stacked-SVD update:
previous singular values times components, centered new rows and a running-mean
correction row. Population variance and mean are updated with the parallel
variance formula, while explained variance uses n_samples_seen_-1. This retains
only the selected old components, so truncated streaming fits depend on batch
boundaries. Full-rank updates reproduce ordinary PCA up to roundoff.

Methods are fit, partial_fit, transform, inverse_transform and fit_transform.
n_components=0 selects min(first_batch_rows,n_features); positive values fix the
rank. batch_size=0 selects 5*n_features for fit; positive values set its batches.
fit merges a short final batch when an explicit component count requires it,
and resets prior fitted state. partial_fit processes the supplied array as one
batch, ignores batch_size and permits single-row updates after initialization.
The first batch must contain at least two rows and at least n_components rows.
Changing feature width or component count between updates is rejected.

whiten defaults false. With whitening, transforms divide by square roots of
explained variance, floored at machine epsilon; inverse_transform restores the
scale. Components use the largest absolute loading as a positive sign anchor,
as in sklearn's right-vector sign convention. Attributes include components_,
mean_, var_, singular_values_, explained_variance_, explained_variance_ratio_,
noise_variance_, n_samples_seen_, n_features_in_, n_components_ and batch_size_
(the latter records fit's resolved batch size). noise_variance_ follows sklearn's
per-update discarded singular-value average, including its dependence on the
current batch size. Constant data returns zero variance ratios instead of NaNs.

Scope: dense finite real64 input, native Jacobi SVD, no sparse arrays, sample
weights, score/covariance helpers or automatic variance-fraction rank selection.
Memory includes dense feature-square SVD vectors, so this port favors moderate
feature counts. Re-exported through sklearn_decomposition and sklearn. Inventory:
117 core types in 50 source modules and 48 native test programs. No public
extension or runtime dependency was added. Original contributions remain MIT
with sklearn's BSD notice retained.


## Feature agglomeration

`feature_agglomeration` implements sklearn 1.6.1 FeatureAgglomeration by fitting
the native agglomerative tree to transpose(X), clustering columns rather than
samples. n_clusters defaults to 2 and must be between 1 and n_features; at least
two input features are required. linkage supports ward (default), single,
complete and average, with Euclidean distances. A complete unconstrained tree
is built and distances_ is always available. No connectivity graph, distance
threshold, alternate metric, caching or early tree stopping is implemented.
The dense exhaustive merge search favors moderate feature counts.

Methods are fit, transform, inverse_transform and fit_transform. pooling_func
is 'mean' by default, with 'median' and 'max' also supported, corresponding to
NumPy pooling callables in sklearn. Arbitrary callables are not supported.
transform pools each sample's values within each learned feature cluster.
inverse_transform repeats each pooled value into the original member columns;
it reconstructs clusterwise constant data and cannot recover discarded detail.
Changing pooling_func after fitting is allowed. Fitted cluster count is retained
separately, so changing n_clusters requires refitting to affect transforms.

labels_ contains zero-based feature cluster IDs ordered by first feature index,
which can permute sklearn's arbitrary cluster IDs. Pooled output columns follow
this order. children_ uses zero-based leaf/node IDs, with the first n_features
IDs denoting feature columns. children_ has n_features-1 rows and two columns;
distances_ contains the corresponding merge distances. Equal-distance merges
follow the existing native tree's scan order and may differ from sklearn ties.
Other attributes are n_features_in_, n_clusters_, n_leaves_ and
n_connected_components_=1. Inputs and queries must be nonempty finite real64
matrices with matching widths. This is a dense transformer, with no predict or
fit_predict API. Exported by sklearn_cluster and sklearn.

Inventory: 118 core types, 50 source modules and 49 native test programs.
Original contributions remain MIT; sklearn BSD attribution is retained.
No public extension or runtime dependency was added.


## Coordinate-descent cross-validation

`elastic_net_cv` and `lasso_cv` implement the single-target dense ElasticNetCV
and LassoCV contracts from sklearn 1.6.1. The objective is the existing elastic
net objective: ||y-Xw-intercept||^2/(2*n) + alpha*l1_ratio*||w||_1 +
alpha*(1-l1_ratio)*||w||^2/2, with an unpenalized optional intercept. Methods are
fit, inherited predict and R2 score. LassoCV fixes l1_ratio=1 on every fit.

cv defaults to 5 and selects contiguous, unshuffled K-fold partitions. The
first mod(n_samples,cv) folds have one extra row. Every candidate is fitted
using only training-fold samples (including fold-specific centering), and its
held-out MSE is recorded. Selection minimizes the unweighted mean of fold MSEs,
then refits on the full data. Equal means choose the earliest l1 ratio and the
largest alpha within that ratio. cv must be between 2 and n_samples.

Set alphas(:) to positive finite candidates; they are sorted descending with
duplicates retained and shared across l1 ratios. Otherwise n_alphas=100 and
eps=1e-3 generate geometric grids independently for each ratio. The maximum
alpha is max(abs(X_centered^T*y_centered))/(n*l1_ratio); centering follows
fit_intercept. Maxima at or below 1e-15 yield constant 1e-15 grids. n_alphas=1
uses only alpha_max; eps is finite in (0,1]. Set l1_ratios(:) to a nonempty
candidate list, or use inherited scalar l1_ratio (default .5). All ratios must
be finite in (0,1]. Pure ridge and zero-alpha candidates are outside this port.
LassoCV overwrites any supplied ratio grid with [1].

alphas_ always has shape (n_alphas,n_ratios); mse_path_ always has shape
(n_alphas,cv,n_ratios), including LassoCV and singleton axes. This differs from
Python's ratio-first/squeezed attributes. alpha_, l1_ratio_ and best_mse_ record
the selection, while inherited alpha/l1_ratio are set to the selected values.
coef_, intercept_, n_iter_ and converged_ describe the full-data refit;
all_cv_converged_ is true only when every fold/candidate fit converged.
n_features_in_ records input width. Repeated fit resets paths and coefficients.

The existing native cyclic coordinate solver starts each candidate at zero;
there is no warm-start path optimization. Its stopping criterion uses maximum
coordinate change < tol*max(1,max(abs(coef))), with package defaults tol=1e-8
and max_iter=10000, rather than sklearn's dual-gap stopping/defaults. Iteration
counts can differ. Capped fits retain their last iterate and expose convergence
flags. No sample weights, custom splitters, shuffled/random selection, positive
coefficient constraint, sparse matrices, multi-target CV or parallelism are
implemented. Original contributions remain MIT with sklearn BSD attribution.
Inventory: 120 core types, 50 source modules and 50 native test programs.
No public extension or runtime dependency was added.


## AdaBoost classification

`adaboost_classifier` implements sklearn 1.6.1 discrete SAMME with native
weighted CART weak learners. Defaults are 50 rounds, learning_rate=1,
max_depth=1, criterion='gini', splitter='best', random_state=0. Configure
max_depth, min_samples_split, min_samples_leaf, integer max_features (0 means
all), min_impurity_decrease, criterion ('gini', 'entropy', 'log_loss') and
splitter ('best', 'random'). These controls correspond to an explicit
DecisionTreeClassifier base in Python; arbitrary estimator objects, sparse
inputs, SAMME.R, class weights and monotonic constraints are not implemented.

fit accepts dense finite real64 data, integer labels (including one class),
and optional finite nonnegative sample_weight with positive total. Trees use
float32 feature comparisons and reject values outside float32 range. Each fit
resets the integer-seeded legacy NumPy RNG. Each round draws a fresh CART seed;
feature traversal, sorting, partitioning and weighted criterion updates follow
the installed sklearn backend. Its runtime FEATURE_THRESHOLD is 0, despite the
bundled Cython declaration specifying 1e-7; validation records the binary hashes
and runtime value. Leaf vote ties select the smallest sorted label.

SAMME uses weighted classification error and
learning_rate*(log((1-error)/error)+log(K-1)). Perfect learners get weight 1
and stop. A learner no better than random guessing is discarded; the first
such learner raises an error. Initial zero weights stay excluded and positive
weights are clipped to machine epsilon each round. Source-associated NumPy
reductions and exp(log(weight)+increment) preserve stopping on nonfinite
sample-weight sums. Infinite learner weights and undefined votes retain
source Inf/NaN results without triggering native floating-point traps. An
existing native correction rescales finite initial weights whose sum overflows;
the Python reference for that case uses the equivalent rescaled public input.

Methods are fit, decision_function, predict, predict_proba, predict_log_proba,
score and staged_decision_function, staged_predict, staged_predict_proba,
staged_score. score and staged_score accept optional sample_weight. Binary
decisions have shape (n,1); multiclass decisions (n,K). Probabilities use the
SAMME softmax conversion. Staged arrays have layouts (n,channels,T), (n,K,T),
(n,T) and (T), respectively; accumulation makes one pass through the retained
learners. Single-class final outputs follow sklearn. Native single-class
stages return constant decisions/probabilities/predictions/scores, correcting
the source generators' ZeroDivisionError; the validator retains those errors.

estimators_ contains the retained fitted CART models; estimator_ is the
unfitted prototype. n_estimators_ is the retained count. estimator_weights_
and estimator_errors_ keep the requested length with unused entries 0 and 1.
feature_importances_ is the learner-weighted mean of full-tree importances.
The retained stump_feature_, stump_threshold_, stump_left_ and stump_right_
arrays summarize each tree's root and immediate child majority labels; they
fully describe a weak learner only when max_depth=1. Features are one-based,
with 0 for leaves. n_features_in_, n_classes_ and sorted classes_ describe
the fitted data. Refit resets learned state.

AdaBoost also participates in the shared native classifier dispatch used by
Pipeline, voting, bagging and stacking. seed_base assigns the explicit nested
CART prototype seed before the AdaBoost seed, matching sklearn's sorted
parameter order. tree_random_state records that prototype seed; individual
learners receive seeds from random_state. The original caller's prototype is
preserved by native composition copies. No public extension or dependency is
added; original implementation remains MIT with retained BSD attribution.


## Stochastic-gradient regression

`sgd_regressor` implements dense single-target SGDRegressor from sklearn 1.6.1,
with fit, partial_fit, predict and R2 score. Supported losses are squared_error
(default, half squared residual), huber, epsilon_insensitive and
squared_epsilon_insensitive. epsilon defaults to .1 and must be finite and
nonnegative. Huber uses the epsilon transition and clips its residual gradient;
the epsilon-insensitive losses use the corresponding dead zone.

penalty defaults to 'l2'; 'none', 'l1' and 'elasticnet' are also supported.
Elastic-net uses l1_ratio=.15, with endpoints zero/one giving L2/L1. alpha
defaults to 1e-4 and is finite and nonnegative. The unpenalized intercept is optional (fit_intercept=true). Each
row computes a pre-update gradient, clipped to [-1e12,1e12], applies L2 decay
max(0,1-eta*alpha), then adds -eta*gradient*sample_weight times the row. Optional
sample weights must be finite/nonnegative and match row count. Zero-weight
rows still decay L2 coefficients, contribute to the unweighted stopping loss,
and advance the sample clock; all-zero weights are allowed as in sklearn.

learning_rate supports 'invscaling' (default), 'constant', 'optimal' and
'adaptive'. Optimal requires positive alpha and permits eta0=0; adaptive
reduces eta by five after exhausted patience while eta>1.e-6. eta0 defaults .01;
power_t defaults .25 and must be nonnegative. Inverse scaling uses
eta0/t_^power_t. t_ starts at 1 and increments once per row, including across
partial_fit calls. Cold fit resets coefficient, intercept, averaging state
and the clock. Warm fit semantics are detailed in the later SGD fit batch. partial_fit initializes on its first batch, runs exactly one
epoch and retains standard coefficients/clock thereafter. Feature width and
averaging configuration must remain fixed during partial_fit; fit can reset
them. max_iter applies to fit and defaults to 1000.

average=0 disables averaging; a positive integer starts averaging post-update
iterates at that global sample number. average=1 matches sklearn average=True.
coef_ and intercept_ publish averaged values once the threshold is reached,
while private standard iterates drive future updates, including new batches.
Averaging includes zero-gradient and zero-weight rows. n_iter_ reports epochs
in the latest call; n_features_in_ records input width. loss_ is the final
mean unweighted pre-update data loss (excluding regularization).

tol defaults to 1e-3; tol=-1 disables stopping (sklearn tol=None). Other negative
values are rejected. With stopping enabled, an epoch whose unweighted loss sum
is greater than best_loss-tol*n increments the stale counter; improvements
reset it. n_iter_no_change defaults to 5. Reaching patience sets converged_=true
except when an adaptive rate reduction is still possible;
exhausting max_iter leaves it false. partial_fit ignores stopping and leaves
converged_ false. The loss comparison is strict, including at tol=0.

shuffle defaults false in this port (sklearn defaults true) to provide a
deterministic ordered update stream. When enabled, scalar-seeded shuffling now
uses sklearn's local NumPy seed draws and forward xorshift Fisher-Yates pass.
The index permutation is reshuffled in place each epoch with the same seed;
fit/partial_fit seed a new local stream. Both ordered and shuffled trajectories
are compared directly with sklearn, including per-class OVR streams.

Scope: finite dense real64 matrices and one numeric target; sparse data,
validation-set early stopping and metadata routing remain unsupported. Warm
fits and explicit initialization are described in the later SGD fit batch.
The source follows sklearn's stochastic gradient and loss contracts; original
contributions remain MIT with retained BSD attribution. Exported through
sklearn_linear_model and sklearn. Inventory: 122 core types, 52 modules and
52 native test programs. No dependency or public extension was added.


## Stochastic-gradient classification

`sgd_classifier` implements sklearn 1.6.1 SGDClassifier's binary/one-vs-rest
classification contracts using the shared native SGD engine. Supported losses
are hinge (default), squared_hinge, perceptron, log_loss, modified_huber,
squared_error, huber, epsilon_insensitive and squared_epsilon_insensitive.
Methods are fit, partial_fit, decision_function, predict, predict_proba, predict_log_proba and
accuracy score. At least two known classes are required. fit discovers sorted
integer classes from its target labels and resets all fitted state.

The first partial_fit call requires classes(:), listing the complete distinct
class set; its batch may contain a single class. Later batches can omit classes,
or supply the same set. Unknown targets and changed class sets are rejected.
partial_fit runs one epoch, retaining private standard coefficients, averaging
state and sample clock. It fixes feature width and the averaging configuration.
fit can reset these. All-zero sample weights are allowed, with the same L2
shrinkage, averaging and clock behavior as SGDRegressor.

Binary coef_ has shape (1,n_features), intercept_ (1) and decision_function
(n,1). Positive binary scores select the larger sorted label, zero the smaller.
Multiclass coef_ is (K,n_features), intercept_ (K), decisions (n,K); maximum
score wins and ties select the first class. Each OVR head optimizes its signed
binary targets. n_iter_per_class_, converged_per_class_ and loss_ record native
per-head diagnostics; n_iter_ is the maximum head epoch count and converged_
is their conjunction. t_ advances by n_samples*maximum_epochs, as in sklearn;
heads use that shared clock when a later partial_fit resumes. n_features_in_
records the trained width. Partial fits leave convergence flags false.

Optional sample_weight must be finite/nonnegative and match the sample count.
Optional class_weight(:) is positive finite and aligned with sorted classes_;
it applies only to that fit/partial_fit call, so pass it again for later
weighted batches. class_weight_balanced=true computes n/(K*class_count) from
unweighted fit label frequencies and is mutually exclusive with explicit
weights. Balanced weighting is rejected for partial_fit; supply explicit
weights instead. Binary weights multiply both classes' row costs. In
multiclass OVR, only each channel's positive rows receive that class's weight;
negative rows retain weight one, matching sklearn's reduction convention.

Averaging is supported for binary/multiclass fit. Averaged partial_fit requires
fit_intercept=false; ordinary unaveraged partial_fit supports intercepts. The
1.6.1 classifier reference does not consistently retain the standard intercept
after publishing averages (and multiclass updates can overwrite the average
intercept). The port rejects averaged streaming with an intercept instead of
reproducing that state behavior. Averaged no-intercept streaming is supported
for both binary and multiclass problems and compared directly with sklearn.

predict_proba is available only for log_loss and modified_huber. Logistic
probabilities use sigmoid scores; multiclass normalizes the OVR sigmoid values.
Modified Huber clips scores to [-1,1], maps to [0,1], then normalizes multiclass
rows. An all-zero probability row becomes uniform (also a finite fallback for
complete logistic underflow). These probabilities are not separately calibrated.
Other losses reject the probability method with an explicit diagnostic.

Shared controls follow SGDRegressor: none/L1/L2/elastic-net penalties, finite
alpha>=0, all four rate schedules, eta0>0 (zero allowed for optimal), power_t>=0,
optional unpenalized intercept,
max_iter, tol=-1 to disable stopping, training-loss patience and integer
average thresholds. Defaults are hinge, alpha=1e-4, eta0=.01, invscaling,
power_t=.25, tol=1e-3, max_iter=1000, n_iter_no_change=5, average=0 and
shuffle=false. These ordered inverse-scaling defaults differ from sklearn's
shuffle=true, optimal rate, power_t=.5 and eta0=0 defaults. Explicitly enabled
shuffling now reproduces scalar-seeded sklearn permutations.

Scope: finite dense real64 arrays, integer labels and nine classification losses;
sparse feature input, validation-set early stopping and parallel OVR remain
unsupported. Warm fits, explicit initialization and log probabilities are
described in the later SGD fit batch. Exported through sklearn_linear_model and
sklearn. Original contributions remain MIT with sklearn BSD attribution.
Inventory: 123 core types, 52 modules and 53 native tests. No dependency or
public extension was added.


## Group and explicit-fold cross-validation

`group_kfold%split(groups)` implements the unshuffled largest-group-first
sample-balancing algorithm. `n_splits` defaults to 5 and must be between 2 and
the number of distinct groups. Integer group labels may be negative or noncontiguous.
Each group stays wholly in one test fold. Equal-sized groups are processed in
descending numeric label order; NumPy's unstable argsort can choose a different
order, so tied group allocations may differ from sklearn. Shuffled GroupKFold
is not supported.

`leave_one_out%split(n_samples)` requires at least two samples and holds out
each sample in ascending order. `leave_one_group_out%split(groups)` requires
at least two distinct groups and holds them out in ascending label order.
`predefined_split%split(test_fold)` accepts assignments on each call; every
unique integer except -1 is a test fold, in ascending order. -1 marks samples
included in every training set. Empty assignments and all -1 return zero folds;
a single test label may produce an empty training set, matching sklearn.

All four return allocated `index_split` arrays with ascending **1-based** sample
indices. They materialize every train/test pair, so LeaveOneOut uses quadratic
index storage. They do not add generic estimator cloning or new cross_val_score
dispatch; callers can fit/score manually as in `example/group_validation.f90`.


## SequentialFeatureSelector

`sequential_feature_selector%fit(x,y,estimator)` accepts a real regression
response and a `linear_regression`-family estimator, tested with OLS, ridge,
lasso and elastic net. Each candidate gets an independent estimator copy and
unshuffled K-fold mean R2 score; ties choose the first feature in input order.
The caller's estimator is not fitted or modified. Classification, custom
scorers/splitters, fractional feature counts, sample weights, metadata routing,
parallel candidates and unsupervised estimators are outside this initial subset.

`direction` is `forward` (default) or `backward`. Positive
`n_features_to_select` must be less than the original feature count. Zero means
sklearn's `auto`: without tolerance, forward selects floor(p/2), while backward
removes floor(p/2) and retains ceil(p/2), following sklearn 1.6.1 even for odd p.
Set `use_tol=.true.` for an explicit tolerance (`tol=0` by default). With auto,
selection stops before accepting a step whose score improvement is below tol;
the first step always happens. Forward tol must be nonnegative; backward tol
may be negative. With a fixed count, tol is validated but does not stop selection.

`cv` defaults to 5 and must leave at least two rows in every test fold, ensuring
R2 is defined; at least two input features are required. Inputs must be finite.
Fitted `support_`, `n_features_in_` and `n_features_to_select_` describe the chosen
features. `get_support`, `transform`, `fit_transform` and `inverse_transform`
are available; inverse fills unselected columns with zero. The selector stores
no fitted estimator, and downstream fitting on the selected data is explicit.


## Recursive Feature Elimination and RFECV

`rfe%fit(x,y,estimator)` supports the existing `linear_regression` family with
real regression targets (tested OLS, ridge, lasso and elastic net), and
`logistic_regression` with integer binary/multiclass targets. Each elimination
fits an independent estimator copy and removes the smallest squared coefficient
importances; multiclass importances sum squares over classes. The caller's
estimator is unchanged. `n_features_to_select=0` means floor(p/2); positive
integers select that many columns, and counts above p retain all columns.
Fractional feature counts, arbitrary importance getters, tree importances,
weights, sparse arrays and multiple targets are outside this initial subset.

`step` is a positive real: values below 1 remove max(1, int(step*p)) features
per round, using the ORIGINAL feature count; values at least 1 remove int(step).
The last removal is clipped to the requested count. Equal importances eliminate
earlier input columns first; NumPy's unstable argsort can choose a different
order on ties. `support_` preserves original column order; selected columns have
`ranking_=1`, and earlier eliminated columns receive larger ranks.

`rfecv` adds `min_features_to_select=1` and `cv=5`. It scores complete RFE paths
in each unshuffled fold, chooses the best mean score (fewest columns on exact
score ties), and repeats elimination plus final fitting on all data. Regression
uses KFold/R2 and requires at least two test rows per fold. Classification uses
StratifiedKFold/accuracy, encoding labels by first appearance as sklearn does.
Each class must contain at least two samples, and at least one class must have
cv members; classes smaller than cv are otherwise supported. Custom scorers,
custom CV, parallelism and metadata routing are not implemented. RFECV uses
min_features_to_select; its inherited n_features_to_select is ignored.

`cv_results_` is an `rfecv_results` record with ascending `n_features`,
`mean_test_score`, population `std_test_score`, and `split_test_score` shaped
(number of feature counts, number of folds). It is a dictionary replacement,
not an additional counted sklearn estimator. A minimum above p evaluates only
all features. Fitted `n_features_`, `n_features_in_`, support and rankings are
available, along with transform/fit_transform, get_support and zero-filled
inverse_transform. Fitted copies are exposed as regression_estimator_ or
classifier_estimator_ and refitting clears the other estimator kind.

Fortran `predict` returns real regression values; `predict_classes` returns
integer labels. Classification also exposes predict_proba and decision_function
(binary decisions have shape n-by-1). `score` dispatches by target type. Inputs
must be finite, and fitting requires at least two original features. Parameters,
elimination paths and predictions inherit the supported estimator's numerical
scope. Original MIT contributions retain the sklearn BSD attribution.


RFE reference comparisons use unmodified sklearn coefficient ordering for OLS,
ridge and logistic models. Elastic-net/lasso reference configurations supply an
importance_getter that ranks sklearn squared coefficients with stable input-order
ties, matching the native tie rule; final fitted coefficients remain unchanged.
Multinomial intercepts and decisions are compared after removing their common
class offset, which is not identifiable under softmax. Probabilities, class
predictions, coefficients, masks, rankings and CV paths are checked directly.


## Least-angle regression and LassoLars

`lars` implements dense least-angle regression with `n_nonzero_coefs=500`,
`fit_intercept=.true.`, `eps=epsilon(1._dp)` and `fit_path=.true.`. The iteration
limit is the requested maximum number of path steps (as in sklearn), and need
not equal the final number of nonzero coefficients. `lasso_lars` uses the same
active-set homotopy with coefficient dropouts/re-entry and alpha interpolation;
it adds `alpha=1`, `max_iter=500` and `positive=.false.`. Its inherited
n_nonzero_coefs is ignored. Fits center X and y for intercepts but do not scale
features, matching sklearn 1.6.1. Real single-target responses and finite dense
real64 arrays are required. Sparse inputs, multiple targets, sample weights,
jitter and caller-supplied Gram/Xy arrays are outside this initial subset.

Fitted coef_, intercept_, n_features_in_, n_iter_ and alphas_ are available.
With fit_path true, coef_path_ is (n_features, n_knots), and active_ gives final
active column indices in entry order, **1-based**. With fit_path false,
coef_path_ and active_ are cleared and alphas_ contains only the endpoint. The
solver currently constructs an internal full path even when storage is disabled.
Prediction and R2 scoring use the final coefficients.

`lars_path(x,y,method,max_iter,alpha_min,eps,positive)` provides uncentered paths
for method `lar` (default) or `lasso`. It returns a lars_path_result record with
alphas, coefficients, active, n_iter and n_skipped; this record is an API adapter,
not an additional sklearn estimator. Defaults are max_iter=500, alpha_min=0,
eps=machine epsilon and positive=false. Only lasso allows positive constraints.
max_iter=0 returns the initial zero-coefficient knot. The equality tolerance for
alpha stopping follows sklearn's float32 epsilon, even with float64 arrays.

The native implementation forms a dense Gram matrix, recomputes active
Cholesky factors, solves equiangular directions and rounds inactive correlations
to float64's 15 decimal digits. It skips candidate columns with a Cholesky pivot
below 1e-7 and exposes their count in direct path results. Duplicate/degenerate
columns and tied correlations can yield a different active representation than
sklearn; no Python warning machinery is reproduced. Paths and predictions for
nondegenerate dense, correlated, wide and dropout fixtures are reference-tested.


## Least-angle model selection

`lars_cv` and `lasso_lars_cv` select alpha using unshuffled K-fold held-out mean
squared error. Defaults are cv=5, max_iter=500, max_n_alphas=1000 and inherited
fit_intercept=true/eps=machine epsilon; LassoLarsCV adds positive=false. Each
training fold supplies its own centering and complete coefficient path.
The CV alpha grid is the ascending exact union of fold knots, sampled with
sklearn's floor-based stride, so max_n_alphas is a target rather than a hard
cap. Held-out residuals are interpolated before squaring and averaging; final
fitting uses the chosen alpha on all data. Exact MSE ties favor the first
(smallest) grid alpha. cv must be between 2 and n, including one-row test folds.

Fitted alpha_, cv_alphas_ and mse_path_ (n_alpha_grid, cv) supplement the
inherited final-model attributes and full published path. These estimators
ignore inherited n_nonzero_coefs and always retain a final coefficient path.
LAR knots may be locally nonmonotonic: interpolation uses a stable knot sort
and sklearn 1.6.1's inserted endpoint convention, including when the subsampled
grid ends below a fold's first alpha. A single-alpha grid is evaluated by
endpoint clamping; this avoids sklearn 1.6.1's duplicate-boundary interpolation
failure with max_n_alphas=1. That setting is checked using sklearn fold paths
and a manually scored/refitted reference. Custom CV, sample weights, parallel
folds and metadata routing are outside this subset.

`lasso_lars_ic` selects a Lasso path knot using criterion='aic' (default) or
'bic', with max_iter=500 and optional positive constraints. noise_variance=-1
means automatic estimation; explicit estimates must be positive and finite.
Automatic estimation requires n > p + intercept and uses full-feature SVD
least squares, or active-set nonnegative least squares when positive=true.
The residual sum of squares is divided by n-p-intercept, matching sklearn.
Zero/nonfinite estimated variance is rejected because log-variance criteria
would be undefined. Sparse/multiple-target inputs, jitter, Gram/Xy arguments
and zero variance are not supported.

The criterion is n*log(2*pi*noise_variance_) + RSS/noise_variance_ + penalty*df,
where penalty is 2 (AIC) or log(n) (BIC), and df counts path coefficients larger
than float64 epsilon in magnitude. The first minimum wins, hence larger alpha
on exact criterion ties along a descending path. Fitted alpha_, criterion_,
noise_variance_, alphas_, coef_, intercept_ and n_iter_ are retained; n_iter_
describes the complete path rather than the selected knot. IC does not publish
coef_path_ or active_, and ignores inherited alpha/fit_path/n_nonzero_coefs.
All three models retain the existing single-target dense-real64 numerical
scope and use predict/score on the selected final regression model.


Default rank-deficient LarsCV may select a different alpha and final predictions
from sklearn because nearly singular active-column admissions differ.
The model-selection reference suite uses eps=.1 for complete LAR paths when CV
training folds are rank deficient; this avoids matching unstable nearly-singular
admission decisions in sklearn. Full-rank LAR, truncated LAR and all Lasso CV
comparisons use default machine epsilon, including wide Lasso data. A native
test separately exercises default-epsilon rank-deficient LarsCV and checks
finite predictions and minimization of its recorded error surface.


## SparsePCA and dictionary learning

`sparse_pca` supports method='lars' (default) and 'cd', alpha=1, ridge_alpha=.01,
tol=1e-8, max_iter=1000, random_state=0 and n_components=0 (auto = input feature
count). Fits center X and learn sparse loadings by alternating Lasso coding of
X.T with dictionary updates. Components are sign-oriented and normalized to
unit norm; all-zero components remain zero. transform uses ridge least-squares
projection onto the nonorthogonal components; ridge_alpha=0 uses a pseudoinverse.
inverse_transform reconstructs with components and adds mean_. Fitted mean_,
components_, error_, n_iter_, n_components_ and n_features_in_ are available.
fit_transform returns the ridge projection, as sklearn does.

`dictionary_learning` uses the same objective .5*||X-code*dictionary||^2 +
alpha*||code||_1, with atom row norms <= 1. Defaults are n_components=0 (auto =
input feature count), alpha=1, max_iter=1000, tol=1e-8, fit_algorithm='lars',
random_state=0, and transform_max_iter=1000. Code updates use native Lasso-LARS
or cyclic Gram coordinate descent with sklearn's 1e-4 inner stopping tolerance
and dual-gap check. Sequential dictionary updates enforce the unit-ball
constraint; unused atoms are replaced by a sampled data row plus small noise.
positive_dict clips atoms; positive_code requires fit_algorithm='cd', matching
sklearn's estimator validation. Positive sparse coding is available for Lasso
and threshold transforms, including direct sparse_encode with lasso_lars.

DictionaryLearning fit_transform returns the final learned training code;
transform performs fresh coding and can differ. The default transform is OMP;
other algorithms are lasso_lars, lasso_cd, lars and threshold. Controls are
transform_alpha=-1 (auto = training alpha), transform_n_nonzero_coefs=0
(auto = min(max(floor(n_features/10),1),n_atoms)), transform_max_iter=1000,
positive_code and split_sign. split_sign affects transform outputs, not the
learned code returned by fit_transform. Reconstruct a training code with
matmul(code, components_). Fitted components_, error_, n_iter_ and
n_features_in_ are retained.

Both learners accept overcomplete dictionaries and zero outer iterations.
Paired code_init/dict_init (DictionaryLearning) or u_init/v_init (SparsePCA)
replace SVD initialization. Dictionary code/dictionary shapes are (n_rows,r)
and (r,n_features); SparsePCA U/V shapes are (n_samples,r) and (r,n_features).
Initial factors are truncated or zero-padded to the requested component count.
A lone initializer is ignored, following sklearn. Zero iterations preserve the
initial factors; DictionaryLearning atoms are projected only during updates.
Degenerate SVD subspaces can produce different initial bases and local optima,
especially coordinate descent warm starts on zero singular directions.

`sparse_encode` exposes all five coding algorithms and optional finite init
for coordinate descent. It takes rows of X and rows of dictionary atoms and
returns (n_samples,n_atoms). alpha defaults to 1; n_nonzero_coefs=0 means auto;
max_iter=1000 and positive=false. Positive constraints exclude lars and omp.
`sparse_coder` applies those transforms to its supplied dictionary, with the
same transform controls and optional split-sign output. fit is a no-op and
fit_transform performs encoding. Matrix dimensions are obtained from dictionary
rather than cached SparseCoder metadata properties.

All inputs are finite dense real64 arrays. Sparse matrices, parallel jobs,
callbacks, state serialization and noninteger random_state objects are outside
this subset. Scalar nonnegative default-integer seeds are supported (through
2147483647 on the current compiler); the native default is deterministic zero.
The internal sklearn_legacy_rng helper supplies NumPy RandomState-compatible
MT19937 uniforms, masked bounded indices and cached polar Gaussians for atom
resampling. It is not a counted sklearn estimator or a public extension.
Its 28 NumPy stream comparison groups are recorded separately from the sklearn
algorithm result counts. NumPy source notices are retained under LICENSES.


## AdaBoost regression

`adaboost_regressor` implements sklearn 1.6.1 AdaBoostRegressor's AdaBoost.R2:
weighted bootstrap sampling, linear/square/exponential losses, learning-rate
scaled estimator weights, weighted-median predictions, staged predictions and
R2 scores, perfect-fit/poor-learner stopping, and weighted tree importances.
The default learner is a squared-error CART with max_depth=3. Depth,
min_samples_split, min_samples_leaf, integer max_features (zero means all), and
min_impurity_decrease can be configured. Arbitrary estimator objects,
pruning, monotonic constraints, missing values, sparse matrices and multioutput
regression are outside this subset. Predict/score sample weights are not exposed;
fit accepts weights. Nonnegative default-integer random seeds are supported.

A private dense CART builder follows sklearn's float32 feature conversion,
Xorshift feature ordering, constant-feature caching, squared-error split proxy,
partitioning, and depth-first construction. This builder is local to AdaBoost
regression; existing native tree/forest behavior is unchanged. The existing
NumPy-compatible legacy RNG supplies estimator seeds and weighted-bootstrap
uniforms. Query features are also rounded to float32; out-of-range values are
rejected. Targets, errors and accumulated weights use real64. Feature sorting uses the same three-way introsort and heap fallback as sklearn.
Numerical parity is established for the reference fixtures, not every possible
floating-point tie pattern.

Initial zero-weight samples remain excluded; positive weights are clipped at
machine epsilon before each boosting round, matching sklearn. If the first
learner has error >=0.5, it is retained with zero ensemble weight, matching
sklearn's prediction behavior; native feature_importances_ is zero in this case
rather than sklearn's division-by-zero NaNs. Training stops on a nonpositive or
nonfinite updated weight sum. Parameters and finite dense inputs are validated.


## Quantile regression

`quantile_regressor` implements sklearn 1.6.1 QuantileRegressor's weighted
pinball-loss plus alpha * L1 objective, optional unpenalized intercept, dense
single-target fit/predict and unweighted R2 score. Quantile must lie strictly
between zero and one; alpha may be zero. Sample weights must be finite,
nonnegative and have a positive finite sum. Weights are normalized to sum one,
and zero-weight rows are removed before optimization. The objective matches
sklearn's sum-weight-rescaled penalty convention.

The independent native solver is a dense primal simplex with a residual-slack
initial feasible basis, Bland entering/leaving index rules, row sign changes,
feature/target scaling and an iteration limit. It optimizes the full piecewise
linear objective; no smoothed approximation or external LP runtime is used.
For n positive-weight samples and p parameters (including an intercept), the
tableau has n rows and 2*p+2*n+1 columns. Storage is quadratic in n, and simplex
can take many pivots. This implementation targets moderate dense problems.
SciPy/HiGHS solver selection and solver_options, sparse arrays, multioutput,
metadata routing and estimator cloning are outside this subset. There is no
claim that pivot counts or nonunique optima match HiGHS.

Native max_iter (default 10000) and tol (default 1e-10) control pivot count and
scaled reduced-cost tolerance. Inspect converged_ before using a result as an
optimum; an iteration limit or unreliable numerical pivot returns the current
iterate with converged_=false. n_iter_ counts native pivots. Native diagnostic
attributes objective_, dual_coef_, dual_gap_ and optimality_ expose the
normalized weighted objective, training-row dual variables, nonnegative
primal-minus-dual objective gap, and maximum negative scaled reduced cost.
Certificates are meaningful as optimality evidence when converged_ is true.
The intercept and coefficients are returned in the original input units.

Independent and duplicate columns, zero features, exact fits, underdetermined
systems, extreme quantiles and strongly regularized solutions are covered by
reference fixtures. Nonunique solutions can have different coefficients or
predictions from HiGHS while attaining the same optimum. Validation checks the
objective and primal-dual conditions in these cases, rather than asserting
identical arbitrary solutions. Tiny sample weights are normalized before the
HiGHS reference call in the weight-rescaling fixture to prevent absolute
solver tolerances from treating the entire objective as negligible; the native
call retains the original tiny weights.


## Nu support-vector classification

`nu_svc` translates sklearn 1.6.1 NuSVC's dense binary and one-versus-one
multiclass classification with linear, RBF, polynomial, sigmoid and precomputed
kernels. It supports nu, scalar sample weights, gamma scale/auto/numeric rules,
degree, coef0, tolerance and per-pair iteration limits. Default multiclass
scores use sklearn's OVR vote/confidence transform; OVO scores and OVR
break_ties are supported. Binary decision scores have shape (n_query,1) and
favor the second sorted class when positive. Multiclass OVO scores favor the
first class of each ordered pair when positive, matching sklearn's convention.

The independent native optimizer minimizes the two-label-group box-constrained
nu dual with fixed weighted alpha mass in each group, gain-based pair updates,
periodic full gradient reconstruction and LIBSVM's margin/intercept rescaling.
Sample weights are capacities and are not normalized. Every class needs positive
total weight; each pair must satisfy nu*(weight_a+weight_b)/2 <= min(weight_a,
weight_b). Zero-weight samples are excluded. Kernel calculations and gradients
use full real64 matrices, while sklearn's LIBSVM uses a float kernel cache.
Dense quadratic storage targets moderate datasets. Sigmoid or indefinite
precomputed kernels can be nonconvex; KKT convergence indicates stationarity,
not a guarantee of a global minimum for those kernels.

Fitted classes_, support_, n_support_, support_vectors_, dual_coef_, intercept_,
and linear-only coef_ follow the class-grouped LIBSVM layout. Native support
indices are 1-based original training row indices. Native n_iter_ and
kkt_violation_ have one entry per ordered class pair; converged_ and fit_status_
report aggregate convergence. Native pair_alpha_ (training rows by class pairs)
and pair_scales_ expose unscaled dual variables and margin normalizers for
feasibility checks. n_features_in_ is cached. Native max_iter defaults to 200000;
examine convergence after a limited fit. Kernels and their parameters are saved
at fit time; decision_function_shape and break_ties may be changed for queries.

When a group has no free dual variables, its margin level can lie in a
nontrivial interval. Native levels use the interval midpoint, ignoring
machine-roundoff artifacts at capacity bounds. LIBSVM may select a different
valid level after leaving a coefficient within machine epsilon of a bound.
This can change normalized coefficients, intercepts and scores while preserving
the unscaled optimum. Reference validation checks both levels' KKT intervals
and unscaled kernel projections in these cases. Meaningful RBF support metadata
is compared above a 1e-10 coefficient threshold; full dual comparisons retain
all entries. This avoids treating LIBSVM coefficients around 1e-15 as distinct
substantive support vectors. Native exposed support arrays retain actual
nonzero coefficients, without that validation-only threshold.

For zero-weight precomputed references, validation explicitly reduces the
positive-weight training Gram and query columns before calling sklearn. This
is mathematically equivalent to dropping the zero-weight rows and avoids a
verified sklearn 1.6.1 wrapper mismatch between compacted support indices and
unreduced query columns. Native precomputed queries retain all original
training columns; original sample indices remain valid. Training precomputed
matrices must be finite, square and symmetric. Zero or unbounded margin
normalization is rejected explicitly rather than returning nonfinite model
coefficients. This includes degenerate constant kernels and some fully
saturated nu endpoints that also fail in sklearn.

Probability calibration, class_weight, callable kernels, sparse matrices,
shrinking/cache controls, metadata routing and sklearn's unlimited max_iter
sentinel are outside this subset. Native positive finite tol and positive
max_iter are required; break_ties requires OVR scores. No public extension or
external runtime dependency was added.


## Online one-class SVM

`sgd_one_class_svm` translates sklearn 1.6.1 SGDOneClassSVM's dense linear
one-class updates, including its default optimal learning rate, shuffled
training, L2 decay, one-class offset update, optional intercept, constant /
inverse-scaling / adaptive schedules, averaging, fit / partial_fit,
coefficient/offset initialization and warm-start fits. score_samples returns
X @ coef_; decision_function subtracts offset_; predict labels nonnegative
scores +1 and negative scores -1. fit_predict returns training labels after fit.

The internal hinge prediction is X @ w + b with b=1-offset. Hinge updates
include equality at margin one. Coefficients decay by max(0,1-eta*nu/2), and
the dense intercept update is update-eta*nu. Sample weights scale the hinge
update. Zero-weight rows are retained: they still advance the sample clock,
apply decay and the offset term, contribute to unweighted stopping loss and
participate in averaging. All-zero weights are accepted. This differs from
kernel NuSVC's capacity/zero-row convention. Native weights must be finite
and nonnegative. Targets are not used.

Lazy coefficient scaling and averaging follow the native Cython weight-vector
contracts, including scale resets below 1e-9. The existing NumPy-compatible
legacy RNG generates the Cython shuffle seed. A forward Fisher-Yates pass uses
sklearn's Xorshift stream; each epoch reuses the seed by value and permutes the
current ordering. Integer random_state is reproducible; native default is zero.
Each partial_fit call starts a fresh local row ordering and local shuffle stream,
as sklearn does for integer seeds, while preserving coefficients, average
buffers and the cumulative t_ sample clock. Averaging uses zero for off, one
for immediate averaging, or a later integer update index; the setting must
remain fixed during partial_fit. fit resets t_ and averaging buffers. With
warm_start, its initial parameters come from the previous public coef_/offset_,
including an averaged result when enabled. Explicit initializers override them.

Native controls are nu, fit_intercept, max_iter, tol (-1 disables stopping),
shuffle, random_state, learning_rate, eta0, power_t, warm_start and average.
n_iter_no_change is also exposed (sklearn's inherited internal setting is five).
Defaults match the mathematical training settings apart from deterministic
random_state=0. Fitted arrays/attributes are coef_, scalar offset_, n_features_in_,
n_iter_ and t_. Native converged_ indicates the training-loss stopping predicate
fired, and loss_ is last-epoch mean pre-update hinge loss. Reaching max_iter or
performing partial_fit does not assert convergence; source-style warnings are
replaced by these inspectable attributes. Sparse/float32 paths, early-stopping
validation, verbose callbacks, random-state objects and signed sample weights
are outside this subset. Hidden passive-aggressive learning-rate modes are not
exposed. Extremely small nu that underflows nu/2 is rejected explicitly.

Reference fixtures cover the actual defaults, all four learning rates, seeded
shuffling, adaptive reductions/stopping, zero/all-zero weights, initialization,
with/without intercepts, warm fits, delayed averaging, scale resets, full-fit
then partial-fit transitions and three-batch streaming. Coefficients, offsets,
query scores and labels, every recorded streaming/warm state and exact sample
clock/epoch counts are compared with installed sklearn. Prior reference states
are copied before later calls, since sklearn mutates coefficient buffers in
place. Native initializers copy their inputs; no caller-array aliasing is used.


## Isolation forests

`isolation_forest` translates sklearn 1.6.1 IsolationForest's dense finite-data
algorithm: random isolation trees, automatic/count/fraction row and feature
sampling, bootstrap multiplicities, sample weights, contamination thresholds,
fit_predict and warm-start growth. score_samples returns negative path-length
scores (lower means more anomalous); decision_function subtracts offset_;
predict returns +1 for nonnegative decisions and -1 otherwise. The default is
100 trees, min(256,n_samples) rows, all features, no bootstrap and offset_=-0.5.

Integer max_samples=0 selects auto; a positive count is clipped to the input
row count as in sklearn. Alternatively set max_samples_fraction in (0,1] and
leave max_samples=0. Integer max_features=0 selects all features; alternatively
set max_features_fraction in (0,1], with its truncated feature count clamped to
at least one. Count and fraction controls for the same axis cannot both be
active. Fractions default to zero (disabled). contamination=-1 is auto;
a fraction in (0,.5] selects the linear-interpolated training-score percentile.
Native random_state is a nonnegative default-integer seed and defaults to zero.

Sampling follows sklearn's tracking-selection, RandomState permutation and
reservoir branches, including their exact RNG consumption and order. Seeded
random targets, estimator seeds and splitter Xorshift states match the pinned
sklearn/NumPy streams. Tree features are float32-rounded, random cuts use the
Cython endpoint convention, constant features are cached, and weighted target
impurities/early leaves follow dense ExtraTreeRegressor. The original shared
native tree/forest builders are unchanged. Query features are also rounded to
float32. Nonfinite/out-of-float32-range features are rejected. Missing values,
sparse arrays, signed weights, parallel backends and metadata routing are
outside this subset.

Sample weights must be finite and nonnegative; zeros are excluded from each
tree's active rows. Bootstrap counts multiply the weights. Node sample counts
and path corrections use unique active rows, while score normalization uses
the requested max_samples_, matching sklearn. All-zero weights can therefore
produce empty weighted roots. Their native value, impurity and importances
are zero instead of sklearn's NaNs; forest path scores still agree. A sample
fraction that truncates to zero is rejected without bootstrap (sklearn also
fails that path); with bootstrap it produces empty trees and scores -0.5, as
sklearn does. Singleton sampling also scores -0.5.

Fitted n_estimators_, n_features_in_, max_samples_, max_features_ and offset_
are exposed. estimators_ contains native decision_tree_regressor storage with
1-based node/feature indices, 0 leaf links and root depth zero. It supports
apply/get_depth/get_n_leaves and node inspection; sample arrays returned by the
forest use original training row indices starting at one. Tree features are
local indices into each bag's feature list. estimators_features_(i)%values and
estimators_samples_(i)%values expose the actual selected feature/row indices,
including bootstrap duplicates. Unlike sklearn's reconstructed warm-start
sample property, native metadata retains the actual draws for every tree,
including those fitted on earlier calls.

Warm starts retain existing trees and consume/skip the same initial estimator
seeds before appending new trees. n_estimators cannot shrink, and feature
count must remain unchanged. A warm fit without added trees retains the model
and recalculates its threshold on the supplied data. Stored per-tree feature
lists are always applied during prediction; tree training controls are not
reapplied to old trees. Warm fits can keep distinct per-tree sampling lists;
the latest max_samples_ supplies the current normalization, as in sklearn.
Path-length accumulation preserves the source operation order, which matters
at the exact auto threshold for constant data. Numerical parity is established
for the reference fixtures rather than claimed for all floating-point ties.


## Robust covariance and elliptic envelopes

`min_cov_det`, `elliptic_envelope` and `fast_mcd` translate sklearn 1.6.1's
FastMCD robust covariance path. The implementation includes the univariate
shortest-half algorithm, 30 random starts/two-step screening followed by ten
candidate refinements for up to 500 rows, and the larger-data subset/merge/full
search. Above 500 rows it partitions shuffled data into roughly 300-row groups,
performs 500 total random trials (at least ten per group), merges candidates
on up to 1500 rows and, where required, refines on all original rows. Scalar
nonnegative default-integer seeds use the existing NumPy-compatible legacy RNG.
All source search stages are present; the larger cases do not fall back to
ordinary covariance or a small-data approximation.

MinCovDet first stores raw location/covariance/support, applies the normal-model
median/chi-square consistency factor to distances, and reweights at the .975
chi-square quantile. Reweighted location/covariance, supports and training
squared Mahalanobis distances are exposed. assume_centered changes the raw
covariance after support selection and fixes reweighted/raw location at zero,
as in sklearn. store_precision controls caching; get_precision works in both
modes. Native symmetric eigendecomposition supplies Moore-Penrose precision,
including singular covariance, and native regularized-gamma evaluation and
bisection supply the chi-square quantiles. No SciPy/LAPACK runtime is needed.

Native support_fraction=0 selects automatic ceil((n+p+1)/2) support; a fraction
in (0,1] selects floor(fraction*n) rows. The resulting raw support count must
lie between one and n. At least two input rows and finite dense real64 data
are required. A zero raw covariance with a proper subset, a nonpositive or
nonfinite consistency correction, or an empty reweighting mask is rejected
explicitly. rank_ exposes the final precision rank. Singular Gaussian log
likelihood returns negative infinity, matching the source convention.

Native correction_ and reweight_threshold_ expose postprocessing diagnostics;
n_features_in_ is cached. Fitted sklearn-counterpart fields are location_,
covariance_, optional precision_, support_, dist_, raw_location_,
raw_covariance_ and raw_support_. Methods include fit, get_precision,
mahalanobis and Frobenius/spectral error_norm; MinCovDet.score returns Gaussian
log likelihood. Standalone fast_mcd returns allocated raw location, covariance,
logical support and squared-distance arrays. Sparse data, custom covariance
callbacks, random-state objects, public standalone C-step/candidate APIs and
separate correction/reweight method calls are outside this subset.

EllipticEnvelope uses the same robust fit, sets offset_ to the contamination
percentile of negative fitted distances, and exposes score_samples,
decision_function, predict and fit_predict. contamination must be in (0,.5].
Labels are +1 for nonnegative decisions and -1 otherwise. score computes
unweighted or finite nonnegative-weighted classification accuracy, while
MinCovDet.score is the Gaussian likelihood. Fitting itself does not accept
sample weights, consistent with sklearn.

Raw estimates can be nonunique at distance or determinant ties. Native support
selection uses stable ascending distance order. In the univariate shortest-half
path, two boundary observations can be exactly equidistant from the midpoint;
NumPy's unstable sort may choose a different one. Validation explicitly checks
tied boundary eligibility, selected variance/distances and consistency factor,
then still compares corrected/reweighted estimates and outlier labels directly
against sklearn. All remaining fixtures compare exact support masks, including
singular covariance with an exact zero column. Numerical equality of arbitrary
raw tied choices is not claimed for every platform or dataset.


## Bisecting K-means

`bisecting_kmeans` translates sklearn 1.6.1 BisectingKMeans's dense weighted
hierarchy. It repeatedly splits the leaf with biggest weighted inertia or
largest unweighted row count, preserving depth-first tie order. Each split
uses two centers, random or greedy k-means++ initialization, multiple restarts,
and a native Lloyd or Elkan solver. Elkan maintains per-row upper/lower bounds
and skips distance evaluations using center separation; it does not delegate
to the Lloyd path. Empty-center relocation, largest-cluster center fallback,
strict-label/tolerance stopping and final assignment updates follow the source.
The split restart selection preserves sklearn's relative 1e-6 improvement rule.

Default controls are n_clusters=8, n_init=1, max_iter=300, init='random',
algorithm='lloyd', bisecting_strategy='biggest_inertia' and tol=1e-4. Native
random_state=0 is deterministic. Initialization consumes the existing
NumPy-compatible legacy stream, including weighted no-replacement random seed
selection and the two local k-means++ candidate trials. Source bisections use
the supplied tol directly as the total squared-center-shift threshold, rather
than the variance-scaled tolerance used by ordinary KMeans. Training uses an
unweighted global mean subtraction and restores centers to original units.
With n_clusters=1, the pinned source retains the unweighted root center even
when fitting weights; native behavior follows this convention.

predict routes observations down the saved binary split tree. Final labels
are zero-based and assigned in depth-first leaf order. transform returns
Euclidean distances to every final leaf center. The inherited sklearn score
contract uses global nearest-final-center inertia, not the hierarchical route;
therefore score(x,weights) need not equal -inertia_. fit_predict returns fitted
hierarchical labels and fit_transform returns final-center distances after fit.
The native tests include a fixture where tree routing differs from global
nearest-center labels.

Finite nonnegative sample weights are supported during fit and score. Largest
cluster selection counts rows, including zero-weight rows; the other strategy
uses weighted SSE. Random initialization requires two positive-weight rows in
each selected leaf. k-means++ can handle a single positive-weight row. A leaf
selected for another split must have positive total weight. All-zero weights
are accepted for the no-split, one-cluster case and for score. Duplicate data
can create empty leaves, which remain part of the requested hierarchy, as in
sklearn. n_clusters must lie between one and the input row count. Inputs must
be nonempty finite dense real64 arrays; signed weights, sparse/float32 paths,
callable initializers, verbose/parallel controls, estimator cloning and
in-place copy_x=False semantics are outside this subset. Native inputs are
preserved.

Fitted sklearn-counterpart fields are cluster_centers_, labels_, inertia_ and
n_features_in_. Native tree diagnostics expose tree_centers_ in original units,
children_ (1-based node links, zero for leaves), tree_leaf_labels_ (-1 for
internal nodes), tree_sample_counts_ and tree_scores_. Nodes are stored in
split-creation order. Native n_iter_per_split_, split_inertia_ and
converged_per_split_ describe the best restart at each bisection; converged_
aggregates these flags. The source does not expose a global n_iter_ for this
estimator. Existing native KMeans behavior is unchanged. Floating-point ties
and parallel reductions can affect equivalent cluster choices; seeded reference
fixtures use one source OpenMP thread and compare the hierarchy explicitly.


## MiniBatchKMeans

mini_batch_kmeans implements dense sklearn MiniBatchKMeans fitting and streaming
partial_fit. fit draws training batches uniformly with replacement, applies
weighted running-mean updates, and monitors variance-scaled center motion and
exponentially weighted batch inertia. The first update is excluded from stopping
checks. max_no_improvement=0 stops at the second step; -1 disables this check.
A positive tol is multiplied by the mean population feature variance, as in the
source. A zero tolerance disables the center-motion check. n_iter_ is the ceiling
of processed batch rows divided by training rows; n_steps_ counts updates in a
64-bit integer. Iteration budgets and initialization/reassignment clock products
use wide integer arithmetic before division or capping.

Initialization supports weighted random sampling without replacement, greedy
k-means++ with 2+floor(log(K)) local trials, or explicit initial_centers. Native
n_init=0 implements sklearn 'auto': one k-means++ start or three random starts.
Multiple starts select the smallest inertia on a common sampled validation set;
they do not run complete fits before selection. init_size=0 implements automatic
sizing. A positive size smaller than K is replaced by 3*K, and the effective size
is capped by the number of rows. Initialization subsets and validation sets are
sampled with replacement. Explicit centers still consume source-compatible
initialization subsampling draws and force one start. The legacy NumPy-compatible
RandomState generator supplies seeded batch draws, weighted choices and uniform
permutations; no Python or NumPy runtime dependency is introduced.

partial_fit initializes only on its first call and updates using every supplied
row, without resampling. Its RNG state and accumulated cluster weights persist
between calls. The first batch must have at least K rows; later batches can be
smaller. fit resets centers/counts and batch/epoch counters for a new optimization.
As in the source, fit uses a local seeded RNG rather than advancing an existing
partial_fit RNG. Training labels/inertia after partial_fit describe only its
latest supplied batch. predict assigns nearest final centers, transform returns
Euclidean distances, and score returns negative weighted nearest-center SSE.

Low-count centers can be reassigned uniformly to observations from the current
batch. Reassignment is checked when a count is zero or at least 10*K effective
batch rows have elapsed. At most floor(current_batch_rows/2) centers are reset;
reset counts become the minimum retained count. reassignment_ratio=0 disables
this operation. Equal counts at the cap use increasing native cluster index;
NumPy's unstable argsort can choose another tied center. Such ties and floating-
point near-ties between duplicate centers can lead to different assignments,
optimization histories or local minima. Reference fixtures compare unambiguous
rankings; an explicit weighted fixture exercises the cap and count reset.

Finite nonnegative weights, including zeros, are supported. Random initialization
requires K positive-weight rows in its sampled subset; k-means++ needs positive
subset total weight. Explicit initialization also allows all-zero weights.
Sparse/float32 matrices, callable initializers, parallel/verbose controls,
cloning and metadata routing are outside this subset. Inputs are preserved.
Original contributions retain MIT licensing with upstream BSD notices.

Native counts_ exposes the source's private accumulated weights. converged_
records early stopping during fit, not a streaming convergence test. n_iter_
starts at zero and is set by fit; partial_fit only increments n_steps_. With
compute_labels=false, fit omits labels and estimates inertia from the smoothed
batch cost. This estimate is unavailable for a single update; the native
has_inertia_ flag is false and inertia_ is a placeholder zero, whereas sklearn
1.6.1 raises when it tries to use its uninitialized smoothed cost in this corner.
A fresh partial_fit with labels disabled similarly has no inertia. Subsequent
partial_fit calls with labels disabled retain previously available label/inertia
state, matching the source; query methods remain available. fit_predict requires
compute_labels=true and reports a deliberate diagnostic otherwise.


## SpectralCoclustering and SpectralBiclustering

spectral_coclustering implements Dhillon's paired row/column spectral partition.
Dense finite input is shifted by -min(X) when its minimum is negative, then
scaled independently by inverse square roots of row and column sums. It retains
1+ceil(log2(K)) leading singular triplets, bounded by the smaller matrix
dimension, discards the leading vectors and stacks scaled left/right vectors.
Joint centroid clustering of this embedding yields row_labels_ and
column_labels_ with shared zero-based cluster IDs. Each row and column belongs
to exactly one of K paired biclusters. K must be between two and n_rows, with
at least one singular vector remaining after the leading vector is discarded.
The source accepts K=1 at parameter validation but fails to fit its zero-feature
embedding; the native code reports this case deliberately.

spectral_biclustering implements Kluger's checkerboard algorithm with scale,
bistochastic and log normalization. Scale uses the same independent margins.
Bistochastic repeatedly applies scale normalization, stopping at a Frobenius
change below 1e-5 or 1000 iterations, as in the source dense helper. Log shifts
input to a minimum of at least one, takes logs, subtracts row and column means,
and adds the overall mean. Scale/bistochastic discard the leading singular
vectors; log retains them. Singular vector counts are capped by matrix size,
including requests beyond the smaller dimension. n_best is positive and cannot
exceed configured n_components; the actual number selected is capped by the
number of retained vectors.

Each left and right singular vector is clustered in one dimension to obtain a
piecewise constant approximation. The n_best vectors with the smallest Euclidean
approximation errors are selected independently on each side. The original,
unshifted training matrix is projected onto selected right vectors to cluster
rows, and its transpose onto selected left vectors to cluster columns. The
result is a Cartesian grid of K_row*K_column blocks, in row-group-major order.
Each row belongs to K_column blocks and each column to K_row blocks. Native
n_row_clusters/n_column_clusters override scalar n_clusters when both are
positive; leave both zero to use the scalar. Row counts are bounded by n_rows;
column counts are bounded by both n_rows (the source parameter rule) and the
available columns (the subsequent centroid fit rule).

Both estimators use a full native thin Jacobi SVD, factoring the transpose for
wide matrices. Zero left singular columns are completed to an orthonormal basis
with twice-applied Gram-Schmidt projections, completing only requested vectors.
The basis search stops at a well-conditioned coordinate; its scans are bounded
by the current basis rank instead of the number of rows. An arbitrary null-space choice is
unavoidable for rank-deficient data. This is an exact dense backend, without
randomized/ARPACK selection or n_svd_vecs controls. Repeated singular values,
null-space bases, ties at the singular-vector cutoff, piecewise-error ties and
floating-point centroid ties can yield different partitions or local minima
from Python. Validation uses source randomized SVD with enough oversampling to
span the complete smaller dimension. Singular vector signs are aligned; cluster
IDs are compared after relabeling, with paired IDs preserved for co-clustering
and independent row/column IDs for checkerboards.

Centroid fitting uses source-style greedy k-means++ or weighted-uniform random
initialization, scalar-seeded legacy NumPy RNG draws, repeated Lloyd fits,
variance-scaled 1e-4 stopping, a 300-iteration cap, farthest-point empty-center
relocation and source same-partition restart selection. mini_batch=true uses
the existing native MiniBatchKMeans with the wrapper's init/n_init/seed and its
usual defaults. Each vector or projection fit starts with the configured scalar
seed, matching the source wrapper. Existing public KMeans and MiniBatchKMeans
bodies are unchanged. Empty-center relocation resolves exact distance ties by
native first index; the source's partition-based ordering may choose differently.
Explicit centroid arrays and callable initialization are outside this subset.

Fitted source counterparts are row_labels_, column_labels_, rows_, columns_
and n_features_in_. rows_ and columns_ have shape (n_biclusters,n_rows) and
(n_biclusters,n_columns). get_biclusters returns copies of this pair, matching
the source biclusters_ property. get_indices/get_shape/get_submatrix take a
zero-based block ID. Returned row/column positions are one-based Fortran indices;
submatrices preserve input values and can have empty dimensions. get_submatrix
requires a finite dense matrix with the fitted training shape; Python-style
negative block indexing is unsupported. There is no inductive predict/transform
method in these source estimators.

Native diagnostics include n_rows_, n_biclusters_, normalized_data_, leading
singular_values_ (including the discarded leading value), retained left_vectors_
and right_vectors_, n_normalization_iter_ and normalization_converged_. A
nonconverged 1000-step normalization still feeds the fit, as in sklearn.
Co-clustering additionally exposes margin scalings, embedding_, centroid
inertia_ and n_iter_. Checkerboard diagnostics include piecewise approximation
errors, selected zero-based indices into retained vectors, and row/column
projections. Sparse/float32 inputs, estimator cloning and metadata routing are
outside this subset. Finite dense inputs with negative values are supported
through the source shifts. Shift/margin overflow and nonpositive margins for
scale/bistochastic normalization receive deliberate diagnostics. Log supports
constant matrices through a completed null-space SVD basis. Original
contributions retain MIT licensing with full upstream BSD notices.


## BayesianGaussianMixture

bayesian_gaussian_mixture implements sklearn's variational Gaussian mixture
updates with dirichlet_process (default) and dirichlet_distribution weight
priors, and full/tied/diag/spherical covariance forms. It estimates Beta stick
or finite-Dirichlet concentrations, Gaussian mean precision, Wishart degrees of
freedom, posterior means/covariances and Cholesky precision factors. Responsibilities
use expected log weights and expected log precision, including mean uncertainty.
score_samples returns the source variational log score, and score its mean;
these are not ordinary Gaussian-mixture densities obtained by taking log(weights_).
The simplified source lower bound, with its constant terms removed, drives
convergence and restart selection. lower_bound_ is a total objective, rather
than the per-row score returned by score.

All four source initializers are supported: kmeans, random (row-normalized
uniform responsibilities), random_from_data (uniform seeds without replacement),
and k-means++ (greedy data seeds). Scalar-seeded legacy NumPy RNG state continues
across initialization attempts. K-means initialization uses one native Lloyd fit
with the source defaults and a shared centroid core. That core was extracted
from the existing biclustering implementation without changing its update math;
all 2,992 biclustering reference groups were rerun. Public KMeans, MiniBatchKMeans,
GaussianMixture and the RNG helper bodies are unchanged.

Native weight_concentration_prior=0 selects the source None default 1/K;
positive values select an explicit prior. mean_precision_prior defaults to one,
the resolved source None value. degrees_of_freedom_prior=0 selects p; explicit
values must exceed p-1. Allocate mean_prior(p) for an explicit mean. Allocate
covariance_prior(p,p) for an explicit covariance: full/tied require a finite
symmetric positive-definite matrix; diag requires a positive diagonal matrix;
spherical requires a positive scalar multiple of the identity. This expands the
source's vector/scalar representations into a consistent Fortran matrix. Without
an explicit covariance prior, the unbiased sample covariance/variances (ddof=1)
are used, with spherical variance equal to their mean. A default prior can be
singular, as in the source, provided regularization yields positive-definite
posterior covariances. Nonfinite, incompatible or indefinite explicit priors
receive deliberate diagnostics.

n_init fits independent initializations using one continued RNG stream and
retains the largest final bound. tol is an absolute bound change; zero disables
tolerance stopping. max_iter=0 performs initialization only, retains the last
initialization when n_init>1, sets n_iter_=0/converged_=false/lower_bound_=-infinity,
and leaves an empty bound history. Posterior queries remain available. History
storage grows with actual iterations, rather than the configured budget; an
int64 loop counter supports large budgets with early stopping.

With warm_start=true, a previously fitted model continues from its posterior
and previous bound, using one run and recomputing default priors for the current
data. The first warm-start fit still uses n_init attempts, following source
behavior. Changing component count, feature count, covariance form or weight
prior family requires a fresh fit with warm_start=false. Fresh fits may change
feature/component counts and covariance form; explicit prior arrays must be
reset to compatible shapes. Queries reject changes to fitted component/covariance/
weight-family controls until refit. Fitted attributes otherwise follow the
usual read-only convention.

weights_, means_, mean_precision_, degrees_of_freedom_, weight_concentration_,
covariances_, precisions_, precisions_cholesky_, n_iter_, converged_, lower_bound_
and n_features_in_ expose source counterparts. Native covariance/precision
arrays always have shape (p,p,K): tied matrices are repeated, and diag/spherical
forms have zero off-diagonal entries. degrees_of_freedom_ repeats the tied scalar
K times. weight_concentration_ has shape (K,2) for Beta (a,b) parameters or (K,1)
for a finite Dirichlet vector. Resolved covariance_prior_ is likewise a full
matrix. This consistent layout uses O(K*p*p) storage even in diagonal/spherical
modes; their factorization avoids cubic work, and tied factorization is shared.
Additional diagnostics are lower_bounds_ for the selected run and per-component
log_det_precision_. The best-run history is checked against source per-iteration
bounds; its final value matches lower_bound_.

sample(n_samples,x,labels) draws from weights_, means_ and normalized
covariances_, matching the source's Gaussian sampling distribution and grouping
rows by zero-based component ID. It does not draw Student-t predictive samples.
Each call restarts the configured scalar RNG seed. Native multinomial counts
use independent categorical draws and full covariance sampling uses Cholesky
roots, so the samples do not match NumPy's seeded multinomial/SVD stream bit for
bit. Determinism, grouped labels, component frequencies and Gaussian-mixture
first/second moments are verified separately from exact posterior comparisons.

Native digamma uses positive-argument recurrence and a high-order asymptotic
series; log-Gamma uses the Fortran intrinsic. No SciPy/Python runtime dependency
is introduced. Fit requires at least two nonempty finite dense real64 rows,
with 1<=K<=n_rows. tol/reg_covar are finite and nonnegative; scalar priors must
be finite and valid, and extremely small digamma arguments that cannot have a
finite real64 result are rejected. Sparse/float32 matrices, sample weights,
verbose output, cloning and metadata routing are outside this subset. Original
contributions retain MIT licensing, with sklearn/NumPy attribution preserved.


## HDBSCAN

hdbscan implements the dense sklearn 1.6.1 brute backend: self-including core
neighbor distances, mutual reachability, Prim linkage, condensed-tree pruning,
excess-of-mass (eom) or leaf selection, membership strengths and DBSCAN cuts.
min_samples=0 resolves to min_cluster_size; max_cluster_size=0 means no cap.
min_cluster_size must be at least two but may exceed the sample count when an
explicit smaller min_samples is supplied. At least two finite rows are needed.
max_cluster_size affects EOM selection; leaf selection ignores it. Selection
epsilon can merge clusters beyond the size cap, following source behavior.
allow_single_cluster controls root selection, with source root-lambda filtering.

Metrics are euclidean, sqeuclidean, manhattan/cityblock, chebyshev, minkowski
(p>=1), cosine and precomputed. The fixed brute backend divides the entire
distance matrix by alpha before computing core distances, matching source brute
semantics; the source auto/tree backend can behave differently when alpha!=1.
Core order statistics use quickselect. Dense mutual reachability replaces the
distance workspace in place; this O(N**2) workspace is freed after building the
hierarchy, before center computation. Inputs are preserved. Precomputed matrices
must be symmetric, nonnegative and square; positive infinite distances are
accepted. Their supplied diagonal is preserved. Alpha scaling that would overflow
a finite distance is rejected explicitly.

Finite feature rows are clustered. Rows with infinity receive label -2 and
strength zero; rows with NaN receive label -3 and NaN strength. A row containing
both infinity signs is classified as missing (-3), matching source semantics
without invalid arithmetic. Ordinary noise is -1 with strength zero. Occupied
clusters use contiguous labels starting at zero, and n_clusters_ counts occupied
clusters. selected_clusters_ may also contain an unoccupied selected root.
Finite-row membership strengths lie in [0,1]. Degenerate zero-distance lifetimes
can produce infinite or NaN raw stabilities_, explicitly represented even with
floating-point traps enabled.

store_centers accepts none, centroid, medoid or both. Centers use membership
strengths as weights, with the first point breaking medoid ties. Centers cannot
be requested with precomputed distances. A refit clears unrequested center
attributes; no clusters gives a (0,p) center matrix. Native centers handle
excluded nonfinite rows, correcting the source 1.6.1 mask-shape failure. Native
raw-tree remapping also handles exactly two finite rows among excluded rows,
using the known finite root rather than the source's last-raw-point assumption.

dbscan_clustering(cut_distance,min_cluster_size) uses strict height<cut_distance;
its optional minimum defaults to five independently of the fitted minimum.
Positive infinite cuts are supported; nonfinite row markers are preserved.
children_ has (raw_N-1,2) one-based child IDs: leaves 1..raw_N and merge node
raw_N+i for row i. distances_ and cluster_sizes_ describe this full tree,
including excluded leaves appended at infinite height. core_distances_, condensed
and stability arrays, and selected cluster IDs refer to the finite subset:
finite_indices_ maps its leaves to raw rows, and its condensed root is finite_N+1.
stabilities_ indexes consecutive cluster IDs starting at that root. mst_edges_
records raw point IDs in the brute Prim visitation path; mst_distances_ records
frontier keys, which need not equal distances between the recorded endpoints.

Native MST and equal-height sorting use deterministic first-index/stable ties.
Source NumPy sorting can order equal heights differently. In 120 reference
configurations, 114 direct fits match source labels and strengths. Six differ
on single-cloud or cosine plateaus/roundoff ties. All 120 match source linkage
ultrametrics and sorted heights; source tree routines applied to each native
tied hierarchy match its selection, labels, strengths and condensed tree. Thus
no universal bitwise or label identity is claimed at tied linkage heights.

The implementation is real64 CPU dense only. Tree/automatic backends, sparse
inputs, callable metrics, parallel controls and in-place copy=False are absent.
There is no inductive prediction method. See compare_hdbscan.py for the direct
and hierarchy-conditioned checks and test_hdbscan.f90 for analytic fixtures.


## BernoulliRBM

bernoulli_rbm implements sklearn 1.6.1 Bernoulli restricted Boltzmann machines
with dense binary visible and hidden units. fit uses stochastic maximum
likelihood (persistent contrastive divergence), retaining a hidden Markov chain
of batch_size rows. Training uses consecutive unshuffled batches and performs
one visible/hidden Gibbs transition per update. The final positive batch may be
shorter than the persistent negative chain; both gradients are divided by the
actual positive batch length, matching source behavior. A batch larger than the
dataset likewise retains its full negative chain. n_iter=0 initializes only.

components_ has shape (n_components,n_features_in_), intercept_hidden_ and
intercept_visible_ are vectors, and h_samples_ has shape
(initial_batch_size,n_components). transform returns hidden conditional means;
fit_transform performs a fresh fit then transforms. partial_fit initializes on
its first call and otherwise continues one update on the entire supplied array,
without splitting it and independently of n_iter. Changing batch_size between
partial calls preserves the existing chain size, following source. Changing
n_components during partial_fit is rejected explicitly; use fit to reset shapes.
Fresh fit supports changed data dimensions and component/chain counts.

gibbs samples one hidden then visible transition and returns real64 zeros/ones.
It advances a private sampling stream shared with partial_fit. A scalar-seeded
fit uses a separate local RNG and resets model arrays without resetting an
already-created partial_fit/Gibbs stream, matching the source integer-seed
contract. Thus a first partial_fit or Gibbs call after fit starts from the
original seed rather than the end of the fit RNG. Assigning a fitted Fortran
object copies its private stream, permitting reproducible continuation.
score_samples computes stochastic pseudo-likelihood from stable free energies,
flipping one randomly chosen feature per row. With scalar integer seeds it
recreates the scoring RNG each call, so repeated equal queries give equal
scores and do not advance the sampling stream. This is pseudo-likelihood,
not normalized log likelihood. Hidden logits and softplus/free-energy
calculations avoid exponential overflow, retaining small softplus tails.

The existing legacy MT19937/uniform/Gaussian/bounded-integer helper reproduces
NumPy RandomState draw order, including row-major logical array draws followed
by Fortran storage. All 80 reference configurations match source parameters,
persistent binary chains, transforms, scores and Gibbs samples; numeric
comparisons use atol=rtol=2e-10, binary samples/metadata are exact. Tests cover
mixed fit/partial/Gibbs calls, changed seeds/rates, fresh refits, zero iterations,
short/oversized/full batches, fractional and signed finite values and extreme
query logits. Analytic native tests check conditional probabilities, uniform
pseudo-likelihood, a one-step PCD gradient, saturated logits and object copies.

Inputs are dense finite real64 CPU arrays. Bernoulli modeling expects binary
values (or probabilities in [0,1]); as in source, finite values outside that
range are accepted without range checks. Sparse/float32 inputs, RandomState
objects, unseeded global RNG behavior, verbose training logs, feature-name
metadata and automatic classification are absent. random_state defaults to the
native deterministic seed zero and must be nonnegative. learning_rate must be
positive and finite; batch_size/n_components positive; n_iter nonnegative.
No inverse transform or prediction method is supplied by the source estimator.


## LatentDirichletAllocation

latent_dirichlet_allocation implements dense single-worker sklearn 1.6.1 batch
and online variational Bayes topic modeling. Inputs are nonnegative finite
real64 document/word count matrices; fractional counts and empty documents are
accepted. components_ is (n_components,n_features_in_), containing topic-word
Dirichlet pseudocounts. Normalize each row to obtain topic-word means.
exp_dirichlet_component_ stores exp(E[log beta]), which is not a normalized
probability matrix. transform returns document-topic means, or unnormalized
gamma parameters with normalize=false. fit_transform supports both forms.

Native doc_topic_prior=-1 and topic_word_prior=-1 resolve independently to
1/n_components. Explicit priors must be in (0,1]. Zero priors, although accepted
by source parameter validation, define improper Dirichlet distributions and are
rejected here. The E-step matches the source AS 103 positive-digamma approximation,
self-consistency updates, machine-epsilon denominator and mean absolute change
stopping test. max_doc_update_iter=0 preserves initial gamma values: deterministic
ones for inference and seeded Gamma draws for training. max_iter=0 initializes
model parameters and computes fitted perplexity without an EM update.

fit resets latent parameters and its private scalar-seeded RNG. Batch mode uses
one full-data update per epoch; online mode uses consecutive unshuffled chunks,
weight=(learning_offset+n_batch_iter_)**(-learning_decay), and the actual corpus
size for scaling. partial_fit always uses online updates, regardless of
learning_method, splitting the supplied data by batch_size and scaling to
total_samples. It preserves its RNG and resolved priors; changed seed/prior
controls take effect on a fresh fit. Changed component counts require fit;
changed batch size, decay, offset and total_samples affect future online steps.
Copying an object copies its arrays and stream. Inference and scoring use
constant initialization, are deterministic and do not consume training draws.
The private shape-100 Gamma sampler follows NumPy's legacy rejection algorithm
and existing MT19937/Gaussian helper without adding a runtime dependency.

score returns the full variational evidence bound. perplexity returns
exp(-bound/word_count), optionally using population subsampling compensation;
the topic prior contribution is included only once after scaling document terms.
bound_ stores fitted training perplexity, not the evidence bound. It is NaN for
partial-fit-only models and is not refreshed by partial_fit after fit.
evaluate_every>0 checks absolute perplexity changes against perp_tol;
nonpositive values disable evaluation. n_batch_iter_ starts at one and counts
actual EM updates plus one using int64. n_iter_ follows source behavior: it
excludes the epoch triggering early stopping and is not incremented by streaming.

Native Dirichlet bounds algebraically remove large linear terms using a stable
log-Gamma/digamma remainder and a Stirling expansion for arguments >=16. Exact
one-dimensional and prior-equal distributions contribute zero. This avoids
cancellation when online updates represent very large populations. Infinite
perplexity is returned explicitly on exponential overflow. Zero-word/zero-evidence
perplexity is explicitly NaN, including roundoff-level prior equality; source
can return NaN, zero or infinity from its rounded 0/0 calculation. Zero-word
queries with a strictly negative bound return infinity. Native perplexity
validates the fitted feature count and preserves metadata, correcting the source
1.6.1 reset_n_features=True path that can accept truncated feature arrays.

The 120 reference configurations check 6,734 result groups: 6,627 ordinary
source comparisons at atol=rtol=3e-8 (metadata exact), 26 source score comparisons
with a log-Gamma roundoff budget, 26 independent 60-digit mpmath evidence-bound
certificates at the usual tolerance, and 55 explicit zero-word definition checks.
The roundoff budget is 32*eps times the summed magnitudes of log-Gamma component
and row-total terms, propagated through log perplexity. Observed bound-scale
source/native differences reach 4.19e-5 for large populations; high-precision
certificates validate the stable native result. No identity is claimed for
undefined source perplexity. Fixtures cover both learning modes, streaming,
priors, fractional/zero counts, unused columns, tiny priors, zero inner/outer
iterations, early stopping, huge populations, single words/topics, more topics
than documents, dimensional refits and mixed training/inference calls.

The backend is dense real64 CPU with one worker. Sparse/float32 input, n_jobs,
verbose logs, object/global RNGs, feature-name metadata and persistence are
outside scope. random_state defaults to deterministic zero. There is no inverse
transform or inductive classification contract. CountVectorizer output can be
passed directly as the document/word matrix. mpmath is a validation dependency
only; no additional Fortran runtime dependency or public extension is introduced.


## MiniBatchDictionaryLearning and MiniBatchSparsePCA

mini_batch_dictionary_learning implements sklearn 1.6.1 dense online dictionary
learning: LARS/CD Lasso codes, sufficient-statistic forgetting, constrained atom
updates/resampling and source minibatch convergence logic. n_components=0 selects
the feature count. A supplied dict_init is truncated or zero-padded to this
count; otherwise initialization follows source randomized SVD. Initial atoms
are singular values times right singular vectors, without premature unit
normalization. Constraints apply after each update: optional nonnegative atoms
and row norms <=1. Unused atoms (A(j,j)<=1e-6) are replaced by a random batch row
plus Gaussian noise of 0.01 times its population standard deviation (or 0.01
for constant rows). Positive training codes require the CD solver.

fit shuffles rows once before cycling fixed consecutive batches across epochs.
The final short batch uses its actual length in both the cost and forgetting
schedule. Statistics a_ (K,K) and b_ (p,K) correspond to source private _A/_B.
partial_fit performs exactly one update on the entire supplied matrix,
independently of batch_size/max_iter, and continues the cached RNG/statistics.
The initialized fit algorithm is retained across partial calls, as in source;
changed algorithm/initializer/seed controls take effect on fresh fit. Changing
the component count during partial_fit is rejected explicitly rather than
silently ignoring it. A fresh fit resets all state and supports new dimensions.
Input arrays and supplied initializers are preserved; source can alias and
mutate a writable initializer. Copies of native objects copy their streams.

max_no_improvement=-1 disables patience, while zero stops at the first eligible
smoothed-cost check. tol=0 disables dictionary-motion stopping. Source warmup
uses min(100,N/actual_batch_size) steps, followed by an exponentially weighted
cost with weight actual_batch_size/(N+1). Dictionary motion is its Frobenius
norm divided by K. n_steps_ counts updates; fit sets n_iter_ to the ceiling of
steps/steps_per_epoch. Streaming does not refresh n_iter_ or convergence state;
partial-only models expose n_iter_=0. ewa_cost_/ewa_cost_min_ use NaN for source
None state, and no_improvement_ records the patience counter. Counters/budgets
use int64, without allocating histories proportional to max_iter.

transform supports lasso_lars, lasso_cd, lars, omp and threshold, with existing
sparsity/penalty controls and optional split_sign output (positive then negative
parts, yielding 2*K columns). transform_alpha=-1 defaults to training alpha;
transform_n_nonzero_coefs=0 selects source-style automatic sparsity.
fit_transform fits then recodes the entire input using transform settings.
No inverse-transform method is supplied by the source dictionary estimator.

mini_batch_sparse_pca centers features, learns a mini-batch dictionary on X.T,
recodes X.T and normalizes nonzero component rows. Its defaults match source,
including batch_size=3 and tol=0.001. transform uses a ridge/pseudoinverse least
squares projection onto the sparse components; inverse_transform reconstructs
and restores mean_. Zero components and ridge_alpha=0 are supported. There is
no source partial_fit contract for this wrapper. n_steps_ is a native diagnostic
in addition to source n_iter_, n_components_ and n_features_in_.

The private randomized SVD core follows source defaults: ten oversamples,
seven power iterations when K<0.1*min(shape) and four otherwise, automatic
transposition for wide matrices, pivoted-LU normalization, final economic
Householder QR and a thin Jacobi SVD of the projected matrix. Sign selection
uses the original left singular vectors. Logical row-major Gaussian draws and
subsequent RNG state match NumPy RandomState. Null/tied singular subspaces can
have different bases; tests compare meaningful nonzero directions and
reconstruction, without claiming arbitrary null-vector identity. No randomized
SVD tuning parameters are added to the estimators, which use source defaults.

The existing batch DictionaryLearning atom update and SparsePCA ridge projection
were extracted into shared private helpers without changing their arithmetic.
Their existing 744 reference groups and 28 separate NumPy RNG groups are rerun
for this refactor. New validation covers 3,468 groups across 200 configurations:
100 dictionary fits/streams, 68 SparsePCA fits and 32 randomized-SVD cases over
four seeds, including default training/solver controls with fixed scalar seeds.
Source comparisons use
atol=rtol=3e-8, with exact counters and subsequent RNG values; orthogonality
certificates additionally check sampled singular bases. Cases include all coding
methods, positivity, padding/truncation, dead atoms, constant data and oversized/unit/short
batches, zero/huge iteration budgets, patience/motion stopping, changed cached
controls, dimensional refits, constant features, zero ridge and overcomplete
models. The backend is dense real64 CPU; n_jobs, callbacks, verbose output,
object/global RNGs and float32 arrays are not implemented. Nonnegative scalar
seeds are supported, with native random_state=0 by default. No runtime dependency
or public algorithm extension is introduced.


## MiniBatchNMF

mini_batch_nmf implements dense sklearn 1.6.1 MiniBatchNMF with multiplicative
beta-divergence updates, optional L1/L2 penalties, persistent numerator/denominator
statistics, fresh coefficient restarts and minibatch early stopping. Native
beta_loss is real: 2, 1 and 0 correspond to source frobenius, kullback-leibler
and itakura-saito aliases; other finite real values use the general beta loss.
The MM exponent is 1/(2-beta) below one, 1/(beta-1) above two and one otherwise.
Updates preserve the source float32 epsilon (2**-23) denominator/power safeguards,
float64 small-factor cutoffs and loss masking, even for real64 data.

n_components=0 maps to source 'auto': infer the row count of custom H, otherwise
the input feature count. init='auto' maps to None, selecting nndsvda when K fits
within min(shape), otherwise random initialization. random uses scaled absolute
Gaussian draws for H then W in logical NumPy row-major order. nndsvd/nndsvda/
nndsvdar reuse the existing default randomized-SVD core, split positive/negative
singular directions, and threshold values below 1e-6. The latter two fill zeros
with the data mean or small absolute Gaussian values. NNDSVDAR restarts its
scalar-seeded RNG for filling, matching source integer-seed behavior. Null
singular directions are handled as zero factors before filling, avoiding invalid
normalization with the native SVD's zero null columns. Explicit custom W/H must
match (N,K)/(K,p), be finite/nonnegative and each contain a positive value. Native
copies preserve supplied arrays. Other init modes ignore optional W/H as source
does. NNDSVD requires K<=min(shape); random/custom may be overcomplete.

fit/fit_transform use fixed consecutive batches without shuffling. Coefficient
rows persist between visits unless fresh_restarts=true, which solves each batch
from its deterministic mean-based start and performs a final full-data coefficient
solve. fit_transform returns the fitted training factors; these need not equal a
subsequent transform's freshly optimized factors. alpha_w is nonnegative;
alpha_h=-1 means 'same as alpha_w'. Penalties scale by feature count for W and
actual batch size for H, with l1_ratio in [0,1]. Batch costs follow the source's
penalty scaling, and reconstruction_err_ excludes regularization and equals
sqrt(2*beta_divergence), with source nonnegative rounding clamping.

The cached forgetting weight is forget_factor**(clamped_batch_size/initial_N).
Partial updates reuse this weight even when later array sizes or controls change.
partial_fit performs one update on the entire supplied array and always solves
fresh coefficients. Initial custom W is validated then discarded for this path,
as in source; the native implementation avoids retaining an unused coefficient
batch. Cached component count, beta/MM exponent, forgetting weight, batch size
and transform iteration limit remain from initialization; fresh fit resets them.
Regularization, tolerance and fresh-restart solve limits are read for each call.
Copies of an object copy its accumulated statistics and reproduce continuation.

transform starts from sqrt(mean(X)/K) and solves W against fixed components_.
transform_max_iter=0 selects max_iter at initialization; positive values override
it. inverse_transform is the linear product of finite coefficient rows and H.
components_numerator_/components_denominator_ correspond to source private online
statistics. n_steps_/n_iter_ use int64; fit sets n_iter_ to the ceiling of steps
per epoch, while partial_fit leaves it and reconstruction_err_/convergence state
stale. Partial-only models expose n_iter_=0 and NaN error/EWA state where source
attributes are absent. The first fit step skips convergence monitoring, then
EWA cost, relative H motion and patience follow source rules. tol=0 disables
motion stopping; max_no_improvement=-1 disables patience, while zero is invalid
for this estimator. Zero-norm motion cannot trigger convergence, matching source
NaN comparisons without floating-point traps. Large iteration budgets allocate
no histories proportional to the budget.

Inputs for fitting/transformation must be nonempty, finite and nonnegative;
nonpositive beta requires strictly positive X. Native applies this domain check
consistently to streaming and transformation, including source paths that omit
it. max_iter/fresh_restarts_max_iter must be positive. forget_factor lies in
[0,1], and zero/one endpoints are supported. Only nonnegative scalar seeds are
supported, with native default zero. Sparse/float32 arrays, global/object RNGs,
verbose output, feature-name metadata and persistence are outside this scope.
No existing NMF or randomized-SVD implementation is changed by this addition.

compare_minibatch_nmf.py directly checks 6,500 source result groups across 164
configurations and four seeds, including default training controls. It compares
factors/components, both online accumulators, fitted/resolved state, convergence
counters, transforms, inverse transforms and reconstruction errors at
atol=rtol=3e-8, with exact metadata. Fixtures cover losses from -0.5 through 3,
all five initializers plus automatic choice, custom auto rank, penalties,
forgetting endpoints, short/unit/oversized batches, fresh restarts, huge-budget
early stopping, streaming cached-control changes, dimension refits, tiny/sparse
values, constant zeros and wide inputs. Native tests independently verify exact
rank-one Frobenius/KL/IS fixed points, analytic accumulated-statistic recurrences,
object copies, input preservation, zero-norm patience and null NNDSVD handling.


## MultiTaskElasticNetCV and MultiTaskLassoCV

multi_task_elastic_net_cv and multi_task_lasso_cv select joint multi-response
penalties with sklearn 1.6.1-style alpha paths, fold errors and cold full-data
refitting. Targets remain a rank-two (N,T) matrix, including T=1. Folds center
features/targets using training means only, warm-start coefficients along the
descending alpha path, restore the fold intercept and average squared errors
over test rows and targets. Selection averages folds equally, even when sizes
differ, and retains the first minimum in supplied ratio/descending-alpha order.
Fresh fits always reset path models and cold-refit the selected penalty.

Scalar l1_ratio defaults to .5 for ElasticNetCV; allocate l1_ratios(:) to supply
several ratios in preference order. LassoCV fixes the ratio to one and ignores
inherited ratio controls. Explicit alphas(:) are finite/nonnegative, sorted
without removing duplicates, and shared across ratios. Otherwise n_alphas
(default 100) values are generated from full-data centered group correlations,
with alpha_max=max_j ||X(:,j).T Y||_2/(N*l1_ratio) and a geometric range controlled
by eps (default 1e-3). Grids below 1e-15 collapse to that resolution floor.
A zero mixing ratio requires explicit alphas; zero alphas are accepted by both
CV and base multi-task estimators. The base kernel now supports [0,1] mixing,
including pure L2. As in source, zero L1 strength can make the L1-feasibility
dual-gap certificate insensitive to an already accurate least-squares/Ridge
solution, so n_iter_ may reach the limit; analytic normal-equation cases are
also tested. tol remains finite/nonnegative and max_iter positive.

cv defaults to five nonshuffled KFold splits, with remainder rows in the first
folds. Allocate test_fold(N) for PredefinedSplit semantics instead: -1 means
training-only; other integer labels, including negative labels, are sorted into
folds. This path permits a single holdout and ignores the integer cv control.
Each fold must have nonempty train/test sets. General Python splitter objects,
iterables of arbitrary train/test pairs, shuffle/group/time-series CV adapters,
sample weights, parallel jobs, metadata routing and sparse/float32 inputs are
outside this wrapper's current scope. Input arrays and supplied grids/fold maps
are preserved. CV prediction and uniform-average multi-target R2 are inherited.

Native alphas_ is always (n_alpha,n_ratio), and mse_path_ is always
(n_alpha,n_fold,n_ratio), retaining singleton axes that source squeezes.
fold_n_iter_ and fold_dual_gaps_ expose corresponding per-solution diagnostics,
with gaps divided by each fold's training count. fold_labels_, fold_test_sizes_
and n_splits_ describe validation partitions. alpha_, l1_ratio_ and best_mse_
report selection; coef_, intercept_, dual_gap_, eps_, n_iter_, n_features_in_
and converged_ describe the full-data refit. all_cv_converged_ is native and
checks every fold/path dual gap. Inherited warm_start does not warm CV refits.
For positive eps>1, source's ascending stored grid/descending solver-path
association is reproduced; conventional eps<=1 gives the intended decreasing
regularization range. Native storage does not squeeze dimensions.

Both CV types and both base estimators support selection='cyclic'/'random'.
Random mode samples features with replacement. A NumPy-compatible scalar RNG
draws the source 31-bit seed for each coordinate solve, followed by sklearn's
uint32 XorShift update and feature modulo. CV resets the scalar RNG per fold
and ratio but advances it between alpha solves; final refitting uses a fresh
scalar-seeded stream. Base warm fits similarly reseed each call. Native
random_state defaults to zero; object/global RNGs are not exposed. Cyclic update
arithmetic and existing warm-start/dual-gap behavior remain unchanged.

compare_multitask_cv.py checks 7,960 groups over 272 seeded scenarios: 192 CV
and 80 base-model cases. Of these, 7,957 compare directly with source, including
exact fold/refit iterations, and three records describe the same zero-alpha
selection whose inactive ratio differs (native .5, source 0) from last-bit CV
error ties. Both effective penalties are identically zero, the selected ratio
belongs to the supplied candidates, and final parameters/predictions and all
path errors/iterations/gaps remain source comparisons. No active-penalty
selection differences are accepted. Numerical tolerance is atol=rtol=3e-8;
metadata is exact. Source paths are observed while retaining their computation.
Fixtures include scalar/vector ratios, explicit/auto/duplicate/zero penalties,
random single sweeps and warm fits, predefined and leave-one-out folds, single
rows per holdout, zero/constant/tiny targets, zero/duplicate features, one target,
wide designs, repeated dimension changes and source defaults. Native tests
independently verify analytic fold MSEs, selected/full-data coefficients, exact
ties, holdout semantics, zero/L2 solutions and zero-target huge-budget stopping.
The existing 192 multi-task reference groups are rerun for the shared kernel.


## OrthogonalMatchingPursuitCV (2026-11-20)

`orthogonal_matching_pursuit_cv` is available from `sklearn_linear_model`,
`sklearn_omp_cv` and the facade. Dense real64 X has at least two features;
y is one real target per row. It implements the installed sklearn 1.6.1
forward Cholesky OMP path, fold-local centering, path truncation to the shortest
fold, equally weighted fold mean squared errors, first minimum sparsity
selection, and cold full-data refitting. The refit uses a Gram matrix when
N>P, matching the source auto-precompute branch; fold paths use direct X.
Input arrays are preserved. No SciFort or other runtime dependency is added.

Controls are `fit_intercept` (true), `cv` (5), and `max_iter` (0 denotes the
source default/zero behavior: min(max(P/10,5),P)). Positive max_iter is a cap
on candidate active features and must not exceed P. Outputs include `coef_`,
`intercept_`, `n_nonzero_coefs_`, `n_iter_`, `n_features_in_`, `n_splits_`,
`best_mse_`, `mse_path_(step,fold)`, `fold_n_iter_`, `fold_train_sizes_` and
`fold_test_sizes_`. There is no zero-feature candidate. `predict` and `score`
provide ordinary linear prediction and R2 through the existing metric helper.

Unallocated partition controls give contiguous, unshuffled KFold with larger
first folds for remainders. `test_fold(:)` gives PredefinedSplit labels, with
-1 reserved for training-only rows and all other labels sorted, including
negative labels. A single predefined holdout is supported. Alternatively,
allocate both `train_mask(N,F)` and `test_mask(N,F)` to represent arbitrary
partitions, including time gaps, rolling windows, groups and shuffled holdouts.
Each column is a fold; true marks included rows. The two mask arrays must have
matching nonzero fold counts and N rows. Every training and test set must be
nonempty. Masks and test_fold are mutually exclusive. Masks consume rows in input order and cannot encode permutations or
repeated copies of a row; Python splitter objects/metadata routing and parallel
execution are not implemented. Source copy flags are replaced by preserving
all inputs. Sample weighting and multi-output targets are outside this API.

Exact dependence and residuals at machine precision can change coefficient
representations or unused tail iterations between BLAS/LAPACK and native
arithmetic. This is recorded separately from direct source equality. The
reference suite has 1,936 groups across 88 configurations (four seeds, two
successive fits each), with 1,921 direct comparisons, one certified equivalent
coefficient representation, and 14 path-tail counter differences. The latter
require the same shortest path length; all affected folds have saturated their
training rank or belong to the explicitly verified exact sparse-signal fixture.
All retained fold errors, selected sparsities, final iteration counts, fitted
predictions and query predictions still compare to source. The coefficient
exception requires rank deficiency and zero training/query action of the
coefficient difference. Queries in that fixture obey the same feature relations;
outside the data's affine span, nonunique representations may predict differently.
Both checked and release comparisons passed. The optimized release report
has 1,922 direct source checks and 14 unused tail-counter differences, with no
coefficient exception. A zero-step fold (for example centered constant targets or all constant X) is
rejected explicitly; the source CV reduction likewise cannot select from an
empty path. This differs from the standalone base OMP estimator's zero model.

The analytic native test checks known coefficients, first-step and selected
fold errors, common early stopping, exact sparse signals, repeated shorter fits,
input preservation, single predefined holdouts and arbitrary training windows.
The reference matrix also covers defaults with P>50, max_iter=1/2/P, unequal
folds, leave-one-out, two features, wide matrices, duplicate/constant/dependent
columns, near dependence, scaling, noiseless targets, grouped/time-series/
shuffled partitions and changing fitted dimensions. Thirty new rejection cases
cover data/query shapes and finiteness, fold controls, masks and empty paths.


## Statistical univariate selectors (2026-11-20)

SelectPercentile, SelectFpr, SelectFdr, SelectFwe and GenericUnivariateSelect are
implemented in sklearn_univariate_selection and re-exported by
sklearn_feature_selection and the facade. Native controls and defaults reproduce
source percentile/tie rules, strict false-positive/family-wise thresholds,
inclusive Benjamini-Hochberg step-up thresholds, and generic percentile/k_best/
fpr/fdr/fwe modes. Generic k_best supports zero, values above P and the native
k_all flag for source 'all'. Percentile accepts real percentages. Controls are
read lazily on every projection, as in source get_support, including changes
after fitting. Native support indices are one-based. Shape changes on refitting
replace score/probability state, including clearing old p-values when omitted.

Integer targets default to ANOVA f_classif, real targets to centered
force-finite f_regression, following the project's existing generic fit
convention; source Python classes default to f_classif independently of dtype.
Classification can instead use chi2, and regression can disable centering.
Scores and p-values use installed sklearn 1.6.1's moment/correlation/count
contracts, including constant/degenerate NaNs, zero-correlation regression
fallbacks, maximal finite exact-correlation scores and the negative-F rounding
endpoint. Regression's stats.f.sf returns 1 below support, while ANOVA's
special.fdtrc returns NaN there; these paths are distinguished. Explicit IEEE
branches preserve these results under invalid/zero/overflow traps. Older
score-only functions and SelectKBest bodies are unchanged.

F survival uses a native regularized beta fraction with symmetry and separately
computed complementary coordinates, and chi-square survival uses the
integer/half-integer upper-gamma recurrence with erfc and log-domain positive
terms. Formulas follow NIST DLMF 8.17.4/22/23 and 8.8.6; modified Lentz iteration
has a convergence diagnostic. This adds no SciFort, SciPy or Python runtime
dependency. The numerical routines are private implementation helpers.

fit_scores(scores,pvalues) is the native custom-score entry point; percentile
and k_best also work without p-values. This represents callback results without
requiring a Python callable ABI. scores_, pvalues_ and n_features_in_ are public.
Score NaNs/infinities follow source cleaning, sorting, comparison and percentile
interpolation behavior, including extreme interpolation overflow without raising
a native floating trap. Dense X and real regression targets must be finite;
classification labels are integers. Sparse/object arrays, pandas/feature names,
metadata routing, Python callbacks and unimplemented statistical score families
are outside the wrapper. Regression needs positive residual degrees of freedom.
Class-degenerate ANOVA/chi-square scores retain source NaNs rather than warnings.

transform projects columns; inverse_transform restores omitted columns to
zero. For an N-by-zero selected matrix the native inverse returns N-by-P zeros,
consistent with the existing native selector helpers. The source dense inverse
rejects zero features. Those inverse checks verify a zero-fill identity and
explicitly confirm the source rejection, rather than claim source equality.
Very large regression F scores amplify last-bit correlation differences. When
both source and native F exceed 1.e10, an otherwise failing score comparison
must instead match F/(F+df), calculated stably as 1-df/(F+df), within 64 machine
epsilons. Other score entries still use ordinary tolerances; probability arrays,
log tails, masks, transformed data and zero-filled results remain independently
checked. Such groups are explicitly identified in the report. Severe moment
cancellation and perfectly correlated rounding can remain platform-sensitive,
as in source. No claim of bitwise statistical equality is made.

The native test has analytic masks, strict/inclusive boundaries, stable tie
behavior, perfect/constant/degenerate scores, post-fit parameter changes,
custom score refits, input preservation and inverse zero filling. Independent
F(1,2), F(2,2) and chi-square(2) tail identities verify the probability backend.
The reference suite covers four seeds, all five types and generic modes,
ANOVA/chi-square/regression/custom scores, balanced/unbalanced/many/single/
one-sample-per-class labels, single/wide/changing feature counts, large N,
centering, scaling, extreme/NaN/infinite/tied scores, very small probabilities,
zero/full projections and threshold equalities. It includes logarithmic checks
for every positive normal-range tail. Forty-two new rejection cases check
controls, malformed/missing scores, p-values, modes, degrees of freedom,
nonnegative counts, fitting state, data/query shapes and finiteness.

## PolynomialCountSketch (2026-11-20)

Added the core sklearn.kernel_approximation.PolynomialCountSketch counterpart,
exported as polynomial_count_sketch by sklearn, sklearn_kernel_approximation
and its focused module. Dense real64 fit/transform/fit_transform implement
augmented polynomial TensorSketch: sqrt(gamma) feature scaling, an optional
sqrt(coef0) constant, independent degree hash/sign tables, per-degree count
sketches and the inverse FFT of their spectral product. Scalar seeds reproduce
NumPy legacy RandomState integer draws, including C-order traversal and the
one-bucket case. Public index_hash_ values remain zero-based, with table shape
(degree,augmented_features); bit_hash_ stores -1/+1 signs. Refit replaces all
hash tables and fitted dimensions. Inputs are preserved.

Positive arbitrary component counts use radix-2 FFTs or Bluestein convolution,
with a plan reused across rows/degrees within each transform. The MIT SciFort
FFT helpers are adapted from local src/signal_spectral.f90, identified by SHA-256
77283c0c5c14214e9a9c95c04ac9d737ee5153444a1027d38acd2f2d14f082a6.
The SciFort copyright/header and complete MIT license are retained, and the
modifications and Bluestein citation are recorded in THIRD_PARTY_NOTICES.md.
The plan is implementation support, excluded from the sklearn facade/core type
count. No SciFort runtime dependency or public algorithm extension is added.

Controls default to gamma=1, degree=2, coef0=0, n_components=100 and native
random_state=0. Fitted hash tables and n_features_in_/n_features_out_ are public.
Source live controls are retained: gamma/coef0 magnitudes, a prefix degree and
component counts that contain all active buckets can change after fitting.
Actual output width follows current n_components; n_features_out_ stays at the
fitted value, matching source. Changing zero/nonzero coef0 status requires a
refit because it changes augmented width. Negative finite coef0 is accepted
but produces source-compatible all-NaN features, implemented explicitly to
avoid invalid floating traps. Native transform checks invalid changed controls
and inconsistent fitted hashes with explicit diagnostics. Supported seeds are
nonnegative default integers. Sparse input, Python RNG objects/global streams,
feature-name methods and metadata routing are outside this wrapper.

compare_polynomial_count_sketch.py checks 2,950 direct sklearn 1.6.1 result
groups across five seeds, 140 scenarios and 590 fit/transform operations.
Seeds include zero and the largest supported default integer. Each operation
checks exact bucket/sign tables and fitted/output metadata, plus features and
Gram matrices at atol=rtol=3e-8 with equal NaNs. Fixtures cover one/power-of-two/
composite/prime component counts through 2,048, degrees 1 through 7, gamma=0,
zero/positive/negative offsets, zero rows, one/wide/changing feature counts,
constant inputs, live controls, repeated fitting and independent query data.
All groups use the source oracle without identity-based substitutions.

Native tests independently enumerate an exact degree-two tensor sketch and
compare complex FFTs to direct Fourier sums for short lengths. Eighteen FFT
lengths through 1,009 check forward/inverse round trips, including radix-2,
composite and prime sizes. Additional checks cover input preservation, signed
constant maps, deterministic seed/refit behavior and negative-offset NaNs.
Thirty-six new isolated diagnostics cover parameter domains, empty/nonfinite
data, unfitted transforms, query/augmented dimensions, invalid changed
controls, hash shapes/buckets, integer work-size overflow and FFT plan misuse.
The polynomial_kernel_features example uses the public facade to compare a
mapped dot product with the corresponding exact polynomial kernel.

Inventory: 165 core types, 77 source modules, 79 native test programs,
58 reference suites and 91,321 result groups, plus 28 separate NumPy stream
groups. Isolated invalid-input inventory is 1,555. Original contributions
remain MIT licensed with complete upstream notices and specific attribution.

## LabelBinarizer and MultiLabelBinarizer (2026-11-20)

Added label_binarizer and multi_label_binarizer as core sklearn preprocessing
counterparts, exported by sklearn, sklearn_preprocessing and the focused
sklearn_label_binarization module. Both implement fit, transform, fit_transform
and integer/real inverse_transform overloads, with allocated integer outputs.
The public label_set container represents a ragged row of integer labels;
it is data representation, excluded from the algorithm count like index_split.
No public algorithm extension or runtime dependency is added.

LabelBinarizer discovers sorted classes from a nonempty rank-one integer target,
preserves source one/two-class versus multiclass encoding, configurable integer
negative/positive values and unknown-label negative entries. Default neg_label=0
and pos_label=1 can change after fitting. One-class binary queries are entirely
negative. Source query-dependent target typing is retained: three distinct
query labels produce all fitted class columns even after binary fitting, and
can mark a known single class positively. Refit replaces classes_/y_type_.
Binary inverse uses a strict midpoint/explicit threshold, using the second
column for two-column scores. Multiclass inverse picks the first maximal score
and clips an extra-column winning index, as in source. Native midpoint
arithmetic avoids default-integer overflow for extreme encoding values.
Empty inverse row counts are supported. Native inverse requires finite scores,
positive width and a finite binary threshold; source's absent single-class
positive index becomes an explicit diagnostic. Rank-two multilabel-indicator
LabelBinarizer fitting, sparse output and string/object labels are unsupported.

MultiLabelBinarizer discovers a sorted union or preserves a unique constructor
classes ordering. Constructor changes wait for the next fit; classes_ is a
fitted snapshot. Repeated labels are idempotent. Unknown labels are ignored
with source indicator values, without Python warnings. Unallocated/empty row
values, empty datasets, empty fitted/configured vocabularies and zero-column
indicator matrices are supported. Inverse validates exact 0/1 integer or real
indicators and returns allocated label sets in fitted class order. Inputs are
preserved. Python iterable/object/string APIs, sparse formats and feature-name
integration are outside this wrapper.

compare_label_binarization.py checks 6,285 exact sklearn 1.6.1 result groups
across 555 scenarios and five seeds. The 375 LabelBinarizer scenarios check
one/two/many classes, known/unknown/multiclass queries, zero/negative/positive/
extreme integer encodings, fit and fit_transform parity, query dimensions,
target type, both inverse overload paths, threshold equalities, tied maxima,
changed score widths, live encoding controls and dimension-changing refits.
The 180 MultiLabelBinarizer scenarios check automatically discovered, reversed,
subset and empty class orders, duplicates, unknown labels, empty rows/datasets,
integer and real inverse indicators, complemented indicators, fitted snapshots
and rediscovery on refit. All groups use the source oracle and exact integer
array equality, including ragged counts and flattened inverse values.

Native tests independently assert binary class ordering, strict thresholds,
query-dependent widths, zero positive encoding, first-max ties and index
clipping, single-class negatives, empty inverse rows, duplicate/empty label
sets, explicit class ordering, fitted snapshots, zero-column inverse and input
preservation. Twenty-six new isolated diagnostics check invalid encoding
controls, empty/unfitted label methods, inverse widths and nonfinite values,
missing single-class positive labels, duplicate constructor classes and invalid
integer/real indicators. example/label_encoding.f90 exercises both types through
the public facade. sklearn source attribution and complete BSD notices remain
retained; original contributions use the project MIT license.

Inventory: 167 core types, 78 source modules, 80 native test programs,
59 reference suites and 97,606 result groups, plus 28 separate NumPy stream
groups. Isolated invalid-input inventory is 1,581.

## Mutual-information and Pearson scoring (2026-11-20)

Added four core sklearn functions: mutual_info_classif, mutual_info_regression,
r_regression and mutual_info_score. Feature scorers are exported through sklearn
and sklearn_feature_selection; the clustering metric through sklearn_metrics;
all are available in the focused sklearn_mutual_info module. They add algorithms
without increasing the estimator/transformer/splitter type count. No public
extension, SciFort dependency or external runtime is introduced.

MI implements Kraskov continuous/continuous Chebyshev neighbor radii, Ross
continuous/discrete class-wise radii and singleton removal, and categorical
contingency MI. Integer digamma differences use harmonic tables. Negative
estimates are clipped to zero. Dense all-pairs distance/count passes use O(N)
scratch per feature, quickselect for kth radii and O(N**2) distance work. The
categorical metric stores observed joint pairs rather than a dense K-by-L table.
Source preprocessing scales without centering, distinguishes feature-array and
scalar-target near-zero scales, and adds 1.e-10 Gaussian perturbations with
NumPy-compatible scalar-seeded C-order draw traversal. The private pairwise sum
adapts NumPy 2.2.6's eight-accumulator/128-element/aligned-recursion reduction.
For class-wise Euclidean radii, native arithmetic follows source's auto backend
choice: ordered Gram terms for brute force and squared differences otherwise.
Radius counts use reduced squared distances, retaining source boundary rules.

Dense real64 X and regression targets, integer classification/cluster labels,
positive n_neighbors and nonnegative default-integer seeds are supported.
discrete_features accepts scalar logical, logical mask or nonempty native
one-based column indices. Fractional real category codes are allowed. Continuous
neighbors must be fewer than N; class-wise counts adapt to available repeats,
ignoring singleton labels. Copy=False, sparse data, Python RNG/callback objects,
joblib parallelism and string/object labels are outside the native interface.
Inputs are preserved. MI vectors can feed score-only univariate selectors.

r_regression retains signed correlations, centered/uncentered moment formulas,
division order and force_finite semantics: NaNs become zero by default, while
signed infinities remain. Explicit IEEE branches handle undefined/underflowed
denominators with invalid/zero/overflow traps enabled. Exact constant columns
have undefined mathematical correlation; BLAS summation can change source/native
outputs among NaN, zero, signed infinity or tiny moment-rounding artifacts.
Those cases are reported separately rather than asserted to be source-equal.
mutual_info_score accepts nonempty integer label pairs and preserves zero-entropy
and label-permutation invariance; precomputed contingency input is not exposed.

compare_mutual_info.py checks 2,196 result groups over six seeded datasets with
N=7,16,31,64,127,257, including seed 0 and the largest supported scalar seed.
There are 1,158 score groups and 1,038 exact input-preservation identities.
The pre-build checked run has 1,150 direct source score groups and eight groups
with an explicit exact-constant Pearson certificate. Such groups still compare
every nonconstant column directly. Constant columns must be exactly equal down
the input column; both outputs must be IEEE undefined/zero or finite artifacts
bounded by 32*sqrt(N*epsilon). force_finite=True cannot return NaN. Other source
comparisons use atol=rtol=3e-8, with equal NaNs and signed-infinity agreement.
The report records affected columns, source/native values and bounds.

Fixtures cover all continuous/discrete combinations, mixed features, every
native mask/index/scalar overload (with repeated/reordered indices),
k=1/2/3/N-1 and adaptive counts above N,
single/wide features, singleton categories, repeated/constant/rounded data,
small scales, offsets, constant targets, one-to-four-sample MI edges, recursive
reductions, both Pearson options, tied/duplicate features, underflowed norms,
and empty/single/permuted clustering labels. Empty metric inputs are explicitly
confirmed as source rejections rather than source score comparisons. Four source
rejections cover that case, singleton-only continuous/discrete samples, unique
discrete regression features and invalid continuous neighbor counts.

Native tests assert categorical entropy/independence, label permutation,
single-sample categorical input, mask/index equivalence, deterministic seeds,
input preservation, signed Pearson values, uncentered cosine, undefined NaNs,
finite fallbacks, one-sample centering and preserved underflow infinities.
Thirty-four new isolated diagnostics cover invalid controls, masks/indices,
data sizes/finiteness, neighbor counts, absent repeated labels and metric sizes.
example/nonlinear_feature_scoring.f90 combines MI ranking with score-only
selection, Pearson scores, classification MI and the discrete metric via sklearn.

Original contributions remain MIT; complete sklearn/NumPy BSD notices and the
NumPy summation source attribution are retained. Research references follow
sklearn: Kraskov, Stogbauer and Grassberger (2004), and Ross (2014).
Inventory: 167 core types, 79 source modules, 81 native test programs,
60 reference suites and 99,802 result groups, plus 28 separate NumPy stream
groups. Isolated invalid-input inventory is 1,615.

## Clustering information and pair agreement (2026-11-20)

Added normalized_mutual_info_score, adjusted_mutual_info_score,
fowlkes_mallows_score, contingency_matrix, pair_confusion_matrix, entropy,
expected_mutual_information and homogeneity_completeness_v_measure. All eight
are sklearn.metrics.cluster counterparts, exported through sklearn_metrics and
sklearn, with focused access through sklearn_cluster_information. Existing
homogeneity_score/completeness_score/v_measure_score now delegate to the same
backend, adding source empty-input conventions and optional V-measure beta.
These add metrics/helpers rather than estimator types. No public extension or
runtime dependency is introduced.

Sorted label bins and observed joint cells use heap sorting and O(N) histogram
storage. Observed information follows source log/epsilon conventions with
NumPy-compatible pairwise reduction. Pair confusion stores ordered distinct
sample counts in int64, with rows TN/FP and FN/TP; Fowlkes-Mallows uses the
geometric mean of pair precision/recall, returning zero with no true positives.
The explicit contingency function allocates a dense sorted-class count matrix;
its two-argument overload returns int64, while real64 eps selects an allocated
real64 matrix with finite nonnegative smoothing added to every cell. Metrics
themselves do not materialize a dense K-by-L matrix.

Normalized and adjusted MI implement min/geometric/arithmetic/max entropy
averages, defaulting to arithmetic. Expected MI follows sklearn's hypergeometric
sum over feasible joint counts with cached intrinsic log factorials. AMI
retains negative values and source signed-epsilon denominator flooring. The
public expected-MI helper accepts default/int64 nonnegative count matrices,
requires their sum to match nonnegative n_samples, supports zero margins and
nonempty all-zero tables, and guards invalid extents/work sizes and sum overflow.

Dense integer label arrays must have equal lengths. Empty NMI/AMI and H/C/V
return one, entropy(empty) returns one, contingency(empty) is 0-by-0, pair
confusion(empty) is zero and empty/singleton Fowlkes-Mallows is zero, following
source. Beta is finite/nonnegative. Source's beta=0, zero-completeness division
error becomes an explicit native diagnostic; independent partitions with H=C=0
still give V=0. Sparse matrices, dtype overrides, object/string labels and the
deprecated ignored FMI sparse flag are not exposed. Inputs are preserved.

compare_cluster_information.py checks 2,720 groups over five seeds, 200 paired
partition fixtures and 125 manual count matrices. The initial checked run has
2,510 direct source groups, 200 exact input identities and ten AMI groups with
explicit zero-chance-gap certificates. Every ordinary AMI value and unaffected
normalization remains directly source compared at atol=rtol=3e-8. Expected MI
is directly compared at atol=rtol=5e-11; all integer tables, pair counts, shapes
and input copies are exact. Source beta-zero division and empty expected-table
rejections are explicitly verified, not reported as numerical equalities.

The adjustment denominator is mathematically zero for min-normalized AMI when
one partition is entirely singleton clusters (and for every average when both
are singleton partitions). Intrinsic/SciPy/libc log-gamma rounding can then
substantially change the final source/native score. Certificates are limited to
those exact partition conditions: source/native MI and expected MI must agree
with the other entropy within 64*N*epsilon*max(1,H), and each reported adjusted
score must lie in the source epsilon-guarded ratio interval obtained from its
observed NMI/entropy/expected MI, allowing four epsilon units for reconstructing
the observed MI. All nondegenerate components remain source checked. Reports
record conditions, source/native values, component witnesses and ratio bounds.

Fixtures cover N=0,1,2,3,4,5,7,16,31,64,127,257,513, independent/identical/
renamed/refined/unbalanced/singleton partitions, all four averages, zero through
1.e300 beta, eps 0/1.e-12/.25/10, empty and zero-entropy limits, zero matrix
margins, count overloads and expected MI up to N=1,200. Native tests independently
check sorted contingency orientation, smoothing, analytic ordered pairs/FMI,
binary entropy, the exact balanced hypergeometric EMI log(2)/3, AMI=-.5 on
balanced independent partitions, perfect-label scores, empty conventions,
weighted V-measure wrappers and 70,000-sample counts above 32-bit range.
Thirty-four new diagnostics cover lengths, methods, eps/beta controls,
zero denominators, negative/inconsistent counts, empty matrices and overflow.
The clustering_agreement example exercises all eight functions through sklearn.

Source paths/hashes, sklearn and NumPy BSD notices, and Vinh/Epps/Bailey (2010)
and Fowlkes/Mallows (1983) research references are recorded in
THIRD_PARTY_NOTICES.md. Original contributions retain the project MIT license.
Inventory: 167 core types, 80 source modules, 82 native test programs,
61 reference suites and 102,522 result groups, plus 28 separate NumPy stream
groups. Isolated invalid-input inventory is 1,649. The original integration
comparison is rerun because the existing H/C/V numerical wrappers changed;
compare_sklearn.py now also accepts a prebuilt executable for profile reuse.

## Randomized, repeated and exhaustive resampling batch

Added seven core splitter types: shuffle_split, stratified_shuffle_split,
group_shuffle_split, repeated_kfold, repeated_stratified_kfold, leave_p_out,
and leave_p_groups_out. Both sklearn_model_selection and sklearn export them;
the new sklearn_resampling backend also owns index_split, whose existing
model_selection import remains source compatible. All seven expose split and
get_n_splits. Randomized splits reproduce scalar-seeded NumPy RandomState
indices via the existing native legacy RNG. Repeated folds keep a single
stream across repeats and return sorted sample indices. Stratified shuffling
uses source quota allocation and randomized ties, while repeated stratified
folds encode classes by order of first appearance. Group holdouts operate on
distinct sorted labels and prevent group leakage. Exhaustive holdouts follow
lexicographic combinations with checked binomial counts before allocation.

Shuffle controls support independent integer counts or real64 fractions,
source ceil/floor/complement rules, and partially unused samples/groups.
GroupShuffleSplit defaults to five iterations and 20% of groups; other shuffle
types default to ten iterations and 10% of samples. All random seeds default
to zero; Python None/object seeds are outside the native API. Repeated
stratification allows rare classes when another class meets n_splits, returning
the source indices without Python's warning. Returned folds are eager arrays,
so exhaustive splitting retains combinatorial memory/time costs. Core API and
limitations are documented in API.md; no new extension or dependency is added.

The new reference suite checks 23,280 ordered-index/count/identity groups over
820 fixtures and five seeds (0, 7, 42, 2026, 314159): 20,820 sklearn groups and
2,460 input-preservation/repeatability groups. Cases include populations
2--513, unequal class/group sizes, a single class, rare classes, randomly
ordered negative labels, quota ties, rounded/fractional/count/mixed controls,
unused samples/groups, uneven repeated folds, up to 127 folds per repeat,
and exhaustive p=1/2/n-1 combinations. Every native index is compared exactly
with sklearn's one-based translated output; no tolerance or alternative RNG
oracle is used. New native tests independently check defaults, source counts,
no sample/group leakage, balanced class allocations, sorted repeated indices,
once-per-repeat sample coverage and lexicographic exhaustive combinations.

Forty-five additional isolated diagnostic cases cover invalid samples,
counts/fractions/conflicts, nonfinite fractions, empty training sets, class
constraints, group constraints, repeat controls, invalid seeds and split-count
overflow. A resampling example demonstrates random/repeated/group holdouts.
Baseline source paths/hashes and retained BSD notices are recorded in
THIRD_PARTY_NOTICES.md. Original contributions remain MIT.
Inventory: 174 public core types, 81 source modules, 83 native test programs,
62 reference suites and 125,802 result groups, plus 28 separate NumPy RNG
stream groups. Isolated invalid-input inventory is 1,694. Existing numerical
splitter implementations are unchanged; the original integration comparison
and group-splitter reference suite are rerun to check the index_split move.

## Dense exact t-SNE and trustworthiness batch

Added tsne and trustworthiness to sklearn_manifold and the sklearn facade,
with sklearn_tsne and internal sklearn_tsne_core implementations. Dense exact
t-SNE supports PCA, random and explicit float64 initialization; the local
NumPy-compatible Gaussian stream; randomized-SVD loading sign conventions;
source float32 probability search and single/mixed/double optimizer paths;
Student degrees of freedom above three embedding dimensions; adaptive gains;
two exaggeration/momentum phases; 50-step gradient/progress checkpoints;
automatic/explicit learning rates and fitted embedding, KL, iteration,
feature count and resolved rate attributes. The largest-finite-KL/iteration
convention for a full max_iter=250 run is preserved. Stable scaled gradient
norms prevent underflow-based premature stopping. Source precomputed squaring
is reproduced on a copy, preserving caller input. Trustworthiness computes
neighborhood rank penalties with native stable index tie choices.

This is the exact dense algorithm, not Barnes-Hut. Quadratic memory/pair work,
no out-of-sample transform, nonnegative default-integer seeds, float32 working
ranges, native metric support and tie differences are explicit in API.md.
No new dependency or public extension is added. Internal diagnostic helpers
and trace outputs are not exported by the sklearn facade.

The new reference suite checks 1,620 result groups over 95 fixtures and five
seeds (0,7,42,2026,314159). It covers 1/2/4 dimensions, sample counts 7--31,
Euclidean/Manhattan/cosine/precomputed input, explicit/random/PCA starts,
normal and exaggerated probabilities, fixed/automatic rates, float32 and
float64 gradients/short trajectories, full two-phase fits, checkpoint stops,
max_iter=250 and unchanged inputs. Affinities, source objective/gradient at
initial/final native positions, short optimizer steps, stable full trajectories
and unique-neighborhood trustworthiness are compared directly to sklearn.
All explicit-initialization paths additionally replay the public fit exactly
and verify each native step against the source objective/gradient and NumPy
update formulas, including gains, stopping and the pre-update reported KL.
Small reduction errors can change near-zero gain signs or amplify over a
nonconvex trajectory; such fields require recorded step certificates rather
than broader coordinate tolerances. Tied embedded-neighborhood differences
require the independent native stable-rank identity and valid rank bounds
for both source and native scores, with unique original-space ranks.

The initial checked prototype recorded 1,306 direct sklearn groups,
195 replay-identity groups, 95 input-identity groups, 15 sklearn/stable-tie
certificates and nine sklearn/trajectory certificates. Its 65 trajectories
contained 18,300 source-checked steps and 73 cancellation-bounded gain choices;
maximum scaled-gradient/update/position/objective differences were about
1.1e-15/5.1e-16/1.8e-15/3.6e-14. Final checked/release reports retain their own
profile-specific oracle/cancellation counts. No broad geometric alignment or
relaxed final-coordinate assertion substitutes for step-level verification.

Native tests independently check uniform joint probabilities, zero objective
and gradient at uniform collapse, finite-difference gradients, fixed points,
very-small-gradient stopping without norm underflow, deterministic refits,
input preservation, both float32 initialization paths, source max_iter=250
and checkpoint behavior, analytic trustworthiness and collapsed ties.
Forty-six new diagnostic cases cover invalid controls, seeds, initialization,
metric shape/negativity, probability/objective domains and working ranges.
The nonlinear_embedding example demonstrates PCA-initialized exact t-SNE and
trustworthiness. Upstream source hashes, research attribution and retained BSD
notices are recorded in THIRD_PARTY_NOTICES.md; original work remains MIT.
Inventory: 175 public core types, 83 source modules, 84 native test programs,
63 reference suites and 127,422 result groups, plus 28 separate NumPy RNG
stream groups. Isolated invalid-input inventory is 1,740. Existing numerical
implementations are unchanged; manifold re-exports the new implementation.

## Hessian eigenmapping and modified LLE batch

Extended the existing locally_linear_embedding type with method='hessian'
and method='modified', completing all four sklearn LLE methods. New finite,
nonnegative hessian_tol=1.e-4 and modified_tol=1.e-12 controls follow source
validation. Hessian fitting builds centered tangent/monomial neighborhoods,
full Householder QR complements and tolerance-normalized alignment blocks.
Modified fitting builds full local eigensystems, fixed trace-regularized
affine weights, median spectral ratios, cumulative-energy/searchsorted
almost-null-space sizes and Householder multiple-weight blocks. The native
backend reuses Jacobi SVD/eigen routines and completes local orthogonal bases.
The shared fit/fit_transform/transform and analytic constant-mode removal
remain intact; public reg affects fitting only for standard LLE and affects
query interpolation for every method. No public type or extension is added.

The new reference suite checks 3,535 groups over 95 fixtures and five seeds
(0,7,42,2026,314159), covering 18--32 samples, 3--12 input features,
6--12 neighbors, 1/2/3 embedding dimensions, both SVD and local covariance
branches, odd/even median selection, default/modified tolerances and nondefault
query regularization. Local source alignments are captured from the installed
sklearn implementation before global eigenanalysis. Source local spectral
values, regularized weights, thresholds, nullities, ordinary fitted/query
geometry and reconstruction error are compared; native local bases also
satisfy independent orthogonality and covariance eigen-residual identities.
Every modified alignment is independently rebuilt using the source formulas
on its validated native local eigensystem.

Nonunique global eigenspaces use source-matrix residual/lowest restricted
energy, orthogonality, zero-mean and source barycentric-query certificates.
Spectral-cutoff differences require measured near-equality bounds and a
counterfactual native-frame assembly using source nullities that matches the
source alignment. The initial checked prototype recorded 695 direct sklearn
groups, 2,620 spectral identities, 95 input identities, 50 source-formula/native
frame assemblies, 51 nonunique-eigenspace comparisons and 24 cutoff-certified
comparisons. There were 17 nonunique spectral fixtures and four single-row
cutoff fixtures. Final profiles retain their own complete oracle/certificate
accounting; no broad coordinate tolerance replaces these checks.

Native tests independently check alignment symmetry, positive semidefiniteness,
constant-mode annihilation, translation/scale invariance, affine constrained
regularized weights, embedding orthonormality and energy, deterministic refit,
input preservation, query transforms and the public reg control's distinct
fit/query behavior. Twenty-five new diagnostics cover tolerance domains,
method-specific neighborhood limits, zero spectra/empty nullspaces, zero
Householder denominator, malformed neighbor arrays and invalid feature data.
All 25 also passed a standalone checked probe before full fpm validation.
An lle_variants example demonstrates both methods and query coordinates.

The existing compare_manifold.py additionally accepts --executable, preserving
its default fpm invocation while enabling checked-profile regression testing
without a redundant build. Standard LLE, LTSA and MDS source comparisons are
rerun because the LLE type/interface was extended. Existing standard/LTSA
numerical bodies are unchanged. Dense spectral/rank/tie/working-range
limitations and source paths/hashes/research attribution are documented.
Original work remains MIT with retained sklearn BSD notices.
Inventory: 175 public core types, 84 source modules, 85 native test programs,
64 reference suites and 130,957 result groups, plus 28 separate NumPy RNG
stream groups. Isolated invalid-input inventory is 1,765. No dependency is added.

## Ranking curves, calibration and weighted-score batch

Added six public sklearn functions: roc_curve, precision_recall_curve,
det_curve, auc, calibration_curve and brier_score_loss. Added sklearn_ranking
and sklearn_calibration modules, metric re-exports and the calibration facade
import. Existing roc_auc_score and average_precision_score now use the shared
stable O(N log N) threshold backend, retain their first three native arguments,
and add sample_weight; ROC AUC also adds standardized max_fpr partial area.
The backend follows source stable-ascending/reversed tie order, zero-weight
filtering, independent weighted cumulative counts, curve thinning, endpoint
and DET trimming conventions. Explicit positive-label and single-class rules,
quiet IEEE undefined outputs and signed-weight arithmetic follow source
semantics. Partial AUC specifically preserves the source rejection when
zero weights remove all negative support. Calibration uses uniform/quantile
bins, source linear interpolation and left-edge assignment, and drops empty
bins. Brier uses source numeric-label inference and weighted squared error.
Native score/probability/AUC-coordinate inputs are finite dense real64 values;
working-range and invalid-domain diagnostics are explicit.

The new reference suite checks 5,584 groups over 365 fixtures and five seeds
(0,7,42,2026,314159): 5,215 direct sklearn groups, 365 unchanged-input checks
and four source/native partial-AUC rejections. Cases cover 1--513 samples,
unique/tied/constant scores, uniform/random/log-scale/zero/signed weights,
removed classes, singleton and custom labels, explicit one-vs-rest curves,
all thinning choices, increasing/decreasing/constant AUC coordinates,
partial limits .01--1, 1--11 uniform/quantile bins, repeated probabilities,
empty calibration bins and IEEE NaN/infinite outputs. Curve thresholds and
array shapes are compared exactly; other values use tight 3.e-12 tolerances
with matching NaNs. No alternative numerical oracle replaces source checks.

Independent native tests verify analytic ROC/PR/DET outputs, trapezoidal and
partial ROC areas, AP step integration, calibration edge/duplicate behavior,
weighted and singleton Brier rules, singleton/zero-support NaNs, signed AP
and a 100,000-sample closed-form rank identity. Fifty-three new diagnostic
cases cover every public path's sizes, finite data, labels, weights, partial
limits, coordinate monotonicity and calibration controls. All passed a
standalone checked probe before full fpm testing; two fixture literals were
corrected to real64 to match the max_fpr interface. The ranking_curves example
demonstrates threshold curves, scoring and calibration. Existing numerical bodies outside
ROC AUC/AP remain unchanged. Provenance and complete upstream notices are
retained; original work stays MIT. No new public type, extension or dependency
is added; the existing native positive-label AUC convenience is documented.
Inventory: 175 core types, 86 source modules, 86 native test programs,
65 reference suites and 136,541 result groups, plus 28 separate NumPy RNG
stream groups. Isolated invalid-input inventory is 1,818.


## Self-training classifier batch

Added self_training_classifier, exported by sklearn_semi_supervised and the
sklearn facade. The sklearn 1.6.1 loop supports native logistic_regression,
gaussian_nb and kneighbors_classifier bases, copied without mutating the
input estimator. It implements threshold and k_best selection, strict
threshold comparison, original-label preservation, per-sample iteration
history, transduction, zero/finite/unlimited iteration controls, source
stopping-rule precedence and mandatory final refitting. Inductive class
predictions, probabilities, accuracy and available base log-probabilities
and decision functions use the final fitted estimator. One-class fits are
supported where the base supports them. Other estimator types, sparse data,
metadata/sample-weight routing and verbose output are outside this port.

Confidence ties use stable input order; source argpartition leaves the tied
cutoff choice unspecified. Rounding of saturated base probabilities can also
change tied selections. The reference suite checks all native iterations
against source base fits: correct labels, exact threshold choices, k-best
counts and confidence ordering. Tied cutoff comparisons allow only 32 double
precision epsilons for Gaussian NB/KNN arithmetic; logistic ordering is exact.
Histories that differ from ordinary sklearn require an independently verified
tied cutoff. Final source-base fits on the certified transduction validate
query predictions and probabilities. Direct source histories remain required
for ordinary cases. Multiclass logistic decisions are centered per row to
remove their unidentifiable common intercept; probabilities are compared
directly. The stable tie policy itself has an analytic native test.

The suite has 4,905 groups across 495 fixtures and five seeds
(0,7,42,2026,314159), with one through four classes and all three supported
bases: 3,694 direct sklearn groups, 990 input identities and 221 groups from
30 tied-cutoff cases certified using source fits. Gaussian NB and KNN include
single-class cases; logistic requires at least two initial classes. Uniform
and distance-weighted KNN, both selection rules, zero/unlimited/capped fits,
no-change stopping, fully labeled inputs and query decision/probability
outputs are covered. Native tests additionally verify analytic transduction,
strict threshold equality, all stopping-rule precedence, state replacement,
deep copies, unchanged fitted/unfitted input estimators and mandatory final
refits. Nineteen new isolated diagnostics cover invalid controls, input
sizes/data, missing initial labels, unsupported bases, unfitted prediction,
feature mismatches and unavailable delegated methods.

Source provenance and retained BSD attribution are in THIRD_PARTY_NOTICES.md;
original contributions remain MIT. No extension or dependency is added.
Inventory: 176 core types, 87 source modules, 87 native test programs,
66 reference suites and 141,446 result groups, plus 28 separate NumPy RNG
stream groups. Isolated invalid-input inventory is 1,837.


## Cross-validated probability calibration batch

Added calibrated_classifier_cv, exported by sklearn_calibration and sklearn.
It implements sigmoid and isotonic calibration using native logistic,
Gaussian NB or KNN bases. Integer CV uses source first-appearance class
encoding and unshuffled stratified allocation. Ensemble fitting averages
individually normalized fold classifier/map pairs; nonensemble fitting
collects out-of-fold responses and refits the final base on all rows.
Prefit calibration copies an already fitted base without refitting it and
retains its output classes, including missing/unknown calibration labels.
Binary response selection/complement, one-vs-rest multiclass targets,
zero-mass uniform fallback and final probability handling follow source.

The initial weighted CV scope was KNN, with weighted prefit calibration for
all three bases. The weighted/streaming classifier batch below subsequently
adds weighted logistic/Gaussian NB CV fitting. Weights are finite,
nonnegative and bounded by the documented native working range. All-zero
sigmoid weights retain source .5 map outputs; isotonic fitting requires
positive weight support. General estimator/default LinearSVC cloning,
FrozenEstimator, custom CV splitters, sparse arrays, metadata routing,
parallelism and arbitrary fit parameters remain outside this interface.

Sigmoid maps minimize the source Platt objective with weighted Bayesian
smoothing and response scaling at maximum absolute score >=30. A native
damped two-variable Newton solver uses a rank-one pseudoinverse direction
for constant designs and a gradient-decrease certificate when summed loss
roundoff obstructs a near-optimal step. Isotonic maps reuse weighted PAV and
clipped interpolation, explicitly bounding binary-target fits to [0,1] to
contain endpoint roundoff. The shared logistic solver now centers fitted
multiclass intercepts: this removes an arbitrary common softmax offset that
otherwise varies across fold fits and contaminates pooled raw scores.
Its probabilities retain the same model; binary fitting is unchanged.

The new suite checks 11,680 groups across 690 fixtures and five seeds
(0,7,42,2026,314159), with 2/3/4 classes, both map methods, all three bases,
3/5-fold CV, ensemble/nonensemble/prefit paths, uniform/distance KNN,
weighted/zero-weight fits, missing/unknown calibration labels and very large
logistic decision scores. There are 6,086 direct sklearn groups, 3,520
source-calibrator certificates, 1,380 input identities, 690 composition
identities and four machine-precision argmax-tie groups. Each sigmoid map's
independent normalized gradient must be at most 2.1e-9, and its objective
cannot exceed a source optimizer fit by more than 2e-10. Each isotonic map
must match a fresh source fit on native response values to 3e-12 over the
union of both knot sets and query responses, certifying the full piecewise
linear function. Raw base responses and final probabilities compare directly
to source; the latter use 2e-6 tolerances (3e-6 for logistic). Argmax exceptions
require both native and source gaps within 64 machine epsilons and are
recorded individually; all four observed source gaps are exactly zero.

Native analytic tests cover smoothed versus empirical constant-score targets,
weighted priors and zero support, complements, deep copies, source fold
encoding, mandatory full-data refitting, input estimator preservation,
canonical softmax gauge, refit state replacement and uniform multiclass
fallback. Twenty-three isolated diagnostics cover invalid controls, data,
weight domains, unsupported bases/weighted CV paths, prefit/fitted state,
class counts, feature mismatches and zero-weight isotonic fits. A checked
standalone probe passed the new native tests, all 23 diagnostics and the
complete reference suite before full package validation.

Original contributions remain MIT with retained sklearn BSD attribution.
No dependency or public extension is added. Inventory: 177 core types,
88 source modules, 88 native test programs, 67 reference suites and 153,126
result groups, plus 28 separate NumPy RNG groups. Isolated diagnostic
inventory is 1,860.


## Weighted and streaming Gaussian/logistic classifier batch

Extended existing Gaussian NB and logistic fit APIs with optional real64
sample weights. Gaussian NB adds partial_fit, optional fixed priors and
n_features_in_. Initial streaming calls declare the full class set; repeated
class declarations normalize order/duplicates and later calls reject changed
classes, unknown targets and feature counts. fit resets accumulated state.
Weighted population moments use Chan/Golub/LeVeque merging. The pinned source's
unweighted current-batch smoothing, current-epsilon subtraction and <=1e-8
weighted-mass moment skip are retained, including count accumulation for
skipped updates. Priors use source length/nonnegative/np.isclose validation
and are retained without normalization. Zero empirical mass and zero
variances preserve source IEEE undefined posteriors; zero priors and unseen
classes are handled explicitly. Gaussian prediction uses joint likelihoods,
including the source's first-NaN argmax behavior. This replaces the former
native zero-variance fit rejection.

Weighted logistic uses mean weighted loss plus 1/(C*sum(weights)) L2 strength,
diagonal equilibration of its Newton system and safe combined-penalty
formation. Tiny C can be supported with large weights even if 1/C alone
would overflow. Binary/multiclass prediction now uses source raw decision
scores, preserving labels when sigmoid/softmax probabilities round to .5
or a uniform vector. Self-training prediction delegates to its base; its
confidence/probability paths require finite base probabilities. Existing
unweighted fit calls remain supported. Inputs/weights are finite dense
real64 arrays; weights are nonnegative and bounded by documented numerical
working ranges. Signed/scalar weights and logistic class_weight are outside
this interface.

CalibratedClassifierCV now forwards training-fold weights to Gaussian NB
and logistic and forwards all weights for the nonensemble full-data refit.
All three supported bases therefore allow weighted CV; KNN retains source
calibration-only weighting. Two obsolete unsupported-weighted-CV diagnostic
cases now check weight-length errors. Forty-two additional diagnostics cover
streaming class/state/feature rules, priors, weights, finite data, moment and
penalty working ranges, zero-mass logistic rejection and undefined base
responses in calibration/self-training. All 42 passed isolated checked probes.

The new reference suite has 16,536 groups across 681 fixtures and five seeds
(0,7,42,2026,314159), including 1,441 Gaussian fitted/streaming/refit states.
Cases span one through four observed classes (plus an additional declared
unseen class in selected streaming fixtures), 1/4 features, uniform/integer/random/
log-scale/tiny/zero weights, missing or declared unseen classes, fixed/zero/
approximately normalized priors, changing batch scales and smoothing,
constant data and tiny-C/large-weight logistic fits. Counts and epsilon use
zero absolute tolerance with 4e-11 relative tolerance; other statistics use
4e-11 absolute/relative, Gaussian posterior outputs 3e-8 and logistic outputs
3e-6. Prediction labels are exact. All 481 logistic fits additionally satisfy
an independent normalized gradient bound of 3e-8.

Oracle accounting: 15,309 ordinary direct sklearn groups, 681 input
identities, 481 independent optimality groups, 60 limiting-intercept groups
from 30 zero-weight-class cases, and five direct sklearn Newton-Cholesky
groups from one certified Newton-CG recovery. Zero-weight classes can have
no finite unregularized intercept optimum; their coefficients, probabilities,
predictions and gradient remain checked, while finite intercept/decision
representations are recorded separately. Conditional softmax common-offset
comparison is available for nonunique intercept gauges; none was needed in
the final checked prototype corpus.

The saved synthetic line-search regression has source Newton-CG normalized
gradient 5.58358e-6 despite requested tolerance 1e-12. Native gradient is
7.37199e-12; source Newton-Cholesky reaches 1.11022e-16. Native and recovered
source losses both equal .6924822701505227, below the stalled source loss
.6924822702129585. Recovery requires an independently measured source stall,
a much smaller native gradient and source-recovered loss agreement; it does
not increase public-output tolerances or remove the difficult case.

The expanded calibration suite has 13,760 groups over 810 fixtures: 7,166
direct source groups, 4,160 source-map certificates, 1,620 input identities,
810 composition identities and four machine-precision argmax exceptions.
It adds weighted ensemble/nonensemble CV for Gaussian NB and logistic.
Native analytic tests cover weighted moments, pooled/streaming equivalence
at zero smoothing, zero-weight batches, source mass-cutoff boundaries,
fixed priors, quiet undefined outputs, integer-weight replication, tiny-mass
intercept fitting, safe C rescaling and raw-score prediction/delegation.
A runnable weighted_streaming_classifiers example demonstrates both models.

Original contributions remain MIT with retained sklearn BSD attribution;
no dependency or public extension is added. Core type count stays 177.
Inventory: 88 source modules, 89 native test programs, 68 reference suites
and 171,742 core result groups, plus 28 separate NumPy RNG groups.
Isolated invalid-input inventory is 1,902.


### Independent multioutput wrappers (2026-11-21)

Added core counterparts of MultiOutputRegressor and MultiOutputClassifier,
with independent fitted copies per target, refit/deep-copy behavior, ragged
per-output class probabilities, exact subset classification accuracy and
uniform-average regression R2. Streaming supports native SGD regression,
Gaussian NB and SGD classification; training weights follow each native
base's actual capabilities. The supported base families, first-call class
lists, source partial_fit metadata/shrink quirks and inherited limits are
specified in API.md. Chains and a general estimator protocol remain outside
this implementation. See example/multioutput.f90.

The 3,030-group suite spans 255 fixtures across five seeds, six regression
base configurations and eight classification base configurations. It checks
single/multiple outputs, constant targets, nonconsecutive labels, weighted
fits and scores, pretrained template preservation, streaming first batches
with unseen classes, shrink/refit, averaged SGD, hinge prediction, ragged
probabilities, fitted coefficients/counters and input preservation.

Ordinary floating comparisons retain 3e-8 prediction/state/score and 3e-7
probability tolerances. Exceptions have explicit certificates: source-stump
and exhaustive SSE checks establish equal-gain CART splits at every native
node and reconstruct predictions; Complement NB differing class selections
must lie within 64 machine epsilons (scaled by source score magnitude) of
the source maximum. Weighted constant-target source R2 roundoff is bounded
and checked against source R2 on identically translated targets/predictions.
No difficult fixtures are removed or ordinary tolerances widened.


This batch passed both checked and release runs of all 90 native programs
and 3,030 reference groups, the original 192-group integration comparison,
all 1,934 isolated input diagnostics and the checked example. Checked and
release reports match exactly, including certificate accounting. Inventory:
179 core types, 89 modules, 90 native programs, 69 core reference suites and
174,772 core groups, plus 28 separate NumPy RNG groups. Other reference
suites retain their previously recorded results and were not all rerun here.


### RegressorChain and source lazy SGD scaling (2026-11-21)

RegressorChain joins the independent multioutput wrappers. Native fitting
supports observed preceding targets or cross-validated predictions, explicit
one-based ordering, seeded NumPy-compatible random ordering, integer KFold
and explicit index partitions. Query outputs are restored to original target
order. Native base families and training-weight limits are inherited from
multi_output_regressor. Full fits receive weights; source-style internal
cross_val_predict fits are unweighted when metadata routing is disabled.
Single-output chains do not evaluate unused CV folds. Explicit iterable
training indices preserve source behavior, including repeats or overlap.
See API.md for configuration, metadata and scoring contracts.

The new reference suite covers 430 fixtures over five seeds and six native
base families. It checks ordering, teacher-forced versus out-of-fold training,
weighted fits/scores, singleton queries, fitted coefficients and counters,
refits and preserved pretrained templates/inputs. Tree validation certifies
every full and CV-fold tree against source stumps and exhaustive SSE scans,
then reconstructs both training features and sequential query predictions.
One duplicated-training-row fixture selects a different equal-gain split;
its two prediction/score groups carry a separate source tie certificate.
Ordinary prediction/state/score tolerance remains 3e-8; no difficult fixture
was removed or tolerance widened.

A weighted cross-validated SGD chain exposed roundoff accumulation in the
former eager L2 updates. The shared SGD implementation now uses source
WeightVector64 lazy scaling, ordered dot products, scale reset below 1e-9
and final materialization. The saved synthetic fixture
validation/fixtures/regressor_chain_lazy_l2.json is checked against generated
inputs so later fixture edits cannot silently drop it. On that input, the
former code's maximum scaled component error was 4.3424e-8, failing the
original 3e-8 tolerance; lazy scaling reduced it to 7.9580e-9. Existing dense
averaging behavior is retained. This remains original MIT work with retained
sklearn BSD attribution; no dependency or public extension is added.


For SGD chains the suite independently reconstructs native out-of-fold
feature matrices, checks every generated column against source fold fits
and predictions, and compares full fitted coefficients/counters to source
fits on those identical matrices. Full-chain query predictions and scores
still compare against the ordinary sklearn chain. All comparisons retain
3e-8 tolerance. This separates amplification of tiny feature-rounding
changes from a fitting or routing error: the saved fixture's release
coefficient discrepancy against the unconditioned source chain was about
4.0191e-8, while identical-design source fits reproduced native stages
exactly in the diagnostic probe. Rounding witnesses are recorded separately
from the consistently checked conditioned-stage groups; their presence can
differ by optimization profile. The full checked/release native test runs
were completed before adding these reference-driver witnesses; numerical
production sources were unchanged afterward.


This batch passed all 91 native programs in both profiles, the new
3,211-group chain suite, the affected 3,030-group multioutput suite,
216-group SGD regression suite and 306-group SGD classification suite.
The original 192-group integration comparison, all 1,963 input diagnostics
and the checked example passed. Inventory: 180 core types, 89 modules,
91 native programs, 70 reference suites and 177,983 core groups, plus
28 separate NumPy RNG groups. Other suites retain their recorded results
and were not all rerun in this batch.


### ClassifierChain (2026-11-21)

Added ordered and seeded-random classifier chains with native label,
probability, log-probability and decision-score feature responses, ordered
method preferences, observed-target or out-of-fold training features,
weighted full fits and restored target-column query outputs. Integer CV
uses source-compatible first-appearance StratifiedKFold, including minority
classes smaller than the fold count. Probability CV pads absent fold
classes before selecting the second global class column. Explicit iterable
folds preserve repeated/overlapping training rows and require test partitions.
Single-output chains ignore unused CV specifications. API.md describes
base-family capabilities, binary query-response restrictions and scoring.

The 8,665-group reference suite uses 785 fixtures over five seeds and eight
base families. It covers all four response methods and capability fallback,
log-loss/hinge/modified-Huber SGD, averaging, weighted fits and signed query
weights, rare and constant classes, multiclass label chains, shared nonzero
indicator labels, single outputs/queries, refits and input/template copies.
Source CV plans are supplied independently to the reference driver's feature
witnesses. Fitted means/variances, priors, log-feature probabilities,
coefficients, counters, KNN stored arrays and class/order metadata are
checked against source fits on those actual feature matrices.

Tree witnesses check every full and CV-fold node using source stumps and
exhaustive Gini scans, including genuinely distinct equal-optimum candidates,
and reconstruct all native CV and sequential query paths. Complement NB
alternatives require source log-score gaps within a scaled 64-epsilon bound;
this also checks any near-tied labels propagated into later chain features.
Ordinary probabilities/decisions/linear state and CV feature comparisons
retain 3e-7 tolerances, NB moments/log parameters use 3e-8, and score/tree
identity checks use 3e-12. These exceptions are explicitly reported by oracle
and witness rather than widening tolerances or removing difficult inputs.


This batch passed both profiles of all 92 native tests and 8,665 new
classifier-chain groups, plus 3,211 regressor-chain and 3,030 multioutput
regression groups. All 2,015 input diagnostics, the original 192-group
integration comparison and the checked example passed. Inventory: 181 core
types, 89 modules, 92 native programs, 71 reference suites and 186,648
core groups, plus 28 separate NumPy RNG groups. Other suites retain their
recorded results and were not all rerun in this batch.


## Multiclass reductions

The three sklearn multiclass wrappers are available through sklearn_multiclass
and the sklearn facade. Supported native classifiers include linear/SGD,
naive Bayes, nearest-neighbor, CART and SVM families; OVR/OVO also accept the
supported regression bases. OVR supports multilabel indicator matrices.
Gaussian NB/SGD classification support streaming for OVR/OVO. Output-code
classification uses seeded row-major legacy NumPy codebooks and explicit
Euclidean decoding. Constant heads retain source response/tag semantics.
See API.md for exact interfaces, weighting, shapes and streaming restrictions.
Binary Perceptron now exposes one coefficient row and one decision column.

All 93 native programs passed checked and release profiles. Both profiles
passed 27,490 new reference groups, 8,665 classifier-chain regression groups
and 3,030 multioutput regression groups; all 2,112 input diagnostics, the
192-group integration comparison and the checked example passed. Core
inventory is 184 types, 90 source modules, 72 reference suites and 214,138
retained result groups, plus 28 separate NumPy RNG groups. These are subset
API counterparts, not a percentage of full sklearn feature parity.


## Voting ensembles

VotingClassifier and VotingRegressor counterparts combine named heterogeneous
native base estimators, with dropped slots, fresh fits, encoded classifier
labels, weighted hard/soft votes or regression averages, per-base transforms,
probability tensors, fit_transform, scores, refits and deep copies. Finite
signed voting weights follow source, including zero mass for hard voting
and bincount limits for negative votes. Soft/regression predictions require
nonzero active weight mass. Training weighting and probability availability
follow each base. API.md lists supported families and rank adaptations.

Both profiles passed all 94 native programs and 16,965 new reference groups
over 1,200 fixtures, plus the 192-group integration comparison. All 2,150
input diagnostics and the checked example passed. Inventory: 186 core types,
91 modules, 73 core reference suites and 231,103 retained result groups, plus
28 separately counted NumPy RNG groups. Sparse paths, arbitrary estimator
protocols and scheduling remain outside this dense native subset.


## Permutation feature importance

The core inspection function supports scalar and multiple-score callbacks,
raw importances, population means/deviations, repeat controls, source-compatible
legacy seeding, all three subsampling branches and exact cumulative shuffles.
Callbacks receive one-based original row identities and optional subset
weights, allowing any fitted native model and target/scoring shape. There
is no automatic estimator-score dispatcher; see API.md for the callback
contract. The baseline uses all rows, even with subsampled permutations.
NaN features, empty fractional subsets, signed finite scoring weights and
quiet nonfinite score reductions follow documented source behavior.

Both profiles passed 17,820 new groups across 600 fixtures, checking every
callback's exact rows, feature matrix and weights as well as its scores and
all importance summaries. All 95 native programs, 2,171 input diagnostics,
the 192-group integration comparison and the checked example passed. The
result is a container rather than a new estimator: the core type count
remains 186. Inventory: 92 source modules, 74 core reference suites and
248,923 retained core groups, plus 28 separate NumPy RNG groups.


## Brute partial dependence and ICE

Partial dependence evaluates scalar/multiple numeric responses through a
prediction callback, exposing averages, individual curves or both. Sorted
unique/categorical axes and Cunnane percentile grids follow source, with
last-axis-fast Cartesian ordering and last substitution for duplicate feature
indices. Signed weighted means preserve raw individual curves. Ragged axis
vectors describe the flattened grid dimension. See API.md for response
selection, missing-data behavior, fixed-rank shapes and callbacks. Tree
recursion, Python response dispatch and display objects are not implemented.

Both profiles passed all 96 native programs and 13,395 new groups across
835 fixtures, plus 17,820 existing permutation-importance groups. Thirty
observed source binary-response reshape failures per report are explicitly
recorded; 60 result groups use actual source-prediction mean/ICE identities.
All other comparisons retain their ordinary tolerances. The 192 integration
comparisons, all 2,202 input diagnostics and the checked example passed.

Core type count remains 186: response/axis result containers are excluded.
Inventory: 93 source modules, 75 core reference suites and 262,318 retained
core groups, plus 28 separately counted NumPy RNG groups. Original work stays
MIT, with retained sklearn and SciPy source BSD notices and no new dependency.


## Weighted CART and standalone ExtraTrees

`extra_tree_classifier` and `extra_tree_regressor` are the scikit-learn
ExtraTreeClassifier/ExtraTreeRegressor counterparts. They expose the inherited
single-tree controls and methods: `fit`, `predict`, `apply`, `score`,
`get_depth`, `get_n_leaves`, classifier `predict_proba`, and learned
`nodes_`, `feature_importances_`, `n_features_in_` and `max_features_`.
The classifier supports Gini, entropy and log-loss; regression supports squared
error. With configured `max_features=0`, ExtraTree classification uses
floor(sqrt(n_features)), while ExtraTree regression uses all features. Positive
integer feature counts override these defaults. These single estimators are
separate from the existing `extra_trees_classifier`/`extra_trees_regressor`
ensembles.

Both decision trees and ExtraTrees accept optional `sample_weight` in `fit`
and `score`. Weights must be finite and have one entry per row. Zero fitting
weights remove rows from node cardinalities, including min_samples_leaf tests;
`weighted_n_samples` stores weight mass separately. Signed fitting weights
follow sklearn's child-mass and impurity rules. Degenerate total masses can
produce NaN/Inf responses and importances, as in sklearn; prediction preserves
those values. Regression score rejects nonfinite predictions or targets;
singleton R2 returns quiet NaN. Score weight sums must be nonzero when an
average is required.

Decision trees and standalone ExtraTrees use float32 feature
geometry, the observed zero split tolerance of the installed sklearn 1.6.1
binary, and seeded legacy NumPy /
xorshift feature selection. Training features must be finite and within
float32 range. Queries may contain NaNs: each missing feature follows the
larger active-row child, with ties going right, matching a source tree trained
without missing values. Query infinities and float32 overflow are rejected.
NaN training, sparse input, pruning, monotonic constraints, fractional feature
counts, alternate regression criteria, and multioutput single-tree fitting
remain outside this subset. Native max_depth=0 retains its existing root-only
meaning; sklearn requires a positive depth.

DecisionTree fitting without weights uses unit weights in the same source-based
kernel. This replaces the former real64/native RNG path and also changes the
underlying trees in existing unweighted ensembles. Zero-gain splits allow a
later split to learn XOR. Prefix recomputation can select a different optimal
split from sklearn's incremental C criterion at floating-point ties; an exact
source tree topology is not guaranteed in those cases.

Voting, independent multioutput estimators, regressor/classifier chains, and
multiclass reductions accept ExtraTree bases through their decision-tree
interfaces and forward training weights to both tree families. Tree streaming
(`partial_fit`) remains unavailable. The new internal `sklearn_ieee_math`
module handles quiet nonfinite arithmetic in weighted trees and composite
reductions; it adds no public estimator or external runtime dependency.


## BaggingClassifier and BaggingRegressor

`bagging_classifier` and `bagging_regressor` are available from `sklearn`,
`sklearn_ensemble`, and `sklearn_bagging`. Call
`fit(x,y,estimator,sample_weight)` with an optional native base template;
omitting it selects the corresponding DecisionTree. Classification targets
are integer label vectors; regression targets are real64 vectors. The fitted
template and supplied arrays are copied/preserved. `predict`, classifier
`predict_proba`, and weighted `score` expose the combined model.

Controls are `n_estimators=10`, nonnegative `random_state=0`, `bootstrap=.true.`,
`bootstrap_features=.false.`, `warm_start=.false.` and `oob_score=.false.`.
Configure integer `max_samples`/`max_features`, or their real64
`max_samples_fraction`/`max_features_fraction` counterparts. Zero means unset;
when neither count nor fraction is set the whole population is used. A count
and fraction cannot both be set for an axis. Positive fractions truncate to an
integer row count; feature counts clamp to at least one, as in source. A tiny
positive row fraction can select zero rows; whether that fit is defined depends
on the base. Counts may not exceed the population even with replacement.
Sampling with/without replacement follows sklearn's RandomState draws,
tracking/permutation/reservoir switching and feature-before-row draw order.
The base seed is derived independently from each member seed.

Supported bases match the native voting families: logistic regression,
Gaussian NB, SGD classification, KNN, decision/Extra trees, multinomial,
Bernoulli and Complement NB, perceptron, passive-aggressive classification,
ridge classification, LinearSVC, SVC and DummyClassifier; and the linear
regression hierarchy, SGD regression, KNN, decision/Extra trees, KernelRidge,
and DummyRegressor. Each base retains its native numerical conventions and
supported options. If the native base supports weighted fit, bagging trains on
all rows with bootstrap multiplicities (or a subsampling mask) multiplied by
optional finite fitting weights. Bagging also provides a weighted least-squares adapter for exact
`linear_regression` and `ridge` bases, using the full original row count for
centering and numerical rank decisions. This adapter accepts nonnegative
weights; the public standalone base fit signatures are unchanged. Other bases
train on sampled rows; explicit fit weights are rejected for those bases. This is the native capability
boundary, including bases whose Python counterpart additionally accepts
weights. The resulting sampled-row fit can have different learned class
metadata or smoothing from a source weighted fit. Sparse data, arbitrary
Python-style estimator protocols, metadata routing, parallel jobs, decision
functions and log-probability aggregation are outside this initial API.

Fitted `estimator_` stores the copied template. `estimators_(j)%estimator`
contains each fitted dynamic model. `estimators_features_(j)%values` and
`estimators_samples_(j)%values` hold actual one-based draws, including repeats;
`seeds_`, `n_estimators_`, `n_features_in_`, `max_samples_` and `max_features_`
record the fit. Classifiers add sorted `classes_` and `n_classes_`. Per-member
class probabilities are aligned to global classes; bases without probability
methods contribute hard votes. Native decision trees retain their depth limit
of 100 when the source default is unbounded. Fit inputs must be finite. NaN
queries are forwarded to bases that support them; infinities are rejected.

Warm starts append new fitted members and skip prior ensemble seed draws.
An unchanged count preserves the ensemble; reducing the count is rejected.
Feature width and classification labels must stay fixed during warm starts.
Stored sampling records always describe every actual fitted member. In
sklearn 1.6.1, `estimators_samples_` after warm start exposes only the latest
batch, because its generator uses the latest seeds. Native records preserve
all fitted draws; validation identifies this source property discrepancy.
Sampling records also retain original draws when later warm-start controls
change, rather than regenerating them from the latest configuration.

OOB scoring requires bootstrap and forbids warm start. Classifiers expose
`oob_decision_function_` and `oob_score_`; uncovered rows have NaN probabilities
but participate in accuracy through the first encoded class, as in sklearn.
Regressors expose `oob_prediction_` and `oob_score_`; uncovered rows retain zero
predictions for R2. OOB scores are unweighted. The source classifier OOB
implementation directly adds local probability arrays without class
alignment: a single local class broadcasts across global classes, while other
incompatible class widths fail. This source behavior is retained and the
incompatible-width case is diagnosed explicitly. Regression scoring rejects
nonfinite predictions/targets and returns NaN for a singleton. Refit without
OOB deallocates old OOB response arrays; consult the arrays' allocation status
before using the scalar OOB score.


## StackingClassifier and StackingRegressor

`stacking_classifier`, `stacking_regressor` and the named `stacking_estimator`
container are available from `sklearn`, `sklearn_ensemble` and
`sklearn_stacking`. The container is the same native type as voting_estimator,
with `name`, allocatable polymorphic `estimator`, and `drop` components.
Call `fit(x,y,estimators,final_estimator,sample_weight,splits)`; the final
estimator, weights and explicit index_split arrays are optional. Targets are
integer label vectors for classification and real64 vectors for regression.
Input arrays and supplied estimator templates are preserved.

By default, `cv=5` uses nonshuffled StratifiedKFold for classification and
KFold for regression. Classification allocation follows source first-appearance
label ordering and permits minority counts below cv when not every class is
too small. Supplied one-based splits must have nonempty, bounded train/test
indices and test sets that partition every row exactly once. Explicit train
order, test order and repeated training indices are retained; train/test
overlap is permitted as in source's iterable split protocol. `prefit=.true.`
uses copied fitted base states and their full training responses instead of CV;
it cannot be combined with explicit splits. Prefit bases retain their original
label responses, including when stacking encodes the final classifier targets.

Full fitted bases are used by `transform` and prediction. Separate fold fits
produce the out-of-fold design used to train the final estimator. Classifier
`stack_method='auto'` prefers predict_proba, then decision_function, then predict;
these three names can also be requested explicitly. Binary probability stacks
drop their first column. Multiclass probability and decision columns are kept;
missing fold probability classes are padded with zero, and source's unsupported
imbalanced decision-fold cases are diagnosed. `passthrough=.true.` appends the
original features after the base responses. Regressors stack scalar predictions.
`fit_transform` returns the full fitted-base transform, not the OOF design.

The default final classifier is native LogisticRegression with C=1, Newton
optimization, tolerance 1e-10 and at most 500 iterations. The default final
regressor is RidgeCV with leave-one-out selection over [.1,1,10]. The existing
`ridge_cv` type now uses LOO/GCV when `cv=0`; its historical standalone default
remains five-fold selection. Positive finite LOO alphas are required. LOO
weights are supported through the native ensemble dispatch; the standalone
inherited fit signature remains unweighted. GCV reports selected `alpha_`,
`best_score_` and per-candidate `cv_scores_`, including quiet singleton NaNs.

Supported base/final families are the explicit native voting/bagging dispatch
families for the appropriate target kind. Weights require native weighted fit
support; exact OLS/Ridge use the shared weighted SVD adapter, and LOO RidgeCV
supports weighted selection. In prefit mode only the final estimator is fitted,
so its weight capability is checked independently of the bases. Tree fits use
the weighted float32/xorshift builder with implicit unit weights when weights
are omitted, matching source split geometry and random feature ordering.
Existing base optimization choices and floating-point tie conventions apply.

Learned fields include `estimators_` (active named fitted slots),
`active_indices_`, polymorphic `final_estimator_`, `stack_method_`,
`n_feature_outs_`, `n_features_in_`, and `n_features_out_`. Classifiers add sorted
`classes_` and `n_classes_`. `training_meta_features_` retains the actual final
training design; `cv_splits_` and `cv_estimators_(fold,base)%estimator` retain the
fold plans and fitted states for inspection. These are native introspection
containers, not additional algorithms. They increase memory use relative to
source's transient fold fits. Passthrough layout is captured at fit time.

`predict`, `transform` and weighted `score` are supported; classifiers also
expose predict_proba/decision_function when the final estimator supports them.
Binary decision responses are real64 matrices with one column. Regression
score uses source force-finite R2 and returns NaN for a singleton. Fit features
must be finite. Query NaNs are forwarded to bases, and any passthrough NaNs
then reach the final estimator's own checks. Infinities and invalid widths are
diagnosed. Meta designs exceeding the native numerical working range are
rejected. Sparse data, multilabel indicators, arbitrary estimator protocols,
metadata routing and parallel fitting remain outside this subset.


## Target encoding

`target_encoder` is exported by `sklearn_preprocessing` and `sklearn`. Dense
real64 numeric category matrices support NaN as a learned category; infinity
is rejected. `fit(x,y)` learns full-data encodings and `transform(x)` applies
them, using the full target mean for unseen categories. Integer and real64
vector targets are accepted and must be finite. String/object features, sparse inputs, multiple
targets and feature-name generation are outside this native interface.

`target_type='auto'` selects continuous regression for fractional targets
or floating targets outside the NumPy int64 label range,
binary for one/two integral labels, and multiclass for more integral labels.
Explicit `'continuous'`, `'binary'` and `'multiclass'` are available. Sorted
numeric `classes_` define the encoded class order. Binary uses one encoded
column per input feature, including source's explicitly forced binary mode
with more than two classes (ordinal class indices). Multiclass produces
feature-major class columns; explicit multiclass with two classes is diagnosed
because the source implementation fails on that configuration. One-class
multiclass follows source's all-zero LabelBinarizer result.

`auto_smooth=true` (default) uses source empirical Bayes shrinkage from global
variance and within-category sums of squared deviations. Constant targets and
categories absent from a training fold fall back to its target mean. Set
`auto_smooth=false` and `smooth>=0` for fixed smoothing; zero gives unsmoothed
category means for observed categories. Fixed smoothing is evaluated as a
convex mixture to avoid overflowing the product of smoothing and target mean.
Finite target magnitude is bounded by `sqrt(huge(real64))/(4*n_samples)` so
variance calculations cannot overflow under the checked floating-point profile.

`fit_transform(x,y)` returns out-of-fold features and retains the full-data
encodings for future `transform` calls. This distinction prevents training-row
target leakage; it is intentional that `fit_transform(x,y)` differs from
`fit(x,y); transform(x)`. Continuous targets use KFold and classification uses
StratifiedKFold with source's first-appearance class ordering. Defaults are
`cv=5`, `shuffle=true`, `random_state=0` (the native deterministic scalar seed,
compared to Python's unspecified random state). Seeded shuffles match NumPy's
legacy RandomState. With `shuffle=false`, the native seed is ignored (the
reference recipe passes Python `random_state=None` for that configuration).
Minority classes smaller than cv are allowed if at least
one class has cv rows. Classification cross-fitting requires integral labels,
matching the source splitter's target validation; fit alone permits continuous
numeric labels in explicitly forced binary mode. `cv_splits_` retains sorted
1-based training/test indices as a native inspection container.

Automatic categories are sorted finite numeric values followed by NaN, when
observed. To specify categories, allocate `categories(n_features)` and assign
`categories(j)%values` (a `target_category` record). Each list must be nonempty,
sorted, unique, with optional NaN last; unused specified categories are allowed.
`categories_` and `encodings_` use the same ragged `values(:)` representation.
`encodings_` is feature-major and then class-major for multiclass. The vector
`target_mean_` has one element for continuous/binary targets and one per class
for multiclass; `n_features_in_` and `n_features_out_` describe learned widths.
Changing constructor controls after fitting does not alter transforms. Refit
replaces learned state and clears earlier cross-fitting diagnostics.

The native implementation preserves two sklearn 1.6.1 explicit-category edge
behaviors: omitted training values contribute to the first category's learned
statistics, while their transformed values still use the global mean;
`fit_transform` overwrites all such unknown rows with the final fold's target
mean, regardless of their held-out fold. These apply only when user-specified
categories exclude observed training values. Full automatic categories avoid
these source quirks. `target_category` and `cv_splits_` are data containers,
not additional estimator algorithms or extensions beyond sklearn. Category discovery
uses the existing dense insertion sort and lookup scans category lists; this
implementation is intended for moderate sample sizes and category counts.


## Numeric ordinal encoding

`ordinal_encoder` is exported by `sklearn_preprocessing` and the `sklearn`
facade. It provides `fit(x)`, `transform(x)`, `fit_transform(x)` and
`inverse_transform(z,unknown_mask,infrequent_mask)` on dense real64 matrices.
Numeric categories may include NaN; infinity and empty data are rejected.
Codes are zero-based, stored as real64, and each input feature produces one
output feature. String/object categories, integer output dtypes and feature-name
metadata are outside this native interface.

`categories` is optional: allocate one `ordinal_category` record per input
feature and assign its `values(:)` to provide sorted unique numeric categories,
with optional NaN last. Unallocated categories select automatic discovery.
Unused specified categories are retained. `categories_` and
`infrequent_categories_` use the same ragged representation; an empty
infrequent vector means no grouped category, including disabled grouping. `ordinal_category` aliases the
numeric container already used by TargetEncoder, not an additional algorithm.
`n_features_in_` records the fitted width. `mappings_` retains category counts,
zero-based codes, 1-based infrequent indices and the 1-based missing index
(zero means absent) as native inspection data.

`handle_unknown='error'` is the default and requires unallocated `unknown_value`
(the native representation of Python None). Use
`handle_unknown='use_encoded_value'` and assign the allocatable scalar
`unknown_value` an integral real64 value or NaN to encode unseen categories.
The allocatable `encoded_missing_value` defaults to unallocated, meaning NaN;
assign an integral real64 value or NaN to override it. NaN is treated as missing
only if included in the learned categories; otherwise it is unknown. Unknown
training values excluded by explicit categories are allowed only in the
use_encoded_value mode and do not contribute to category counts.

`min_frequency_count=0`, `min_frequency_fraction=0` and `max_categories=0`
disable their respective controls (Python None). A positive integer count or
a fraction strictly between zero and one marks categories whose counts are
strictly below the threshold. Count and fraction are mutually exclusive.
Fractional thresholds use all training rows, including missing/unknown rows.
A positive `max_categories` caps finite category codes including the grouped
infrequent code. Missing and unknown sentinels do not count toward the cap.
NaN is excluded from frequency grouping. Source stable count sorting is
retained: ties favor larger sorted numeric categories as frequent categories.
Grouped finite categories receive the last code after frequent categories.
Grouping parameters are consumed at fit time; missing/unknown controls remain
live for transforms, matching valid source parameter changes. Refit replaces
learned categories and mappings, including changes of feature width.

Inverse transforms return real64 values. Source None values and the literal
`infrequent_sklearn` have no numeric value, so both use NaN placeholders;
request the optional allocatable logical `unknown_mask` and `infrequent_mask`
outputs to distinguish them. Actual numeric NaNs have both tags false. Source
inverse ordering is retained: replace the missing sentinel, detect unknown
and infrequent codes, and let the infrequent marker win if markers coincide.
Sentinel collisions admitted by source's grouped-cardinality validation are
also admitted here and may discard category identity. Native inverse transforms
preserve their input; source inverse may mutate its encoded array.

The inverse follows source NumPy indexing conventions, including truncating
fractional codes toward zero and wrapping negative indices. Invalid indices
produce a diagnostic before array access. sklearn 1.6.1 has a grouped/missing
index inconsistency: with a NaN category, its inverse may decode a grouped code
as a numeric/NaN category or raise IndexError for the original missing index.
The native implementation preserves observable source results and diagnoses
those source index failures. Forward encodings and learned grouping remain
usable for these configurations. The reference suite retains and tests these
source exceptions explicitly. Category discovery uses the existing dense
insertion sort; transform category lookup uses binary search.


## GraphicalLasso cross-validation

`graphical_lasso_cv` is exported by `sklearn_covariance` and the `sklearn`
facade. `fit(x)` selects an alpha and fits the full-data sparse inverse
covariance. It inherits covariance, precision, location, objective/duality-gap
history, iteration/convergence state, Mahalanobis distances and covariance-error
norms from the existing GraphicalLasso family. Its `score(x)` uses the fitted
location and precision with the source Gaussian likelihood, including negative
infinity for a singular precision. Inputs are finite dense real64 arrays with
at least two features. Fitting bounds their magnitude by
`sqrt(huge(real64))/(16*n_samples*n_features)` to protect covariance arithmetic.
Only `mode='cd'` and `covariance='empirical'` are supported; LARS, sparse data,
metadata routing and parallel jobs remain outside this native interface.

Unallocated `alphas` selects a refined grid: `n_alphas=4` points per refinement,
`n_refinements=4`, initial upper bound equal to the maximum absolute off-diagonal
full-data empirical covariance and initial lower bound one hundredth of that.
Set allocatable `alphas(:)` for an explicit positive finite grid; supplied order
and duplicates are retained during each fold's warm-start path, after which all
results are stably sorted descending. Explicit grids use one refinement. Both
grid forms require at least two points because source 1.6.1 fails with a single
point. Refinement follows source neighboring-score and failed-candidate rules;
exact score ties select the last candidate in descending order, favoring the
smaller alpha. `alpha_` is the selected value; inherited `alpha` mirrors it after
fitting. Native `best_score_` records the selection score.

`cv=5` uses nonshuffled KFold. Allocate constructor `splits(:)` with native
`index_split` records for explicit folds; indices are 1-based, nonempty and
bounded. Their order, repeated rows and overlap are allowed, and they need not
partition the data. One explicit fold is supported. `cv_splits_` retains the
actual folds. Every refinement starts a fresh path per fold; within a path,
each successful covariance initializes the next candidate with source's 0.95
off-diagonal shrinkage. Failed candidates have score negative infinity and
retain the preceding successful warm start. Fold paths use
`int(0.1*max_iter)` iterations, including the source zero-iteration convention.
The final model uses `max_iter` (default 100). Nonnegative max_iter, positive
finite tol/enet_tol/eps and positive n_refinements are required.

An important source convention is preserved: positive-alpha CV paths center
both training and test data independently, regardless of assume_centered.
The full-data covariance and location honor `assume_centered`. The final
alpha-zero report row instead uses an EmpiricalCovariance fold fit and scores
test observations relative to that fold's training mean. That appended row is
never considered for selection. For an exactly diagonal full covariance,
source's automatic grid itself can contain alpha zero and NaNs; the native
implementation preserves that grid, its failures and possible zero selection.
A singular direct alpha-zero fit is diagnosed, matching source's unhandled
inverse failure. Successful zero-alpha fits have n_iter_=0 and one cost pair;
a positive-alpha zero-iteration fit has n_iter_=1 and an empty cost history.

`cv_results_` contains `alphas(:)`, `split_test_score(candidate,fold)`,
`mean_test_score(:)` and population `std_test_score(:)`. Negative-infinity means
and NaN standard deviations are preserved for failed folds. Native inspection
arrays `empirical_covariance(:,:,fold)`, `empirical_precision(:,:,fold)` and
`empirical_test_covariance(:,:,fold)` retain the actual alpha-zero endpoint
matrices; these are data containers, not extra estimators or algorithms.
Refit replaces the result arrays and fitted model. The existing scalar
GraphicalLasso remains unchanged except for source's zero-Gram-diagonal CD skip,
needed when a CV training fold has constant features.

Candidate numerical failures are handled with a local IEEE exception scope,
like NumPy errstate: invalid/divide-by-zero/overflow halting modes and their
flags are restored on return. Native tests exercise restoration after NaN-grid
and failed-path calculations. Invalid input, a singular required direct inverse,
or a failed final fit still uses the package's normal diagnostic/error-stop API.

The numerical backend is the existing dense Jacobi eigensolver and native
partial-pivot determinant calculation. Near-singular empirical endpoints can
differ numerically from SciPy's eigensolver/LU at the same input precision.
Validation retains such cases and verifies covariance moments, inverse backward
errors or all four Moore-Penrose equations, and conditioned likelihoods. For a
rank-deficient precision, a partial-pivot growth/Weyl bound checks determinant
roundoff when its sign is numerically indeterminate. These checks apply only
to the appended, unselected empirical endpoint; positive-alpha path scores,
selection and final fits retain ordinary direct sklearn comparisons. Certificates
and source warnings are retained in the reference reports; no ordinary numeric
tolerance is widened for these endpoints. Which singular folds require a
certificate may vary across optimization profiles because determinant signs
near zero are roundoff-sensitive.

## LogisticRegressionCV

`type(logistic_regression_cv)` extends `logistic_regression` in
`sklearn_linear_model` and is re-exported by `sklearn`. It implements dense L2
binary and joint multinomial classification with a native full Newton solver.
Set `cs(:)` to positive candidate inverse regularization strengths, or use
`n_cs` (default 10) for a logarithmic grid from 1e-4 to 1e4. Candidate order and
repetitions are preserved. `cv` defaults to five nonshuffled stratified folds;
optional `splits(:)` supplies explicit 1-based `index_split` train/test indices.
Explicit folds may overlap or contain repeated rows and need not partition data.

`fit(x,y,sample_weight)` accepts nonnegative sample weights. Optional
`class_weight(:)` follows sorted `classes_`; alternatively set
`class_weight_balanced=.true.` for global unweighted inverse class frequencies.
Supported `scoring` values are `accuracy`, `balanced_accuracy`, `neg_log_loss`,
and binary `roc_auc`. Scoring uses raw sample weights, without class weights.
`score(x,y,sample_weight)` uses the current scoring control; ordinary
`logistic_regression%score` also accepts optional accuracy weights.

`scores_(fold,C)` stores the shared joint classification scores.
`coefs_paths_(model_class,fold,C,parameter)` stores feature coefficients followed
by the intercept when fitted; binary has one model class (the positive label),
and multinomial has one per sorted class. `path_classes_` identifies those rows.
`cs_`, `c_`, `class_weight_`, `cv_splits_`, `best_c_indices_`, and
`n_features_in_` expose fitted choices and metadata. `n_iter_path_(fold,C)` and
`converged_path_` describe native optimizer work, which is not sklearn's
newton-cg iteration count or tensor layout.

With `refit=.true.`, the first best mean-score candidate is selected and a
full-data fit starts from the mean selected fold parameters. With
`refit=.false.`, each fold selects its own first best candidate and the final
parameters and C are arithmetic means, without another fit. `refit_init_`
records those mean parameters. Inherited `n_iter_` counts only the final fit
(zero when refit is false). Zero iterations are supported; convergence flags
report the resulting unconverged paths. NaN scores follow NumPy argmax's first
NaN selection rule. Multinomial intercepts are centered to remove their common
additive null direction.

Training folds must retain every multinomial class. With an intercept and
positive iteration cap, every global class needs positive effective training
mass; a finite intercept optimum is required. Binary one-class training folds
are supported without an intercept. L1/elastic-net penalties, OvR solver modes,
multiple l1 ratios, and sklearn metadata-routing APIs are not implemented.
The CV subtype works through existing logistic-capable voting, bagging,
stacking, multiclass, multioutput, calibration, self-training, and nested
`cross_val_score` paths. See `example/logistic_cv.f90`.

## Binary decision thresholds

`fixed_threshold_classifier` and `tuned_threshold_classifier_cv` are exported
by `sklearn_threshold` and the `sklearn` facade. Call `fit(x,y,base)` with a
supported native binary classifier template. Templates and fitted bases are
copied. Integer labels, finite dense real64 input/queries, `predict`, weighted accuracy
`score`, and forwarding `predict_proba`/`decision_function` follow native
classifier conventions. Forwarded responses retain the base's original class
order and margin sign. These wrappers are not base templates for the existing
voting/multiclass dispatcher.

Both types have `response_method='auto'`, which prefers `predict_proba` and
falls back to `decision_function`. An explicit method must be supported by the
base. Assign the optional integer `pos_label` to select either fitted class;
the associated probability column or oriented binary margin is thresholded.
Fixed prediction reads the current `pos_label` and `threshold` controls.
An unallocated real scalar `threshold` selects 0.5 for probabilities or zero
for margins. Explicit finite or infinite cutoffs are supported; NaN is rejected.
A response equal to the cutoff maps to the positive class (`>=`).

Tuned controls are `cv=5`, `n_thresholds=100`, `refit=.true.`,
`scoring='balanced_accuracy'`, `store_cv_results=.false.`, and
`random_state=0`. Set `validation_fraction` in (0,1) for one stratified shuffled
holdout, or assign explicit 1-based `splits(:)`; repeats, overlap and ordering
are preserved. Otherwise nonshuffled stratified K-folds are used. Explicit
splits take precedence over a validation fraction. Set `prefit=.true.` and
`refit=.false.` to optimize a previously fitted base on the supplied validation
data. With `refit=.false.` and a single ordinary split, the final base is fitted
on that split's training rows. Multiple ordinary folds require refitting on all
data. Fixed also accepts `prefit=.true.` to copy a fitted base without retraining.
Prefit validation targets must match the two fitted classes for tuned fitting.

Set `thresholds(:)` to a finite nondecreasing candidate grid (duplicates are
allowed), or use `n_thresholds` to generate a linspace spanning each fold's
responses. The global automatic grid spans all fold extrema. Fold score curves
are linearly interpolated onto the common grid with constant extrapolation;
their arithmetic mean selects the first maximum. Curves whose endpoints satisfy
NumPy's default isclose test are rejected, including one-point grids. With an
explicit nonconstant grid, constant base responses are allowed, as in source.

Built-in objectives are `balanced_accuracy`, `accuracy`, `f1`, `precision`,
`recall`, and `jaccard`. Binary precision/recall/F1/Jaccard default to positive
label 1; assign `pos_label` for other labels. Accuracy/balanced accuracy default
to the larger fitted label for response orientation. An optional `objective`
procedure in tuned `fit` accepts `(integer y(:), integer predicted(:), optional
real(dp) sample_weight(:))` and returns a finite real64 score to maximize.
Assign `pos_label` when a custom objective needs another response orientation.

Optional nonnegative `sample_weight` is sent to base fitting and, by default,
threshold scoring. Optional `score_weight` independently overrides validation
weights, allowing weighted scoring with an unweighted base. Positive total
training/test mass is required where used. Base-specific fitting limits apply.
The custom objective receives raw validation weights. The public `score` method
always returns accuracy and accepts optional signed finite score weights with
a nonzero total, independently of the tuning objective.

Learned metadata includes sorted `classes_`, `n_features_in_`, `estimator_`,
`positive_label_`, `response_method_used_`, `best_threshold_`, `best_score_`, and
`cv_splits_`. The tuned positive label is fixed at fitting. With
`store_cv_results=.true.`, `thresholds_` and `mean_test_score_` store the common
curve, while `fold_thresholds_(fold,candidate)` and
`fold_scores_(fold,candidate)` store fold curves. Refitting with storage disabled
clears those optional arrays. See `example/threshold_tuning.f90`.

## RandomTreesEmbedding

`random_trees_embedding` is exported by `sklearn_tree_embedding`,
`sklearn_ensemble`, and `sklearn`. `fit(x,sample_weight)` and
`fit_transform(x,sample_weight)` draw a seeded uniform random target, fit
ExtraTree regressors with one sampled feature per node, and fit a dense
`one_hot_encoder_` to their leaf assignments. `transform` returns real64 binary
codes, with one active column per fitted tree. This corresponds to source
`sparse_output=False`; sparse storage is not implemented.

Controls are `n_estimators=100`, `max_depth=5`, `min_samples_split=2`,
`min_samples_leaf=1`, `min_impurity_decrease=0`, `random_state=0`, and
`warm_start=.false.`. Depth -1 means unbounded; depth zero retains the native
root-only convention. Sample counts are integers and impurity decrease is
finite/nonnegative. Finite sample weights, including zero/signed weights,
follow the underlying tree rules. Fits require finite float32-representable
features. Queries may contain NaNs, using learned missing routing in `apply`.

Warm fits preserve prefix trees and skip their seed draws before appending.
Changing controls affects newly added trees. Shrinking the forest or changing
feature width during a warm fit is rejected. A no-growth fit keeps all trees
but rebuilds the encoder from the current training rows. Consequently, an old
leaf absent from that dataset becomes an unknown category on later transform,
as in sklearn. The encoder's unknown-category policy resets on every fit.

`apply(x)` returns 1-based leaf IDs, and numeric `one_hot_encoder_%categories_`
use the same IDs. `get_feature_names_out()` returns source-style
`randomtreesembedding_TREE_LEAF` names using zero-based tree/node numbers.
Optional input names are accepted for width validation. `decision_path(x,ptr)`
returns a dense integer indicator; optional allocatable `ptr(:)` and learned
`node_offsets_` are zero-based cumulative node counts. In sklearn 1.6.1,
NaNs go right in dense `decision_path`, even if `apply` goes left using learned
missing routing. This source distinction is preserved and tested.

Learned metadata includes `estimators_`, `estimator_seeds_`, `random_target_`,
`n_features_in_`, `n_features_out_`, `n_estimators_`, `n_samples_fit_`,
`n_added_`, `n_outputs_=1`, `node_offsets_`, and `feature_importances_`.
Importances average non-root-only trees and normalize, or return zeros when
all trees are roots. Degenerate signed weights may produce quiet nonfinite
importances, as in source. `estimators_samples()` returns every current fit
row for every tree. Like the source property, it reflects the latest row count
even when warm prefix trees were trained on earlier data.

## KernelCenterer

`kernel_centerer` is exported by `sklearn_kernel_centering`,
`sklearn_preprocessing`, and `sklearn`. `fit(kernel)` accepts a finite square
matrix; symmetry or positive semidefiniteness is not required. It stores
`k_fit_rows_` (training column means), `k_fit_all_` (their mean), and
`n_features_in_`. `transform(kernel)` accepts rectangular query-by-training
kernels and subtracts training column means and query row means, then adds
the training grand mean. `fit_transform` combines those steps.

Transformations allocate new arrays and preserve the caller's input.
`get_feature_names_out()` returns `kernelcenterer0`, `kernelcenterer1`, etc.,
with optional input-name width validation. Reduction order follows the tested
C-contiguous sklearn/NumPy reference, including NumPy's blocked pairwise
contiguous sums. IEEE helpers retain source NaN/Inf results if finite inputs
overflow intermediate sums/subtractions, with floating-point traps enabled.
Different NumPy input layouts may differ in reduction roundoff.
See `example/random_feature_maps.f90` for both transformers in a classifier
and centered-kernel workflow.


## KNeighborsTransformer and RadiusNeighborsTransformer

`kneighbors_transformer` and `radius_neighbors_transformer` are exported through
`sklearn_neighbor_graph`, `sklearn_neighbors`, and `sklearn`. The latter two also
export `csr_matrix`, `csr_from_dense`, and `csr_from_arrays`. These are storage
helpers for core sklearn graphs, not additional machine-learning algorithms.

`fit(x)`, `transform(x)`, and `fit_transform(x)` accept dense real64 features or
precomputed distances (`metric='precomputed'`). Precomputed training distances
must be square; query columns correspond to training samples. CSR precomputed
inputs use only stored entries: missing edges are unavailable neighbors, while
stored zeros are valid zero-distance edges. Repeated columns are retained and
can supply multiple stored neighbors. Sorting is stable by stored distance and
never modifies caller arrays. Finite nonnegative distances are required.

Controls: `mode='distance'` or `'connectivity'`, `n_neighbors=5` for the nearest
graph, `radius=1` for the radius graph, and `metric` / `p=2`. Supported metrics
are euclidean, sqeuclidean, manhattan, cityblock, chebyshev, minkowski with finite
p>=1, cosine, and precomputed. Search is brute force; there are no tree-search,
parallel, callable-metric, weighted-metric, fractional-p, or sparse-feature
interfaces. Dense queries currently materialize their full pairwise distance
matrix and CSR copy. This is not a large-data sparse search implementation.
Use well-scaled inputs whose distance arithmetic stays in real64 working range.

Distance-mode KNeighborsTransformer requests **k+1** neighbors for every query,
including held-out queries; connectivity mode requests k. Enough fitted samples
and enough stored CSR entries in every row are required at transform time.
Radius graphs include d<=radius, permit zero/infinite radius, and may have empty
rows. Distance rows are sorted by distance. Dense connectivity radius rows use
training index order; precomputed CSR rows retain distance order. Equal-distance
neighbors use stable input order. Sklearn's optimized heaps/NumPy partitions can
choose or order different exact ties; exact index parity is not promised there.

Calling `transform()` without x means an implicit training query: self indices
are excluded, as in Python `transform(None)`. K selection first requests one
additional candidate and removes self (or the first candidate if self is absent).
Repeated self entries in the selected CSR candidates are rejected for this
implicit K query; explicit queries retain duplicates. `fit_transform(x)` is an
explicit training query, so self edges are eligible. For dense features,
`transform(x,self_query=.true.)` explicitly requests the training-array identity
semantics used by cosine's zero-vector diagonal; x must equal the fitted values.
Ordinary `transform(x)` treats x as an independent query. Fitted arrays and the
effective metric/p are value copies; refitting replaces prior dense/CSR state.
Learned metadata is `n_features_in_`, `n_features_out_`, `n_samples_fit_`.
`get_feature_names_out()` uses sklearn's class prefix and zero-based positions.

CSR row pointers and column indices are **1-based**, including terminal
`indptr(n_rows+1)=nnz+1`. `validate()` checks structure; `nnz()` counts stored
entries including zeros/duplicates; `to_dense()` sums duplicate columns;
`stored_mask()` reports presence independently of value; `matmul(rhs)` multiplies
a dense right-hand side, also summing repeated entries. `csr_from_dense(x)` omits
zeros by default; set `store_zeros=.true.` to retain them. Generic CSR storage
permits nonfinite data, but graph estimators require finite distances. There is
no claim that other existing estimators accept CSR. See
`example/neighbor_graphs.f90` for a public-facade example.


## IterativeImputer: Bayesian chained regression

`iterative_imputer` is exported by `sklearn_iterative_impute`,
`sklearn_preprocessing`, and the `sklearn` facade. It implements sklearn's
experimental core IterativeImputer algorithm, rather than a public extension.
`fit(x)`, `fit_transform(x)`, and `transform(x)` accept dense real64 matrices.
Each selected feature is regressed on the other current feature estimates using
observed target rows, and only missing target values are replaced. The complete
fitted sequence of regressors is replayed at transform time. Observed caller
values are preserved, including values outside the imputation bounds.

The `estimator` prototype is a native `bayesian_ridge`, defaulting to sklearn's
BayesianRidge controls. Its public controls can be changed before fitting; each
step gets a fresh value copy. Arbitrary Python estimators, other native base
families, sample weights, metadata routing, sparse inputs and callbacks are not
part of this interface. Posterior uncertainty uses the existing Bayesian ridge
covariance and noise precision. The package has no Python/SciPy runtime dependency.

Controls include `max_iter=10`, `tol=1e-3`, `initial_strategy` (mean, median,
most_frequent or constant), `fill_value=0`, `imputation_order` (ascending,
descending, roman, arabic or random), `skip_complete`, `keep_empty_features`,
`add_indicator`, `sample_posterior`, `random_state=0`, and
`n_nearest_features=0` (zero means Python None/all other features). Correlation-
weighted neighbor sampling without replacement follows NumPy RandomState draw
batching. Zero-variance correlations get the source exploration floor 1e-6.
Random update order follows the same initial shuffle and per-round reshuffles.
The private MT19937 state persists through transforms and subsequent fits,
matching sklearn's reused random_state_ behavior; copied fitted models have
independent copies of that state. Configure controls before fitting.

NaN marks missing values by default. For a finite marker, set
`missing_is_nan=.false.` and assign `missing_values`. Other NaN/Inf inputs are
rejected in finite-marker mode; infinities are rejected in NaN mode. Original
missing masks determine appended indicators, including indicators for dropped
empty columns. Empty columns are dropped unless `keep_empty_features=.true.`;
retained empty columns use initial constant fill or zero for other strategies.
Complete-at-fit columns get regressors by default so they can be imputed later;
`skip_complete=.true.` instead uses their initial statistics for new missingness.
A one-column fit, max_iter=0, or an entirely missing effective matrix returns
initial imputation without constructing a regression sequence. An entirely
missing query matrix also uses initial imputation, matching source's fast path.

Deterministic fits clip predictions to scalar `min_value`/`max_value`, or optional
full-input-width `min_values(:)`/`max_values(:)` vectors; dropped columns are
removed from these vectors. Defaults are the finite real64 extremes, equivalent
to unbounded imputation for ordinary finite results; IEEE infinite limits are
also accepted. Bounds must be ordered and cannot contain NaN. Posterior mode
uses inverse-CDF sampling from the corresponding truncated Gaussian, clamping
posterior means that lie outside bounds as in source. It runs all rounds without
early stopping, and repeated transforms advance the RNG. Deterministic early
stopping uses the maximum row sum of absolute changes, strictly below tol times
the largest absolute observed value. `converged_` reports that early-stop test;
no convergence warning is printed by the native library. Use well-scaled
matrices whose regression and covariance arithmetic stays in real64 range.

Learned fields include `n_features_in_`, `n_features_with_missing_`, `n_iter_`,
`valid_features_`, `indicator_features_`, `initial_statistics_`, and
`imputation_sequence_`. The missing-feature count follows source and includes
complete features when skip_complete is false. Each `imputer_triplet` stores
`feat_idx`, `neighbor_feat_idx`, and its fitted `estimator`; indices are 1-based
positions in the retained feature matrix. Valid/indicator feature indices refer
to original input columns. Initial statistics use zero for dropped empty
columns, whereas Python's internal SimpleImputer stores NaN there.

`get_feature_names_out()` returns original x0/x1/etc. names, or accepts optional
input names for width validation, followed by missingindicator_NAME indicator names.
Returned names have fixed length 128. A tested sklearn 1.6.1 naming quirk is
preserved: constant initialization with keep_empty_features=false drops empty
data columns but retains their names through its initial imputer. Names can
therefore outnumber data columns for that specific source case. Fitted-model
assignment deep-copies the sequence and RNG; refitting replaces the learned
sequence, statistics, masks and bounds. See `example/chained_imputation.f90`.


Posterior values inherit the Bayesian ridge thin-SVD convention documented
above: in wide rank-deficient regressions, the retained null-space completion
is nonunique. Different valid SVD bases can change predictive uncertainty and
therefore posterior draws, later regressions and transformed values. Exact
sklearn-default posterior parity is not promised for this case. The retained
hard fixtures validate covariance spectra and action on the observed row space,
then replay sklearn with that certified native covariance. Direct-default
output discrepancies remain in the report; they are not hidden by larger
numerical tolerances. This affects four of the current 324 fixtures, while
all other fixtures compare directly with sklearn's default BayesianRidge.


## Histogram gradient boosting (numeric features)

`hist_gradient_boosting_regressor` and `hist_gradient_boosting_classifier` are
exported by `sklearn_hist_gradient_boosting`, `sklearn_ensemble`, and `sklearn`.
These implement sklearn's numeric histogram/Newton boosting family, including
native missing-value routing. Input features are dense real64 and may include
NaN and +/-infinity. Targets must be finite. Regression losses are squared_error
(default), absolute_error, quantile, poisson and gamma; quantile=0.5 by default.
Poisson targets must be nonnegative with at least one positive value; Gamma
targets must all be positive. Poisson/Gamma predictions use an exponential link.
Classification uses log_loss, arbitrary integer labels and one Newton tree per
class for multiclass problems (one tree for binary and single-class problems).

Common controls: max_iter=100, learning_rate=0.1, max_leaf_nodes=31,
max_depth=0 (Python None), min_samples_leaf=20, max_bins=255,
l2_regularization=0, random_state=0, optional integer monotonic_cst(:), and
warm_start=false. max_leaf_nodes=0 similarly means Python None. Binning uses
feature-wise distinct-value midpoints or midpoint quantiles, excluding NaNs.
Above 200,000 rows it uses a seeded sample of that size without replacement.
Thresholds are clipped above at 1e300, with a dedicated final missing-value bin.
Gradients and Hessians are float32 as in sklearn, while histogram sums, Newton
values, gains and predictions are float64. Numeric best-first growth includes
histogram subtraction, depth/leaf limits, 1e-3 minimum child Hessian sums,
regularized leaf values and monotonic child bounds. For absolute/quantile losses,
terminal values are refitted to residual medians/quantiles and ignore L2, as in
source. Odd medians and these quantile baselines avoid an unnecessary mean
calculation, preserving valid constant large targets such as 1e308.

NaN routing is learned by scanning both directions when a feature had missing
training values. If none were seen, new NaNs follow the child with more rows;
ties go right. Infinite numerical thresholds distinguish missing-only splits.
Sample weights are optional, finite, nonnegative and must have positive total
mass; zero-weight rows still count toward minimum leaf sizes and binning.
Classification accepts class_weight='none' or 'balanced', plus optional
class_weights(:) in sorted classes_ order. An explicit vector takes precedence.
Signed sample weights and arbitrary class-weight dictionaries are outside this
native interface. Monotonic constraints must be -1, 0 or 1, one per feature;
any constraint array is rejected for multiclass classification, matching source.

This increment implements numeric splits with all features available at each
node. Categorical splits, dtype/category discovery, max_features subsampling,
interaction constraints, custom loss/scorer objects, parallel backends and
recursive partial dependence are not yet provided. The default native
`early_stopping=.false.` explicitly disables stopping. Set it true for
training-loss stopping with n_iter_no_change=10 and tol=1e-7; this corresponds
to Python early_stopping=True, scoring='loss', validation_fraction=None.
Sklearn's default auto mode and validation holdouts are not reproduced here.
The stop test requires that none of the latest n scores exceed the earlier
reference score plus tol; higher scores mean smaller loss. No convergence
warning is emitted. Check early_stopped_ and n_iter_.

Warm starts retain the original baseline and prefix trees, refit the bin mapper,
and append up to max_iter. Shrinking below n_iter_, changing feature width,
changing loss family, or changing classifier labels/classes is rejected.
Other growth controls affect appended trees. As in source, warm training raw
predictions use the new bin map with existing bin cuts, while public prediction
uses stored numerical thresholds. Fit parameters should normally be kept stable
when continuing training. Fitted assignment deep-copies all trees and bin maps.

Learned fields: n_iter_, n_features_in_, n_classes_, n_trees_per_iteration_,
baseline_prediction_, predictors_, bin_mapper_, train_score_, early_stopped_,
and classifier classes_. predictors_ has shape (iteration, tree/class).
Each internal histogram_tree holds a depth-first nodes_ array with feature,
bin_threshold, numerical threshold, left/right child indices, count, depth,
missing_go_to_left, value and gain. Indices and bin cuts are 1-based; zero
feature/children mark leaves. Leaf values include shrinkage, while internal
values follow the unshrunk source grower convention. A budget-finalized leaf
can retain a positive candidate gain, as in sklearn. Internal helper modules
sklearn_histogram_binning and sklearn_histogram_tree support this family and
are not additional estimator algorithms or general public tree APIs.

`predict(x)` returns a real vector or integer class labels. `predict_proba(x)`
returns binary two-column or multiclass probabilities. For a single fitted
class sklearn still returns two probability columns; this behavior is retained.
`decision_function(x)` and `raw_predict(x)` return (rows, trees-per-iteration)
matrices, including a one-column matrix for binary classification. Staged
methods materialize arrays: staged_predict returns (rows, iterations),
staged_predict_proba returns (rows, probabilities, iterations), and
staged_raw_predict returns (rows, trees-per-iteration, iterations). Score uses
the existing unweighted R2/accuracy implementations and their limits.

Overflowing baselines, nonfinite gradients and gradients outside float32 range
are diagnosed. Use well-scaled targets and weights whose loss arithmetic stays
in the working range. The independent uint32 binning seed can span the full
0..2**32-1 range; internal seeding uses the new standalone
seed_legacy_uint32(rng,seed_int64) helper. Existing legacy_random_state%seed
keeps its original nonnegative default-integer interface. This helper extends
NumPy stream support without introducing another ML algorithm or dependency.
See `example/histogram_boosting.f90` for both estimators.

Class labels are sorted in O(n log n) time. Binary training loss retains the
source softplus cutoffs and stable negative tails, including zero-tolerance
stopping on a one-class, heavily weighted fixture. Existing composite estimator
dispatch (including voting, bagging, stacking and threshold wrappers) does not
yet accept these two estimator types.


## Image patch extraction and reconstruction

`sklearn_image` and the `sklearn` facade provide `patch_extractor`,
`extract_patches_2d` and `reconstruct_from_patches_2d`, corresponding to sklearn's
image feature extraction algorithms. Inputs and outputs use real64 arrays.
Axes follow Python: (height,width[,channels]) for one image,
(images,height,width[,channels]) for a batch, and
(patches,patch_height,patch_width[,channels]) for patches. The color overloads
retain their channel axis even when its length is one; the grayscale overloads
return arrays without that axis. This is a static-rank Fortran API distinction
from Python's automatic singleton-channel squeeze.

Set `patch_extractor%patch_size=[height,width]`; its default [0,0] selects
[image_height/10,image_width/10] with integer division. Images smaller than ten
in either direction require an explicit positive size. Set `max_patches` to a
positive integer count, or `patch_fraction` to a real value strictly between
zero and one. Both default to zero, meaning all patches, and cannot both be
nonzero. A fraction is truncated toward zero and may produce an empty patch
array. Integer limits are capped at the total number of spatial positions.
Whenever a limit or fraction is specified, sampling is with replacement,
even when that count equals all possible positions. Rows are drawn first,
then columns, with the legacy NumPy RandomState stream. `random_state=0` is the
native deterministic default. Each transform resets this integer seed and
shares the stream across images in batch order. Mutable external RandomState
objects and a process-global random stream are not exposed.

`fit(x)` only validates controls and stores no fitted data. `transform(x)` and
`fit_transform(x)` accept grayscale rank-three or color rank-four batches;
transform does not require fit. All image dimensions must be nonempty and
extraction inputs finite. Exhaustive enumeration advances width first, then
height, then image. Outputs own their storage and do not modify inputs.
Standalone `extract_patches_2d(image,patch_size,...)` takes a rank-two or
rank-three image and the same optional max_patches, patch_fraction and
random_state controls; it requires an explicit positive patch size.

`reconstruct_from_patches_2d(patches,image_size)` returns a rank-two or rank-three
image. It places patches width first, then height and sums their overlapping
entries. Extra patches are ignored; missing patches leave zero contributions.
The source denominator is retained exactly: at pixel (i,j), using 1-based
indices, divide by min(i,patch_height,height-i+1) times
min(j,patch_width,width-j+1). In particular, this denominator is independent of
how many patches were supplied and can exceed the actual overlap count when
patches span more than half an image. Such outputs intentionally follow sklearn
and need not reconstruct the original image. Reconstruction accepts IEEE
NaN/Inf patch values and quietly propagates arithmetic overflow, preserving
source behavior even with runtime floating-point traps enabled. Image and patch
dimensions must be positive, patches must fit, and channel counts must agree.
Image graph builders, sparse pixel graphs and arbitrary-dimensional patch
extraction are not provided by this increment.


## Neighborhood components analysis

`neighborhood_components_analysis` is exported by `sklearn_nca`,
`sklearn_neighbors` and the `sklearn` facade. It learns a dense linear embedding
by maximizing the expected number of correctly classified stochastic
leave-one-out neighbors. The diagonal neighbor probability is zero. Squared
Euclidean distances feed a row-wise stable softmax; fitting uses the analytic
matrix gradient and unbounded L-BFGS-B with ten corrections.

Controls mirror sklearn NCA: `n_components=0` means None/default input width,
`init='auto'`, `warm_start=.false.`, `max_iter=50`, `tol=1e-5`, and native integer
`random_state=0`. tol controls both projected-gradient and relative-function
stopping, as in scipy.optimize.minimize(tol=...). The optimizer's default
15,000 evaluation limit and 20 line-search evaluations are retained. max_iter
must be positive; zero iterations are not an initialization-only API.

Initialization supports identity, random Gaussian, PCA, LDA and auto. Auto
chooses LDA if output width <= min(input width,classes-1); otherwise PCA if
output width < min(input width,samples); otherwise identity. Assign an
allocatable `init_matrix(rows,input_width)` to use an explicit transform,
taking precedence over init. If n_components is nonzero, its value must equal
the matrix's row count. With an existing warm-start model, the previous
components are reused after validation, even if a different n_components is
requested, matching source behavior. Feature width changes are rejected.
PCA uses native Jacobi SVD; LDA implements the source within-class standard
scaling and between-class SVD. Nonunique SVD bases/signs can differ. A deficient
LDA between-class rank can reduce the actual output width; zero usable rank is
diagnosed. Automatic/explicit LDA needs more samples than classes. One-class
and all-singleton labels remain valid with identity/random initialization.

Methods are `fit(x,y)`, `transform(x)` and `fit_transform(x,y)`, using dense
finite real64 feature matrices and integer labels. Fit needs at least two
samples. Transform returns raw X times transpose(components_), without
centering; the result has one column per fitted component. Inputs are not
modified. Learned state includes components_, initial_components_, gradient_,
objective_ (the positive objective being maximized), n_features_in_,
n_optimizer_iter_, n_function_evaluations_, converged_ and optimization_message_.
`n_iter_` follows sklearn's callback counter: one plus the number of completed
optimizer iterations, including 1 for immediate convergence. The diagnostic
objective/gradient evaluation after termination is excluded from the optimizer
evaluation count. The latest line-search status, relative step, bracket bounds
and bracket flag are available in line_search_status_, line_search_step_,
line_search_bracket_ and line_search_bracketed_.

An optional procedure pointer `callback` receives (transformation,iteration)
after each accepted optimizer step. The real64 vector is flattened in Python
row-major order, and iteration starts at one. Callback arguments are read-only;
a callback may inspect the current line-search diagnostics. Python callback
mutation, verbose console formatting, metadata routing and feature names are
not implemented. Explicit matrix initialization and deep-copy warm starts are
covered by native tests. See `example/neighborhood_components.f90`.

NCA has O(samples**2) working memory and a nonconvex objective. Native Fortran
reductions and the vendored Fortran optimizer can produce different long
optimization trajectories from NumPy/SciPy's C/BLAS implementation. This can
materially change final components and query transforms on sensitive fixtures;
identical matrices or iteration counts are not promised. Validation retains
direct fitted-output comparisons where they agree and reports every divergent
fixture separately. It checks source objective/gradient formulas at native
parameters, source transforms, initialization metrics, accepted-step decrease,
Wolfe curvature or the optimizer's explicit bracket-warning termination, and
source stopping conditions. It does not mark divergent matrices as direct
matches. SVD initialization comparisons use the induced metric A.T*A to account
for orthogonal basis freedom; source fitting is replayed from the checked
native initialization matrix when comparing optimization outputs.

Large intermediate products outside the native working range are diagnosed;
finite input alone cannot guarantee representable SVD, squared-distance or
gradient arithmetic. `nca_objective_gradient` in sklearn_nca is an internal
numerical/validation helper and is not re-exported by the sklearn facade.
The L-BFGS-B 3.0 numerical kernel is a pinned, unmodified local fpm dependency
under vendor/lbfgsb with retained BSD-3-Clause terms; original project code
remains MIT. Its four vendor modules are excluded from core source-module and
estimator counts. No external BLAS binary or network fetch is required.


## Dictionary vectorization and feature hashing

`sklearn_feature_extraction` and the `sklearn` facade export `dict_vectorizer`
and `feature_hasher`, with helper records feature_row, feature_item and
feature_string. These records represent native ordered feature streams; they
are storage helpers, not additional estimator algorithms. Both vectorizers
produce real64 CSR matrices with 1-based pointers and column indices.
`transform` and `fit_transform` return csr_matrix; `transform_dense` and
`fit_transform_dense` explicitly materialize dense real64 arrays. This method
choice replaces Python's dynamically typed sparse/dense return value; there
is no native sparse or dtype constructor switch. General estimator support
for CSR inputs remains limited to the APIs listed elsewhere in coverage.

A feature_row has allocatable items(:) or tokens(:). Each feature_item has an
allocatable key, a numeric value (default 1), optional category string, optional
categories(:) of feature_string, and is_none flag (default false). Assigning a
category creates a unit indicator; a category list creates unit indicators for
each entry, including repeats, and an empty list creates none. Numeric values
use the key unchanged; is_none maps Python None to NaN for DictVectorizer.
A feature_string stores its exact bytes in allocatable text. Keys and strings
must be allocated, but empty strings are valid. A row with neither collection
allocated represents an empty mapping/token list. Supplying both collections,
both category forms, or a categorical None value is diagnosed. Item order is
preserved and keys may repeat, corresponding to a .items() stream or pair
input. Fortran fixed-length padded strings retain their trailing blanks; use
trim explicitly when those blanks are not part of the intended name.

DictVectorizer defaults to sort=true and separator='=' (an unallocated native
separator means the default). Assign a deferred-length separator for arbitrary,
empty or multibyte separators. fit learns numeric keys and key+separator+value
categorical names; a hash table provides vocabulary lookup without scanning
all learned names. sort=false preserves first encounter order; sort=true
orders strings lexicographically by unsigned bytes. For valid UTF-8 strings
this matches Python Unicode ordering. No Unicode normalization or case folding
is performed. String keys are supported; arbitrary Python object keys and
alternative numeric dtypes are outside this native API.

Methods include fit, transform, fit_transform, the dense variants,
get_feature_names_out, vocabulary_index, inverse_transform and restrict.
feature_names_ is an array of feature_string and n_features_out_ is its width.
vocabulary_index returns a 1-based column or zero for an unknown name.
Transform ignores unknown names, retains explicit numeric zeros and retains
repeated CSR entries for categorical lists or colliding expanded names.
CSR rows are sorted by column. Changing separator after fit changes how query
categories are expanded; sort is consumed only when fitting. fit(empty_rows)
learns an empty vocabulary; fit_transform and transform require a nonempty
sample sequence. Empty mappings may produce a valid zero-column result.

restrict accepts a logical mask of fitted width or a list of unique 1-based
indices, retaining the requested column order and updating lookup. Negative,
zero, duplicate and out-of-range indices are diagnosed. Source's malformed
vocabulary behavior for repeated restrict indices is not reproduced.
inverse_transform accepts a dense matrix or csr_matrix of fitted width and
returns numeric feature_row items using expanded names; it does not recover
the original categorical fields. Inputs must have positive dimensions and
finite stored values. Dense inverse omits zeros. Sparse inverse includes a
column when any of its original stored entries is nonzero, even if duplicate
entries sum to zero, matching source. Duplicate values use NumPy 2.2.6
contiguous pairwise reduction, as in SciPy sparse scalar lookup. Dense
materialization uses sequential CSR addition; these two source operations can
produce different answers on ill-conditioned duplicates. Output items are ordered by column.
Nonfinite numeric values are accepted in forward DictVectorizer transforms,
as in sklearn, and rejected on inverse input. Original arrays and learned
models are independent under normal Fortran allocatable assignment.

FeatureHasher is stateless and supports input_type='dict' (default), 'pair' or
'string'. Dict/pair modes use items; string mode uses tokens and unit counts.
String-valued items hash key+'='+category with value one. Category lists and
None values are rejected by FeatureHasher. Numeric zero inputs are skipped;
collisions are summed, retaining a stored zero after cancellation. Output CSR
rows have sorted unique indices. Defaults are n_features=1048576 and
alternate_sign=true. The sign is the sign of the signed MurmurHash3 result;
columns are abs(hash) modulo n_features, including the INT32_MIN case. The
positive native integer n_features range matches source's int32 column limit.
fit only validates controls and transform does not require a fitted model.
The native byte implementation matches sklearn's MurmurHash3 x86_32 at seed
zero. Categorical keys represent UTF-8 text. Numeric item keys and string-mode
tokens may also contain raw bytes corresponding to Python byte names. Native
records do not infer Python repr(bytes_key) when combining a byte key with a
string category. The numerical helper murmurhash3_32 additionally accepts a uint32 seed
through int64, returns the signed result in int64 and is available only through
sklearn_murmurhash, not the sklearn facade. Its modular arithmetic avoids
signed integer overflow and is independent of host endianness.

IEEE NaN/Inf numeric values follow the installed forward-transform behavior,
including FeatureHasher's implementation despite its finite-value docstring.
Quiet arithmetic preserves IEEE results with floating-point traps enabled.
Native row sorting is stable. SciPy can use different equal-index sorting or
reduction orders, so finite ill-conditioned collision sums are also checked
against explicit forward-error bounds for diagnosis if a direct comparison
differs. The retained suite requires direct numerical comparisons within its
stated tolerance, with no accepted roundoff exceptions. Sparse inverse uses
the measured source pairwise order directly.
All hash values and CSR structure comparisons remain exact. CSR counts are
limited by the existing native integer storage capacity. See
example/structured_features.f90 for a minimal numeric/categorical example.


## Callable transforms and target wrappers (2026-11-21)

Added the sklearn FunctionTransformer and TransformedTargetRegressor
counterparts. Core inventory is now 211 public estimator/transformer/splitter
types in 117 source modules, plus the existing four vendored optimizer modules.
There are 115 native test programs and 95 retained core reference suites.
The two new suites contain 6,114 groups across 972 fixtures: 1,410 groups /
300 FunctionTransformer fixtures and 4,704 groups / 672 transformed-target
fixtures. Total retained core groups are 398,006; the separate 42 legacy RNG
groups remain excluded. The new isolated registry has 33 cases, bringing the
ten registries to 2,831 cases. These counts measure API subsets, not complete
sklearn class ports or a percentage of sklearn functionality.

FunctionTransformer covers identity and numeric callbacks, optional dense
validation, feature metadata, generated/explicit/callback feature names,
fit-free transforms, changed-width inverse inputs and inverse diagnostics.
The target wrapper covers cloned native transformer prototypes, paired target
functions, default or supported native regressors, weighted SVD linear fits,
scalar/matrix targets, original-scale scoring and independent multiple outputs.
Unsupported joint multi-target bases are diagnosed rather than replaced with
independent models. Full callback/rank/weighting limits are in docs/API.md.
No dependency or public algorithm extension is added; original code remains
MIT with sklearn-derived BSD attribution retained.

Reference gates use installed sklearn 1.6.1, NumPy 2.2.6 and existing native
transformers. Function results use 3e-12 relative/absolute tolerance; target
results use 3e-9 except inherited PowerTransformer fits at 3e-6. Transformer
state can have large absolute differences for very large power-fit parameters;
reports retain maximum absolute differences, fixture contexts and fractions
of the allowed relative/absolute tolerance. Every transformed training target
is also checked directly. Metadata, names and output shapes are exact checks.

The target validator retains all fixtures. If a native CART split differs,
it uses the existing source-conditioned certificate: source sklearn stumps
verify every node's response, impurity and admissible split within a measured
floating-point accumulation bound. Predictions and inverse mapping are then
checked against these certified source leaf responses. Constant-target R2
is discontinuous at exact prediction equality; numerical discrepancies are
recorded explicitly and the native score is checked against sklearn r2_score
on native predictions. This is separate from direct prediction comparisons.
Strict JSON reports record source/binary hashes, tree runtime provenance,
warning counts, certificates and constant-target score differences.

Reproduce after building the desired profile:

```
python validation/compare_function_transformer.py --profile checked
python validation/compare_transformed_target.py --profile checked
python validation/check_target_wrappers_invalid_inputs.py --profile checked
```

Use --profile release for the optimized gate. --executable accepts a completed
native executable, avoiding another fpm invocation. Do not run simultaneous fpm
builds in this checkout. The transformed_targets example shows log/exp target
regression and a standalone callable transform through the sklearn facade.


## Dense column and parallel feature composition (2026-11-21)

Added sklearn ColumnTransformer and FeatureUnion counterparts with 17 existing
native child transformer types and recursive composition of both new graph
types. Selection supports ordered/repeated 1-based positions, logical masks,
explicit feature names and fit-time callbacks. Column transforms clone models,
retain empty selections without fitting, exclude all explicitly selected columns
from remainder, and append a dropped/passthrough/transformed remainder. Fitted
union lists follow sklearn's update convention. Both support branch weighting,
feature names and empty results; ColumnTransformer additionally exposes output
indices, custom name formatting and named query reordering/missing-drop columns.
Native tests cover independent graph copies, prototype cloning, changed-row
single branches and current-weight transforms. Production/native sources were
frozen before the final checked/release fpm gates.

Core inventory is now 213 public estimator/transformer/splitter types in 119
source modules, plus the existing four vendor optimizer modules. There are 117
native test programs and 97 retained core reference suites. The new suites
contain 2,160 groups / 360 ColumnTransformer fixtures and 1,200 groups / 240
FeatureUnion fixtures, covering all 17 supported existing child types, both
nested graph types, callback expansion, selector variants, remainder modes,
fit versus fit_transform, prefit prototypes and generated/explicit feature
names. Total retained core groups are 401,366; the separate 42 legacy RNG
groups remain excluded. The new isolated registry contains 43 cases, bringing
the eleven registries to 2,874 cases. Counts describe API subsets, not full
sklearn class ports or a percentage of sklearn's functionality.

The reference engine uses installed sklearn 1.6.1, NumPy 2.2.6 and pandas for
named-column reference semantics. Numeric comparisons use 3e-9 relative/absolute
tolerance, except inherited PowerTransformer fits at 3e-6. Metadata, output
shapes, names, selector call counts and native/source output slices are exact.
No fixtures are filtered. Reports retain source hashes, canonical native binary
hashes, warning counts, numeric differences and full per-fixture checks.
Numeric category names include subnormal and maximum finite values, exponent
layout boundaries, negative zero, full-precision random values and ordinary
decimals. KNNImputer naming preserves sklearn 1.6.1's all-missing-column omission
even when the numeric output retains the columns. Active zero-width blocks
still participate in concatenation row checks; a single active branch may
change row count, as in source.

These are dense, serial, unsupervised composition APIs. Existing child limits
continue to apply. No generic external estimator objects, supervised y routing,
sparse stacking, pandas output containers, metadata routing or general
sequential Pipeline is added. Python slicing/scalar column selection is
represented with explicit native position vectors. Explicit name vectors
provide named matrix indexing without importing a Python container library.
The API and example document native storage/indexing conventions. Original
contributions remain MIT, with sklearn BSD attribution retained. No new
dependency or public extension algorithm is added.

Reproduce the new gates with one fpm build at a time:

```
python validation/compare_column_transformer.py --profile checked
python validation/compare_feature_union.py --profile checked
python validation/check_composition_invalid_inputs.py --profile checked
```

Use --profile release for optimized gates, or --executable to inspect completed
canonical executables without another fpm invocation. Example:
`fpm run --example feature_composition`.


Native empty-container adaptations: FeatureUnion accepts an empty branch list
and returns zero output columns, while sklearn 1.6.1 rejects an empty list on
fit. Both support all-dropped lists. The native API returns an empty name vector
for unprefixed empty/all-dropped unions; the source unprefixed all-dropped
naming path raises a NumPy empty-concatenate error. These differences are
identified native container conventions, not public extension algorithms.


## Sequential Pipeline and recursive dense transformer bridge (2026-11-21)

Added the sklearn Pipeline counterpart with typed classifier/regressor
prediction, sequential fit/fit_transform, supervised selector/TargetEncoder
steps, supported native estimator tails, original final-estimator scoring,
inverse transforms, feature names and independent contiguous slices. A new
internal abstract dense_transformer protocol lets pipelines appear inside
ColumnTransformer and FeatureUnion without a module cycle. Pipelines can also
contain either parallel graph type or another pipeline. No additional algorithm
is represented by this protocol or pipeline_step storage.

The implementation preserves intermediate fit_transform versus final fit
ordering. TargetEncoder training encodings are cross-fitted; subsequent
transform/predict uses the fitted full-data mappings. Final estimator type,
not target dtype, controls classification/regression. Real integral labels
and integer regression targets have native adapter tests. Weights go to the
final estimator. Scoring transforms input once and delegates the correct score
family, including GLM D2. Existing component numerical/weighting restrictions
remain explicit. The dense API is fit-first and excludes caching, metadata
routing, arbitrary Python estimators and sparse paths; detailed method/rank,
name-container, slicing and matrix-target limits are in docs/API.md.

Inventory is now 214 core public types, 120 core source modules, 119 native
test programs and 98 retained core reference suites totaling 409,922 result
groups. The separate 42 legacy RNG groups remain excluded. The new pipeline
suite has 8,556 groups across 756 unfiltered fixtures: three seeds, 13
preprocessing families, transformer/regression/classification/matrix/GLM tasks,
four configurations, twelve additional final-handoff fixtures, weighted tails,
binary/multiclass and integer/real labels,
fit versus fit_transform, names, inverse outputs, nested graphs and callback
call counts. Matrix-target cases omit supervised single-target transformations
that are not part of that API. New isolated checks add 58 cases, bringing twelve
registries to 2,932 cases. These counts measure API subsets, not full sklearn
class ports or a percentage of sklearn functionality.

Reference tests use installed sklearn 1.6.1 with 3e-8 relative/absolute numeric
tolerance, except native/source logistic and Poisson optimization at 3e-6.
Classifier labels, names, classes, dimensions and callback counts are exact.
Binary decision scores are compared in native (samples,1) form. Inverse groups
cover available native component inverses; components whose native inverse is
absent produce an explicit empty placeholder, not an asserted Python inverse.
The source imputer's inverse also requires add_indicator, absent in the native
simple imputer. Source KNN references explicitly select brute search to match
the existing native backend.

KNN cases additionally compare actual fitted training features and target order.
Rare default-source differences from ties and Euclidean distance cancellation
are not excluded or accepted under wider prediction tolerances. A recorded
source-conditioned certificate validates native neighbor indices/distances
against direct Euclidean geometry and relative accumulation bounds, verifies
source-space admissibility under measured input perturbations, and bounds the
source optimized distance kernel's squared error. Sklearn precomputed-distance
KNN then independently reconstructs weighted predictions/probabilities and
scores for the certified selected neighbors. The report marks affected groups
with this oracle and retains changed-neighbor counts, perturbations and kernel
error/bound witnesses. All other groups keep direct sklearn comparisons.

The dispatcher change also requires rerunning both existing parallel-composition
reference suites against updated binaries. The existing 192-group integration
gate is rerun on the checked executable. Reports retain strict JSON, full
fixture/check lists, validator/certificate/source hashes and canonical native
executable hashes. Original contributions remain MIT with sklearn BSD notices;
there is no new external dependency or public extension algorithm.

Reproduce with one fpm build at a time:

```
python validation/compare_pipeline.py --profile checked
python validation/check_pipeline_invalid_inputs.py --profile checked
python validation/compare_column_transformer.py --profile checked
python validation/compare_feature_union.py --profile checked
```

Use --profile release for optimized gates or --executable for a completed
canonical binary. The public example is `fpm run --example sequential_pipeline`.


Final verification of the corrected build passed all 119 native test programs
in both checked and optimized profiles. Each profile passed 8,556 Pipeline
groups across 756 fixtures, 58 Pipeline rejection checks, 3,360 existing
composition reference groups and 43 composition rejection checks. Four KNN
fixtures per profile required the retained geometric certificates. The checked
integration gate passed all 192 comparisons. Both canonical example binaries
reported training R2 0.9999792, feature numeric__x0 and 20 predictions, matching
the independent sklearn example (R2 0.9999792424116486).

The final audit verified complete strict reports, matching profile fixture
configurations, executable/validator/source hashes and fpm object content
digests for 243 source/test/vendor files plus four new app/example files in
each profile. Two byte-identical sources per profile had newer timestamps
than their cached objects; their content digests matched. Retained evidence
is in validation/results/pipeline_final_audit.json and the Pipeline reports
and native/integration logs. Older unrelated reference suites were retained,
not all rerun for this increment. The broad sklearn translation remains ongoing.


## Model-based feature selection (SelectFromModel)

This increment adds select_from_model, bringing the inventory to 215 core
public types, 121 core source modules and 120 native test programs. Ninety-nine
retained core reference suites total 413,810 result groups; the separate
42-group NumPy legacy RNG suite remains outside that count. The new registry
adds 47 isolated rejection checks, bringing 13 registries to 2,979 cases.
These are implemented API subsets rather than complete sklearn class ports.

The dense selector clones supported coefficient/tree-importance estimators,
accepts prefit models, aggregates matrix coefficients with nonzero integral or
infinite norms, interprets mean/median/scaled/numeric thresholds, applies stable
descending feature caps, and provides zero-filling inverse transforms, feature
names and support indices. Native callbacks provide custom importances and
feature caps. SGD partial_fit retains successive fitted state while first
cloning only estimator parameters. Sequential Pipeline dispatch supports this
supervised transformer, including integer regression targets, names and inverse
transforms. The existing native estimator fit/weighting restrictions still apply.

The new source suite retains 432 fixtures and 3,888 result groups over three
seeds, twelve estimator/importance families and twelve control configurations.
It checks support, importance norms, thresholds, fitted metadata, names, callback
counts and full fit/query/inverse matrices, including empty selection, exact
ties, positive/negative infinite thresholds and direct prefit transforms.
Families cover explicit vector/matrix prefit coefficients, weighted Ridge,
Lasso, binary/multiclass logistic regression, regression/classification CART,
MultiTaskLasso, three-batch regression/classification SGD, and vector/matrix
importance callbacks. The existing SGD solver supports L2/no penalty; no new
SGD loss or penalty algorithm is added by this selector.

All 72 fitted tree fixtures additionally receive independent certificates of
node membership, weighted impurity, globally optimal split loss and normalized
feature contributions. Six fixtures in the development checked run differed
from sklearn in feature attribution on existing tied split paths. Those retain
all selection checks against sklearn using the certified native fitted
importance vector. Reports identify the selection oracle for every fixture,
retain the source importance difference and numerical certificate witnesses,
and hash the helper. They do not loosen feature-mask or transform comparisons.
The remaining fixtures compare directly with the fitted source estimator.

New implementation/adapters remain MIT with retained sklearn BSD attribution;
no external repository or dependency is imported. Sparse arrays, arbitrary
Python attribute paths, metadata routing and general estimator reflection are
not implemented. Importances and numeric dense inputs must be finite; norm
overflow raises a native diagnostic. Empty transform results are supported,
while inverse_transform requires a nonempty selected matrix, as in sklearn.
Naming requires fit even though direct prefit get_support/transform works.
See docs/API.md and example/model_based_selection.f90 for native signatures.

Reproduce after building one profile at a time:

```
python validation/compare_model_selector.py --profile checked
python validation/check_model_selector_invalid_inputs.py --profile checked
```

Use --profile release for optimized validation, or --executable to select an
already-built canonical executable without invoking fpm again. Full verification
results for the frozen implementation are recorded below when complete.


Final verification of the frozen implementation passed all 120 native test
programs in checked and optimized fpm builds. Each canonical profile passed
3,888 selector groups across 432 fixtures and all 47 selector rejection checks,
8,556 existing Pipeline groups plus 58 rejection checks, and 3,360 existing
composition groups plus 43 rejection checks. All 72 tree certificates passed
per profile; six fixtures per profile used the certified native-tree selection
oracle. The checked integration gate passed all 192 comparisons.

Both canonical public example binaries retained columns 2 and 4 and printed
training R2 0.9999949, matching the independent sklearn value
0.9999949201575922. The final audit verified complete strict reports, identical
profile fixture configurations, executable/validator/certificate/source hashes,
and all 248 compiled content digests per profile (245 core/test/vendor files
plus the two new drivers and example). Evidence is retained under
validation/results/model_selector_final_audit.json, model_selector_*.json,
and the native/integration logs. Existing unrelated reference-suite reports
were retained rather than all rerun for this increment. The broad sklearn
translation remains ongoing.


## General parameter search (GridSearchCV and RandomizedSearchCV)

This increment adds grid_search_cv and randomized_search_cv through sklearn_search
and the sklearn facade. To avoid a cycle through the existing model-selection
and estimator-dispatch modules, sklearn_model_selection does not re-export these
new types. There are now 217 core public estimator/transformer/splitter types,
124 core source modules and 122 native test programs. Parameter storage/grid/
result/scorer helpers are excluded from estimator counts. Two new reference
suites bring 101 retained core suites to 419,854 result groups. The separate
42-group legacy RNG suite remains excluded. The new 61-case registry brings
14 rejection registries to 3,040 cases.

Typed scalar parameter spaces support real, integer, logical and text values;
axes sort by name and rightmost values vary fastest. Lists of grids preserve
candidate order, including empty dictionaries. Finite random sampling matches
sklearn's permutation, tracking-selection and reservoir branches without
replacement. If any axis has an rvs callback, grid choices and sorted axes are
sampled with replacement. Large finite spaces are indexed lazily: the test
samples a million-point grid without creating a million candidate objects.
Integer sizes and seeds use native default-integer limits; random_state defaults
to deterministic zero. None/array/estimator-valued parameters require an explicit
native representation/setter callback rather than Python object reflection.

Searches clone native models for each candidate/fold, slice fit weights, use
unshuffled KFold or first-appearance-encoded StratifiedKFold, accept explicit
1-based folds, and retain complete split/mean/std/rank tables and measured
fit/score times. Scoring can be a built-in name, a scalar callback, or named
multiple scorers. Refit can use the best metric, be disabled, or use a result
callback; ties choose the first minimum-rank candidate. Optional train scores
and refitted prediction/probability/decision/score delegation are available.
Nested Pipeline, SelectFromModel estimator and transformed-target regressor
parameter paths are supported. Existing native child fit/weighting/solver
restrictions still apply; Pipeline weights go to its final estimator.

The supported scope is serial, dense finite X and vector regression/class labels.
Original project adapters remain MIT and sklearn BSD notices are retained. No
external repository/dependency or extension algorithm is imported. The native
error_score default is 'raise' and only that policy is supported: existing
child estimators use terminating require diagnostics, so failed fits cannot yet
be recovered as numeric replacement scores. Parallel jobs, general metadata
routing, sparse data, unsupervised/matrix targets and arbitrary Python clone/
parameter reflection remain outside this increment. Timing measurements use
native wall-clock resolution and are not expected to match Python timings.

The parameter-space suite passes 1,976 candidate checks across 112 fixtures:
four seeds, heterogeneous values, empty/list grids, distribution callbacks,
sampling ratio boundaries, oversubscribed sampling and lazy large-grid indexing.
The search suite retains 4,068 groups across 270 fixtures over three seeds,
ten model/scorer families and nine control configurations. It checks full
candidate/fold results, ranks, refit metadata and score availability, candidate
parameters, split indices, callback fit/set counts, timing shape/nonnegativity
and refitted predictions/responses. Families include Ridge, ElasticNet, binary/
multiclass logistic regression, Gaussian NB, KNN, scaling pipelines, nested
model-based selection, nonfinite scalar scorers and custom fit/set callbacks.

All numerical score tables retain direct sklearn comparisons. Ranking is also
checked exactly with scipy.stats.rankdata applied to the emitted native means.
Twenty-eight metric/fixture records in the development checked run needed
roundoff certificates: any changed pair ordering must lie within
128*n_splits*machine_epsilon times the source score scale, including the source/
native score differences. Those records retain the changed pairs and bound.
If such an ordering changes the selected candidate, the source refit/prediction
oracle uses that certified selection. NaN/infinity ranking and one-row R2
folds remain covered. No rank check is dropped and score tolerances are not
relaxed to conceal a changed ordering.

Reproduce after building one fpm profile at a time:

```
python validation/compare_parameter_space.py --profile checked
python validation/compare_search.py --profile checked
python validation/check_search_invalid_inputs.py --profile checked
```

Use --profile release for optimized gates or --executable with a completed
canonical binary. See docs/API.md and example/parameter_search.f90 for usage.
Final verification of the frozen implementation is recorded below.


Final verification of the frozen implementation passed all 122 native programs
in checked and optimized fpm builds. Each canonical profile passed 1,976
parameter-space checks across 112 fixtures, 4,068 search checks across 270
fixtures, and all 61 isolated rejection checks. Twenty-eight metric/fixture
roundoff certificates passed in each profile. The checked integration gate
passed all 192 comparisons.

Both canonical public example binaries selected alpha 0.01 from three candidates
and printed mean CV score -1.104558e-6, matching the independent sklearn value
-1.1045576422766414e-6. The final audit verified complete strict reports, identical
profile check configurations, canonical executable/validator/certificate and
installed sklearn/SciPy source hashes, plus all 254 compiled content digests
per profile (250 core/test/vendor files plus the three new drivers and example).
Evidence is retained under validation/results/search_final_audit.json,
search_*.json, parameter_space_*.json and the native/integration logs. Existing
unrelated reference suites were retained rather than all rerun for this
increment. The broader sklearn translation remains ongoing.


## General cross-validation and out-of-fold responses

This increment adds cross_validate and cross_val_predict, plus general scalar
cross_val_score and a typed class-label prediction adapter, in sklearn_validation.
The facade exports cross_validate, cross_val_predict and cross_val_predict_class;
it retains the older typed cross_val_score from sklearn_model_selection. Import
the broader scalar helper directly from sklearn_validation (rename it on import
when also importing the facade). No new estimator types are counted: the core
inventory remains 217 estimator/transformer/splitter types, with 125 source
modules and 124 native test programs. Two new reference suites bring 103 core
suites to 424,408 retained result groups. The separate 42-group legacy NumPy RNG
suite remains excluded. Fifty new rejection cases bring 15 registries to 3,090.

Cross-validation supports native estimator dispatch, real/integer vector targets,
unshuffled KFold or first-appearance-encoded StratifiedKFold, explicit 1-based
folds, sliced fit weights, scalar/multiple scoring, train scores, returned fitted
models/indices, measured fit/score times and custom fit/score callbacks. No final
full-data fit occurs. The scalar helper disables train scoring and returned
artifacts, as sklearn cross_val_score does. No-target fitting supports the
existing composition transformer adapters with explicit scoring callbacks;
other native no-target fits can use a fitter callback.

Out-of-fold predictions require the held-out indices to partition all samples
exactly once, while ordinary cross_validate permits repeated/nonpartition tests.
The output always restores original sample order. Supported classifier response
methods are predict, predict_proba, predict_log_proba and decision_function.
Response methods encode global labels before fitting each fold and place missing
class columns in global sorted-label order, using zero probabilities and minimum
finite real64 log/margin placeholders. Incompatible missing-class decision shapes
are rejected. Stable native NB/boosting log probabilities are retained rather
than reconstructed from underflowed probabilities. Known estimators without a
log-probability API, including KNN, raise a native diagnostic. Scalar/binary
responses are stored as n-by-1 matrices; the class-label adapter returns integers.

Integer regression targets retain their original representation in native
Pipeline fitting, including supervised selector dispatch. Native fit/scoring
callbacks receive normalized real regression or integer class targets; encoded
class targets are visible to OOF response fit callbacks. Custom response and
class-metadata hooks support native wrappers. Returned models are independent
value copies; procedure pointer lifetimes remain the caller's responsibility.

The supported scope remains serial dense finite X, vector targets and existing
native child fit/weight/solver restrictions. Only error_score='raise' is supported
because child diagnostics terminate execution. Sparse/precomputed pairwise slicing,
metadata routing, parallel jobs, recoverable fit/scoring errors, multioutput/
multilabel targets, and no-target cross_val_predict are not implemented here.
No-target default scoring requires a callback. Native callback response matrices
must be allocated and correctly sized. None of these additions is a public
extension algorithm; original contributions remain MIT with sklearn BSD notices.
No external repository or dependency is imported.

The cross_validate source suite retains 3,888 groups across 288 fixtures:
three seeds, twelve model/scorer families and eight configurations. It compares
complete test/train scores, scalar wrapper scores, callback counts, returned
model predictions/probabilities/classes, explicit indices and timing shapes.
Families include Ridge, ElasticNet, logistic regression, Gaussian NB, KNN,
scaling/selection pipelines, no-target scaling and nonfinite scalar scores.
Custom CV includes repeated test rows and one-row R2 folds. The OOF suite retains
666 groups across 222 fixtures covering all supported response capabilities,
nonmonotone partitions, missing classes, weighted fitting, encoded fit targets,
custom response callbacks and original row order. Unsupported capability cases
are retained in the isolated rejection registry. All numerical/label comparisons
use direct sklearn 1.6.1 oracles; no roundoff certificate is needed for these
suites.

Reproduce with one fpm build at a time:

```
python validation/compare_cross_validate.py --profile checked
python validation/compare_cross_val_predict.py --profile checked
python validation/check_validation_invalid_inputs.py --profile checked
```

Use --profile release for optimized runs or --executable for completed canonical
binaries. See docs/API.md and example/out_of_fold_predictions.f90 for usage.
Final verification of the frozen implementation is recorded below.


Final verification of the frozen implementation passed all 124 native test
programs in checked and optimized fpm builds. Each canonical profile passed
3,888 cross_validate groups across 288 fixtures, 666 cross_val_predict groups
across 222 fixtures, and all 50 isolated rejection checks. The checked
integration gate passed all 192 comparisons. No roundoff certificate was used
for either new reference suite.

Both canonical public example binaries reported four folds, 40 out-of-fold
predictions and R2 0.9915055, matching the independent sklearn value
0.9915054641718475. The final audit verified complete strict reports, identical
profile check configurations, canonical executable/validator/entry and installed
source hashes, and all 256 compiled content digests per profile (253 core/test/
vendor files plus the two new drivers and example). Evidence is retained under
validation/results/general_cv_final_audit.json, cross_validate_*.json,
cross_val_predict_*.json, validation_invalid_*.json and native/integration logs.
Existing unrelated reference suites were retained rather than all rerun for
this increment. The broad sklearn translation remains ongoing.


## Learning curves and permutation significance tests

`sklearn_validation` and the `sklearn` facade now expose `learning_curve`,
`validation_curve` and `permutation_test_score`, adapted from sklearn 1.6.1
`model_selection/_validation.py` and `_split.py`. These functions add no estimator
classes: the inventory remains 217 core types and 125 source modules, with 126
native test programs. There are now 106 core reference suites retaining 429,685
result groups, plus the separate 42-group legacy NumPy RNG suite. The new
34-case rejection registry brings 16 registries to 3,124 cases.

Learning curves accept real or integer targets, or no targets with a scoring
callback. `learning_curve_options` extends `cross_validation_options`; supply
fractional `train_sizes` in (0,1] or integer `train_counts`, never both. Defaults
are five evenly spaced fractions from 0.1 to 1. Sizes are sorted and deduplicated
against the first fold's training length. Later shorter folds clip their prefixes;
`actual_train_sizes` records the resulting counts. Seeded optional training-index
shuffles use the existing NumPy-compatible legacy RNG. Results contain folds,
sizes, and train/test score matrices indexed by size then fold. Optional timing
matrices report individual fits, with test scoring time for batch fits and both
train/test scoring time for incremental fits, matching the pinned source.

Batch fits start afresh for each size/fold. Incremental learning is available for
SGDRegressor, SGDClassifier and GaussianNB through
`exploit_incremental_learning`; fitted prototype state is cleared, each fold gets
one model, and successive chunks update it with partial_fit. Classifiers receive
the full target class set. Custom models can supply `partial_fit_func` and
`incremental_init_func`; the latter must clear any custom fitted state. Pinned
sklearn 1.6.1 forwards fit parameters unsliced on its incremental path, so full
vector sample weights normally fail the child learner's chunk-length check here
as well. Empty chunks on uneven folds also retain the source learner's rejection.

Validation curves take a parameter name and typed `parameter_value` array made
with `param(...)`. Parameter order and duplicates are retained; each value/fold
gets a fresh fit. Native nested Pipeline parameters and an optional
`set_params_func(model, parameter_set)` callback use the existing search adapters.
The result stores the parameter range, folds, and train/test score matrices.

Permutation tests return the observed mean CV score, all permuted scores, and
`pvalue=(1+count(permuted_score >= observed_score))/(1+n_permutations)`. A positive
permutation count is required (default 100); the seed defaults to zero. Optional
integer groups constrain target shuffling within each group. Default classifier
folds are recomputed for each permuted target; explicit folds remain fixed.
One cloned model is reused across folds within each observed/permuted run,
matching the pinned source, including custom fit callbacks. Fit weights retain
their original row association and are sliced per fold. NaN comparisons do not
count as exceedances.

Scope remains serial dense finite X, vector targets, scalar scoring and
`error_score='raise'`, with the existing child estimator restrictions. Sparse
inputs, metadata routing, parallel execution, recoverable errors and multioutput
targets are not provided. Native callbacks receive normalized targets; ordinary
Pipeline fits retain original integer regression targets. Procedure pointer
lifetimes remain the caller's responsibility. No extension algorithms or new
dependencies were introduced. Original contributions remain MIT, with retained
sklearn BSD attribution.

The direct source suites retain 2,733 learning-curve groups over 360 fixtures,
1,584 validation-curve groups over 264 fixtures and 960 permutation groups over
240 fixtures. Learning fixtures include 21 matching source rejections for
one-class prefixes or empty incremental chunks. Comparisons cover complete
score arrays, prefix sizes, fold indices, fit traces, callback counts, parameter
order, incremental resets, seeded shuffles, groups, nonfinite scores and exact
p-values. Timing checks compare shapes and nonnegative values, not wall times.
See `example/model_learning_curves.f90` for a complete public API example.


## Seeded regression, manifold and quantile sample generators

Seven additional sklearn 1.6.1 datasets are available from `sklearn_datasets`
and the `sklearn` facade. Implementation is in `sklearn_sample_generators`,
adapted from `datasets/_samples_generator.py` with retained BSD-3-Clause
attribution. There are now 126 core source modules and 127 native test programs;
the estimator/transformer/splitter inventory remains 217. The new reference suite
brings 107 core suites to 430,357 retained result groups. The separate 42-group
NumPy RNG suite remains excluded. Twenty-one rejection cases bring 17 isolated
registries to 3,145 cases. No public extension or dependency was added.

All outputs are allocated by the subroutine: X has shape (samples,features),
regression/latent targets are real vectors and classification labels are integer
vectors starting at zero. Samples must be positive. The interfaces are:

```
call make_friedman1(n_samples,n_features,x,y,noise,random_state)
call make_friedman2(n_samples,x,y,noise,random_state)
call make_friedman3(n_samples,x,y,noise,random_state)
call make_sparse_uncorrelated(n_samples,n_features,x,y,random_state)
call make_s_curve(n_samples,x,t,noise,random_state)
call make_swiss_roll(n_samples,x,t,noise,random_state,hole)
call make_gaussian_quantiles(n_samples,n_features,x,labels,n_classes,mean,cov,shuffle,random_state)
```

Trailing parameters are optional. Noise defaults to zero and must be finite and
nonnegative; an omitted seed deterministically selects zero. These seven routines
use independent local NumPy-compatible legacy RandomState streams and do not
change the Fortran global RNG. Required sample/feature sizes follow the existing
native dataset API rather than Python's optional size defaults. The four earlier
native dataset routines retain their existing RNG behavior.

Friedman1 requires at least five features; only the first five enter its nonlinear
response. Friedman2 and Friedman3 share the four-feature ranges of the electrical
circuit benchmark, using square-root and arctangent responses respectively.
Sparse uncorrelated regression returns a dense Gaussian design, four informative
linear terms and unit Gaussian response noise. At least four features are required;
the pinned Python implementation fails by indexing for smaller positive counts,
while native code provides an explicit diagnostic.

S-curve and Swiss roll return noisy three-dimensional coordinates plus the clean
latent coordinate t. Swiss roll supports the source `hole` option, default false,
including its discrete corner draws and parameter order. Even a single sample
returns a length-one native t vector, whereas Python squeezes t to a scalar.
Gaussian quantiles support a finite mean vector (default zero), scalar isotropic
covariance (default one, including zero), class count (default three), and optional
sample shuffling (default true). Samples must be at least the class count. Radial
sorting precedes label allocation; any remainder belongs to the last class.

Draw order follows the source, including row-major feature draws, cached Gaussian
values, manifold noise generated in (3,samples) order, Gaussian quantile draws
before sorting, and legacy Fisher-Yates shuffling. Source comparisons check full
feature and target arrays, not only moments. The suite retains 672 groups over
336 fixtures: seven generators, four seeds and twelve configurations, including
one-row outputs, odd lengths, high feature counts, zero/nonzero noise, hole/no-hole,
zero covariance, shifted means, singleton classes and uneven class allocation.
It also runs 21 isolated invalid-input cases.

Run `python validation/compare_sample_generators.py --profile checked`, or use
`--profile release` / `--executable` for an existing binary. The public example is
`example/synthetic_sample_generators.f90`. Python is used only for validation.
Original native contributions remain MIT; sklearn-derived portions retain the
existing upstream BSD notice and the existing RNG notices remain applicable.


## Clustered classification and Hastie benchmark generators

`make_classification` and `make_hastie_10_2` are now exported by
`sklearn_datasets` and the facade. They adapt sklearn 1.6.1
`datasets/_samples_generator.py` in `sklearn_classification_datasets`. The
`classification_dataset_options` type stores generation parameters; it is not
counted as an estimator. The inventory remains 217 core estimator/transformer/
splitter types, with 127 source modules and 128 native test programs. There are
108 core reference suites retaining 430,965 groups, plus the separate 42-group
legacy RNG suite. Thirty-three new isolated rejection cases bring 18 registries
to 3,178 cases. No extension algorithm or new dependency is introduced.

```
type(classification_dataset_options) :: options
options%n_informative=4
options%n_redundant=2
options%n_repeated=1
options%n_classes=3
options%weights=[.2_dp,.3_dp]
options%random_state=42
call make_classification(120,9,x,labels,options)
call make_hastie_10_2(120,x,labels,random_state=42)
```

Both routines allocate real X matrices and integer label vectors. Native sample
and feature counts are required, following the existing dataset API; an omitted
seed deterministically uses zero rather than Python's global random state.
`make_classification` accepts optional options with Python's remaining defaults:
2 informative, 2 redundant, 0 repeated features; 2 classes and 2 clusters per
class; flip_y=0.01; class_sep=1; hypercube=true; shift=0; scale=1; shuffle=true.

The generator covers distinct binary hypercube vertices, scaled random polytopes,
per-cluster random covariance, redundant linear combinations, repeated columns,
uninformative Gaussian columns, weighted cluster allocation, randomly replaced
labels, affine transforms, sample shuffling and independent feature shuffling.
Informative dimensions above 30 follow the source's leading random bits plus
30-bit distinct vertices. Empty clusters still consume their covariance draws.
Full and partial class weights retain source truncation and cyclic remainder
allocation. Weights are not normalized: totals below one receive the remaining
samples, and overallocated clusters are clipped to the requested row count, as
in the source. A missing last class weight is inferred from one minus the sum.

Options `feature_shift(:)` and `feature_scale(:)` represent Python vector
transforms and override the corresponding scalar fields. `random_shift` and
`random_scale` represent Python shift=None and scale=None. Random flags cannot
be combined with explicit vectors. Scalars and vectors must be finite; scalar
scale must be positive, while vector scales may include zero and negative
entries, matching the source array parameter path. Native weights must be finite
and nonnegative, including an inferred last class; negative-weight slice behavior
accepted incidentally by Python is outside the native scope. Cluster products and
weighted cluster sizes must fit native default integers. Noise fractions must be
in [0,1], class separation positive, feature blocks must fit, and the requested
vertices must be representable by the informative dimensions.

Hastie 10.2 generates ten standard Gaussian features with labels +1 where squared
radius exceeds 9.34, and -1 otherwise. Labels are native integers rather than
Python's float array, with identical class values. Both generators use independent
local legacy RandomState-compatible RNG streams and leave caller options intact.

The shared bagging sampling helper now allocates its population pool only for
its permutation branch. Tracking and reservoir selection retain their previous
random draws without a needless population-sized allocation; this makes sampling
small vertex sets from a 2**30 population practical. Native tracking membership
uses a linear search, so very large tracked subsets can still be expensive.

The new source suite retains 608 result groups over 304 fixtures: 256 clustered
classification fixtures (32 configurations, eight seeds including the largest
native positive default integer) and 48 Hastie fixtures, including 12,000-row
benchmark draws. Complete features and labels are compared against sklearn.
Cases cover each sampling branch, empty and overallocated clusters, inferred/
zero/unnormalized weights, all feature blocks, one class, all/no label flips,
high informative dimensions, random polytopes and scalar/vector/random affine
transforms. There are also 33 isolated invalid-input checks.

Run `python validation/compare_classification_datasets.py --profile checked`;
use `--profile release` or `--executable` for an existing canonical driver.
See `example/classification_sample_generators.f90` for public usage. Original
native contributions remain MIT with retained sklearn BSD and existing RNG
notices. No external repository was imported.


## Gaussian-process covariance kernels and fitted kernel snapshots

`sklearn_gp_kernels`, also exported by `sklearn`, provides RBF, Matern,
RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel and WhiteKernel
counterparts through a value-type `gp_kernel`. Constructors are
`rbf_covariance(length_scale,fixed)`, `matern_covariance(length_scale,nu,fixed)`,
`rational_quadratic_covariance(length_scale,alpha,fixed_length,fixed_alpha)`,
`periodic_covariance(length_scale,periodicity,fixed_length,fixed_periodicity)`,
`dot_product_covariance(sigma_0,fixed)`, `constant_covariance(constant_value,fixed)`
and `white_covariance(noise_level,fixed)`. Parameters have sklearn's defaults.
RBF and Matern accept scalar or per-feature length scales. Matern supports all
positive finite nu values and positive infinity, with closed forms for 0.5, 1.5,
2.5 and infinity and a log-scaled Bessel K evaluation otherwise.

Kernel expressions support sum, product and integer/real powers, including
nonnegative real64 constants on either side of + or *. Flat internal expression
storage gives independent value copies without recursive allocation ownership.
The public methods are `%matrix(x[,y])`, `%diag(x)`, `%gradient(x)`, `%theta()`,
`%with_theta(theta)`, `%n_dims()` and `%is_set()`. Gradients are with respect to
log hyperparameters, in source order: left before right in compositions, alpha
before length scale for RQ, length scale before periodicity for periodic kernels.
Fixed flags omit whole parameter groups from theta and gradient width, including
all coordinates of a fixed anisotropic length scale. Power exponents and Matern
nu are not included in theta, matching sklearn. `%with_theta` returns a copy.

White noise contributes only on the diagonal of auto covariance. A supplied Y
always produces zero white cross covariance, even for identical arrays. Kernel
diagonals use a direct O(samples * expression_nodes) path. Gradients accept only
auto covariance, represented by their x-only signature. Length scales, RQ alpha
and periodicity must be finite and positive; constants, white noise and dot-product
sigma_0 finite and nonnegative. Zero active scalar values use negative-infinite
log theta, as in the source. Invalid real powers raise a diagnostic; gradients
at zero bases use zero for nonnegative powers. Inputs remain nonempty finite
dense real64 matrices. Hyperparameter bounds and optimizer integration,
PairwiseKernel/CompoundKernel adapters and sparse/object inputs are not provided.

Both `gaussian_process_regressor` and `gaussian_process_classifier` accept an
optional `%kernel` expression. This replaces their legacy amplitude/length-scale
RBF configuration (and GPR white-noise field). When no expression is set, existing
scalar controls retain their meaning. Fit clones the configured kernel into an
internal snapshot, so later input-kernel or scalar edits do not alter an existing
fit; a subsequent fit uses the changed configuration. This also fixes the older
GPR behavior of consulting live scalar kernel controls during prediction.
GPR now supports unfitted prior predictions: zero means, kernel covariance and
kernel standard deviations. Posterior covariance is returned without diagonal
clipping, matching sklearn; standard deviations clip negative roundoff variances.
The existing scalar alpha, vector-target regression, binary/OVR classification,
normalization and fixed-parameter training scope remains. Automatic hyperparameter
optimization, GPR multiple targets, vector alpha and posterior sampling remain
outside this increment. Set `optimizer=None` for the Python reference equivalent.

General Matern values use an adapted MIT SciFort Bessel helper from
`c:/fortran/scifort/src/special/bessel_k.f90`, source SHA256
`b03ec89063b912c46e2d0899d79f10eb0375b81cc978ffb2808fc82798d0a53d`.
Its copyright header is retained and the full license is in LICENSES/scifort-MIT.txt.
Only the implementation support module is imported; SciFort is not linked and
its additional Bessel APIs are not exported by the facade. General Matern length
gradients use an analytic Bessel-ratio identity. The pinned source approximates
these with a 1e-10 forward difference; comparisons therefore also use independent
1e-5 central differences of sklearn kernel values. This numerical choice adds no
extension algorithm. All kernel algorithms have sklearn counterparts; original
native interfaces remain MIT with retained sklearn BSD notices.

The inventory remains 217 core estimator/transformer/splitter types, with 129
source modules and 130 native test programs. Kernel expression/evaluation storage
is counted separately from that estimator inventory. There are 109 core reference
suites retaining 440,097 groups, plus the separate 42-group legacy RNG suite.
Twenty-seven new rejection cases bring 19 isolated registries to 3,205 cases.

The new suite retains 9,132 groups over 876 fixtures: 300 standalone kernel,
288 GPR and 288 GPC fixtures. It compares auto/cross matrices, diagonals, theta,
fixed groups, gradients, cloning, shifted parameters, prior/posterior predictions,
uncertainty, Cholesky/dual state, binary/multiclass probabilities and likelihoods,
fitted snapshots and refits. There are also 45 independent general-Matern central
gradient groups and 48 SciPy log-Bessel support points within the stated group
count. Duplicate rows, zero scalar kernels, anisotropy and nested expressions are
retained. The development central-gradient maximum error was 3.82e-10; source
forward-difference gradients differed by up to 1.22e-4 due to their small step.
The source comparison uses 3e-4 tolerance for those noisy gradients and stricter
central-difference checks, rather than claiming bitwise equivalence.

Run `python validation/compare_gp_kernels.py --profile checked`; use
`--profile release` or `--executable` for completed binaries. The existing
`compare_gp_classification.py` now accepts the same profile/executable arguments,
so regression gates never need concurrent fpm invocations. See
`example/gaussian_process_kernels.f90` for a complete public example.


The GP kernel suite additionally retains three constant-kernel multiclass tie
certificates per profile. Each has three equally populated classes and a constant
Gram/cross matrix: the OVR heads are permutation-equivalent, so exact normalized
probabilities are 1/3 for every class. Source and native probabilities must both
satisfy this symmetry within 64*epsilon*sum(abs(Laplace_erf_coefficients)), and
reported labels must belong to the analytically tied class set. Observed source
and native labels, class counts and errors are retained in the reports. Optimized
reductions select a different tied class in these fixtures; all other label
comparisons remain exact. These three certificate groups are included in the
9,132-group total; no fixture is dropped or relabeled.


## Multi-target GP regression and marginal-likelihood kernel fitting

`gaussian_process_regressor` now supports matrix targets, observation-specific
noise, log-marginal likelihood evaluation/gradients and opt-in bounded L-BFGS-B
kernel fitting with seeded restarts. Algorithms adapt sklearn 1.6.1
`gaussian_process/_gpr.py`, with existing kernel derivatives and the already
vendored L-BFGS-B 3.0 implementation. Native defaults retain optimizer='none';
set `%optimizer='fmin_l_bfgs_b'` to learn free parameters in an explicit kernel.
Legacy scalar RBF controls remain fixed, matching the no-free-parameter default
kernel convention. GPC optimization is not added in this increment.

Use `%fit(x,y)` with a real vector or (samples,targets) matrix. Scalar `%alpha`
is retained; an allocated `%alpha_values(:)` overrides it and accepts one value
(broadcast) or one per training row, all finite and nonnegative. Noise is added
in normalized target units without target-scale adjustment, as in sklearn. The
same covariance hyperparameters are shared across independent output columns,
and joint likelihood is the sum of their likelihoods. `%n_targets=0` means
automatic; a positive count configures prior output width and validates fitted
width. `%n_targets_` and `%n_features_in_` describe the fitted data.

`%predict_multi` and `%predict_std_multi` always return (samples,targets)
matrices; `%predict_cov_multi` returns (samples,samples,targets). This explicit
rank remains for a one-column matrix, whereas Python squeezes one-target
predictions. Existing vector prediction/std/cov methods work with one target and
raise a clear diagnostic for multiple targets. Matrix `%score` averages target
R2 scores uniformly, with optional finite nonnegative sample weights; one-row
R2 returns NaN. Covariance and standard deviation scale independently with each
target's normalization. Matrix normalization clips near-zero scales; vector
normalization replaces only exact zeros, following the source distinction.
Scalar/one-column fast-axis reductions use the existing NumPy-compatible pairwise
sum contract, including the non-binary constant-target regression case.

New fitted storage includes normalized `%y_train_`, `%alpha_multi_`, target mean/
scale vectors, and public `%kernel_`. `%alpha_`, `%y_train_mean_` and
`%y_train_std_` remain authoritative for single-target prediction compatibility;
matching matrix fields retain copies for diagnostics. Input `%kernel` remains
unchanged by fitting; learned parameters are in `%kernel_`.

`value=model%log_marginal_likelihood(theta,gradient,clone_kernel)` evaluates the
joint likelihood and optional derivative with respect to log kernel parameters.
With no theta it returns the cached fit likelihood and cannot return a gradient.
The default clone_kernel=true leaves fitted state intact; false updates fitted
kernel parameters, matching the source's side effect, without rebuilding cached
prediction factors. A non-positive-definite candidate returns negative infinity
and a zero gradient, while an invalid final fit raises a diagnostic.

`gp_kernel%bounds()` returns log bounds in (n_dims,2) form, including (0,2) for an
all-fixed expression. `%with_bounds(log_bounds)` returns a copy. Initial free
bounds default to log([1e-5,1e5]) for each parameter; composed kernels preserve
child bounds and theta clones retain them. This typed log-matrix API represents
Python's per-hyperparameter positive-domain constructor bounds. Ordered finite
or correctly signed infinite endpoints are accepted; random restarts require
finite bounds. Bounds remain in the same order as theta/gradient.

The optimizer uses ten L-BFGS correction vectors, default projected-gradient
tolerance 1e-5, ftol=2.220446049250313e-9, and 15,000 iteration/function limits,
matching SciPy defaults. `%n_restarts_optimizer` adds log-uniform initial points
from the local legacy RNG, seeded by `%random_state` (default zero). The first
run starts from the configured theta; the lowest objective is selected with
first-best tie behavior. Controls `%tol`, `%ftol`, `%max_iter` and `%max_fun`
allow explicit solver settings. Diagnostics include run points/objectives,
iteration/evaluation totals, convergence and the chosen termination message.
Optimizer paths and iteration counts need not be bit-identical to SciPy's C
translation. General Matern derivatives retain the prior analytic implementation
rather than sklearn's noisy 1e-10 numerical difference.

The inventory remains 217 core estimator/transformer/splitter types and 129 core
source modules, with 132 native test programs. A new suite brings 110 core
reference suites to 448,167 retained groups, plus the separate 42-group legacy
RNG suite. Twenty-eight new rejection cases bring 20 registries to 3,233 cases.
The suite retains 8,070 groups over 384 fixtures, including 96 optimization
comparisons across nine kernel configurations, scalar/vector noise, vector/
matrix/one-column targets, normalization, priors, weighted scores, bounds,
likelihood clone semantics and non-PD candidates. Posterior state is compared
strictly at the native learned theta; learned objectives and predictions are
also compared with independent SciPy optimizations, including every restart.
Likelihood gradients are checked with independent sklearn central differences.
No fixture is dropped for an optimizer-path difference.

Reproduce with `python validation/compare_gpr_advanced.py --profile checked`,
or use `--profile release` / `--executable` for an existing driver. See
`example/gaussian_process_regression.f90` for multi-target fitting with learned
kernel parameters and heteroscedastic noise. Posterior sampling, sparse data,
metadata routing and general GP classifier kernel optimization remain outside
this increment. No new dependency or public extension algorithm is introduced.
Original contributions remain MIT with the existing sklearn, NumPy and optimizer
notices retained.


## GP classifier likelihood gradients, kernel fitting and warm starts

`gaussian_process_classifier` now evaluates binary Laplace marginal-likelihood
gradients and optionally learns kernel parameters using bounded L-BFGS-B with
seeded restarts. Multiclass one-vs-rest heads are optimized independently and
retain separate fitted `%kernels_(:)` in sorted class order; binary fits have
one kernel. Prediction uses each head's own covariance and diagonal. Original
scalar RBF controls remain supported and fixed. Native optimizer='none' remains
the default; set `%optimizer='fmin_l_bfgs_b'` for free parameters of an explicit
kernel expression. No classifier algorithm outside sklearn is added.

Controls `%n_restarts_optimizer`, `%random_state`, `%max_iter`, `%max_fun`, `%tol`
and `%ftol` follow the GP regressor's native interface, using the existing
L-BFGS-B 3.0 implementation with ten correction vectors, gtol=1e-5,
ftol=2.220446049250313e-9 and default 15,000 iteration/evaluation limits. Each
head receives the same scalar RNG seed, matching sklearn's independently cloned
binary estimators. There is a pinned-source distinction: sklearn 1.6.1 GPC
exponentiates its uniform draw in log bounds before passing it as optimizer
theta. This differs from GPR, which passes the log-uniform draw directly. Native
GPC deliberately preserves that exponentiation and subsequent bound clipping;
it does not silently substitute the GPR convention. Raw initial points are
retained in `%optimization_initial_points_` and compared directly with sklearn.

`%warm_start` reuses the last posterior mode while evaluating candidate kernels.
It defaults false. A new fit resets caches, as sklearn's outer classifier
recreates its binary estimators. Fit performs the source's likelihood evaluation
and separate predictive posterior-mode calculation. Fitted mode-control snapshots
preserve the settings used at fit even if parent controls are edited later.
`%posterior_mode_` retains fitted prediction state; `%mode_cache_` tracks the
separate likelihood-evaluation cache, including updates on cloned evaluations.
`%fit_initial_cache_` records the cache entering the final predictive mode solve.

`value=model%log_marginal_likelihood()` remains callable from an intent(in) model
and returns the cached fit value. With a theta vector, optional allocated output
`gradient` returns the binary derivative, and optional `clone_kernel` defaults
true. Cloned evaluation preserves fitted kernel/prediction factors while updating
the mode cache, as in the source. False updates fitted kernel parameters without
rebuilding prediction factors. Multiclass evaluation accepts either one shared
theta or class-major concatenated head parameters and returns mean likelihood;
multiclass gradients raise a diagnostic, matching sklearn. Requesting a cached
gradient without theta is rejected via `eval_gradient=.true.`.

The likelihood derivative implements GPML Algorithm 5.1, including the third
logistic derivative and posterior-mode correction. Posterior iteration follows
Algorithm 3.1 and retains the pre-update pi/W/L and previous likelihood at the
source stopping point. NumPy-compatible pairwise sums, source multiplication
grouping and a standard C-runtime log1p call reduce numerical differences without
adding a package dependency. General Matern gradients retain the existing analytic
Bessel implementation; independent central differences additionally validate them.

Diagnostics distinguish posterior convergence from optimizer convergence and
retain per-head iteration/evaluation totals, per-run points/objectives, per-run
termination messages and convergence flags, and the selected run's status.
`%optimization_eval_cache_` retains the input cache associated with each returned
run objective, including a solver-restored accepted point after line-search stops.
Warm objectives depend on cache history and truncated Newton convergence, so
validation replays sklearn's mode solve from the same recorded input state for
endpoint and final-prediction transitions. Independent full sklearn fits still
check learned likelihoods and probabilities, every restart result and initial
point. No fixture is removed for a path difference. Explicit likelihood evaluations leave the cached fit likelihood unchanged.

The inventory remains 217 core estimator/transformer/splitter types and 129 core
source modules, with 134 native test programs. A new suite brings 111 core
reference suites to 457,011 retained groups, plus the separate 42-group legacy
RNG suite. Thirteen new rejection cases bring 21 registries to 3,246 cases.
The new suite retains 8,844 groups over 324 fixtures: two/three classes, nine
kernel configurations, three seeds and six cold/warm/optimized/truncated-mode
configurations. It includes 96 independent optimization comparisons, 288 restart
endpoint checks, analytic/central gradients, posterior state, shared/packed theta,
clone/mutation semantics, refits and exact labels. Development learned-probability
differences were below 5e-6 and selected likelihood differences below 8e-7;
conditional posterior and cache transitions agreed to near machine precision.

Run `python validation/compare_gpc_advanced.py --profile checked`; use
`--profile release` or `--executable` for completed binaries. See
`example/gaussian_process_classification.f90` for learned multiclass kernels.
One-vs-one GPC, multiclass likelihood gradients, object/sparse kernels and metadata
routing remain outside this increment. Original native interfaces and diagnostic
storage remain MIT, with sklearn, NumPy and the existing optimizer notices
retained. No external repository or dependency is imported.


## Gaussian process one-vs-one classification

Set `model%multi_class='one_vs_one'` before fitting a
`gaussian_process_classifier`. The default remains `'one_vs_rest'`. For more
than two classes, heads follow lexicographic pairs of sorted classes: (1,2),
(1,3), ..., (c-1,c). Each head trains only on its two classes, preserving input
row order. All existing covariance expressions, optimizer controls, seeded
restarts and warm-start likelihood evaluations apply independently to each pair.
Binary fits retain the same prediction and probability behavior in both modes.

`predict` combines pair predictions with sklearn's bounded confidence correction:
each head votes by the sign of its latent mean, while its positive-class
probability is subtracted from the first class's confidence and added to the
second's. The correction `confidence/(3*(abs(confidence)+1))` breaks vote ties
without overturning a one-vote lead; exact final ties select the first class.
For multiclass one-vs-one fits, `predict_proba` raises a diagnostic, as in sklearn.

`%class_pairs_(2,heads)` contains one-based sorted class indices (zero for OVR
and binary fits). `%n_training_samples_(head)` gives each head's row count;
`%training_indices_(:count,head)` maps those rows to the original training array.
Head-specific pi, square-root weights, posterior modes, alpha and cache columns
use their leading `count` rows. Cholesky factors use the leading count-by-count
block. Trailing rows/blocks are zero padding, including optimizer cache storage.
The global `%x_train_` retains the original training array. OVR and binary fits
continue to use all rows, with their previous state shapes and behavior.

The cached likelihood is the mean over binary heads. Explicit evaluation accepts
one shared theta or a pair-major concatenation of every fitted head's theta.
Clone/mutation and cache semantics match the existing GPC interface. Multiclass
gradients remain unsupported. sklearn 1.6.1's outer packed-theta check mistakenly
uses the number of classes rather than the number of pairs; for four or more
classes it cannot evaluate a full packed OVO kernel. The native interface accepts
the full pair count and is validated against the independent binary estimators.
This is a documented Fortran interface adaptation of the same core algorithm.

Refitting resets all subset maps and caches, including transitions between OVO,
OVR and binary fits. The fitted decomposition is retained if `multi_class` is
edited after fit; call `fit` to change it. See `example/gaussian_process_ovo.f90`.


## Gaussian process prior and posterior sampling

`draws=model%sample_y(x,n_samples=1,random_state=0)` returns a matrix with shape
`(n_query,n_samples)` for a single target. Use `sample_y_multi` for shape
`(n_query,n_targets,n_samples)` with one or more targets. Unfitted models sample
the prior and use the configured `n_targets` (zero means one automatic target).
Fitted models sample the posterior using the fitted kernels, target count and
normalization scales, even if the configured kernel is subsequently changed.
Zero requested draws return a zero-length final axis. Negative draw counts or
seeds, nonfinite predictive moments and outputs exceeding default-integer array
capacity are rejected. The single-target method rejects multiple targets.

Sampling uses the existing scalar-seeded NumPy RandomState-compatible generator,
with seed zero by default independently of the fit-time `random_state`. Each
call owns its RNG state and leaves Fortran's intrinsic RNG and the fitted model
unchanged. The draw order is target-major, then sample-major, then query-major;
all query normals are consumed even for rank-deficient or zero covariance.
Different targets are sampled independently, matching sklearn's multioutput GP.

The covariance factor uses the existing native Jacobi SVD, with matrix scaling
before decomposition to avoid overflow/underflow when covariance entries are
squared. Zero covariance is handled directly; singular covariances do not need
jitter. Nonnegative singular values follow NumPy's default SVD sampling rule.
Native singular-vector signs and bases can differ from NumPy's LAPACK backend,
so individual seeded trajectories are not promised to match sklearn. Native
calls with the same inputs and seed are reproducible. The sampled Gaussian law,
normal-stream structure, cross-target independence and predictive moments are
validated independently of basis orientation; scalar paths also match directly.

sklearn 1.6.1 has a singleton-query prior shape bug: `predict` squeezes the mean,
then `sample_y` fails for both one and multiple targets. The Fortran methods
retain the explicit shapes above. Those source failures are recorded and the
intended scalar draws are compared with NumPy's direct multivariate-normal call.
See `example/gaussian_process_sampling.f90`.


## FactorAnalysis randomized fitting and orthogonal rotation

`factor_analysis` adds `%rotation='none'|'varimax'|'quartimax'` and
`%svd_method='lapack'|'randomized'`. Native defaults remain `'none'` and `'lapack'`
to preserve existing behavior; `'lapack'` selects the native full Jacobi SVD,
without a LAPACK dependency. Set `%iterated_power` (default 3) and `%random_state`
(default 0) for seeded randomized fitting. Power counts zero through two use
unnormalized power steps; higher counts use pivoted-LU normalization, matching
sklearn's automatic rule. Ten oversamples, automatic wide-matrix transposition,
Householder range QR and SVD sign selection reuse the existing native core.
The RNG is seeded once per fit and advances across fitting iterations. Zero
factors produce the same diagonal noise model in either solver mode.

Rotation is applied after fitting, preserving noise variance, fitted covariance,
precision, likelihood history and iteration count up to roundoff. Latent
coordinates and loading interpretation change. Varimax subtracts the normalized
column-energy term; quartimax omits it. The orthogonal polar-factor iteration
uses sklearn's fixed 100-step limit and relative 1e-6 criterion. Native SVD's
missing left null-space columns are completed orthogonally before forming a
polar factor, so rank-deficient or zero loadings retain an orthogonal rotation.

`%rotation_matrix_` has shape `(factors,factors)` with
`components_ = matmul(transpose(rotation_matrix_), unrotated_components)`. `%n_rotation_iter_` and `%rotation_converged_` describe the separate
rotation solve. With no rotation or zero factors they are zero/true and the
matrix is identity (empty for zero factors). All-zero nonempty loadings can
reach the 100-step limit with an unchanged model. Refit resets these diagnostics.
Controls/noise initialization must be finite and valid. The existing component
limit `0 <= n_components <= min(n_samples,n_features)` and `-1` full-feature
selection remain; exactly zero singular values retain the likelihood floor.

SVD signs and rotation axis permutations are not unique. Reference checks align
signed factor permutations and compare covariance, precision and scores directly.
At tol=0 capped fits, last-step improvements within an explicit floating-point
bound can give different convergence flags; those fixtures retain certificates.
One-feature, one-factor varimax has identically zero criterion, and cubic rounding
can make the source helper stop in two steps rather than reach its limit. This
iteration-only difference is recorded while factors and orthogonality remain
strictly checked. See `example/factor_analysis_rotation.f90`.


## Probabilistic PCA and automatic dimension selection

`pca` adds `%dimension_mode='count'|'variance'|'mle'|'zero'` and fitted
`%n_components_`. Default `'count'` keeps the existing integer `n_components`
interface, including its legacy zero-means-all convention. Other modes override
that integer field: `'variance'` uses `%variance_fraction` (default .95, strictly
between zero and one), `'mle'` applies Minka's likelihood-based rank selection,
and `'zero'` selects sklearn's explicit zero-component model. MLE requires
samples >= features. Fraction selection uses a strict cumulative-variance
threshold (right-sided search), so an exactly met threshold retains another
component. Zero total variance selects one component in fraction mode and can
select zero in MLE mode, following sklearn 1.6.1. The existing zero ratio
convention remains: undefined 0/0 variance ratios are stored as zero.

`%mle_log_likelihood_(:)` is allocated only for MLE fits. Entry 1 represents rank
zero and is negative infinity; entry r+1 assesses rank r. The final feature rank
is not assessed, as in sklearn. Candidates whose rank eigenvalue is below 1e-15 return negative infinity.
Exact spectral ties can make Minka's logarithm undefined; fitting raises an
explicit diagnostic, corresponding to sklearn's math-domain failure. Refitting
in another dimension mode clears the diagnostic array. Full native Jacobi SVD
remains the PCA backend for this increment.

New `get_covariance`, `get_precision`, `score_samples(x)` and `score(x)` methods
implement sklearn's probabilistic PCA contracts. The covariance combines the
retained principal directions with isotropic `noise_variance_`. The pinned
sklearn 1.6.1 implementation scales the directions by sqrt(explained_variance)
again when whitening is enabled; the native generative covariance deliberately
preserves that behavior. Precision and likelihood use a Cholesky factor of this
covariance. They require a proper positive-definite covariance, and raise a
clear diagnostic for zero covariance or noise-free reduced-rank models. There
is no singular-Gaussian pseudodensity. Covariance remains available for such
models, as do their transforms and inverses. A zero-dimensional inverse returns
the fitted mean for each input row; transform results have a zero-width axis.

SVD directions are not unique in zero or tied eigenspaces. Reference checks keep
those fixtures, verify orthonormality, eigen equations and membership in the
independent sklearn eigenspaces, then evaluate source operators in the certified
native basis. Exactly zero-variance whitening magnifies floating-point projection
residuals; training projections are compared before whitening in those certified
null directions. Existing `fit_transform` remains fit followed by transform; it
does not reproduce LAPACK U-based null-whitening coordinates. Gaussian moments and selected dimensions remain independently
checked. See `example/probabilistic_pca.f90`.


## PCA solver selection

`pca%svd_solver` accepts `'full'` (the preserved native default), `'randomized'`,
`'covariance_eigh'` and `'auto'`. `%fit_svd_solver_` records the resolved backend.
Full SVD remains the native Jacobi implementation. Covariance eigendecomposition
forms the uncentered Gram matrix, subtracts n times the mean outer product
and divides by n-1 before the symmetric Jacobi solve, clips negative roundoff eigenvalues and derives singular values.
It avoids explicitly centering a data copy. Its full feature spectrum, including
wide-matrix zero eigenvalues, is used for the source-specific residual-noise mean.
As in sklearn, forming covariance squares the conditioning sensitivity.

Randomized fitting uses `%iterated_power=-1` for automatic 4/7 powers or a
nonnegative explicit count, `%n_oversamples=10` (positive),
`%power_iteration_normalizer='auto'|'none'|'LU'|'QR'` and `%random_state=0`.
Automatic normalization is none for up to two powers, LU otherwise. The local
legacy NumPy-compatible RNG is seeded once per fit; it leaves the intrinsic RNG
unchanged. Automatic transposition, QR range extraction and projected SVD reuse
the existing core. PCA requests unflipped factors and resolves signs from the
right vectors, as in sklearn. Missing zero-singular vectors are completed
orthogonally in this PCA caller without changing older consumers' null handling.

`fit_transform` for randomized fitting returns the projected left singular
coordinates (`U*S`, or `U*sqrt(n-1)` when whitening). A later `transform(x)` uses
the fitted right directions and can differ at low power counts, matching sklearn.
Full/covariance backends retain the native fit-then-transform behavior, including
the documented nonunique null-whitening coordinates. Zero-covariance randomized
whitened training coordinates retain an orthogonal left basis even though later
transforms are zero. Singular directions do not have unique coordinates.

Automatic routing follows sklearn 1.6.1: covariance_eigh for at most 1,000
features and n_samples >= 10*n_features; otherwise full for a maximum shape
at most 500 or MLE selection; otherwise randomized for a positive integer count
below .8*min(shape); full for the remaining cases. The first covariance condition
also takes precedence for MLE and variance-fraction modes. Randomized fitting
requires count mode; fraction, MLE and explicit zero modes use full/covariance
backends or automatic routing. Legacy count zero still means all components.
Refits reset the fitted solver and all existing dimension diagnostics.

The internal randomized SVD helper now accepts optional oversampling (including
zero), normalization and sign controls. Its existing 4/7-power, ten-oversample,
LU-normalized, sign-flipped defaults are preserved. Requested ranks larger than
the matrix's smaller dimension are truncated to that dimension, including the
new unnormalized path. See `example/pca_solver_selection.f90`. ARPACK, sparse
inputs and alternate numerical dtypes remain outside this implementation.


## Dense and CSR randomized TruncatedSVD

`truncated_svd` adds `%algorithm='randomized'`, `%n_iter=5`,
`%n_oversamples=10`, `%power_iteration_normalizer='auto'|'none'|'LU'|'QR'` and
`%random_state=0`. The native default `%algorithm='full'` preserves the existing
exact dense SVD backend; this tag is a Fortran compatibility choice, not a
sklearn TruncatedSVD constructor value. Randomized fitting is the sklearn
counterpart. It requires at least two features and a positive requested rank
not exceeding feature count. Requests above sample count are clipped to the
returned smaller matrix dimension, as in sklearn. Native fitted
`%n_components_` and `%n_features_in_` expose the actual shape.

`fit`, `fit_transform` and `transform` accept dense real64 matrices or the
existing `csr_matrix` type. CSR fitting requires the randomized algorithm;
query CSR matrices can be transformed by either fitted backend. Reduced
coordinates, stored components and inverse transforms are dense. The input
matrix is never centered. `fit_transform` returns the actual `X @ components.T`
projection, unlike randomized PCA's projected-left coordinates. Explained
variance is computed from those returned coordinates, not singular-value energy.
Zero total feature variance retains the native zero ratio convention.

The randomized core uses CSR forward/transpose products and dense reduced
workspaces without allocating a dense copy of the input. Wide transposition,
LU/QR/no normalization, seeded draws and sign selection share the existing
dense core. Each estimator call owns its RNG and leaves sparse input storage
unchanged. Zero-singular vectors are completed orthogonally for this estimator.
CSR variance includes implicit zeros and uses two centered passes, aggregating
unsorted duplicate values within each row before computing squared deviations.
Stored zeros and duplicates remain unchanged in the caller's object.

sklearn 1.6.1 TruncatedSVD has a parameter-validation typo: it lists 'OR' where
its documented QR normalizer should be accepted. Native QR works. Reference
fixtures record those source errors and execute the actual sklearn QR algorithm
with parameter validation disabled for those otherwise valid configurations.
Duplicate input references use a canonicalized copy of the same scipy CSR matrix
for numerical matrix semantics. Native inputs need not be sorted or canonical.
SVD null bases are nonunique and are verified against source null spaces;
nonzero coordinates remain compared directly. See
`example/sparse_latent_semantics.f90`. ARPACK, CSC input storage and sparse
inverse outputs remain outside this increment.


## KernelPCA learned pre-images (2026-11-22)

Added sklearn KernelPCA's optional kernel-ridge inverse fitting and latent-query
reconstruction, with alpha regularization and fitted latent/dual diagnostics.
The uncentered latent kernel reuses the forward model's fitted gamma, and the
linear inverse preserves sklearn's omission of the original feature mean.
Refitting without inverse fitting clears stored inverse state. Precomputed
inverse fitting and singular/indefinite ridge systems are rejected explicitly.

`validation/compare_kernel_preimage.py` retains 4,520 groups across 452 fixtures,
using installed sklearn 1.6.1 and 452 independent ridge-equation residual
certificates. Cases cover linear/RBF/polynomial/cosine/sigmoid kernels, three
seeds, tall/wide matrices, regularization including zero, explicit/default
gamma, automatic/explicit component counts, zero/constant input, one-sample
fits, fit versus fit_transform, training/query/arbitrary-latent reconstruction
and clearing inverse state on refit. Fourteen rejection cases cover finite
controls, alpha bounds, disabled/unfitted/stale inverse state, inverse query
shape/finiteness/emptiness, precomputed fitting, zero-dimensional latent space
and singular/indefinite systems. The indefinite sigmoid rejection is confirmed
with the installed source as well as the native implementation.

Inventory: 217 public core types, 129 core source modules, 141 native test
programs and 118 core reference suites with 515,153 retained groups. The
separate 42-group legacy RNG suite remains excluded. There are 28 invalid-input
registries with 3,340 cases. These counts are cumulative; final verification
below identifies the checks actually rerun for this change.

Run `python validation/compare_kernel_preimage.py --profile checked`, or select
`--profile release` / an explicit `--executable`. The original Fortran interfaces
and Cholesky adaptation remain MIT, with the existing sklearn BSD notice
retained. No additional source package, dependency or extension algorithm is
imported. Dense inverse outputs and the existing eigensolver remain the scope.


## SVD linear discriminant analysis (2026-11-22)

Added sklearn's within/between thin-SVD discriminant solver, projection methods,
component selection, rank tolerance and optional covariance storage. The existing
native LSQR default remains; SVD is selected explicitly. Empirical priors and
all retained discriminant axes drive classification, independently of the
requested transform width. Rank-deficient/wide matrices, constant or duplicated
features and singleton classes are supported when within-class rank is nonzero.
The default SVD path allocates no feature-by-feature covariance matrix.

`validation/compare_discriminant_svd.py` retains 11,084 groups over 652 fixtures
and 652 independent within-covariance whitening / between-scatter diagonalization
certificates. Three seeds cover binary and multiclass labels, tall/wide and
single-feature inputs, singleton classes, unequal feature scales, duplicated or
rank-one features, two tolerances, automatic/explicit component counts,
fit/fit_transform, optional covariance, probabilities, predictions, scores,
coefficients and training/query projections. Signs are aligned only for
nonunique SVD axes; binary class-row coefficients/scores are compared in the
source's margin convention. Three coincident-mean cases explicitly preserve
source's undefined (NaN) variance ratios and zero transform rank.

A 12 by 20,000 fixture validates the thin-SVD path with covariance absent in
both implementations; one dense feature covariance would occupy 3.2 GB.
Twenty isolated rejection cases cover controls, shrinkage, dimensions, finite
inputs, fit state, class/sample counts, empty retained rank and stale diagnostics
after LSQR refitting. Two recorded source empty-rank IndexErrors explain the
native diagnostic for zero within variance or an excessively high threshold.

Inventory: 217 core public types, 129 core modules, 142 native test programs,
119 core reference suites and 526,237 cumulative retained groups. The separate
42-group legacy RNG reference suite remains excluded. There are 29 invalid-input
registries with 3,360 cases. Cumulative counts are not a complete rerun claim;
final verification below identifies the checks run for this increment.

Run `python validation/compare_discriminant_svd.py --profile checked`, or use
`--profile release` / `--executable`. Native interfaces and thin-SVD adaptations
remain MIT with existing sklearn BSD notices retained. No new source dependency,
public estimator type or extension algorithm is added. Custom priors and eigen
solver support remain outside this implementation.


## Eigen discriminant projection with shrinkage (2026-11-22)

Added sklearn LDA's generalized eigen solver with scalar covariance shrinkage,
uncentered discriminant projection and full-basis classification. Cholesky
whitening reduces the covariance pair to the existing symmetric eigensolver.
The native default remains LSQR; SVD keeps its thin, covariance-free path.
Eigen fitting retains the within covariance regardless of store_covariance,
with normalized scalings and native generalized-eigenvalue diagnostics.

`validation/compare_discriminant_eigen.py` retains 13,452 groups across 708
fixtures with 708 independent covariance-normalization / generalized-eigen
residual certificates. Three seeds cover binary/multiclass labels, tall/wide and
single-feature input, singleton classes, zero-variance features, shrinkage
0/.15/.7/1, automatic/explicit components, both fit entry points and covariance
storage flags. Coefficients, scores, probabilities and predictions compare
directly with installed sklearn 1.6.1 (binary native class rows use source's
single-margin convention). Transform caps do not reduce the classification basis.

There are 225 source eigenspace certificates for repeated eigenvalues, including
165 crossing the component cutoff. Alignment uses only rotations within source
eigenvalue clusters, with separate metric normalization and eigen-equation
checks; no fixture is discarded. Six zero-between-energy cases retain NaN
variance ratios and full requested eigen transform width. Twenty rejection
cases cover controls, sample/classes, non-positive-definite within covariance,
finite/shape contracts and fitted/refit state. Two singular covariance failures
are confirmed with source's generalized eigensolver as well as native Cholesky.

Inventory: 217 public core types, 129 core modules, 143 native test programs,
120 core reference suites and 539,689 cumulative retained groups. The separate
42-group legacy RNG reference suite remains excluded. Thirty invalid-input
registries retain 3,380 cases. These are cumulative counts, not a full rerun
claim; final verification below states the suites rerun for this change.

Run `python validation/compare_discriminant_eigen.py --profile checked`, or use
`--profile release` / `--executable`. Native interfaces and Cholesky/eigen calls
remain MIT with existing sklearn BSD notices retained. No public estimator type,
extension algorithm or dependency is added. Custom priors and automatic
shrinkage remain outside this increment.


## Automatic discriminant covariance shrinkage (2026-11-22)

Added sklearn LDA's classwise standardized Ledoit-Wolf shrinkage for LSQR/eigen,
with independent total covariance estimation for eigen. The existing native
Ledoit-Wolf estimator is reused; corrected two-pass variance and source's
mean-dependent constant-feature test provide standardization. Ordered class and
eigen-total shrinkage diagnostics are stored and cleared on fixed-mode refitting.
Automatic LSQR uses numerical least squares for singular covariance; eigen
retains its positive-definite requirement. Fixed solver defaults remain unchanged.

Automatic decision scores evaluate centered affine margins relative to the first
class. This preserves the classifier while avoiding cancellation of large common
intercepts. The stored coefficient/intercept rows keep their native interfaces.

`validation/compare_discriminant_auto.py` retains 6,650 groups across 357 fixtures
and 357 covariance/pseudoinverse certificates. Three seeds cover binary and
multiclass labels, tall/wide/single-feature input, singleton classes, unequal
feature units, constant and near-constant features, both solvers and component
caps. Per-class and total estimates compare with separately standardized source
Ledoit-Wolf fits. Six numerical-rank-deficient LSQR cases include zero covariance
and rank-one fits; 134 cases retain negative source generalized eigenvalues.

All fixtures remain. Six extreme near-constant cases use 80-digit classifier
certificates, source cancellation bounds and scaled covariance comparisons.
Six spectral certificates bound small-root uncertainty relative to the whitened
matrix norm, check basis error in covariance units and verify projection identities
against the retained source-certified axes. Seven repeated-basis certificates
and one undefined-ratio certificate are retained. Six classwise isotropy
certificates explain differing shrinkage coefficients with equivalent covariance;
the coefficient has no identifiable effect at an isotropic covariance.
Twenty rejection cases cover modes/controls, unsupported SVD automatic shrinkage,
class/sample counts, singular eigen fitting, finite/shape and fitted contracts.

Inventory: 217 core public types, 129 core modules, 144 native test programs,
121 core reference suites and 546,339 cumulative retained groups. The separate
42-group legacy RNG reference suite remains excluded. Thirty-one invalid-input
registries retain 3,400 cases. These are cumulative counts, not a complete rerun
claim; final verification below identifies the suites actually rerun.

Run `python validation/compare_discriminant_auto.py --profile checked`, or use
`--profile release` / `--executable`. The validator needs mpmath for its 80-digit
certificates. Fortran interfaces and centered evaluation remain MIT, with the
existing sklearn BSD notice retained. No public estimator type, extension
algorithm or external Fortran dependency is added. Custom priors remain outside
this increment.


## Custom discriminant class priors and thin QDA (2026-11-22)

Added `%priors` input for LDA/QDA, preserved caller weights, fitted normalization
status and zero-prior handling. LDA follows source's 1.e-5 renormalization
threshold; QDA retains unnormalized supplied weights. All LDA solvers weight
covariance and geometry by fitted priors, including optional SVD covariance.
Explicit-prior LSQR accepts numerical singularity through least squares.
Automatic decision scores select the first positive-prior reference class.
Negative infinite log priors exclude zero-weight classes safely under traps.

The new reference cases exposed and fixed a prior QDA mismatch for classes with
fewer samples than features. Source regularizes the thin retained SVD subspace,
rather than adding variance in every ambient direction. Native QDA now uses that
thin spectrum and an orthogonal completion of its retained null direction.
Positive regularization is required on this wide path; empirical/fixed LDA
solver defaults remain unchanged.

`validation/compare_discriminant_priors.py` retains 13,594 groups over 841
fixtures and 841 normalization/input-preservation certificates. Three seeds
cover all three LDA solvers, fixed/automatic shrinkage and QDA; uniform, skewed,
nonunit, near-unit, zero, one-positive, tiny and large priors; tall, wide and
single-feature data; both fit entry points and automatic/explicit components.
The QDA extra case accepts finite prior weights whose sum overflows, without
summing them. There are 210 zero-prior posterior/prediction certificates and
27 one-positive-prior SVD undefined-energy certificates.

Forty-eight thin-QDA certificates verify the source regularized spectrum,
projector idempotence/rank and inclusion of the centered training span. The
retained null direction is arbitrary: source query/training scoring uses the
same independently certified native direction. No fixture is removed.
Seventeen rejection cases cover malformed/negative/nonfinite/zero prior vectors,
LDA sum overflow, sample/class limits, singular eigen fitting, empty SVD rank
and unregularized wide QDA.

Inventory: 217 core public types, 129 core modules, 145 native test programs,
122 core reference suites and 559,933 cumulative retained groups. The separate
42-group legacy RNG reference suite remains excluded. Thirty-two invalid-input
registries retain 3,417 cases. Cumulative counts are not a complete rerun claim;
final verification below lists the checks actually rerun.

Run `python validation/compare_discriminant_priors.py --profile checked`, or use
`--profile release` / `--executable`. Original Fortran prior interfaces, safe
logarithms, reference selection and null completion remain MIT, with existing
sklearn BSD notices retained. No estimator type, extension algorithm or external
Fortran dependency is added. Sparse inputs and custom covariance objects remain
outside this interface.


## Weighted discrete NB streaming (2026-11-22)

Added weighted batch and partial fitting for MultinomialNB, BernoulliNB and
ComplementNB, persistent raw feature/class counts, fitted feature width and
Complement total-feature counts. First-call declared classes, missing classes,
class-set/width continuity, fit resets and independent copied state follow the
source streaming contracts. Optional class priors override empirical/uniform
prior selection without renormalization. That increment retained positive scalar alpha; the nonnegative NB smoothing
section below supersedes that restriction. Bernoulli thresholds now reject source-invalid negative values.

`validation/compare_discrete_stream.py` retains 22,150 groups across 824 fixtures
and 1,796 fitted snapshots. Three seeds cover all three estimators, one/two/three
observed classes, tall/wide/single-feature arrays, weighted/unweighted fitting,
multiple thresholds, both Complement normalizations, missing declared classes,
zero/fractional weights, empirical/uniform/fixed priors and nonunit/zero prior
entries. There are 1,796 independent weighted count-conservation certificates
and 324 final streaming/batch equivalence cases. All query and training labels
and scores are checked at every snapshot.

The suite retains 909 field-level NaN certificates for zero-mass empirical
priors, one-feature normalized Complement weights and saturated Bernoulli
likelihoods; no fixture is discarded.
Complement label ties use the existing classifier-chain rounding certificate,
with training accuracy checked against the certified training predictions.
Finite score/probability fields still compare directly with installed sklearn
1.6.1. Twenty-five rejection cases cover first/changed class contracts, labels,
feature widths, finite/nonnegative data and weights, priors, thresholds, alpha,
query contracts and weighted-product overflow.

Inventory: 217 public core types, 129 core modules, 146 native test programs,
123 core reference suites and 582,083 cumulative retained groups. The separate
42-group legacy RNG reference suite remains excluded. Thirty-three invalid-input
registries retain 3,442 cases. These are cumulative counts, not a full rerun
claim; final verification below identifies the suites rerun for this increment.

Run `python validation/compare_discrete_stream.py --profile checked`, or use
`--profile release` / `--executable`. Native state/weight interfaces remain MIT,
with existing sklearn BSD notices retained. No estimator type, public extension
algorithm or external Fortran dependency is added. The nonnegative NB smoothing
section below extends alpha controls; sparse inputs and signed weights remain
outside this native interface.


## CategoricalNB zero smoothing and prior boundaries

CategoricalNB now implements scalar alpha=0 and `force_alpha` (default true),
including the source 1e-10 floor when false. Fixed class priors are retained
without renormalization and override fit_prior; all-zero fixed priors and
zero total sample weight are accepted. Undefined feature rows, empirical
priors and posterior normalization produce explicit NaNs safely under floating-
point traps. Predictions follow source first-NaN/first-maximum behavior.

Streaming `n_categories_` now matches sklearn 1.6.1 latest-batch metadata,
while ragged tables retain their full accumulated widths and remain queryable.
Previously rejected zero alpha, zero weight mass and nonunit priors moved to
success-path checks. The invalid registry instead checks smoothing/weight
range and nonfinite priors, and adds first/later empty-class and cumulative
weight-range cases. Finite nonnegative sample weights, compact integer-valued
categories and scalar alpha remain the native scope; source Python warnings
are not emitted. Sparse arrays and vector alpha are outside this interface.

The new `compare_categorical_smoothing.py` suite compares all fitted tables,
latest-batch metadata, priors, query/training labels, joint likelihoods,
probabilities and scores after every update. Three seeds cover one/two/three
classes, one/three/seven features, six smoothing/floor configurations, batch
and streaming fits, category growth followed by shrinking batch ranges,
scalar/per-feature minima, missing declared classes, zero/fractional weights,
uniform/empirical/nonunit/zero priors. Weighted per-feature class-mass identities
and final streaming/batch equivalence supplement installed sklearn comparisons.
No fixture is removed when source results are undefined. See
`example/unsmoothed_categories.f90`.

Final verification passed in checked and release builds. All 146 native test
programs passed; checked flags were `-fcheck=all -fbacktrace
-ffpe-trap=invalid,zero,overflow`. Each profile passed 76,282 reference groups:
53,460 new categorical-smoothing groups, 480 original CategoricalNB groups,
22,150 discrete-stream groups and 192 integration groups. The new suite retained
1,296 fixtures and 2,916 snapshots, including 2,916 count certificates, 5,563
fields containing source NaNs and 648 streaming/batch equivalence cases. No
label tie certificates were needed. All 25 discrete-stream rejection cases
and all 33 categorical rejection cases passed in both canonical profiles.
The example also passed explicitly through fpm in both profiles.

The full 2,450-case general invalid-input registry passed in each profile.
Its standard checked entry point uses `-fcheck=all -fbacktrace`, without
floating-point traps, linked against the checked library. An initial attempt
with traps stopped at the existing spectral-biclustering
`bcsp_co_margin_overflow` case before its expected diagnostic; the complete
registry was then rerun in its standard configuration. All categorical cases
were additionally rerun against the canonical executable with traps enabled.
No legacy case was filtered out. This increment does not change the spectral
biclustering overflow diagnostic.

The final audit matched cross-profile inventories, executable/validator/source
hashes and all 285 compiled source-content digests per profile. Reports, terminal
logs, the standard checked registry executable and orchestration/audit scripts
are retained in ignored `validation/results`, including
`categorical_smoothing_final_audit.json`.

Cumulative inventory: 217 public core types, 129 core modules, 146 native test
programs, 124 core reference suites and 635,543 retained groups. Thirty-three
invalid-input registries retain 3,445 cases. Only the affected reference suites
listed above were rerun; these cumulative counts do not imply a full rerun.
The separate 42-group legacy RNG reference suite remains excluded and was not
rerun. MIT licensing and sklearn BSD attribution remain; no external package,
estimator type or public algorithm extension is added.


## NearestCentroid class priors and discriminants

NearestCentroid now supports empirical and explicit class priors, their source
normalization tolerance, zero-prior classes and all-zero prior normalization.
It exposes n_features_in_ and class_prior_, Euclidean decision_function and
log/probability methods, and prior-aware Manhattan prediction. Uniform-prior
prediction retains raw-distance routing, which may differ from standardized
probability argmax. Optional priors override uniform/empirical mode. Existing
mean/median and shrunken-centroid fitting remains available.

Zero pooled scales and singleton classes now retain source NaN/Inf diagnostics;
queries reject undefined standardized scales or nonfinite shrunken centroids.
Uniform singleton prediction remains defined. All-global-constant inputs are
rejected, matching sklearn. Explicit safe divisions and range checks avoid
floating-point traps. Source warning objects, sparse inputs and weighted score
remain outside this native interface. See docs/API.md and
`example/prior_weighted_centroids.f90`.

`compare_centroid_priors.py` uses installed sklearn 1.6.1 across seeded binary/
multiclass fits, tall/wide arrays, anisotropic/constant/classwise-constant data,
singleton classes, both metrics, shrinkage and seven prior configurations.
It compares all fitted statistics, query/training labels, accuracy, decisions,
probabilities and their identities. Source NaN/Inf fields are retained, source
method errors are certified against native errors, and roundoff label ties
require source-score bounds. No fixture is discarded for undefined behavior.

Final verification: all 147 native test programs passed with checked flags
`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow` and in release mode.
Each profile passed 22,398 affected reference groups: 22,086 new NearestCentroid
groups, all 120 original graph/neighbor groups and 192 integration groups.
All 30 new rejection cases passed in each profile. The new suite retains 2,520
fixtures, 3,798 nonfinite diagnostic fields, 1,458 source/native method-error
certificates and 756 probability identities. Fourteen bounded label-rounding
certificates were needed per profile. Their scale includes point/centroid
magnitudes to account for cancellation in source Euclidean norm identities;
direct numerical-output tolerances remain unchanged. No fixture was dropped.

The final audit independently reconstructed the required fitted-statistic,
available-method and expected-error inventory for every fixture. It matched
cross-profile inventories, executable/validator/source hashes and all 284
compiled source-content digests per profile. The example passed explicitly
through fpm in both profiles and its output also matched installed sklearn.
Reports, terminal logs and orchestration/audit scripts are archived in ignored
`validation/results`, including `centroid_priors_final_audit.json`.

Cumulative inventory: 217 public core types, 129 core modules, 147 native test
programs, 125 core reference suites and 657,629 retained groups. Thirty-four
invalid-input registries retain 3,475 cases. Only the affected reference suites
listed here were rerun; the cumulative inventory is not a full-rerun claim.
The separate 42-group legacy RNG reference suite remains excluded and was not
rerun. All non-centroid neighbor procedure bodies were verified unchanged.
MIT licensing and sklearn BSD attribution remain; no estimator type, external
Fortran dependency or public extension algorithm is added.


## Nonnegative discrete NB smoothing

`multinomial_nb`, `bernoulli_nb` and `complement_nb` now accept finite scalar
`alpha>=0`. Their `%force_alpha` defaults to true, retaining zero/tiny values;
false floors effective smoothing to 1e-10 without changing input controls.
Multinomial/Complement additionally accept allocated `%alpha_per_feature(:)`,
which overrides scalar alpha and must have n_features finite nonnegative
entries. Flooring applies per feature. Bernoulli retains scalar smoothing;
feature-vector controls are explicitly rejected. Installed sklearn 1.6.1's
Bernoulli vector denominator has class/feature broadcasting artifacts; a new
mathematical variant is not substituted for that source implementation.

Zero priors are accepted even when every fixed prior is zero, matching source;
fixed priors remain unnormalized. `partial_fit` recomputes smoothed probabilities
from raw accumulated counts with the current alpha controls. Changing smoothing
then making a zero-weight update therefore recomputes probabilities without
changing counts. `fit` resets fitted state, preserving input controls.

Dense likelihood arithmetic preserves source NaN/Inf behavior. In particular,
zero times an infinite feature weight is NaN; query rows and classes are handled
individually. Complement normalization retains Inf/Inf and 0/0 NaNs. Positive-
infinite joint scores normalize to NaN at infinite entries and negative infinity
elsewhere, matching source logsumexp behavior. Undefined empirical priors,
zero-total smoothed rows and unit Bernoulli probabilities remain explicit NaNs,
without invalid/zero floating-point traps. Source first-NaN argmax is retained.
Complement counts now follow source addition-before-subtraction order, including
positive smoothing lost at large totals. Python warning objects are not emitted.

See `example/feature_smoothed_nb.f90`. Direct CSR support is described below.
Vector Bernoulli alpha, unthresholded Bernoulli `binarize=None`, signed weights
and weighted score remain outside this interface. Count/smoothing/query intermediates must fit the
native real64 working range; existing explicit count range checks are retained.

`validation/compare_nb_smoothing.py` retains seeded batch/streaming comparisons
for scalar zero/tiny alpha, feature vectors containing zeros, force_alpha on/off,
empirical/uniform/all-zero fixed priors, missing declared classes, zero/fractional
weights and both Complement normalizations. Tall, wide and single-feature data
cover one/two/three observed classes. Large-count cases retain source lost
positive smoothing. All fitted state, query/training labels, scores and
probabilities are checked at each snapshot; count-conservation and final
streaming/single-update identities supplement source comparisons. No NaN/Inf
fixture is dropped. Eleven new rejection cases cover vector controls and
smoothing range checks. The two previously rejected zero-alpha/zero-prior cases
in the streaming registry moved to supported checks, with negative-alpha and
empty-prior rejection cases replacing them.

Final verification: all 148 native test programs passed with checked flags
`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow` and in release mode.
Each profile passed 134,675 affected reference groups: 59,642 new smoothing
groups, 22,150 discrete-stream groups, 16,536 weighted-classifier groups,
27,490 multiclass groups, 8,665 classifier-chain groups and 192 integration
groups. All 11 new and 25 retained rejection cases passed per profile.

The new suite retains 2,378 fixtures and 4,754 fitted snapshots, with 4,754
independent count certificates, 11,359 fields containing source NaN/Inf,
1,188 streaming/single-update equivalence cases and 69 bounded label-rounding
certificates per profile. The final audit independently reconstructed every
snapshot's required fitted-state and prediction fields; none was filtered out.
It matched cross-profile inventories, executable/validator/source hashes and
all 288 compiled source-content digests per profile. The example passed
explicitly through fpm in both profiles and its output matched installed sklearn.
Gaussian NB procedure bodies were independently verified unchanged.

Reports, terminal logs and orchestration/audit scripts are archived in ignored
`validation/results`, including `nb_smoothing_final_audit.json`.
Cumulative inventory: 217 public core types, 129 core modules, 148 native test
programs, 126 core reference suites and 717,271 retained groups. Thirty-five
invalid-input registries retain 3,486 cases. Only the affected reference suites
listed above were rerun; the cumulative inventory is not a full-rerun claim.
The separate 42-group legacy RNG reference suite remains excluded and was not
rerun. MIT licensing and sklearn BSD attribution remain; no estimator type,
external Fortran dependency or public algorithm extension is added.


## CSR discrete Naive Bayes

`multinomial_nb`, `bernoulli_nb` and `complement_nb` overload `fit`, `partial_fit`,
`joint_log_likelihood`, `predict_log_proba`, `predict_proba`, `predict` and `score`
for `type(csr_matrix)`, alongside their existing dense interfaces. Import CSR
storage from `sklearn_sparse` or the `sklearn` facade. Its pointer and column
arrays are 1-based. Optional sample weights, first-call class declarations,
fixed/uniform/empirical priors, smoothing controls and fitted diagnostics use
the same contracts as dense input. Dense and CSR updates can be interleaved in
one fitted model. A new `fit` resets state regardless of representation.

Training traverses stored entries and keeps dense (n_classes,n_features) fitted
statistics. Sparse prediction uses CSR multiplication and returns dense
(n_samples,n_classes) scores/probabilities. Sample matrices are never converted
to dense in the estimator. Inputs are validated and preserved. Empty rows and
zero-stored-entry matrices with positive dimensions are supported. CSR indices
may be unsorted and duplicate columns are retained. Matrix structure and finite
data are checked; Multinomial/Complement stored values must be nonnegative.

Sparse source likelihoods differ from dense likelihoods when weights include
NaN/Inf: omitted entries contribute nothing, while stored zeros participate in
IEEE multiplication. Bernoulli thresholds stored entries independently and
eliminates resulting zeros before counting or prediction; duplicate entries are
not first summed. Signed finite Bernoulli entries below the nonnegative threshold
are therefore removed. These rules match installed sklearn 1.6.1, including
representation-dependent zero-smoothing predictions. The shared source
normalization and first-NaN argmax rules remain unchanged.

Use `example/sparse_document_nb.f90` for a 10,000-feature weighted document fit.
Gaussian/Categorical NB, pipeline/meta-estimator sparse inputs, CSC/COO storage,
vector Bernoulli smoothing, unthresholded Bernoulli `binarize=None`, signed
sample weights and weighted score remain outside this interface. Fitted
statistics still require O(n_classes*n_features) storage, even for sparse data.

`validation/compare_nb_sparse.py` checks canonical and noncanonical CSR input,
explicit/implicit zeros, duplicates, empty rows and all-zero sample matrices.
Seeded one/two/three-class fits span single-feature, tall and wide inputs;
positive/zero/vector/floored smoothing, weights, missing declared classes,
fixed-zero priors, both Complement normalizations and dense/CSR mixed updates
are covered. Signed Bernoulli duplicate cases verify threshold-before-sum
behavior. Every snapshot checks fitted state, sparse and dense query outputs,
training labels and accuracy. Independent event-count conservation and pure-CSR
streaming/single-update identities supplement source comparisons. Thirty-one
new rejection cases cover CSR structure, data and estimator contracts. A native
10,000-by-10,000 training case contains two stored values; production NB was
also inspected to verify it contains no CSR-to-dense conversion.

Final verification: all 149 native test programs passed with checked flags
`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow` and in release mode.
Each profile passed 234,499 affected reference groups: 99,824 new CSR NB groups,
59,642 smoothing groups, 22,150 discrete-stream groups, 16,536 weighted-classifier
groups, 27,490 multiclass groups, 8,665 classifier-chain groups and 192 integration
groups. All 31 new CSR, 11 smoothing and 25 streaming rejection cases passed per
profile. Release primary NB checks used separate stdin/report paths and ran
alongside the checked meta-estimator checks; shared-file release meta-estimator
checks started only after the checked run completed.

The CSR suite retains 2,594 fixtures and 6,050 snapshots, with 6,050 independent
event-count certificates, 35,696 fields containing source NaN/Inf, 864 pure-CSR
streaming/single-update equivalence cases and 82 bounded label-rounding
certificates per profile. Both sparse and dense query representations are
checked after every fit/update. No nonfinite or noncanonical fixture was dropped.
The final audit independently reconstructed every snapshot's required fields,
matched cross-profile inventories and executable/validator/source hashes, and
verified all 290 compiled source-content digests per profile. It also confirmed
that production NB contains no CSR-to-dense conversion. Existing Gaussian,
count-update and dense likelihood procedure bodies were verified unchanged;
the dense log-probability normalizer was shared without changing its arithmetic.

The example passed explicitly through fpm in both profiles and its output
matched installed sklearn. Reports, terminal logs and orchestration/audit scripts
are archived in ignored `validation/results`, including
`nb_sparse_final_audit.json`. Release primary and legacy phase logs are retained
separately. Native large-shape coverage uses a 10,000-by-10,000 sample matrix
with two stored values and dense class-by-feature fitted statistics.

Cumulative inventory: 217 public core types, 129 core modules, 149 native test
programs, 127 core reference suites and 817,095 retained groups. Thirty-six
invalid-input registries retain 3,517 cases. Only the affected reference suites
listed above were rerun; cumulative counts do not imply a full rerun. The separate
42-group legacy RNG reference suite remains excluded and was not rerun.
MIT licensing and sklearn BSD attribution remain; no estimator type, external
Fortran dependency or public algorithm extension is added.

## Coordinate-descent NMF (2026-11-22)

`nmf` adds dense Frobenius-loss coordinate descent with `solver='cd'`:
alternating projected coordinate sweeps, zero-Hessian handling, dimension-scaled
L1/L2 penalties, scalar-seeded coordinate shuffling and projected-gradient
stopping. Initialization supports auto, random, custom and the three NNDSVD
variants through the existing MiniBatchNMF initializer. Fixed-H transform starts
W at zero and preserves fitted state; reconstruction, fitted dimensions,
iteration count and native stopping diagnostics are exposed.

Existing native defaults and multiplicative-update arithmetic are retained;
users select CD explicitly. Dense real64 inputs and conservative numerical
working-range bounds apply. Sparse CD, general beta losses, the free
factorization function, verbose output and Python warning behavior are outside
this addition. Later increments below provide general beta losses and the free
factorization function. No new estimator type or public extension is counted. See
[the API contract](API.md#coordinate-descent-nmf) and
[the example](../example/coordinate_descent_nmf.f90).

The new suite retains 2,022 fixtures and 20,214 reference groups, with 34 isolated
native rejection cases. The cumulative inventory is 217 public core types,
129 core modules, 150 native test programs and 128 core reference suites.
Thirty-seven invalid-input registries retain 3,551 cases. The separate 42-group
legacy RNG suite remains excluded from core reference counts.

Both checked and release builds passed all 150 native tests, 26,906 affected
reference groups (including 6,500 MiniBatchNMF and 192 integration regressions)
and all 34 new rejections per profile. The example matched sklearn in both
profiles. The final audit verified complete field inventories and 287 relevant
compiled-content digests per profile; see [the validation record](VALIDATION.md).

## Beta-divergence NMF multiplicative updates (2026-11-22)

The existing `nmf` MU path now follows sklearn 1.6.1's dense multiplicative/MM
algorithm for Frobenius, generalized KL, Itakura-Saito and general numeric beta
losses. It shares random/custom/NNDSVD initialization, automatic component counts
and dimension-scaled L1/L2 penalties with CD. Source MM exponents, denominator
floors, factor stability cutoffs, fixed-H initialization and ten-iteration
relative-error stopping are implemented. The sklearn 1.6.1 cumulative cached
KL/L1 transform denominator is retained and documented.

This replaces the earlier basic MU update and runtime-RNG initialization;
native default parameter values remain unchanged. Dense real64 range diagnostics
guard powered arithmetic, products, reductions and ratios. Sparse MU, named
string beta selectors and Python warning
behavior remain outside this interface. CD's numerical kernel and the existing
MiniBatchNMF implementation are retained. No new public estimator type or
extension algorithm is added. See [the API contract](API.md#beta-divergence-nmf-multiplicative-updates)
and [the KL example](../example/beta_divergence_nmf.f90).

The new suite retains 2,090 fixtures and 20,844 result groups, including 56
source-matching all-zero-H transform rejections, plus 44 isolated diagnostics.
The cumulative inventory is 217 core public types, 130 core modules, 151 native
test programs, 129 core reference suites and 858,153 retained groups.
Thirty-eight invalid-input registries retain 3,595 cases. The separate 42-group
legacy RNG suite remains excluded from these reference counts.

Final corrected checked and release builds each passed all 151 native tests,
47,750 affected reference groups and 78 rejection checks. Both KL example runs
matched sklearn. The audit verified 290 compiled-content digests per profile and
all cross-profile field inventories, including the 24 strict subnormal boundary
fixtures. See [the validation record](VALIDATION.md).

## Positive linear-model coefficients (2026-11-22)

`linear_regression`, `ridge`, `elastic_net`, `lasso`, `elastic_net_cv` and
`lasso_cv` accept `positive=.true.`. Coefficients are nonnegative while the
intercept remains unrestricted. Weighted OLS/Ridge adapters center and rescale
before constrained fitting; supported pipeline, transformed-target, bagging,
multioutput and regressor-chain clones preserve the flag. Other weight limits
remain unchanged. The default false path retains its existing numerical methods.

Positive OLS uses native Lawson-Hanson NNLS with QR insertion/removal. Positive
Ridge uses an augmented NNLS system for the same constrained quadratic;
augmentation requires O(p**2) storage. Positive OLS avoids the unconstrained
SVD's square V storage and has a 10,000-feature native regression fixture.
ElasticNet/Lasso project coordinate updates at zero, including all CV folds and
the final refit. Native coefficient-change stopping remains in use.

The NNLS QR tail is corrected relative to SciPy 1.15.3: its final below-pivot row
is cleared before later column removal. An explicit wide-case KKT regression
checks the correction against an independent BVLS optimum. Exact Ridge solutions
can differ from public sklearn finite L-BFGS stopping, and nonunique NNLS can
select different coefficient representatives/extrapolations. These cases retain
explicit optimality and strong-convexity certificates rather than asserting
unqualified coefficient parity. See [the API contract](API.md#positive-linear-model-coefficients).

Unsupported inherited positive flags (Bayesian/ARD, OMP, Huber, RANSAC, TheilSen,
PA, GLMs and RidgeCV) receive diagnostics. Sparse input, a public optimizer API
and new direct-fit weighting signatures are outside this addition. MIT original
contributions and retained sklearn/SciPy/LAPACK BSD attribution apply. No new
public estimator type, extension algorithm or runtime dependency is introduced.

New reference inventory: 1,044 core fit/CV fixtures (8,388 groups), 192 positive
composition fixtures (1,608 groups) and 30 isolated diagnostics. Cumulative
inventory: 217 core public types, 131 core modules, 153 native test programs,
131 core reference suites and 868,149 groups. Thirty-nine invalid-input
registries retain 3,625 cases. The separate 42-group legacy RNG suite remains
excluded from core reference counts.

Final checked and release verification each passed all 153 native tests,
55,764 affected reference groups and all 30 new rejections. Both examples matched
sklearn. The final audit verifies 296 compiled source-content digests per profile,
all field inventories and the retained solver-difference certificates; see
[the validation record](VALIDATION.md).

## Free NMF function

`non_negative_factorization` exposes sklearn's dense free-function API alongside
the `nmf` type. It returns W, H and the iteration count, supports CD/MU,
automatic rank inference, custom/random/NNDSVD initialization, regularization
and fixed-H solves. Fixed solves ignore W initializers and preserve H.
The function follows sklearn's solver/iteration/tolerance defaults, with a
native deterministic default seed; the existing type defaults are unchanged.
There is no new estimator type, public extension algorithm or dependency.
Inputs remain dense real64; see the [API contract](API.md#free-nonnegative-matrix-factorization).

The new suite retains 916 fixtures and 3,664 groups plus 12 rejection cases.
Its checks cover both factors, iteration counts, an independent regularized
objective, nonnegativity and exact preservation of fixed H. At machine
stationarity, CD iteration differences require projected-gradient roundoff
certificates for both source and native factors. Cumulative inventory is now
217 public types, 131 core modules, 154 native test programs, 132 core reference
suites and 871,813 retained groups. Forty invalid-input registries retain 3,637
cases. The separate legacy RNG suite remains excluded.

Both final profiles pass all 154 native tests, 44,914 affected reference groups
and 90 diagnostic cases (12 new plus 78 prior CD/MU). The examples match sklearn.
The audit verifies 294 compiled-content digests per profile and unchanged
existing decomposition arithmetic; see the [validation record](VALIDATION.md).

## HashingVectorizer

Added `hashing_vectorizer`, the sklearn 1.6.1 stateless text hashing estimator.
It provides word/character/within-word-character n-grams, English/custom stop
words, lowercasing, signed or unsigned hashes, binary mode and optional L1/L2
row normalization. It uses existing MurmurHash3, FeatureHasher and CSR storage;
there is no vocabulary, extra runtime dependency or public extension algorithm.
Its native `analyze` method exposes the ordered analyzer tokens for inspection.
The default million-feature result is sparse. CSR output feeds existing sparse
Naive Bayes models directly.

Analysis currently supports ASCII bytes only. Non-ASCII text/stop words receive
a diagnostic. File/encoding callbacks, arbitrary token regexes/callables,
Unicode/accent processing and general text pipeline dispatch remain outside
this increment. `token_min_length` encodes an ASCII word regex. Character-array
inputs trim trailing padding; dynamic `feature_string` inputs preserve exact
content. Existing CountVectorizer/TF-IDF behavior is unchanged. See the
[API contract](API.md#hashingvectorizer-text-features) and
[CSR example](../example/hashed_text_documents.f90).

The new suite retains 1,779 fixtures / 14,232 groups and 13 isolated diagnostics.
It checks every analyzer token before hashing, exact CSR shapes/pointers/indices,
numerical values, independent signed-hash reconstruction and normalization,
plus stateless/dense adapter invariants. Cases include stored-zero cancellation,
binary conversion after aggregation, empty rows, all ASCII controls, long tokens,
all 318 English stop words and maximum-integer n-gram lengths. Cumulative core
inventory is now 218 public types, 132 source modules, 155 native tests,
133 reference suites / 886,045 groups and 41 invalid-input registries / 3,650
cases. The separate legacy RNG suite remains excluded.

Final checked and release profiles each pass all 155 native tests, 17,369
affected reference groups and 53 diagnostics (13 new and 40 prior). Both
examples match sklearn, including sparse NB predictions. The audit verifies
296 compiled-content digests per profile, complete analyzer/group inventories,
pinned source/validator/executable hashes and unchanged existing text/hash/CSR
arithmetic. Only the affected suites were rerun; the separate legacy RNG suite
remains excluded. See the [validation record](VALIDATION.md).

## Stateless numeric preprocessing

Added `normalizer` and `binarizer` with dense/CSR fit, transform, fit-transform
and one-to-one feature metadata. Both work through the existing dense transformer
protocol inside Pipeline, FeatureUnion and ColumnTransformer. The existing free
functions now accept CSR; normalization adds both axes and optional returned
norms, preserving sklearn's unsupported sparse L1/L2 return-norm case. Dense
near-zero norms use sklearn's `10*epsilon` guard, correcting the old exact-zero
check. CSR storage behavior is retained: L1/L2 use individual stored entries;
max coalesces duplicates; binarization removes zeros but preserves qualifying
duplicates. Column output is CSR corresponding to source CSC converted to CSR.

The shared base dispatcher forwards existing sample weights for MultinomialNB,
BernoulliNB and ComplementNB and recognizes their weighting capability. This
corrects stale limits in pipeline/voting/bagging/stacking dispatch. Separate
multioutput/chain adapters retain their existing limits. New tests directly
exercise weighted/unweighted pipelines, bagging and voting with all three NB
classes. There is no new dependency or public extension algorithm. Original
adapters remain MIT with retained sklearn/SciPy BSD attribution.

The new core suite has 1,801 fixtures / 19,386 groups and 29 diagnostic cases.
The composition suite has 864 fixtures / 5,328 groups, spanning normalization,
thresholding, weighted Ridge/NB pipelines, unions, reordered columns, transformed
remainders, bagging and voting. Boundary cases include near-zero thresholds,
subnormal and large values, both sparse axes, unsorted duplicates, exact
cancellation and duplicate overflow. Nonfinite source results are checked by
NaN/infinity masks rather than discarded. Source metadata rules require fitting
before feature-name output even though numeric transform works before fitting.
See the [API contract](API.md#normalizer-binarizer-and-sparse-preprocessing-functions)
and [example](../example/stateless_feature_preprocessing.f90).

Cumulative inventory: 220 public core types, 133 source modules, 157 native tests,
135 core reference suites / 910,759 groups and 42 invalid-input registries /
3,679 retained cases. The separate legacy RNG suite remains excluded.

Both final profiles pass all 157 native tests, 57,356 affected reference groups
and 130 diagnostics. The audit verifies 304 compiled-content digests per profile,
full field inventories, IEEE outcome masks and pinned hashes; both examples match
source. Existing regression certificates remain retained, with no new numerical
exceptions. Only affected suites were rerun; see the [validation record](VALIDATION.md).

## PolynomialFeatures estimator and sparse expansion

Added the `polynomial_features` estimator alongside the retained array-call
syntax. It supports degree ranges, interactions, dense/CSR expansion, fitted
widths and names, lazy int64 powers/counts, and dense Pipeline/FeatureUnion/
ColumnTransformer composition. The former degree-12 array-call limit is removed.
Exact reduced binomials and a Vandermonde difference avoid count overflow from
unnecessarily large intermediate totals. Fit-only counts allocate no expansion.
Dense suffix reuse preserves source multiplication association; sparse paths
preserve source explicit-zero and IEEE multiplication behavior. CSR indices
retain the package's default-integer limit. See the
[API contract](API.md#polynomialfeatures-estimator-and-retained-array-api).

The native implementation explicitly corrects two source defects: duplicate CSR
columns are coalesced before expansion, and zero-column interaction results have
empty metadata/output arrays. Reports retain 984 raw noncanonical source
outcomes, 48 empty-metadata exceptions and 36 empty sparse-transform exceptions.
Three huge single-feature exponent cases use symbolic metadata identities;
one extreme interaction fit-only case uses sklearn's exact requested-range
count helper to avoid its expensive unused full-degree auxiliary count.
These are identified source corrections/oracle limits, not extension algorithms.
Original adapters remain MIT with retained sklearn/SciPy BSD attribution.

New reference coverage comprises 1,985 fixtures / 38,020 groups, 31 diagnostics,
243 nonfinite reference fields, and 480 composition fixtures / 2,880 groups.
Cumulative inventory: 221 public core types, 134 source modules, 159 native tests,
137 core reference suites / 951,659 groups and 43 invalid-input registries /
3,710 cases. Types remain API subsets; these counts do not imply complete sklearn
coverage. The separate legacy RNG suite remains excluded.

The final corrected checked and release profiles each pass all 159 native tests,
77,722 affected reference groups and 161 diagnostic cases. Full reports include
128 fresh-process name/refit repetitions per profile; an additional focused
optimized final-source check passed 1,000 repetitions. The audit verifies 307
compiled-content digests per profile, complete inventories, hashes and unchanged
surrounding preprocessing/dispatch code. Both explicit examples match source.
Only affected suites were rerun; see the [validation record](VALIDATION.md).

## Image/grid graphs and quiet subtraction boundaries

Added scikit-learn's `grid_to_graph` and `img_to_graph` helpers with owned real64
CSR output, C-order voxel numbering and selected-voxel compaction. They preserve
bidirectional axis connectivity, explicit zero edges and all diagonal entries.
Image graphs use absolute adjacent differences and image-valued diagonals;
grid graphs use ones. Scalars/vectors/matrices/volumes, empty selections and
zero-volume images are supported. Masks and dimensions have explicit checks.
Construction is linear in grid volume/stored entries, without dense adjacency.
The public contract documents CSR index limits and the absence of dtype/output-
class selectors. No new estimator type, extension algorithm or runtime dependency
is introduced. See the [API contract](API.md#image-and-grid-graph-builders).

The source image mask path squeezes the image but not its mask, so otherwise
same-shaped singleton-axis masks can fail. Native masks preserve their declared
shape. The suite retains 560 raw source exceptions and uses the public function
with a mask matched to its internal squeezed image for the intended graph.
Across 1,832 fixtures / 14,653 groups, it checks CSR structure/data/signs, dense
views, stored-zero masks, sparse multiplication, input preservation and 13
diagnostics. Three large graph cases avoid dense materialization. Reports retain
352 nonfinite reference fields and enforce exact NaN/signed-infinity masks.

Near-maximum image gradients exposed a premature-overflow bug in the shared
quiet-subtraction guard. The corrected extreme opposite-sign path uses
power-of-two scaling, preserving finite rounded results and true overflow without
changing halting modes. Independent NumPy subtraction checks cover 8,703 scalar
pairs / 17,406 groups, including 258 cases the old guard got wrong. Finite value
bits and non-NaN signs match exactly; nonfinite classifications match exactly.
Other quiet primitive/reduction procedure bodies remain unchanged.

Cumulative inventory: 221 public core types, 135 source modules, 161 native tests,
139 core reference suites / 983,718 groups and 44 invalid-input registries /
3,723 cases. Types remain API subsets and helpers do not add estimator types.
The separate legacy RNG suite remains excluded.

Final checked and release profiles each pass 161 native tests, 276,126 affected
reference groups and 308 diagnostics. The shared-helper regression set includes
image patches, structured features, histogram boosting, sparse/smoothed Naive
Bayes, positive regressors, preprocessing, weighted classifiers/trees and
pipeline/column/union integration. Both examples match source. The final audit
verifies 321 compiled-content digests per profile, complete new field inventories,
exact subtraction bits and nonfinite masks, pinned hashes and unchanged
surrounding procedure bodies. Only affected suites were rerun; the separate
legacy RNG suite remains excluded. See the [validation record](VALIDATION.md).


## Seeded AdaBoost CART and shared classifier split fidelity

AdaBoostClassifier now supports configurable native CART weak learners rather
than fixed stumps: depths, split/leaf sizes, feature subsets, impurity cutoffs,
Gini/entropy/log-loss criteria and best/random splitters. Fitted trees and the
unfitted prototype are exposed. New predict_log_proba, weighted score and
staged_score methods complement the existing staged predictions; staged voting
now accumulates each learner once. Integer seeds reproduce sklearn's per-tree
seed sequence and reset on refit. Pipeline and the shared voting/bagging/
stacking dispatcher accept this classifier, including nested seed assignment.

The shared classifier tree builder now follows the source's introsort,
in-place partitions, incremental/reverse class-count updates and final child
impurity recomputation. This addresses concrete discrepancies found when the
same weighted data were supplied directly to both tree implementations.
Regression splits retain their existing path. Existing non-AdaBoost dispatch
branches retain their implementation.

The new reference suite uses 1,460 fixtures and retains 62,417 result groups.
It checks complete fitted tree topology, sample counts, masses, impurities,
leaf probabilities, individual predictions/importances/seeds/depths, ensemble
outputs, all staged APIs, weighted scores, refits and input preservation.
It spans binary, multiclass and single-class targets; three seeds; eight tree
configurations; round counts and learning rates; zero/tiny/overflowing initial
weights; and extreme learner weights. NaN and infinity masks are compared
explicitly. Fourteen invalid-input cases accompany the suite. Source hashes,
compiled tree binary hashes and the runtime feature threshold are recorded.

The report retains 1,152 source single-class generator division errors,
two initial-weight normalization corrections and 20 fields with nonfinite
source results. Native single-class stages return the corresponding constant
outputs. Finite initial weights whose sum overflows are rescaled before
normalization; that reference uses an equivalent rescaled public sklearn fit.
Other nonfinite learner-weight and stopping outcomes follow the source.
These corrections are identified behavior differences within the existing
estimator counterpart, rather than additional public algorithms. Original
implementation remains MIT with sklearn/NumPy BSD attribution; no dependency
or public extension is added.

Cumulative inventory: 221 public core types, 135 source modules, 163 native
programs, 140 core reference suites / 1,046,165 retained groups, and 45
invalid-input registries / 3,736 cases. The cumulative group total includes
22 additional weighted-tree and eight bagging groups retained by the improved
split matching. The former single-class AdaBoost rejection was removed because
that fit is supported by sklearn. Public types remain API subsets.

Final validation passes all 163 native programs in both checked and release
profiles. The checked affected reference run retains 181,543 groups across
AdaBoost, CART, weighted classifiers, bagging, stacking, voting, Pipeline,
gradient boosting, multiclass, multioutput and general integration suites.
The targeted release run retains 72,587 groups across the new AdaBoost CART,
original AdaBoost and weighted-tree suites. Each profile also passes 2,463
main/new diagnostic cases. Numerical checked tests retain bounds checks,
backtraces and traps for invalid, zero and overflow. The legacy main diagnostic
registry uses its standard bounds checks without floating-point traps, so
intentional overflow inputs can reach their expected error messages.

The final audit checks 316 compiled-file digests per profile, reference-field
inventories, source/validator/executable hashes, installed sklearn source and
binary hashes, both example outputs, and unchanged non-AdaBoost dispatch
branches. Full reports and logs are retained under ignored validation/results.
Only the affected reference suites were rerun; the cumulative total also
includes retained results from earlier batches.


## Isomap radius neighborhoods and distance graphs

`isomap` now supports radius neighborhoods and dense/CSR precomputed distances
in addition to dense feature-based fixed-count graphs. Set n_neighbors=0 for
Python's n_neighbors=None and supply a finite radius>=0. The default radius=-1
represents Python None; combining a positive neighbor count and a nonnegative
radius is rejected. p must be finite and >=1. Oversized positive component
counts are clipped by KernelPCA; one-sample radius fits are supported.

Training distance matrices are square, while query distance matrices have one
column per fitted sample. CSR missing entries are unavailable neighbors, not
zero distances. Explicit zero edges and unsorted/duplicate stored columns are
preserved during neighbor selection; repeated selected self entries in a
fixed-count training query raise an error as the source also cannot reshape
that neighborhood. Radius selection includes its boundary and removes training
self edges. Query points without radius neighbors raise a clear diagnostic.
Native stable neighbor ordering can differ from sklearn's backend-specific
ordering when the k-neighbor boundary is tied.

Disconnected feature/dense-distance graphs are connected at the closest pair
for every pair of original components, using the source's component traversal
and first-minimum order. Disconnected precomputed CSR graphs are rejected,
because full cross-component distances are unavailable. The existing native
n_connected_components_ diagnostic records the count before completion.

path_method accepts 'FW', 'D' and 'auto'. 'FW' uses Floyd-Warshall; 'D' runs
binary-heap Dijkstra over an undirected adjacency structure, taking the smaller
weight of reciprocal or duplicate edges and retaining zero edges. The native
auto policy selects FW below 100 samples or when the undirected dense working
graph is at least one-quarter populated, otherwise Dijkstra. path_method_ is a
native diagnostic for the selected backend, rather than an additional
algorithm or a source attribute. Shortest-path results agree numerically across
these choices. Sparse inputs still lead to quadratic distance/kernel storage;
dense graph assembly and the existing native Jacobi eigensolver limit scale.
Sparse feature matrices, tree-based neighbor acceleration, ARPACK/randomized
eigensolvers and parallel execution remain outside this type's current scope.

Fitted metadata include n_features_in_, n_samples_fit_ and n_features_out_.
get_feature_names_out returns feature_string names isomap0, isomap1, etc.; an
optional character input-name array is checked against fitted input width.
Transforms use the fitted search metric, p and neighborhood settings; refit
updates them and supports switching between feature and CSR distance inputs.
Fitted embedding_, dist_matrix_ and kernel_pca_ remain available. Reconstruction
error retains the prior native nonnegative residual clamp: roundoff-negative
source residuals can yield NaN in Python and zero here; the validator tracks
that case and bounds squared-error differences by residual roundoff.
This is an identified numerical difference within the existing counterpart.

The isomap_distance_graphs example demonstrates radius component completion
and CSR precomputed query projection. Original native algorithms remain MIT,
with retained sklearn/SciPy BSD contract attribution. No new dependency,
public estimator type or extension algorithm is introduced.

Cumulative inventory: 221 public core types, 135 source modules, 164 native
programs, 141 core reference suites / 1,050,643 groups, and 46 invalid-input
registries / 3,767 cases. The new suite contributes 362 fixtures / 4,478 groups
and 31 diagnostics. It spans three seeds, Euclidean/Manhattan/Minkowski metrics,
fixed-count and radius graphs, all path choices, dense and CSR fit/query
combinations, parameter changes after fitting, singleton and oversized fits,
asymmetric/nonzero-diagonal distances, duplicate points and stored columns,
zero edges, and the larger sparse auto-path branch.

Nineteen source errors are retained and matched: 16 during fitting (disconnected
CSR graphs and infinite p) and three empty-radius query reductions. Embedding
and query Gram/cross-Gram matrices, positive eigenspaces and eigenvalues check
the geometry without requiring the same valid eigenvector basis. Geodesic
matrices, centering statistics, metadata, output names and input preservation
are checked directly. Reconstruction errors are compared as squared residuals
with an explicit floating-point accumulation bound. These fixtures produce no
source NaN reconstruction errors.

Checked and release profiles each pass all 164 native programs, 4,790 affected
reference groups and 32 diagnostic cases (the new registry plus the existing
Isomap neighbor-count rejection). Both execute the new example. The final
audit verifies 308 compiled-file digests per profile, complete field inventories,
source/validator/executable hashes, installed source provenance, the SciPy
shortest-path binary hash, and unchanged non-Isomap modules and reconstruction
error body. Reports and logs are retained in ignored validation/results. Only
affected reference suites were rerun; cumulative counts retain earlier results.


## KernelDensity trees, metrics and NumPy-compatible sampling

`kernel_density` now implements all six sklearn kernels, adding cosine to
Gaussian, tophat, Epanechnikov, exponential and linear. algorithm accepts
'auto', 'kd_tree' and 'ball_tree'; auto selects KD-tree for the supported metric
family, matching sklearn. Trees use balanced feature-range median partitions,
with leaf_size=40 by default. breadth_first selects the source heap traversal
or depth-first recursion. atol and rtol default to zero; positive finite
values control the source density-bound stopping criteria. Node construction,
box/ball bounds, weighted centroids, traversal and log-bound updates follow
sklearn 1.6.1. The median groups agree with the source, while within-leaf order
can differ from the platform C++ partition routine.

Supported metrics are euclidean/l2, manhattan/cityblock/l1,
chebyshev/infinity and minkowski/p. For Minkowski, p>0 includes fractional
orders; unweighted infinite p follows the source Chebyshev alias. An optional
metric_weights vector maps to metric_params['w'] and requires finite
nonnegative feature weights and finite positive p. Query metrics and feature
weights use the fitted snapshot; changing those parameters requires refitting.
The current kernel and traversal/tolerance settings apply at query time, as in
Python, while bandwidth_ remains fitted. Other BallTree metrics and arbitrary
Python metric objects are not implemented. Inputs remain dense finite real64.

Source KD-tree bounds omit Minkowski feature weights. This can affect pruning,
including a single-point weighted model, and is reproduced by the native
KD-tree backend. Explicit Ball-tree fits use weighted distances in their
bounds. The source cosine normalization formula is retained, including its
nonfinite results for some dimensions; this is not replaced by a different
normalizer. Non-Euclidean scoring retains the source Euclidean volume
normalization and is not generally a normalized density in that metric.
The KD-tree Chebyshev radius diagnostic also follows the source pow(rad,0)
convention, giving a stored value of 1.

fit accepts finite nonnegative sample weights, including all zero. Zero total
weight produces source NaN scores; sampling still selects the first stored
point through the source left-insertion cumulative-weight rule. Overflowing
weight sums and tiny-bandwidth IEEE results are retained without triggering
native floating-point traps. The existing normalized weights_ view remains;
sample_weight_ stores raw weights, weight_sum_ the NumPy-associated total,
and n_features_in_/n_samples_fit_ record fitted dimensions. algorithm_ and
n_nodes_ are native fitted-tree diagnostics. get_tree_arrays returns copies of
one-based point indices, inclusive [first,last,is_leaf] spans, a
(2,n_nodes,n_features) bounds array, radii and node weights. For Ball-tree,
the first bounds plane contains centers and the second is zero. These storage
helpers do not add public estimator types or algorithms.

sample supports integer seeds and a mutable legacy_random_state object through
the same generic method. Integer seeds reset a local stream on every call;
object calls advance the supplied state. Omitted random_state uses a separate
clock-seeded module stream shared by density models. Explicit calls do not
change that stream or the Fortran intrinsic RNG. Zero-sized sample requests
are valid. All mixture uniforms precede normal draws, and raw-weight cumulative
sums use left search insertion, matching RandomState. Gaussian draws use the
selected training center plus bandwidth-scaled normals. Tophat uses the source
regularized-incomplete-gamma transform of normal-vector squared radius, with
no additional radius draw. Both samplers use Euclidean Gaussian/ball geometry
regardless of the scoring metric, as in sklearn. Other kernels do not sample.

The density_seeded_sampling example shows integer and shared-state draws and
the cosine kernel. Original adapters and numerical implementations remain MIT,
with retained sklearn/NumPy/SciPy attribution. No new dependency or extension
algorithm is introduced.


The new reference suite retains 1,144 fixtures and 20,592 comparison groups,
with 34 isolated input diagnostics. It checks all kernels and supported metric
aliases, fixed/Scott/Silverman bandwidths, both tree types and traversals,
leaf sizes, approximation controls, ordinary/zero/tiny/overflowing weights,
support boundaries, high dimensions, parameter changes after fitting,
integer-seed repeats, mutable RNG streams (including a cached normal), empty
samples, and default sample count. Point partitions, spans, bounds/centers,
radii and node masses are compared with fitted sklearn trees. Sample values
and the next RNG outputs are compared with RandomState/SciPy references.

Cross-runtime arithmetic can leave different bound-subtraction residues for
very small tree densities, giving large differences after taking logarithms.
These are not reported as direct log-score matches. The checked run retains
102 explicit density-space roundoff certificates plus 14 log-score sum
identities across 65 fixtures; the largest certified density discrepancy is
1.499e-14. The acceptance bound is 256*machine_epsilon*n_nodes*exp(log_norm).
All fitted-tree structure comparisons still have to pass. Raw native, source
and source single-leaf log outputs are retained. Single-leaf fits are diagnostic:
zero-weight Ball-tree leaves can have NaN centers, so source results can depend
materially on tree configuration and a single-leaf fit is not universally an
equivalent oracle. Total scores in certified fixtures must equal the sum of
the returned native log scores. Other output and IEEE-mask comparisons remain
direct. The report retains 1,708 fields with nonfinite source results.

All 165 checked native programs pass, along with 21,015 affected reference
groups and 39 diagnostic cases. The existing mode/spectral/density validator
now accepts an explicit executable and profile, permitting isolated checked
and release runs without nested fpm calls. The legacy density_weights case
now supplies a negative weight; its former all-zero input is a valid source
fit. Other legacy diagnostics and other core modules retain their bodies.
The checked audit verifies 309 compiled-file digests, complete field inventories,
roundoff records, executable/validator/source and reference-binary hashes,
and the seeded example outputs. Reports and logs remain in ignored
validation/results. Release validation is recorded after its audit completes.


Cumulative inventory: 221 public core types, 135 source modules, 165 native
programs, 142 core reference suites / 1,071,235 groups, and 47 invalid-input
registries / 3,801 cases. These remain API subsets; no public extension type or
new dependency is counted. Only affected reference suites were rerun.

Both checked and release profiles pass all 165 native programs, 21,015 affected
reference groups and 39 diagnostic cases. Each runs the seeded density example
and passes the final 309-file compiled-content audit. The release fixture
inventory matches checked. Raw source/nonfinite outcomes and explicit tree
roundoff records are retained per profile, with hashes of installed tree,
NumPy RNG and SciPy special-function binaries. The cumulative reference total
also retains results from previous batches.

## Barnes-Hut t-SNE batch

`tsne` now defaults to sklearn's `method='barnes_hut'`; select `method='exact'`
for the retained dense algorithm. Barnes-Hut adds one through three embedding
components, angle-controlled spatial summaries, sparse nearest-neighbor joint
probabilities, and dense or CSR precomputed unsquared distances. Existing random,
PCA and explicit real64 initialization, adaptive gains, early exaggeration,
momentum phases and checkpoint stopping are shared. PCA's float32 PC1 standard
deviation now uses NumPy's pairwise summation order. The old exact test and
example explicitly select their original method.

The new internal `sklearn_tsne_bh` module uses a dynamically grown serial
float32 tree, source insertion/barycenter rounding, duplicate leaf masses,
source positive and repulsive force arithmetic, and sparse symmetrization.
Its optional diagnostic snapshots expose parent/child/point/depth/mass/leaf
fields plus bounds, centers, barycenters and squared widths. Integer snapshots
and floating tree snapshots compare exactly with the installed sklearn binary.
The binary exports `EPSILON=0` despite the pxd initializer 1.e-6; provenance
records the source and binary hashes and reads the exported value. This is an
internal source-counterpart helper, not an additional estimator extension.

`compare_tsne_bh.py` retains 4,314 result groups across 1,110 fixtures: 126
sparse-affinity cases, 900 force/tree cases, 12 short optimizer cases and 72
complete fits. Three seeds cover one through three dimensions, two through 100
samples, zero/normal/extreme sparse distances, varying neighborhood counts,
angles zero through one, duplicate/collapsed/tiny/large coordinates, skipped
points and omitted objectives. Complete fits cover explicit float64, random
and PCA initialization; Euclidean, Manhattan, cosine and dense/CSR precomputed
metrics; explicit/automatic rates; exaggeration; 250/300/350 iteration limits;
checkpoint stopping; input preservation and repeatable refits. Complete source
embeddings and KL diagnostics use strict 3.e-8 relative/absolute tolerances,
without trajectory certificates. Tree structure and float32 bounds are exact.
There are 29 additional isolated diagnostic checks.

For skipped-point helper comparisons, the validator calls sklearn's compiled
backend directly: its Python wrapper omits forwarding the argument. The source
also allocates only `n_samples-skip` summary slots, so fixtures use one skipped
point and a duplicate pair to keep source execution within that capacity. Native
recursive traversal has no such summary buffer. Public estimator fits do not
use skipped points.

Barnes-Hut force evaluation normally takes O(N log N), while dense feature
neighbor search still uses quadratic pairwise work/storage. CSR precomputed
input avoids that full matrix and must provide at least neighbor-count-plus-one
stored candidates per row; explicitly stored self zeros are recommended.
Neighbor ties follow the package's stable ordering and can select a different
cutoff subset from sklearn. Sparse feature matrices, parallel execution,
callable metrics and generic metric dictionaries remain unsupported. The
native implementation diagnoses float32 working-range overflow and subdivision
beyond 511 levels. The exact affinity/objective bodies and all unrelated core
source modules are unchanged. No dependency or public extension type was added;
MIT licensing and third-party BSD notices remain intact.

Cumulative inventory: 221 public core types, 136 source modules, 166 native
programs, 143 core reference suites / 1,075,549 retained groups, and 48
invalid-input registries / 3,830 cases. These remain API subsets.

Final verification: both checked (`-fcheck=all -fbacktrace
-ffpe-trap=invalid,zero,overflow`) and release profiles pass all 166 native
programs, 6,246 affected reference groups and 75 isolated diagnostics. The
reference total comprises 4,314 Barnes-Hut groups, 1,620 retained exact t-SNE
groups, 120 graph/neighbor groups and 192 integration groups. The diagnostics
comprise 29 new cases and 46 retained t-SNE/trustworthiness cases. All 72 full
Barnes-Hut embeddings have zero discrepancy from sklearn in both profiles;
all tested force gradients also match exactly. Both example runs produce KL
0.610453724861145 and trustworthiness 0.9211111111111111. The previous exact
example was rebuilt and retains its explicit exact solver.

The final audit verifies 314 compiled source contents per profile, distinct
checked/release driver binaries, complete fixture/field inventories, source
and reference-binary hashes, the exported duplicate threshold, unchanged exact
affinity/objective bodies and unchanged unrelated core modules. Detailed local
reports and audit helpers remain under ignored `validation/results`, including
`tsne_bh_final_audit.json`. The implementation is committed as `2b5c9fb`.

## SGD penalties, schedules and source shuffling batch

The shared SGDRegressor/SGDClassifier engine now supports `l1`, `l2`, `none`
and `elasticnet`, with `l1_ratio=.15` and the source cumulative truncated-gradient
correction. All four public learning-rate schedules are supported: constant,
inverse scaling, optimal initialization with the advancing sample clock, and
adaptive reductions by five when training-loss patience is exhausted. Optimal
requires positive alpha and accepts eta0=0. Adaptive stopping occurs only when
the current rate cannot be reduced further (eta<=1.e-6). Each partial_fit runs
one epoch; truncation corrections and adaptive-rate state reset per call, as
in the source. Standard coefficients, average state and sample clock persist.

Averaging now follows the source WeightVector64 lazy `average_a/average_b`
updates and scale materialization, including resets below 1.e-9. This preserves
the source's averaging-before-L1 order and its resulting non-arithmetic-average
behavior for truncated gradients. Intercepts remain unpenalized. Repeated
zero lazy scales, zero-weight rows and elastic-net endpoints are covered.

Explicitly enabled shuffling now reproduces sklearn's scalar NumPy seed draws,
forward xorshift Fisher-Yates permutations and per-class OVR seed streams.
Permutations are reshuffled in place each epoch using the same seed. The
implementation uses the existing local NumPy-compatible RNG and leaves the
global Fortran RNG alone. The native defaults remain ordered inverse scaling;
there is no new dependency or public extension type.

`compare_sgd_regularization.py` contains 25,119 strict result groups across
4,194 fixtures and seven new isolated diagnostics. Its main Cartesian grid
covers three seeds, four regression losses or five classifier losses, regression
and binary/three-class OVR fitting, all four penalties, all four schedules,
average thresholds 0/1/10, ordered/shuffled data, varied sample weights and
streaming batches. Further cases cover zero/large regularization, elastic-net
endpoints, all-zero weights, repeated lazy-scale resets, nonzero adaptive
updates after rate reductions, optimal eta0=0 at three regularization strengths,
and the adaptive stopping threshold. Coefficients, intercepts, predictions or
decisions, probabilities, labels and per-batch coefficient/clock traces use
3.e-9 relative and 3.e-10 absolute tolerances. No alternative oracle or
trajectory certificate is used. The reference driver streams cases in one
process to avoid Windows launch overhead; diagnostics remain isolated.

Existing sparse-input, warm-start fit, custom initialization, validation-set
early stopping and averaged-classifier partial-fit intercept limitations remain
as documented. The new controls reach supported composition wrappers through
ordinary estimator copies. The original SGD, multiclass, multioutput and
integration reference suites are included in the affected regression run.
All unrelated core modules are unchanged; only the legacy unsupported-penalty
fixture changes from now-supported `l1` to `unknown`.

Cumulative inventory: 221 public core types, 136 source modules, 167 native
programs, 144 core reference suites / 1,100,668 retained groups, and 49
invalid-input registries / 3,837 cases. These remain API subsets. The package
retains its MIT license and attributed third-party BSD sources.

Final verification: checked (`-fcheck=all -fbacktrace
-ffpe-trap=invalid,zero,overflow`) and release profiles both pass all 167 native
programs, 56,353 affected reference groups and 62 isolated diagnostics. The
reference total comprises 25,119 new SGD groups, 216 retained regression groups,
306 retained classification groups, 27,490 multiclass groups, 3,030 multioutput
groups and 192 integration groups. All seven new and 55 retained SGD diagnostics
pass. The largest coefficient discrepancy from sklearn is
5.684341886080802e-14 in each profile. Both example runs produce regression R2
0.9996711850882956 and classification accuracy 0.975.

The final audit verifies 315 compiled source contents per profile, distinct
checked/release driver binaries, complete fixture/field inventories, validator,
parser and upstream source hashes, unchanged unrelated core modules and
unchanged unrelated legacy diagnostics. Detailed local reports and audit
helpers remain under ignored `validation/results`, including
`sgd_regularization_final_audit.json`. Implementation commit: `bf1c19f`.


## SGD warm initialization and complete classifier loss mapping batch

SGD regression and classification now support warm fits and explicit coefficient/
intercept initializers. Every fit restarts the sample clock; partial_fit keeps
advancing it. Cold fits accept the native published coefficient/intercept shapes.
Binary classifiers additionally accept vector coefficients, arbitrary 2D shapes
with the correct element count (flattened in C order), and scalar intercepts.
Multiclass coefficient/intercept shapes must match the class and feature counts.
All initializer arrays and feature inputs are preserved.

The port preserves sklearn 1.6.1's different averaged warm-fit paths. A fitted
warm regressor retains standard and average buffers and ignores new fit
initializers. A warm classifier starts from its published coefficients and
intercepts, with explicit initializers taking precedence, and creates fresh
averaging buffers. Regressor averaging configuration must remain fixed across
warm fits. The regressor's retained-average behavior is checked analytically
and against the actual source, rather than replaced by a conventional average.

SGDClassifier now supports all nine source losses: hinge, squared_hinge,
perceptron, log_loss, modified_huber, squared_error, huber,
epsilon_insensitive and squared_epsilon_insensitive. epsilon=.1 configures the
last three applicable contrasts. predict_log_proba supports log loss and
modified Huber, returning negative infinity for exact zero probabilities without
triggering checked floating-point traps.

The internal sklearn_sgd_clone module supplies parameter-only copies. Cold clone
sites in multioutput/chains, multiclass, voting, bagging, stacking, thresholding,
SelectFromModel, CV/learning curves and search discard fitted SGD state while
retaining constructor controls. Existing streaming-copy adapters now retain
l1_ratio, epsilon and warm_start. Search also exposes epsilon, l1_ratio,
warm_start, average, random_state and n_iter_no_change, including nested pipeline
names. CV/search reset SGD children recursively inside nested pipelines; direct
pipeline fits preserve their warm tails. Existing prefit branches keep their
fitted bases.

The native cloning identity test compares 768 predictions, decisions, masks,
scores and probabilities obtained from unfitted prototypes against those from
fitted warm prototypes with an extra feature. It checks independent outputs,
chains, OVR/OVO and streaming reductions, voting, bagging, stacking, selection,
fixed thresholds, CV, incremental learning curves and nested pipeline search.
It also checks nondefault epsilon/l1_ratio preservation and that fitted caller
coefficients remain unchanged. This tests the clone contract independently of
the implementation's state-reset code.

compare_sgd_fit.py contributes 136,080 strict source groups across 6,480 workflows
and 19,440 fitting stages. Three seeds cover regression and binary/three-class
classification, every applicable loss, all four penalties and schedules, average
thresholds 0/1/10, both shuffle settings, optional initializer combinations,
repeated warm/cold fits, explicit published-array replacement and mixed
partial_fit/warm-fit histories. Coefficients, intercepts, predictions/decisions,
labels, probabilities, log probabilities and clocks use 3.e-9 relative and
3.e-10 absolute tolerances. Negative-infinity and NaN masks are checked explicitly;
there are no trajectory certificates or alternative source oracles. Twenty-one
new isolated input checks cover shapes, nonfinite initializers, warm-width and
averaging constraints, epsilon controls and log-probability availability.

Remaining SGD limits include sparse input, validation-set early stopping,
parallel OVR, general metadata routing and averaged classifier partial fits with
an intercept. Public native defaults remain ordered inverse scaling. No public
extension type or dependency was added. The package stays MIT with retained
third-party BSD attribution; the internal parameter-copy implementation is MIT.

Cumulative inventory: 221 public core types, 137 source modules, 169 native
programs, 145 core reference suites / 1,236,748 retained groups, and 50
invalid-input registries / 3,858 cases. These remain API subsets.

Final verification: checked (`-fcheck=all -fbacktrace
-ffpe-trap=invalid,zero,overflow`) and release profiles both pass all 169 native
programs, 281,953 affected reference groups and 384 isolated diagnostics. The
reference run includes the 136,080 new warm/initializer groups, retained SGD
penalty/regression/classification suites, independent-output and multiclass
reductions, voting/bagging/stacking, search, learning/validation curves,
permutation tests, out-of-fold prediction, selection, pipelines, thresholding
and integration. Diagnostics comprise 21 new cases, 55 retained SGD cases and
308 existing search/pipeline/selector/threshold/validation/curve cases.

Both profiles check 2,621 nonfinite reference log-probability values with exact
masks. The source comparisons retain the documented absolute/relative tolerances
without trajectory certificates. The native clone-isolation identity covers
768 collected outputs. Both example runs improve regression R2 from
0.8469345057054836 to 0.9106907710754828 on the repeated warm fit, restart the
sample clock at 151, and achieve Huber classification accuracy 1.0.

The final audit verifies 334 compiled source contents per profile, distinct
checked/release binaries, complete 6,480-workflow / 19,440-stage field inventories,
source/validator/parser hashes, unchanged SGD loss/update routines and unchanged
unrelated core modules and main diagnostics. Detailed local reports and audit
helpers remain under ignored `validation/results`, including
`sgd_fit_final_audit.json`. Implementation commit: `ebc8cdb`.


## CSR SGD regression and classification batch

Both SGD estimators now accept native CSR input for fit, partial_fit, prediction
and scoring. Classifiers also support sparse decision_function, predict_proba
and predict_log_proba. Dense and sparse batches/queries can be mixed using the
same standard/averaged parameters, feature metadata and sample clock. The shared
parameter preparation retains explicit initialization and warm-fit behavior.
All existing losses, penalties, schedules, sample/class weights and averaging
controls apply to CSR.

The serial kernel visits stored entries in their original order for dot
products, coefficient updates, averaging corrections and L1 truncation.
Unsorted columns, duplicates, explicit zeros and empty rows are supported
without canonicalization or densification. L2 decay remains lazy and applies
even to zero-weight or empty rows. L1 truncation visits stored features only;
implicit zeros and explicitly stored zeros therefore need not give the same
trajectory. Sparse intercepts use the source multiplier 0.01, dense intercepts
use 1.0. Each mixed-format batch uses its own input-format rules. These
source-defined differences are retained rather than forcing dense/CSR equality.

Model storage is O(n_features) per coefficient/averaging channel plus CSR data
and outputs; training never constructs a dense samples-by-features matrix.
Queries use CSR multiplication with dense coefficients. A native analytic test
checks sparse intercept updates and stored-zero L1 behavior, mixed input/query
formats, probabilities/scores and a million-feature CSR model. The sparse text
example feeds signed HashingVectorizer features directly to logistic SGD.

compare_sgd_csr.py contributes 136,080 strict source result groups across 6,480
workflows and 19,440 fitting stages. Three seeds cover every applicable loss,
all penalties/schedules, average thresholds 0/1/10 and both shuffle settings.
Storage patterns include canonical CSR, explicit zeros, unsorted rows,
duplicate/cancelling entries and entirely zero data. Additional histories mix
dense/CSR partial_fit and warm fits, published-array replacement, all native
initializer forms and both query formats. Coefficients, intercepts, predictions/
decisions, labels, probabilities, log probabilities and clocks use 3.e-9
relative and 3.e-10 absolute tolerances with explicit nonfinite masks, without
trajectory certificates or alternative source oracles. Twenty-four new isolated
CSR diagnostics cover structure, finite data, shapes, weights, initialization,
queries and probability availability. Inputs and initializers are preserved.

The full retained dense SGD and affected composition/reference suites are
included in verification because dense and CSR paths share update/state code.
Sparse outer pipelines, CV and composition entry points retain their existing
dense input contracts; direct CSR SGD is available now. General sparse
containers other than the native csr_matrix, validation-set early stopping,
parallel OVR and previously documented averaging restrictions remain outside
this subset. Original code stays MIT with retained sklearn BSD attribution.
No dependency or public extension type was added.

Cumulative inventory: 221 public core types, 137 source modules, 170 native
programs, 146 core reference suites / 1,372,828 retained groups, and 51
invalid-input registries / 3,882 cases. These remain API subsets.

Final verification passed in both checked (bounds/runtime checks and floating-
point traps) and optimized release builds: all 170 native programs, 418,033
reference result groups across 20 affected suites, and 415 diagnostic cases
per build. Both builds ran the sparse text example successfully. The new CSR
suite matched sklearn 1.6.1 with maximum coefficient error
1.7763568394002505e-15; its 37 nonfinite source values matched by explicit
masks. Dense SGD, warm-start and composition regressions passed as well.
The combined audit verified 336 compiled source contents per build, executable
and validator hashes, identical stage/field inventories, preserved source
inputs, and unchanged unrelated core modules and loss routines. Reports and
logs are retained locally under the ignored validation/results directory.


## Source-compatible stochastic multilayer perceptrons

Replaced the original full-batch-only MLP training with sklearn 1.6.1 minibatch
Adam and SGD for dense scalar regression, binary and multiclass classification.
Initialization draws Glorot coefficients in NumPy row-major order, then biases,
using the retained legacy RandomState generator. Epoch shuffles use the same
continuing generator and Fisher-Yates permutation. Binary classification uses
one sigmoid output; multiclass uses softmax. All four hidden activations and
empty, single or multiple hidden layers are supported.

Added auto/explicit batch sizing, configurable Adam beta_1/beta_2/epsilon,
SGD momentum and Nesterov updates, constant/invscaling/adaptive schedules,
and source loss accumulation and stopping. Regularization divides by the
current batch size, and final short batches retain their own denominator.
Adam counts optimizer steps per batch; t_ counts samples per epoch.
Adaptive reductions divide the rate by five and stop only after the source
rate/patience conditions. Cold refits reset initialization and optimizer state.
Defaults now follow sklearn (200 epochs, tol=1e-4, patience=10).

The native layer arrays remain the coefficient/intercept interface. Added
n_features_in_, n_outputs_, n_layers_, t_, best_loss_, out_activation_ and
predict_log_proba, preserving negative infinity for saturated probabilities.
Native tests check seeded two-batch updates, sample clocks, refit isolation,
adaptive stopping, binary parameter shape and probability threshold ties.
The nonlinear regression test explicitly retains its former tighter tolerance
and patience. A new example learns XOR with minibatch momentum SGD.

compare_mlp_stochastic.py checks 48,960 result groups across 2,880 fixtures:
three seeds, regression/binary/multiclass, four activations, eight optimizer
configurations, five batch settings and both shuffle settings. Auto batches
use 203 samples to exercise the 200-sample boundary; other cases include final
short batches and oversized batch clipping. Degenerate zero/constant columns,
repeated samples and no/one/two hidden layers are included. Checks cover every
layer's coefficients, intercepts, Adam moments/variances or SGD velocities,
loss curves, metadata, predictions, class probabilities and log probabilities.
Classifier queries also include extreme positive/negative inputs, checking
source saturation and nonfinite log probabilities. Twenty-five isolated
diagnostics cover controls, shapes, finite data, unfitted
queries and class cardinality.

Most groups use rtol=5e-9 and atol=5e-11. Degenerate zero-input, single-sample-
batch custom-Adam trajectories can amplify rounding over 13 epochs. The three
such regression fixtures also measure componentwise propagation envelopes from
six sklearn runs whose initial values move by one ULP (both directions, all
parameters or one first/last-layer bias). These bounds apply only when the
strict comparison fails; each result remains compared to the original
unperturbed source values. Every other fixture retains the strict tolerance.
The source probes change final coefficients by up to about 4.3e-7; the checked
Fortran development run differs by about 1.4e-7. This is an explicitly recorded
numerical sensitivity check rather than a blanket tolerance increase.

Dense real64 CPU input, integer class labels and one regression response remain
the supported input contracts. Validation-set early stopping, L-BFGS, warm
starts, partial_fit, sparse input, multilabel targets and joint multioutput
regression are not added. The MIT package retains sklearn BSD attribution;
no dependency or public extension estimator was added.

Cumulative inventory: 221 public core types, 137 source modules, 171 native
programs, 147 core reference suites / 1,421,788 retained groups, and 52
invalid-input registries / 3,907 cases. These remain API subsets.

Final verification passed in both checked (bounds/runtime checks and floating-
point traps) and optimized release builds: all 171 native programs, 48,960
MLP reference groups and 25 isolated diagnostic cases per build. The reference
runs matched 3,172 nonfinite source values each. Twelve checked and thirteen
release result groups used the measured one-ULP propagation envelopes; all
remaining groups passed the strict tolerances above. Both XOR example runs
predicted [0,1,1,0] after 463 epochs / 1,852 samples seen, with final loss
0.004167032183886215. The combined audit verified 314 compiled source contents
per build, source/validator/executable hashes and complete layer/field
inventories. All unrelated core modules and native tests were unchanged.
Reports and logs are retained locally under ignored validation/results.


## Full-data L-BFGS multilayer perceptrons

Added solver='lbfgs' to mlp_classifier and mlp_regressor, using the already
vendored L-BFGS-B implementation with unbounded parameters. All coefficients
are packed in NumPy row-major order, followed by all intercepts, as in sklearn
1.6.1 _fit_lbfgs/_loss_grad_lbfgs. Initialization, activations, binary logistic
and multiclass softmax output, clipped logistic losses and squared-error
regression retain the preceding MLP contracts. L2 regularization uses the full
sample count and leaves intercepts unpenalized.

Optimizer defaults match scipy.optimize.minimize's L-BFGS-B adapter: ten
corrections, factr=1e7 and twenty line-search steps, with MLP tol mapped to the
projected-gradient threshold. max_iter bounds accepted iterations;
max_fun=15000 bounds evaluations softly, using the source's strict greater-than
check after accepted steps. Initialization can satisfy the gradient tolerance
with zero optimizer iterations. Stochastic batch sizes, shuffling, momentum
and learning-rate schedules do not affect L-BFGS fits.

The native diagnostic fields n_function_evaluations_, optimization_message_,
gradient_, previous_parameters_ and previous_loss_ expose the returned gradient
and preceding accepted state for verification. Gradient and parameter vectors
use the same packed order. t_ stays zero; loss_curve_ is unallocated and
best_loss_ has no stochastic interpretation for this solver. Cold solver
changes clear stale diagnostic arrays. The new example learns XOR with L-BFGS.

The new reference suite runs 1,080 fixtures across three seeds, three response
kinds, four activations, three network architectures and ten work/tolerance
configurations. Degenerate and duplicate features, initial convergence,
iteration limits and evaluation limits are included. Its 14,688 required result
groups compare loss, gradient and queries to unchanged sklearn routines at the
native returned parameters; check the preceding objective; and verify the
reported termination against source gradient/relative-loss criteria or work
limits. Short and evaluation-limited fits also compare complete source
parameters, predictions and optimizer metadata. Twenty-seven diagnostics cover
controls, data, target/query shapes, unfitted access and class cardinality.

Longer fits additionally retain 2,592 complete endpoint comparisons against
sklearn. In the checked development run, 812 comparisons across 131 fixtures
differed at rtol=2e-6/atol=2e-7. These are recorded as trajectory diagnostics and
excluded from the core required-group count. Long optimization may produce
different coefficients, predictions, losses and work counts. The native and
source losses/gradients at identical parameters still agree: development
maximum loss error was 1.65e-13 and gradient error 1.39e-11. All 1,080 returned
states passed the independent termination checks. No line-search failures
occurred in that run.

Two retained source-only probes move initial parameters by one ULP in either
direction. They change sklearn's final coefficients by up to 0.224 and final
loss by up to 2.57e-4, demonstrating sensitivity in representative logistic and
ReLU networks. These probes are diagnostic records. They do not replace the
source objective or turn differing endpoint comparisons into matches.
The vendored Fortran optimizer and SciPy 1.15.3's C implementation need not
follow identical rounded trajectories.

Native tests also check the no-hidden-layer reduction to ridge normal
equations, the unpenalized intercept, sample-clock/history contracts, zero-step
convergence, work limits, solver changes and multiclass softmax prediction.
Dense scalar regression and integer binary/multiclass classification remain the
supported targets. Validation-set early stopping, warm starts, partial_fit,
sparse input, multilabel classification and joint multioutput regression remain
outside this interface. No dependency or public extension type was added.

Cumulative inventory: 221 public core types, 137 source modules, 172 native
programs, 148 core reference suites / 1,436,476 retained required groups, and 53
invalid-input registries / 3,934 cases. These remain API subsets.

Final checked (bounds/runtime checks and floating-point traps) and optimized
release verification passed all 172 native programs, 14,688 required L-BFGS
groups, 48,960 retained Adam/SGD groups and 52 invalid-input diagnostics per
build. The stochastic regressions retained their twelve checked / thirteen
release rounding-envelope groups. Both XOR examples ran successfully in both
builds. L-BFGS XOR used 122 iterations / 149 evaluations in checked mode and
121 / 148 in release, with losses approximately 0.001359.

The 2,592 long-fit endpoint diagnostics retain 812 differing comparisons across
131 checked fixtures and 819 across 134 release fixtures. All pointwise loss,
gradient, query and stopping/return-status checks passed. Two release ReLU
cases returned explicit line-search nonconvergence; separate installed-sklearn
checks also reported ABNORMAL on those same cases, with final losses within
2.1e-10 of the native values. These outcomes retain converged_=.false.
Maximum pointwise gradient/loss errors were 1.39e-11 / 1.65e-13 checked and
2.29e-12 / 1.73e-14 release.

The combined audit verified 317 compiled source contents per build,
source/optimizer/validator/executable hashes and complete required/diagnostic
field inventories. Unrelated core modules, existing native tests and the
stochastic training/prediction kernel were unchanged. Reports, source probes
and logs are retained locally under ignored validation/results.


## Warm-start multilayer perceptron fitting

Added warm_start to both MLP types for Adam, SGD and L-BFGS fits. The first fit
initializes normally; subsequent warm fits reuse published coefficient/bias
arrays and retain the feature/class contracts. Stochastic fits append loss
history, retain best_loss_ and no_improvement_count_, and accumulate a 64-bit
t_ sample clock. n_iter_ counts epochs in the current call. Switching back to
a cold fit resets these histories, clocks and parameters.

Following sklearn 1.6.1 _fit/_fit_stochastic, each warm Adam/SGD fit constructs
a fresh optimizer: moments, variances or velocities start at zero, optimizer
steps and initial learning rates restart, and the current seed starts a new
shuffle stream without consuming initialization draws. Inverse scaling after
an epoch still uses the accumulated sample clock. Fresh optimizer buffers
follow the actual edited parameter shapes.

Warm L-BFGS calls reconstruct optimizer workspaces and reset optimizer iteration/
evaluation counters while retaining the fitted parameters and any existing
stochastic histories/sample count. Stochastic training can then resume with a
fresh optimizer. A new L-BFGS model has no stochastic history; a subsequent
stochastic warm fit requires an earlier stochastic history, as in the source
interface. Use a cold stochastic fit when that history is unavailable.

Warm classifiers require exactly the previously fitted class set. Feature count
and layer count must stay compatible. Stochastic warm fits retain actual layer
widths when positive public hidden sizes of the same layer count are changed;
a cold fit applies the new architecture. L-BFGS packing additionally requires
public widths to match the actual layers. Numerically edited parameter arrays
and compatible shape replacements are used directly rather than reinitialized.

The new reference suite covers 756 workflows / 4,536 fits and 81,648 result
groups. Three seeds, three response kinds, four activations, three architectures
and seven optimizer configurations exercise repeated warm fits, parameter edits,
cold resets, seed/rate/alpha changes, SGD schedules, and stochastic/L-BFGS
transitions. Checks include all coefficients, biases and optimizer buffers,
loss histories, patience state, sample/epoch metadata and query outputs.
Dedicated histories cross the signed 32-bit sample limit and replace parameter
shapes. Eight diagnostics cover warm feature/class/architecture mismatches,
unavailable stochastic history and revalidation of invalid controls.

Strict comparisons use rtol=5e-9 and atol=5e-10. One all-zero regression workflow
reaches an exact zero bias in SciPy L-BFGS and a roundoff residual in the native
optimizer. Subsequent Adam updates amplify that residual: seven final-stage
fields differ from the unperturbed source trajectory. They are additionally
checked against the componentwise source response to 32 checkpoint-bias
perturbations: signed powers of two from 2^-12 through 2^3 times machine epsilon,
scaled by the preceding parameter working magnitude (at least one). The source
optimizer, history and data remain unchanged in these probes. The original
endpoint values and strict-failure markers remain in the report; the bound
applies only to this workflow/stage. Every other result group keeps the strict
tolerance. This check records cross-solver rounding sensitivity.

Native tests independently check scalar SGD updates, optimizer-state resets,
history and clock accumulation, cold resets, L-BFGS handoffs, edited shapes,
layer-width behavior and a sample count above 2^31. A warm-start XOR example
shows the distinction between current-call epochs and accumulated history.
Partial_fit, validation-set early stopping, sparse MLP inputs, multilabel
classification and joint multioutput regression remain outside this subset.
Original contributions remain MIT with retained sklearn/SciPy BSD notices.
No dependency or public extension type was added.

Cumulative inventory: 221 public core types, 137 source modules, 173 native
programs, 149 core reference suites / 1,518,124 retained required groups, and 54
invalid-input registries / 3,942 cases. These remain API subsets.

Final verification passed in both checked (bounds/runtime checks and floating-
point traps) and optimized release builds: all 173 native programs, 81,648 warm
result groups, 48,960 retained Adam/SGD groups, 14,688 retained required L-BFGS
groups and 60 invalid-input diagnostics per build. Both warm suites matched
5,417 nonfinite source values and used seven checkpoint-response groups. The
retained stochastic suites kept their twelve checked / thirteen release
rounding-envelope groups; L-BFGS endpoint diagnostics and return-status outcomes
were unchanged from the preceding increment.

All three MLP examples ran in both builds. The warm example predicted [0,1,1,0]
with 82 last-call epochs, 482 total history entries, 1,928 samples seen and final
loss approximately 0.0030879462. The combined audit verified 320 compiled source
contents per build, source/optimizer/validator/executable hashes and complete
required/diagnostic inventories. Unrelated core modules and all existing native
tests were unchanged, as were activation, query and objective/gradient kernels.
Reports, source probes and logs remain locally in ignored validation/results.


## Joint MLP regression and multilabel indicator targets

Added rank-two fit overloads to mlp_regressor and mlp_classifier for all three
solvers and warm fits. Multioutput regression uses one shared network with a
coefficient column per response. Its reported squared data loss averages over
samples and responses, while backpropagation divides by the sample count; L2
loss/gradients also divide by the sample count. These are sklearn 1.6.1's
separate loss/Jacobian conventions and are retained explicitly, including in
L-BFGS callbacks. Single-response calculations retain their former values.

Multilabel indicators use independent sigmoid outputs for every label, summed
binary cross-entropies per sample and independent >0.5 thresholds. Probabilities
are positive-event probabilities per label and need not sum to one. This also
applies to two-label matrices even though classes_ then contains [0,1].
classes_ contains zero-based label-column indices. An n-by-1 classification
target delegates to the existing vector classification path; a one-column
regression target retains scalar prediction compatibility.

Fortran's fixed function-result rank requires explicit matrix prediction
methods: predict_multioutput for regression and predict_indicator for multilabel
classification. Existing vector predict calls reject fitted matrix-output
models instead of dropping a response or choosing one indicator. Matrix score
overloads use uniform-average per-response R2 or exact-row-match accuracy.
Constant-response R2 follows force_finite behavior; single-sample MLP R2 is NaN.
Warm fits retain target-kind/output-width checks and existing stochastic state
contracts, and cold fits can change output mode or response width.

The new reference suite covers 1,296 workflows / 3,888 cold/warm/cold fits and
76,464 required groups. Three seeds, regression/indicator targets, one/two/four
responses, four activations, three architectures and six optimizer configurations
cover minibatches, schedules and all solvers. Constant indicators/responses,
degenerate features, one-column targets, extreme classifier queries and
single-sample scoring are included. Coefficients, optimizer states, loss
histories, metadata, predictions, probabilities, log probabilities and scores
are compared at rtol=5e-8 / atol=5e-9. L-BFGS objective and supplied Jacobian
values are additionally compared at native returned parameters with tighter
tolerances. The checked development run required no rounding-envelope exception;
its largest coefficient error was 3.22e-15. Nineteen diagnostics cover empty,
nonfinite and malformed targets, prediction-rank mistakes, scores, warm target
mismatches and unfitted access.

Native analytic tests check a zero-parameter two-response SGD step (loss averages
responses but updates retain source sample-only normalization), uniform-average
R2, undefined one-row R2, independent sigmoid gradients, probability sums above
one, strict threshold ties, exact-row scoring and one-column compatibility.
The new matrix-target example fits two joint responses and overlapping labels.
Original contributions remain MIT with retained sklearn BSD attribution; no
additional dependency or public extension estimator was introduced.

Cumulative inventory: 221 public core types, 137 source modules, 174 native
programs, 150 core reference suites / 1,594,588 retained required groups, and 55
invalid-input registries / 3,961 cases. These remain API subsets.

Final checked (bounds/runtime checks and floating-point traps) and optimized
release builds both compiled the project successfully and passed the five
relevant MLP native programs selected with fpm test. Each build also passed
76,464 new target groups, 81,648 warm groups, 48,960 stochastic groups, 14,688
required L-BFGS groups and 79 invalid-input diagnostics: 221,760 required result
groups per build. The new target suite matched 2,882 nonfinite source values
and needed no rounding-envelope exception. Maximum coefficient errors were
3.22e-15 checked / 3.78e-15 release; supplied-Jacobian errors were at most
3.34e-16 / 3.06e-16. The existing numerical diagnostics retained their previous
counts and return-status outcomes.

The matrix-target example passed both builds, with joint-response R2 above
0.99999999, exact-row multilabel accuracy 1.0 and overlapping positive-label
probabilities above 0.999. The audit verified 151 relevant compiled source
contents per build, complete result-field inventories and source/helper/
validator/executable hashes. All unrelated core modules and existing native
test sources were unchanged. The full native inventory is 174 programs;
execution in this increment selected the five MLP programs. Reports and logs
remain locally under ignored validation/results.
