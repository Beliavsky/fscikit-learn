# Translation coverage

Reference baseline: scikit-learn 1.6.1 installed in
`c:/Programs/Python313/Lib/site-packages/sklearn`.

There are 147 public estimator, transformer, and splitter types. Each is a
**subset** of the corresponding Python API, not a feature-complete class port.
All types have native tests. Forty-seven reference suites additionally check 33,137
result groups across seeded datasets using installed scikit-learn 1.6.1 and
independent numerical identities and optimality certificates.

This document counts core scikit-learn counterparts. Public extensions are
tracked separately in [EXTENSIONS.md](EXTENSIONS.md) and excluded from the
`sklearn` facade and core coverage counts. The [scope policy](SCOPE.md) allows
selective, clearly identified extensions while prioritizing core correctness.

Global scope: dense real64 CPU arrays, integer classification labels, generally one target
per sample; cross-decomposition and multi-task regression explicitly accept multiple response columns. Sparse matrices, pandas integration, Python estimator cloning,
metadata routing, parallel backends and binary model persistence are outside the
initial translation. Fortran identifiers use snake_case rather than CamelCase.

## Implemented domains

| Module (`sklearn_` prefix) | Public types and functions | Scope and numerical choices |
| --- | --- | --- |
| preprocessing | `standard_scaler`, `minmax_scaler`, `maxabs_scaler`, `robust_scaler`, `simple_imputer`, `knn_imputer`, `missing_indicator`, `spline_transformer`, `label_encoder`, `one_hot_encoder`, `power_transformer`, `kbins_discretizer`, `quantile_transformer`; `normalize`, `binarize`, `polynomial_features` | Population variance; linear percentile interpolation; NaN imputation with mean/median/most-frequent/constant; numeric categories; empirical uniform/normal quantile transforms; dense polynomial terms through degree 12 with optional interactions; B-spline features and extrapolation |
| linear_model | `linear_regression`, `lars`, `lasso_lars`, `lars_cv`, `lasso_lars_cv`, `lasso_lars_ic`, `ridge`, `ridge_classifier`, `ridge_classifier_cv`, `multi_task_elastic_net`, `multi_task_lasso`, `lasso`, `elastic_net`, `lasso_cv`, `elastic_net_cv`, `logistic_regression`, `perceptron`, `passive_aggressive_classifier`, `passive_aggressive_regressor`, `sgd_regressor`, `sgd_classifier`, `sgd_one_class_svm`, `bayesian_ridge`, `ard_regression`, `orthogonal_matching_pursuit`, `huber_regressor`, `quantile_regressor`, `ransac_regressor`, `theil_sen_regressor`, `poisson_regressor`, `gamma_regressor`, `tweedie_regressor` | SVD least squares/ridge; least-angle/Lasso homotopy paths with CV and AIC/BIC selection; coordinate descent with K-fold alpha/l1-ratio selection; logistic Newton solver; perceptron; PA-I/II classification/regression with averaging; streaming stochastic-gradient regression/classification and one-class detection; Bayesian precision updates/uncertainty; ARD pruning; OMP; joint Huber coefficient/scale fitting; weighted L1-regularized quantile regression; RANSAC linear consensus; Theil-Sen subset/spatial-median fits; Newton GLMs |
| naive_bayes | `gaussian_nb`, `multinomial_nb`, `bernoulli_nb`, `complement_nb`, `categorical_nb` | Empirical class priors, variance smoothing, strictly positive scalar alpha; Bernoulli threshold; optional complement normalization |
| neighbors | `nearest_neighbors`, `kneighbors_classifier`, `kneighbors_regressor`, `radius_neighbors_classifier`, `radius_neighbors_regressor`, `nearest_centroid`; re-exports `local_outlier_factor`, `kernel_density` | Brute-force search; uniform/distance weights; radius queries/classification/regression; mean/median nearest centroids with optional shrinkage and uniform priors; Euclidean kernel density |
| cluster | `kmeans`, `bisecting_kmeans`, `mini_batch_kmeans`, `dbscan`, `agglomerative_clustering`, `feature_agglomeration`, `mean_shift`, `affinity_propagation`, `spectral_clustering`, `optics`, `birch`; `estimate_bandwidth`, `cluster_optics_dbscan`, `cluster_optics_xi` | Lloyd/k-means++; weighted streaming mini-batch centroids; bisecting centroid hierarchy with Lloyd/Elkan splits; density expansion; single/complete/average/Ward merge trees and feature pooling; flat-kernel mode seeking; damped exemplar messages; normalized-Laplacian clustering with k-means or pivoted QR assignment; OPTICS reachability ordering, Xi hierarchy and DBSCAN extraction; BIRCH streaming CF tree and optional Ward clustering |
| decomposition | `pca`, `incremental_pca`, `truncated_svd`, `nmf`, `sparse_pca`, `dictionary_learning`, `sparse_coder`; `sparse_encode`; re-exports `kernel_pca`, `factor_analysis`, `fast_ica` | Direct Jacobi SVD; PCA whitening and inversion; streaming stacked-SVD IncrementalPCA; exact dense SVD; NMF multiplicative updates; sparse dictionary factorization and SparsePCA; five sparse coding methods; nonlinear kernel PCA; unrotated full-SVD factor analysis; parallel/deflation ICA |
| cross_decomposition | `pls_regression`, `pls_canonical`, `cca`, `pls_svd` | Paired-matrix latent components: NIPALS regression/canonical deflation, CCA pseudoinverse mode, canonical direct SVD and PLSSVD cross-covariance projection; multiple response columns |
| isotonic | `isotonic_regression`; `isotonic_fit`, `check_increasing` | Weighted pool-adjacent-violators, automatic Spearman direction, bounds, repeated-input aggregation and piecewise linear interpolation |
| kernel_approximation | `rbf_sampler`, `nystroem`, `additive_chi2_sampler`, `skewed_chi2_sampler` | Random Fourier RBF maps; sampled-landmark inverse-square-root maps for RBF/linear/polynomial/cosine kernels; additive and skewed chi2 feature maps |
| random_projection | `gaussian_random_projection`, `sparse_random_projection`; `johnson_lindenstrauss_min_dim` | Gaussian/three-point random matrices, dense storage, automatic JL dimension, cached or on-demand pseudoinverse |
| kernel_pca | `kernel_pca` | Centered linear/RBF/polynomial/sigmoid/cosine/precomputed Gram matrices; dense Jacobi eigensolver; out-of-sample transform |
| manifold | `isomap`; re-exports `spectral_embedding`, `mds`, `locally_linear_embedding` | Isomap graph paths; spectral embedding; metric/nonmetric SMACOF; dense standard LLE and LTSA with barycentric query transforms |
| spectral | `spectral_embedding` | Dense symmetric eigensolver; RBF, symmetrized nearest-neighbor or precomputed affinity; degree normalization; connected-component count |
| density | `kernel_density` | Exact Euclidean log densities with Gaussian/tophat/Epanechnikov/exponential/linear kernels; sample weights; fixed/Scott/Silverman bandwidth; Gaussian/tophat sampling |
| outlier | `local_outlier_factor` | Training outlier scores and novelty detection, automatic/percentile threshold, Minkowski and other supported pairwise metrics |
| semi_supervised | `label_propagation`, `label_spreading` | Dense RBF or directed k-neighbor graph, hard/soft clamping, transductive labels and inductive probabilities |
| tree | `decision_tree_classifier`, `decision_tree_regressor` | Exhaustive CART; Gini/entropy/log-loss classification, squared-error regression; depth/leaf/split/feature controls, normalized impurity importance, leaf indices |
| ensemble | `random_forest_classifier`, `random_forest_regressor`, `extra_trees_classifier`, `extra_trees_regressor`, `gradient_boosting_regressor`, `gradient_boosting_classifier`, `adaboost_classifier`, `adaboost_regressor`, `isolation_forest` | Bootstrap forests, randomized feature subsets, random-threshold Extra Trees, squared-error residual boosting; binary/multiclass classification boosting with Newton leaf updates; weighted Gini-stump SAMME; AdaBoost.R2 with depth-controlled CART; random isolation trees and path-length anomaly scores |
| model_selection | `kfold`, `stratified_kfold`, `time_series_split`, `group_kfold`, `leave_one_out`, `leave_one_group_out`, `predefined_split`, `ridge_cv`; `train_test_split`, `shuffle_indices`, `cross_val_score` | Index-based splits; fraction test size; optional stratification; time gap/window; CV for linear-regression family and logistic regression; ridge CV selects negative mean squared error |
| feature_selection | `variance_threshold`, `select_k_best`, `sequential_feature_selector`, `rfe`, `rfecv`; `f_classif`, `f_regression`, `chi2` | Dense scores and masks; greedy forward/backward CV regression selection; coefficient-based recursive elimination and CV for regression/logistic classification; inverse transform fills dropped columns with zero; no p-values |
| feature_extraction_text | `count_vectorizer`, `tfidf_transformer`, `tfidf_vectorizer` | Dense unigram counts; ASCII alphanumeric/underscore tokens of length 2-128; sorted vocabulary; min/max document frequency, binary counts and max features; smoothed/unsmoothed IDF, sublinear TF, l1/l2/no normalization |
| dummy | `dummy_classifier`, `dummy_regressor` | Prior/most-frequent/stratified/uniform/constant classification; mean/median/quantile/constant regression |
| kernel_ridge | `kernel_ridge` | Positive alpha; linear/RBF/polynomial/sigmoid kernels; no intercept |
| discriminant_analysis | `linear_discriminant_analysis`, `quadratic_discriminant_analysis` | LDA follows sklearn `solver='lsqr'` covariance convention with scalar shrinkage; QDA scalar regularization; singular covariance is rejected |
| svm | `svc`, `nu_svc`, `linear_svc`, `linear_svr`; re-exports `svr`, `nu_svr`, `one_class_svm` | Primal L2 squared-loss linear classification/regression; OVO SVC/NuSVC; epsilon/nu support-vector regression and one-class detection; native pairwise dual optimization and weighted capacities |
| mixture | `gaussian_mixture` | EM with k-means initialization, full/tied/diag/spherical covariances; posterior, log likelihood, BIC/AIC |
| neural_network | `mlp_classifier`, `mlp_regressor` | Arbitrary dense hidden layer sizes, relu/tanh/logistic/identity, full-batch Adam, softmax classification or squared-error regression, L2 regularization |
| gaussian_process | `gaussian_process_regressor`, `gaussian_process_classifier` | Fixed isotropic RBF amplitude/length scale plus white noise and diagonal alpha; Cholesky posterior, standard deviation/covariance, log marginal likelihood; optional target normalization; fixed-RBF Laplace binary/OVR classification |
| covariance | `empirical_covariance`, `shrunk_covariance`, `ledoit_wolf`, `oas`, `graphical_lasso`, `min_cov_det`, `elliptic_envelope`; `fast_mcd` | Population covariance, pseudoinverse precision, squared Mahalanobis distance, likelihood, Frobenius/spectral covariance error; L1-penalized sparse precision via block coordinate descent; FastMCD robust covariance and elliptic outlier detection |
| datasets | `make_blobs`, `make_regression`, `make_moons`, `make_circles` | Native synthetic generators; optional deterministic seed; no bundled real-world datasets |

## Metrics and kernels

Regression: mean/root-mean squared error, mean/median/max absolute error,
mean absolute percentage error, mean squared log error, R2, explained variance,
Poisson and Gamma deviances.

Classification: accuracy, confusion matrix, precision, recall, F1, Jaccard,
Hamming loss, balanced accuracy, Matthews correlation, Cohen kappa, log loss,
binary ROC AUC and average precision. Precision/recall/F1/Jaccard support binary,
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
- KernelDensity is an exact dense Euclidean mixture, with no tree backend,
  approximation tolerances or non-Euclidean metrics. Cosine kernels are not yet
  implemented. Zero density under compact kernels returns negative infinity,
  not a finite floor. Its `score` sums log densities. Sample weights must be
  nonnegative with positive total. Bandwidth rules use sample count and feature
  dimension as in sklearn, without additional covariance scaling. Sampling is
  available only for Gaussian/tophat kernels and uses the Fortran RNG.

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
  output, coefficient path or cross-validated OMP estimator.
- FactorAnalysis supports the unrotated full-SVD algorithm (`svd_method='lapack'`
  in sklearn), not randomized SVD or varimax/quartimax rotation. Requested
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
  (possibly regularized) class covariance. Gaussian NB rejects all-zero
  variance when smoothing cannot make it positive.
- SVC exposes one decision column per class pair, positive toward the pair's
  second class. This differs from sklearn's multiclass decision conventions.
  SVR, NuSVC and one-class SVM are separate native types. Platt probability
  scaling and LIBSVM model-file compatibility are not provided.
- MLP uses full-batch Adam, not sklearn's default minibatches. It has no
  validation-set early stopping, SGD/L-BFGS solver, warm start, or multilabel
  outputs. Random initialization and convergence history differ from Python.
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
- Nearest centroid supports uniform priors only, Euclidean means or Manhattan
  medians and scalar shrinkage. It requires more samples than classes and has
  no discriminant probability API or empirical/custom priors. Zero within-class
  scale uses zero deviation rather than producing NaN in that attribute.
- LOF uses `contamination=-1` for sklearn's `'auto'`, otherwise a fraction in
  `(0,0.5]`. `fit_predict` requires `novelty=false`; query scoring/prediction
  requires `novelty=true`. Training scores exclude each point itself; novelty
  queries include matching training points. Neighbor-count clamping is reported
  in `n_neighbors_`; warnings are not emitted. Duplicate/tied-neighbor ordering
  can differ from sklearn's search backend.
- Kernel PCA has no inverse transform, callable kernels or sparse/randomized/
  ARPACK solver. `n_components=0` means all nonzero components; other counts are
  clipped to sample count. Significant selected negative eigenvalues are
  rejected. Repeated eigenvalues can yield different valid bases.
- Isomap supports fixed-count neighbors and ordinary feature inputs only,
  with quadratic storage and cubic shortest-path work. No radius graph,
  precomputed distances or alternative path/eigen solver. Disconnected graph
  completion is reflected in `n_connected_components_` rather than a warning.
- Semi-supervised labels reserve `-1` for unlabeled samples and require at least
  one labeled sample. RBF and KNN kernels are supported, not callable kernels.
  Zero-evidence inductive queries are rejected instead of returning NaN
  probabilities. Unreachable training rows remain zero and transduce to the
  first class, as in sklearn. Inspect `converged_` and `n_iter_`; reaching the
  iteration cap does not emit a Python-style warning.

## Remaining major scikit-learn areas

Not translated: sparse data paths, general estimator base/clone/parameter search
framework, pipelines/column transformers, broader SVM/GLM/SGD options,
histogram boosting, broader AdaBoost and bagging APIs,
t-SNE and Hessian/modified LLE,
outlier detection beyond LOF/OneClassSVM/SGDOneClassSVM/IsolationForest/EllipticEnvelope,
calibration, semi-supervised learning beyond label propagation/spreading,
multiclass/multioutput wrappers,
inspection/permutation importance, further feature-selection wrappers, image features,
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

LLE supports `method='standard'` (default) and `method='ltsa'`, with a native
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
LLE and is zero for LTSA, whose objective uses local tangent projectors. LTSA
uses regularization only in query transforms. Queries use the fitted neighbors
count and barycentric projection of training coordinates; a query equal to a
training point can have a different neighbor set from the training fit.
Neighbor ties, disconnected graphs, unreferenced LTSA observations, or repeated
alignment eigenvalues can make truncated embeddings nonunique across solvers.
No Hessian/modified methods, ARPACK, sparse graph path or inverse transform is
implemented.

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
group soft thresholding. Feature order is cyclic. Each intercept is an
unpenalized response mean minus its coefficient-weighted feature means.

Both provide fit, predict and uniform-average R2 score. coef_ has shape
(n_targets,n_features), intercept_ has n_targets entries, and predictions
have shape (n_samples,n_targets). Controls are alpha (default 1), fit_intercept
(true), max_iter (1000), tol (1e-4), and warm_start (false). ElasticNet's mixing
ratio defaults to .5. This port requires finite alpha>0 and 0<l1_ratio<=1;
pure ridge and zero-penalty least squares are handled by existing estimators.
Tolerance must be finite and nonnegative. Targets and features must be finite.

Stopping first checks the largest coordinate update relative to the largest
coefficient, then checks the primal-dual gap against tol*||centered Y||_F^2.
The gap is also checked at the iteration cap. n_iter_, dual_gap_ (divided by
n_samples), eps_ (unscaled stopping threshold), n_features_in_, and native
converged_ expose the result. An unconverged fit returns its final iterate
with converged_=false. Warm starts reuse coefficients with matching target and
feature dimensions; otherwise a fit starts at zero. Input arrays are not
modified. Zero-norm centered features are skipped, matching the reference.

Sparse matrices, sample weights, random coordinate selection, positivity
constraints, regularization paths and the CV variants are not implemented.
The types are re-exported through sklearn_linear_model and sklearn. The
package now has 109 core types in 44 source modules. No runtime dependency or
public extension was added; MIT and sklearn BSD attribution are retained.


## Categorical Naive Bayes

`categorical_nb` fits a separate discrete probability table for each input
feature, with Laplace/Lidstone smoothing controlled by finite alpha>0 (default
1). Input is a dense real64 matrix containing nonnegative integer category
codes. Categories start at zero; table column one represents category zero.
This port rejects fractional codes rather than silently truncating them.
Classification labels are arbitrary integers, sorted in classes_. Single-class
fits are supported. Sample weights must be finite and nonnegative, with positive
cumulative total. Zero-weight rows still determine the supported category range.

fit resets counters. partial_fit requires the full classes vector on its first
call; later calls may omit it or supply the same set. New categories expand
counters without discarding earlier data. Unknown class labels and changed
feature counts are rejected. n_categories_ records the retained table widths
(the largest observed/reserved range over all batches), even when a later
batch has smaller codes. This makes incremental metadata reflect the retained
tables rather than sklearn 1.6.1's latest-batch width metadata.

min_categories=0 infers widths from data. A positive scalar reserves at least
that many categories for every feature; optional min_categories_per_feature
reserves positive per-feature minima. If both are supplied their maximum is
used. Codes missing inside the fitted/reserved range receive smoothed mass;
query codes outside it are rejected. Category tables use dense storage, so
codes should be compact consecutive integers rather than large identifiers.

fit_prior=true learns empirical weighted class priors; false uses uniform
priors. Optional class_prior supplies a nonnegative finite vector aligned to
classes_, summing to one within 1e-8. Zero priors are represented by negative
infinity in class_log_prior_ and yield zero class probability. An unobserved
class supplied to partial_fit likewise has zero empirical prior until observed.

Methods are fit, partial_fit, joint_log_likelihood, predict_log_proba,
predict_proba, predict and score (accuracy). Fitted fields include classes_,
class_count_, class_log_prior_, n_features_in_ and n_categories_. Ragged tables
category_count_(feature)%values and feature_log_prob_(feature)%values each have
shape (n_classes,n_categories_for_feature), using the public categorical_table
storage type. This helper is not counted as an estimator. Log probabilities
use log-sum-exp normalization; class ties choose the first sorted label.

Sparse data, string categories, automatic encoding, vector smoothing alpha and
alpha=0/force_alpha behavior are not implemented. The type is re-exported
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

`adaboost_classifier` implements sklearn 1.6.1 AdaBoostClassifier's discrete
SAMME algorithm with native weighted depth-one Gini trees. This port fixes the
weak learner to decision stumps: no custom estimator, deeper trees, SAMME.R,
random feature subsets or sparse input is implemented. At least two distinct
integer labels are required; classes_ is sorted. n_estimators defaults to 50
and learning_rate to 1; both must be positive and the rate finite. fit accepts
optional finite nonnegative sample_weight with positive total.

Each round fits a weighted Gini stump and measures weighted classification
error. Accepted weights are learning_rate*(log((1-error)/error)+log(K-1)).
A perfect learner gets weight 1 and terminates, independent of learning_rate.
A learner at or above random-guessing error 1-1/K is discarded; if it is the
first learner, fitting fails. Original zero-weight rows stay excluded. Positive
weights are clipped at machine epsilon each round as in sklearn. Weight updates
use normalized log weights for overflow resistance; extremely small positive
weights can underflow to zero during normalization and are clipped next round.

Stumps consider all features and midpoint thresholds on positive-weight rows.
Features are converted to float32 as in sklearn trees, so inputs beyond float32
range are rejected. Adjacent values within 1e-7 are treated as equal. Equal or
roundoff-level tied split gains choose the earliest scanned feature/threshold,
while sklearn randomizes feature order; geometrically tied fits can differ.
Leaf vote ties choose the smallest sorted label. No RNG parameter is needed
for the native deterministic scan.

Methods are fit, decision_function, predict, predict_proba, score and three
staged methods (staged_decision_function, staged_predict, staged_predict_proba).
Binary decisions have shape (n,1), with twice the normalized signed weighted
vote, matching sklearn 1.6.1. Positive scores select the larger label, zero the
smaller. Multiclass decisions have shape (n,K): each learner contributes +weight
to its class and -weight/(K-1) to other classes, divided by total learner weight.
Probabilities use sklearn's SAMME softmax conversion; they are not calibrated.
score is accuracy. Staged decision/probability results have shape
(n,channels,n_estimators_) and (n,K,n_estimators_); staged predictions are
(n,n_estimators_). These are materialized arrays rather than Python generators.

n_estimators_ is the number of retained learners. estimator_weights_ and
estimator_errors_ retain requested n_estimators length, with unused entries 0
and 1 respectively. stump_feature_ uses one-based feature indices (0 for a
constant tree), stump_threshold_ contains float32-based split midpoints, and
stump_left_/stump_right_ contain original integer labels. Only their first
n_estimators_ entries are fitted. feature_importances_ is the learner-weighted
mean of normalized stump importances; constant stumps contribute zero.
n_features_in_ and n_classes_ record fitted dimensions. Repeated fit resets
all learned arrays. Exported through sklearn_ensemble and sklearn.

Scope: dense finite real64 matrices with float32 tree comparisons; no class
weights, custom weak learners, log-probability API or regression boosting in
this type. Staged computations prioritize clarity over streaming efficiency.
Original contributions remain MIT with sklearn BSD attribution. Inventory:
121 core types, 51 source modules and 51 native tests. No public extension or
runtime dependency was added.


## Stochastic-gradient regression

`sgd_regressor` implements dense single-target SGDRegressor from sklearn 1.6.1,
with fit, partial_fit, predict and R2 score. Supported losses are squared_error
(default, half squared residual), huber, epsilon_insensitive and
squared_epsilon_insensitive. epsilon defaults to .1 and must be finite and
nonnegative. Huber uses the epsilon transition and clips its residual gradient;
the epsilon-insensitive losses use the corresponding dead zone.

penalty is 'l2' by default or 'none'; alpha defaults to 1e-4 and is finite and
nonnegative. The unpenalized intercept is optional (fit_intercept=true). Each
row computes a pre-update gradient, clipped to [-1e12,1e12], applies L2 decay
max(0,1-eta*alpha), then adds -eta*gradient*sample_weight times the row. Optional
sample weights must be finite/nonnegative and match row count. Zero-weight
rows still decay L2 coefficients, contribute to the unweighted stopping loss,
and advance the sample clock; all-zero weights are allowed as in sklearn.

learning_rate supports 'invscaling' (default) and 'constant'. eta0 defaults .01;
power_t defaults .25 and must be nonnegative. Inverse scaling uses
eta0/t_^power_t. t_ starts at 1 and increments once per row, including across
partial_fit calls. fit always resets coefficient, intercept, averaging state
and the clock. partial_fit initializes on its first batch, runs exactly one
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
reset it. n_iter_no_change defaults to 5. Reaching patience sets converged_=true;
exhausting max_iter leaves it false. partial_fit ignores stopping and leaves
converged_ false. The loss comparison is strict, including at tol=0.

shuffle defaults false in this port (sklearn defaults true) to provide a
deterministic ordered update stream. When enabled, native Fisher-Yates shuffling
uses random_state and the package RNG. Each fit/partial_fit call seeds that
native stream; it differs from NumPy/sklearn shuffles. Seeded native repeats
are tested, while exact sklearn comparisons use shuffle=False.

Scope: finite dense real64 matrices, one numeric target; no sparse data, L1/
elastic-net penalties, optimal/adaptive rate schedules, warm-start fit, custom
initial coefficients, validation-set early stopping or metadata routing.
The source follows sklearn's stochastic gradient and loss contracts; original
contributions remain MIT with retained BSD attribution. Exported through
sklearn_linear_model and sklearn. Inventory: 122 core types, 52 modules and
52 native test programs. No dependency or public extension was added.


## Stochastic-gradient classification

`sgd_classifier` implements sklearn 1.6.1 SGDClassifier's binary/one-vs-rest
classification contracts using the shared native SGD engine. Supported losses
are hinge (default), squared_hinge, perceptron, log_loss and modified_huber.
Methods are fit, partial_fit, decision_function, predict, predict_proba and
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

Shared controls follow SGDRegressor: penalty='l2' or 'none', finite alpha>=0,
constant/invscaling rates, eta0>0, power_t>=0, optional unpenalized intercept,
max_iter, tol=-1 to disable stopping, training-loss patience and integer
average thresholds. Defaults are hinge, alpha=1e-4, eta0=.01, invscaling,
power_t=.25, tol=1e-3, max_iter=1000, n_iter_no_change=5, average=0 and
shuffle=false. These ordered inverse-scaling defaults differ from sklearn's
shuffle=true, optimal rate, power_t=.5 and eta0=0 defaults. Native seeded shuffles are
reproducible but do not reproduce NumPy/sklearn permutations.

Scope: finite dense real64 arrays, integer labels, five classification losses;
no L1/elastic-net, optimal/adaptive schedules, regression losses for classifiers,
custom coefficient initialization, warm-start fit, validation-set early stopping,
log-probability API or parallel OVR. Exported through sklearn_linear_model and
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
