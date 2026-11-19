# Translation coverage

Reference baseline: scikit-learn 1.6.1 installed in
`c:/Programs/Python313/Lib/site-packages/sklearn`.

There are 105 public estimator, transformer, and splitter types. Each is a
**subset** of the corresponding Python API, not a feature-complete class port.
All types have native tests. Twenty reference suites additionally check 5,732 result
groups in total across seeded datasets against installed scikit-learn 1.6.1.

This document counts core scikit-learn counterparts. Public extensions are
tracked separately in [EXTENSIONS.md](EXTENSIONS.md) and excluded from the
`sklearn` facade and core coverage counts. The [scope policy](SCOPE.md) allows
selective, clearly identified extensions while prioritizing core correctness.

Global scope: dense real64 CPU arrays, integer classification labels, generally one target
per sample; cross-decomposition explicitly accepts multiple response columns. Sparse matrices, pandas integration, Python estimator cloning,
metadata routing, parallel backends and binary model persistence are outside the
initial translation. Fortran identifiers use snake_case rather than CamelCase.

## Implemented domains

| Module (`sklearn_` prefix) | Public types and functions | Scope and numerical choices |
| --- | --- | --- |
| preprocessing | `standard_scaler`, `minmax_scaler`, `maxabs_scaler`, `robust_scaler`, `simple_imputer`, `label_encoder`, `one_hot_encoder`, `power_transformer`, `kbins_discretizer`, `quantile_transformer`; `normalize`, `binarize`, `polynomial_features` | Population variance; linear percentile interpolation; NaN imputation with mean/median/most-frequent/constant; numeric categories; empirical uniform/normal quantile transforms; dense polynomial terms through degree 12 with optional interactions |
| linear_model | `linear_regression`, `ridge`, `lasso`, `elastic_net`, `logistic_regression`, `perceptron`, `passive_aggressive_classifier`, `passive_aggressive_regressor`, `bayesian_ridge`, `ard_regression`, `orthogonal_matching_pursuit`, `huber_regressor`, `ransac_regressor`, `theil_sen_regressor`, `poisson_regressor`, `gamma_regressor`, `tweedie_regressor` | SVD least squares/ridge; coordinate descent; logistic Newton solver; perceptron; PA-I/II classification/regression with averaging; Bayesian precision updates/uncertainty; ARD pruning; OMP; joint Huber coefficient/scale fitting; RANSAC linear consensus; Theil-Sen subset/spatial-median fits; Newton GLMs |
| naive_bayes | `gaussian_nb`, `multinomial_nb`, `bernoulli_nb`, `complement_nb` | Empirical class priors, variance smoothing, strictly positive scalar alpha; Bernoulli threshold; optional complement normalization |
| neighbors | `nearest_neighbors`, `kneighbors_classifier`, `kneighbors_regressor`, `radius_neighbors_classifier`, `radius_neighbors_regressor`, `nearest_centroid`; re-exports `local_outlier_factor`, `kernel_density` | Brute-force search; uniform/distance weights; radius queries/classification/regression; mean/median nearest centroids with optional shrinkage and uniform priors; Euclidean kernel density |
| cluster | `kmeans`, `dbscan`, `agglomerative_clustering`, `mean_shift`, `affinity_propagation`, `spectral_clustering`, `optics`, `birch`; `estimate_bandwidth`, `cluster_optics_dbscan`, `cluster_optics_xi` | Lloyd/k-means++; density expansion; single/complete/average/Ward merge trees; flat-kernel mode seeking; damped exemplar messages; normalized-Laplacian clustering with k-means or pivoted QR assignment; OPTICS reachability ordering, Xi hierarchy and DBSCAN extraction; BIRCH streaming CF tree and optional Ward clustering |
| decomposition | `pca`, `truncated_svd`, `nmf`; re-exports `kernel_pca`, `factor_analysis`, `fast_ica` | Direct Jacobi SVD; PCA whitening and inversion; exact dense SVD; NMF multiplicative updates; nonlinear kernel PCA; unrotated full-SVD factor analysis; parallel/deflation ICA |
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
| ensemble | `random_forest_classifier`, `random_forest_regressor`, `extra_trees_classifier`, `extra_trees_regressor`, `gradient_boosting_regressor`, `gradient_boosting_classifier` | Bootstrap forests, randomized feature subsets, random-threshold Extra Trees, squared-error residual boosting; binary/multiclass classification boosting with Newton leaf updates |
| model_selection | `kfold`, `stratified_kfold`, `time_series_split`, `ridge_cv`; `train_test_split`, `shuffle_indices`, `cross_val_score` | Index-based splits; fraction test size; optional stratification; time gap/window; CV for linear-regression family and logistic regression; ridge CV selects negative mean squared error |
| feature_selection | `variance_threshold`, `select_k_best`; `f_classif`, `f_regression`, `chi2` | Dense scores and masks; inverse transform fills dropped columns with zero; no p-values |
| feature_extraction_text | `count_vectorizer`, `tfidf_transformer`, `tfidf_vectorizer` | Dense unigram counts; ASCII alphanumeric/underscore tokens of length 2-128; sorted vocabulary; min/max document frequency, binary counts and max features; smoothed/unsmoothed IDF, sublinear TF, l1/l2/no normalization |
| dummy | `dummy_classifier`, `dummy_regressor` | Prior/most-frequent/stratified/uniform/constant classification; mean/median/quantile/constant regression |
| kernel_ridge | `kernel_ridge` | Positive alpha; linear/RBF/polynomial/sigmoid kernels; no intercept |
| discriminant_analysis | `linear_discriminant_analysis`, `quadratic_discriminant_analysis` | LDA follows sklearn `solver='lsqr'` covariance convention with scalar shrinkage; QDA scalar regularization; singular covariance is rejected |
| svm | `svc`; re-exports `svr`, `nu_svr`, `one_class_svm` | OVO SVC; epsilon/nu support-vector regression and one-class detection; native pairwise dual optimization, weighted capacities and PSD kernels for the added models |
| mixture | `gaussian_mixture` | EM with k-means initialization, full/tied/diag/spherical covariances; posterior, log likelihood, BIC/AIC |
| neural_network | `mlp_classifier`, `mlp_regressor` | Arbitrary dense hidden layer sizes, relu/tanh/logistic/identity, full-batch Adam, softmax classification or squared-error regression, L2 regularization |
| gaussian_process | `gaussian_process_regressor` | Fixed isotropic RBF amplitude/length scale plus white noise and diagonal alpha; Cholesky posterior, standard deviation/covariance, log marginal likelihood; optional target normalization |
| covariance | `empirical_covariance`, `shrunk_covariance`, `ledoit_wolf`, `oas`, `graphical_lasso` | Population covariance, pseudoinverse precision, squared Mahalanobis distance, likelihood, Frobenius/spectral covariance error; L1-penalized sparse precision via block coordinate descent |
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
  There is no SVR, NuSVC, one-class SVM, Platt scaling, or libsvm compatibility.
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
framework, pipelines/column transformers, NuSVC and linear SVM variants, quantile
regression, remaining GLM variants, remaining SGD families, histogram boosting, AdaBoost and bagging APIs,
t-SNE and Hessian/modified LLE,
outlier detection beyond LOF/OneClassSVM,
Gaussian-process classification, dictionary learning,
calibration, semi-supervised learning beyond label propagation/spreading,
multiclass/multioutput wrappers,
inspection/permutation importance, feature-selection wrappers, image features,
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
