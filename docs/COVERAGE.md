# Translation coverage

Reference baseline: scikit-learn 1.6.1 installed in
`c:/Programs/Python313/Lib/site-packages/sklearn`.

There are 56 public estimator, transformer, and splitter types. Each is a
**subset** of the corresponding Python API, not a feature-complete class port.
All types have native tests. The reference suite additionally checks 64 result
groups on each of three datasets against installed scikit-learn 1.6.1.

Global scope: dense real64 CPU arrays, integer classification labels, one target
per sample. Sparse matrices, pandas integration, Python estimator cloning,
metadata routing, parallel backends and binary model persistence are outside the
initial translation. Fortran identifiers use snake_case rather than CamelCase.

## Implemented domains

| Module (`sklearn_` prefix) | Public types and functions | Scope and numerical choices |
| --- | --- | --- |
| preprocessing | `standard_scaler`, `minmax_scaler`, `maxabs_scaler`, `robust_scaler`, `simple_imputer`, `label_encoder`, `one_hot_encoder`; `normalize`, `binarize`, `polynomial_features` | Population variance; linear percentile interpolation; NaN imputation with mean/median/most-frequent/constant; numeric categories; dense polynomial terms through degree 12 with optional interactions |
| linear_model | `linear_regression`, `ridge`, `lasso`, `elastic_net`, `logistic_regression`, `perceptron` | Direct Jacobi SVD minimum-norm least squares/ridge; cyclic coordinate descent; L2 binary/multinomial Newton logistic solver with backtracking; deterministic one-vs-rest perceptron |
| naive_bayes | `gaussian_nb`, `multinomial_nb`, `bernoulli_nb`, `complement_nb` | Empirical class priors, variance smoothing, strictly positive scalar alpha; Bernoulli threshold; optional complement normalization |
| neighbors | `nearest_neighbors`, `kneighbors_classifier`, `kneighbors_regressor` | Brute-force search; uniform/distance weights; radius adjacency and dense k-neighbor graphs; exact-match distance weighting |
| cluster | `kmeans`, `dbscan`, `agglomerative_clustering` | Lloyd iterations, k-means++ initialization or explicit centers; density expansion; full single/complete/average/Ward merge trees |
| decomposition | `pca`, `truncated_svd`, `nmf` | Direct Jacobi SVD; PCA whitening and inversion; exact dense SVD instead of randomized backend; nonnegative multiplicative updates, Frobenius objective |
| tree | `decision_tree_classifier`, `decision_tree_regressor` | Exhaustive CART; Gini/entropy/log-loss classification, squared-error regression; depth/leaf/split/feature controls, normalized impurity importance, leaf indices |
| ensemble | `random_forest_classifier`, `random_forest_regressor`, `extra_trees_classifier`, `extra_trees_regressor`, `gradient_boosting_regressor` | Bootstrap forests, randomized feature subsets, random-threshold Extra Trees, squared-error residual boosting |
| model_selection | `kfold`, `stratified_kfold`, `time_series_split`, `ridge_cv`; `train_test_split`, `shuffle_indices`, `cross_val_score` | Index-based splits; fraction test size; optional stratification; time gap/window; CV for linear-regression family and logistic regression; ridge CV selects negative mean squared error |
| feature_selection | `variance_threshold`, `select_k_best`; `f_classif`, `f_regression`, `chi2` | Dense scores and masks; inverse transform fills dropped columns with zero; no p-values |
| feature_extraction_text | `count_vectorizer`, `tfidf_transformer`, `tfidf_vectorizer` | Dense unigram counts; ASCII alphanumeric/underscore tokens of length 2-128; sorted vocabulary; min/max document frequency, binary counts and max features; smoothed/unsmoothed IDF, sublinear TF, l1/l2/no normalization |
| dummy | `dummy_classifier`, `dummy_regressor` | Prior/most-frequent/stratified/uniform/constant classification; mean/median/quantile/constant regression |
| kernel_ridge | `kernel_ridge` | Positive alpha; linear/RBF/polynomial/sigmoid kernels; no intercept |
| discriminant_analysis | `linear_discriminant_analysis`, `quadratic_discriminant_analysis` | LDA follows sklearn `solver='lsqr'` covariance convention with scalar shrinkage; QDA scalar regularization; singular covariance is rejected |
| svm | `svc` | Deterministic two-coordinate SMO, one-vs-one multiclass voting, linear/RBF/polynomial/sigmoid kernels; no probability calibration |
| mixture | `gaussian_mixture` | EM with k-means initialization, full/tied/diag/spherical covariances; posterior, log likelihood, BIC/AIC |
| neural_network | `mlp_classifier`, `mlp_regressor` | Arbitrary dense hidden layer sizes, relu/tanh/logistic/identity, full-batch Adam, softmax classification or squared-error regression, L2 regularization |
| gaussian_process | `gaussian_process_regressor` | Fixed isotropic RBF amplitude/length scale plus white noise and diagonal alpha; Cholesky posterior, standard deviation/covariance, log marginal likelihood; optional target normalization |
| covariance | `empirical_covariance`, `shrunk_covariance`, `ledoit_wolf`, `oas` | Population covariance, pseudoinverse precision, squared Mahalanobis distance, likelihood, Frobenius/spectral covariance error |
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

## Remaining major scikit-learn areas

Not translated: sparse data paths, general estimator base/clone/parameter search
framework, pipelines/column transformers, full SVM family, robust/Bayesian/GLM
linear estimators, SGD families, histogram boosting, AdaBoost and bagging APIs,
manifold learning, spectral clustering, OPTICS/BIRCH/mean shift, outlier detection,
Gaussian-process classification, dictionary learning/ICA/factor analysis,
calibration, semi-supervised learning, multiclass/multioutput wrappers,
inspection/permutation importance, feature-selection wrappers, image features,
real dataset loaders and the full plotting/display API.

SciFort is available at `../scifort` but not linked by this initial package.
An optional SciFort/LAPACK backend and recoverable errors would be useful next
steps before scaling this package to larger production workloads.
