# API conventions and recipes

## Package use

`use sklearn` exposes the core scikit-learn algorithm surface. Any future
public extension must be imported explicitly from a `fsklearn_extra_*` module
and identified in [EXTENSIONS.md](EXTENSIONS.md). See [SCOPE.md](SCOPE.md) for
the distinction between extensions and internal implementation improvements.

From another fpm project, add a path dependency (relative to that project's
manifest):

```toml
[dependencies]
fscikit-learn = { path = "../fscikit-learn" }
```

Use `use sklearn` for the complete public surface, or a focused module such as
`use sklearn_linear_model, only: ridge`. The kind constant `dp` is exported by
the facade and `sklearn_common`. It is `iso_fortran_env`'s `real64`.

An estimator's constructor defaults can be changed by assigning public fields
before fitting. Learned attributes have trailing underscores. `fit` is a
subroutine; `predict`, `transform`, `predict_proba` and `score` are functions.
Most outputs are allocatable arrays that are allocated by ordinary assignment.
There is no Python runtime or external library required to use the package.

All feature arrays have shape `(n_samples,n_features)`. A one-feature input is
still rank two, for example `reshape(values,[size(values),1])`. A regressor's
target/prediction is a rank-one real array. Classification labels/predictions are
rank-one integer arrays. Probability columns follow the fitted `classes_` array.
Regressor `score` is R2; classifier `score` is accuracy. Classification labels may
be arbitrary integers; they do not need to start at zero.

There is no estimator polymorphic base class shared across all domains. Code can
compose preprocessing and estimators with ordinary Fortran calls. Model-selection
`cross_val_score` overloads currently cover `linear_regression` and its subclasses,
and `logistic_regression` only. They copy the estimator, leaving its fitted state
unchanged, but do not build preprocessing pipelines automatically.

## Regression with a held-out set

The full program is in `example/regression.f90`:

```fortran
type(standard_scaler) :: scaler
type(ridge_cv) :: model
real(dp), allocatable :: x(:,:), y(:), train_x(:,:), test_x(:,:), pred(:)
integer, allocatable :: train(:), test(:)

call make_regression(200,6,x,y,noise=2._dp,random_state=42)
call train_test_split(size(y),train,test,test_size=0.2_dp,random_state=7)
call scaler%fit(x(train,:))
train_x=scaler%transform(x(train,:))
test_x=scaler%transform(x(test,:))
model%alphas=[0.01_dp,0.1_dp,1._dp,10._dp]
call model%fit(train_x,y(train))
pred=model%predict(test_x)
print *, 'Selected alpha: ',model%alpha_
print *, 'Test R2: ',r2_score(y(test),pred)
```

The scaler here is fitted before the internal ridge CV. To avoid information
leakage when using CV to estimate the complete preprocessing procedure, fit a
fresh scaler **inside each fold**. The held-out test set above remains separate.

## Nearest neighbors

```fortran
type(nearest_neighbors) :: search
real(dp), allocatable :: distances(:,:)
integer, allocatable :: indices(:,:)
search%n_neighbors=3
search%metric='minkowski'
search%p=2
call search%fit(x_train)
call search%kneighbors(x_query,distances,indices)
```

Both query outputs have shape `(n_query,n_neighbors)` and are sorted by distance.
Indices are one-based training sample indices. Passing training data explicitly
includes the sample itself; there is no implicit leave-one-out query mode.
`radius_neighbors(x,radius)` returns a dense logical adjacency matrix.

## Dimensionality reduction and clustering

```fortran
type(pca) :: projection
type(kmeans) :: clusters
real(dp), allocatable :: embedding(:,:)
projection%n_components=2
projection%whiten=.true.
embedding=projection%fit_transform(x)
clusters%n_clusters=3
clusters%random_state=42
call clusters%fit(embedding)
print *, clusters%labels_
```

K-means supports `call clusters%fit(x,init_centers)` for explicit initial centers,
with shape `(n_clusters,n_features)`. It then performs one initialization run.
Cluster IDs are zero-based, including agglomerative output; DBSCAN noise is `-1`.

## Text features

```fortran
type(tfidf_vectorizer) :: features
type(multinomial_nb) :: classifier
real(dp), allocatable :: x(:,:), query(:,:)
features%tfidf%sublinear_tf=.true.
x=features%fit_transform(documents)
call classifier%fit(x,labels)
query=features%transform(new_documents)
print *, classifier%predict(query)
```

Documents are a rank-one character array. Vocabulary terms are available from
`get_feature_names_out()`. Unknown words are ignored. A document with no known
terms maps to an all-zero row. The ASCII tokenizer accepts letters, digits and
underscore, discarding one-character tokens and punctuation. It does not provide
Python's Unicode regular-expression tokenization. Counts/TF-IDF are dense arrays.

## Gaussian-process uncertainty

```fortran
type(gaussian_process_regressor) :: model
real(dp), allocatable :: mean(:), std(:), covariance(:,:)
model%length_scale=1.5_dp
model%amplitude=2._dp
model%noise_level=0.01_dp
model%alpha=1.e-8_dp
model%normalize_y=.true.
call model%fit(x_train,y_train)
mean=model%predict(x_query)
std=model%predict_std(x_query)
covariance=model%predict_cov(x_query)
```

The fixed kernel is `amplitude * exp(-||x-y||²/(2*length_scale²))` plus
`noise_level` on the covariance diagonal. `alpha` is added only to the training
covariance. Predictive variance includes white-noise variance, and is rescaled
to original target units when `normalize_y` is set. No hyperparameter optimizer
is run. `log_marginal_likelihood_` is the fitted normalized-target likelihood.

## Iterative solvers and reproducibility

Inspect `converged_` and `n_iter_` after fitting elastic net/lasso, logistic
regression, k-means, NMF, Gaussian mixtures, MLPs, SVC, BayesianRidge, ARD and
FactorAnalysis, HuberRegressor and the GLMs. A completed `fit` does
not guarantee convergence if `max_iter` was exhausted. MLP convergence means
training loss failed to improve by `tol` for `n_iter_no_change` iterations;
there is no held-out validation-set stopping criterion.

Set `random_state` for repeatable stochastic fits. Seeding uses the Fortran
runtime's global RNG, not NumPy's RNG; matching integer seeds do not reproduce
Python initialization. Stochastic results need not match across compilers.

See [COVERAGE.md](COVERAGE.md) for the complete type list, supported algorithm
variants, resource limits, omitted parameters and remaining translation work.

## Graph learning and outlier detection

All new types are available from `use sklearn`. Kernel PCA is also exported by
`sklearn_decomposition` (implemented in `sklearn_kernel_pca`), and LOF by
`sklearn_neighbors` (implemented in `sklearn_outlier`). Isomap and label diffusion
have focused `sklearn_manifold` and `sklearn_semi_supervised` modules.

For semi-supervised classification, pass integer training labels with **-1 for
unlabeled rows**. Other integer labels, including negatives other than -1, are
class IDs. `transduction_` contains labels assigned during fitting; `predict(x)`
performs inductive inference and need not equal `transduction_` on training data.
`label_distributions_` is the fitted per-sample class distribution.

```fortran
type(label_spreading) :: model
model%kernel='knn'
model%n_neighbors=7
model%max_iter=1000
call model%fit(x,partially_known_labels)
print *, model%transduction_
```

See `example/semi_supervised.f90` for a complete program. Both label propagation
and spreading expose `converged_` and `n_iter_`. Label spreading uses `alpha` in
`(0,1)` for soft clamping; label propagation hard-clamps known labels.

For Local Outlier Factor, `fit_predict(x)` with default `novelty=false` returns
1 for inliers and -1 for outliers on training data. Set `novelty=true` before
fitting to enable `score_samples`, `decision_function` and `predict` on new data.
`negative_outlier_factor_` always holds training scores. `contamination=-1`
selects the automatic -1.5 offset; a fraction selects an empirical percentile.
Training and novelty-query neighborhoods differ even for identical input rows.

Kernel PCA uses `kernel`, `gamma`, `degree`, `coef0`, `n_components` and
`remove_zero_eig`; its outputs are `eigenvalues_` and `eigenvectors_`.
For `kernel='precomputed'`, fit on a symmetric square training Gram matrix and
transform a `(n_queries,n_training)` cross-Gram matrix. `n_components=0` selects
all nonzero components. There is no learned inverse/preimage transform.

Isomap provides `fit`, `fit_transform`, `transform`, `embedding_`,
`dist_matrix_` and `reconstruction_error()`. Set `n_neighbors` strictly between
zero and the training sample count. Its dense geodesic graph is completed if
disconnected; `n_connected_components_` reports the count before completion.

Radius-neighbor classifiers/regressors use `radius`, `metric`, `p` and
`weights='uniform'` or `'distance'`. Classification defaults to an error for a
query without neighbors; opt in to numeric outlier labels using
`allow_outliers=true` and `outlier_label`. Regression returns NaN for such a
query. Nearest centroid supports `metric='euclidean'` or `'manhattan'`, uniform
priors, and nonnegative `shrink_threshold`; the default zero disables shrinkage.

## Bayesian and sparse regression

`bayesian_ridge`, `ard_regression`, and `orthogonal_matching_pursuit` are in
`sklearn_linear_model`, inherit `fit_intercept`, `coef_`, `intercept_`, `predict`
and R2 `score`, and can be passed to `cross_val_score`.

```fortran
type(bayesian_ridge) :: model
call model%fit(x_train,y_train)
prediction=model%predict(x_query)
uncertainty=model%predict_std(x_query)
```

Bayesian models expose `alpha_` (noise precision), `sigma_` (coefficient
covariance), `n_iter_` and `converged_`. Both use positive `alpha_1`, `alpha_2`,
`lambda_1` and `lambda_2` hyperpriors, plus `tol` and `max_iter`. BayesianRidge
has scalar `lambda_`, `alpha_init=-1` for automatic initialization, and
`lambda_init=1`. ARD has per-feature `lambda_`; features with precision at or
above `threshold_lambda` are pruned. Its `sigma_` contains retained features
only, in original feature order, and can have shape `(0,0)`.

`predict_std` includes observation noise and uses raw query features, matching
sklearn; it does not add intercept uncertainty. See `example/bayesian_regression.f90`.

OMP uses `n_nonzero_coefs=0` for `max(1,n_features/10)` or an explicit positive
count. Set `tol>=0` to stop by residual sum of squares instead; `tol=-1` disables
this rule. Inputs are centered when fitting an intercept but columns are not
normalized. `n_iter_` reports selected features, including early termination on
dependent columns or vanishing correlations.

## Factor analysis and random projections

`factor_analysis` is exported by `sklearn_decomposition` and implemented in
`sklearn_factor_analysis`. It provides `fit`, `fit_transform`, `transform`,
`get_covariance`, `get_precision`, `score_samples`, and mean log-likelihood
`score`. `n_components=-1` selects all features; zero selects a diagonal noise
model with no latent dimensions. Optional `noise_variance_init(:)` must be
positive. Inspect `components_`, `mean_`, `noise_variance_`, `loglike_`, `n_iter_`
and `converged_`. Components have shape `(n_components,n_features)`.

`sklearn_random_projection` exports `gaussian_random_projection`,
`sparse_random_projection` and `johnson_lindenstrauss_min_dim(n_samples,eps)`.
Both types support `fit`, `fit_transform`, `transform` and `inverse_transform`.
`n_components=0` chooses the JL bound from `eps`; specify a positive dimension
to override it. `components_` has shape `(n_components_,n_features)`.
Sparse projections use `density=-1` for `1/sqrt(n_features)`; otherwise choose
a value in `(0,1]`. `density_` stores the fitted value. Storage and outputs are
dense even when entries are zero.

Set `compute_inverse_components=true` to cache `inverse_components_` with shape
`(n_features,n_components_)`. Otherwise `inverse_transform` computes it on
demand. This is a minimum-norm reconstruction; reduced projections generally
lose information. Fitting again clears the previous inverse cache.

## Robust regression and generalized linear models

`huber_regressor`, `ransac_regressor`, `poisson_regressor`, `gamma_regressor`
and `tweedie_regressor` are in `sklearn_linear_model` and the `sklearn` facade.
All support `fit`, `predict`, `score`, `fit_intercept` and `cross_val_score`.
Huber and RANSAC use R2 scoring; GLMs use explained deviance (D2).

Huber jointly estimates `coef_`, `intercept_` and positive `scale_`. Set
`epsilon>=1` (default 1.35), `alpha>=0`, `max_iter` and `tol`. `outliers_` marks
training residuals exceeding `epsilon*scale_`; `loss_` is the total fitted Huber
objective including its L2 penalty. Inspect `converged_` and `n_iter_`.

RANSAC fits ordinary least squares to random subsets and refits the best
consensus. `inlier_mask_` holds that consensus before the final refit;
`n_trials_` and `n_skips_no_inliers_` report sampling diagnostics. Defaults are
`min_samples=0` for `n_features+1`, `residual_threshold=-1` for the unscaled MAD
of targets, `max_trials=100`, and `stop_probability=0.99`. Choose
`loss='absolute_error'` or `'squared_error'`; thresholds apply to that loss.
Set `stop_n_inliers` or `stop_score` for early stopping and `random_state` for
repeatability on a given compiler. Fitted coefficients/intercept are stored
directly on the Fortran object. See `example/robust_regression.f90`.

```fortran
type(poisson_regressor) :: counts
type(tweedie_regressor) :: compound
counts%alpha=0.1_dp
call counts%fit(x_train,nonnegative_target)
prediction=counts%predict(x_query) ! Positive conditional means, not count draws.
compound%power=1.5_dp
call compound%fit(x_train,nonnegative_target)
```

All three GLMs have `alpha>=0` (default 1), `tol>0`, `max_iter`, `converged_`
and `n_iter_`. The intercept is unpenalized. Poisson/Gamma fix their power and
log link during fitting; Gamma requires strictly positive targets. Tweedie
uses `power=0`, `link='auto'` by default. Supported powers are zero and `[1,2]`;
`auto` selects identity for zero and log otherwise. Explicit links must follow
the same pairing. Prediction returns the inverse-link mean. `link_` records
the fitted link, while `coef_` and `intercept_` parameterize the linear predictor.
Scoring requires a nonconstant target. See [COVERAGE.md](COVERAGE.md) for limits.

## Mode seeking, spectral methods and kernel density

`mean_shift`, `affinity_propagation`, `spectral_clustering` and
`estimate_bandwidth` are exported by `sklearn_cluster`. `spectral_embedding`
is exported by `sklearn_manifold` and implemented in `sklearn_spectral`.
`kernel_density` is exported by `sklearn_neighbors` and implemented in
`sklearn_density`. All five types are available through `use sklearn`.

MeanShift has `fit`, `fit_predict` and nearest-center `predict`. A positive
`bandwidth` selects the radius; default `-1` calls `estimate_bandwidth(x)` using
all rows and quantile 0.3. The helper accepts an optional quantile in `[0,1]`.
`bandwidth_` holds the fitted value, including zero if automatically estimated.
Pass alternative seeds as `call model%fit(x,seeds=initial_locations)`.
`cluster_all=false` marks training points farther than the bandwidth from any
center as -1; prediction still assigns every query to its nearest center.
Inspect `cluster_centers_`, `labels_`, `n_iter_` and `converged_`. The iteration
counter matches sklearn's completed-update convention: `max_iter=0` still
performs the first update of each seed.

AffinityPropagation has `fit`, `fit_predict` and, for feature-input fits,
`predict`. Use `affinity='euclidean'` (default) or `'precomputed'` similarities,
`damping` in `[0.5,1)`, `max_iter`, `convergence_iter` and `random_state`.
`preference=huge(1._dp)` selects median similarity; any other finite scalar is
used directly. `cluster_centers_indices_` are one-based training indices,
whereas `labels_` are zero-based. `affinity_matrix_` stores the working
similarities including preference diagonals and tiny perturbations, as in
sklearn. Inspect `converged_` and `n_iter_`; a fit without exemplars produces
empty centers and -1 labels/predictions. Equal-similarity special cases have
`n_iter_=0`. Precomputed fits have no feature-space prediction.

```fortran
type(spectral_clustering) :: model
model%n_clusters=3
model%affinity='rbf'
model%gamma=0.3_dp
model%assign_labels='cluster_qr'
call model%fit(x)
print *, model%labels_
```

SpectralClustering has `fit`, `fit_predict`, `affinity_matrix_`, `embedding_`
and `labels_`. Choose `affinity='rbf'`, `'nearest_neighbors'` or `'precomputed'`.
Nearest-neighbor affinities use `n_neighbors=10` by default and include self
before symmetrization. `assign_labels='kmeans'` (default) uses `n_init` and
`random_state`; `'cluster_qr'` uses deterministic pivoted QR/SVD assignment.

SpectralEmbedding has `fit`, `fit_transform`, `affinity_matrix_` and
`embedding_`. Its default affinity is nearest neighbors, with `n_neighbors=0`
selecting `max(1,n_samples/10)`; `n_neighbors_` reports that value.
For RBF, `gamma=-1` selects `1/n_features`; `gamma_` reports it. Default
`n_components=2` must be below the sample count. Both spectral types report
`n_connected_components_`. They provide training embeddings only; arbitrary
eigenvector signs/rotations and label permutations are not meaningful.

KernelDensity has `fit(x,sample_weight=...)`, `score_samples`, total log-density
`score`, and `sample(n_samples,random_state=...)`. Supported kernels are
`'gaussian'`, `'tophat'`, `'epanechnikov'`, `'exponential'` and `'linear'`.
`bandwidth_rule='fixed'` uses positive `bandwidth` (default 1); `'scott'` and
`'silverman'` choose their dimension/sample-count rules. `bandwidth_` stores the
result. Optional sample weights must be finite, nonnegative and not all zero.
Sampling is supported for Gaussian/tophat kernels. Zero density yields negative
infinity; Gaussian evaluation stays in the log domain to avoid probability
underflow. See `example/clustering_density.f90` for a runnable example.

## FastICA

```fortran
use sklearn, only: dp, fast_ica
type(fast_ica) :: model
real(dp), allocatable :: x(:,:), sources(:,:), reconstructed(:,:)
! Populate x(samples, features) with finite observations.
model%n_components=2
model%algorithm='parallel'  ! or 'deflation'
model%whiten='unit-variance' ! 'arbitrary-variance' or 'none'
model%fun='logcosh'         ! also 'exp' and 'cube'
model%random_state=42
sources=model%fit_transform(x)
reconstructed=model%inverse_transform(sources)
```

`components_` has shape (components, features), `mixing_` has shape
(features, components); the latter is the Moore-Penrose inverse. With reduced
dimension, inverse transformation reconstructs the retained subspace.
`whiten='none'` expects already-whitened input and does not center it.
`converged_` reports whether every extracted component met `tol` before the
iteration limit. See the runnable `independent_components` example.
