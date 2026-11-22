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

Set `random_state` for repeatable stochastic fits. Most stochastic estimators
use the Fortran runtime's global RNG; their matching integer seeds do not
reproduce Python initialization, and results need not match across compilers.
Algorithms explicitly documented with a local NumPy-compatible RandomState
backend, including the new randomized/repeated resampling types, follow that
backend's seeded stream instead. Consult each algorithm's RNG contract.

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

## Isotonic regression and kernel feature maps

```fortran
use sklearn, only: dp, isotonic_regression, rbf_sampler, nystroem
type(isotonic_regression) :: monotone
type(rbf_sampler) :: fourier
type(nystroem) :: landmarks
real(dp), allocatable :: t(:), y(:), x(:,:), fitted(:), z(:,:)
! Populate scalar inputs t and targets y; x is (samples, features).
monotone%auto_increasing=.true.
monotone%out_of_bounds='clip'
fitted=monotone%fit_transform(t,y)
! Optional third argument to fit/fit_transform: sample_weight(:).
fourier%n_components=500
fourier%gamma_scale=.true.
fourier%random_state=42
z=fourier%fit_transform(x)
landmarks%n_components=30
landmarks%kernel='rbf'
landmarks%gamma=0.5_dp
z=landmarks%fit_transform(x)
```

Use `isotonic_fit(y, sample_weight, increasing, y_min, y_max)` for the
standalone ordered-vector fit; all arguments after `y` are optional.
`check_increasing(t,y)` returns the Spearman-sign direction. The type's
`increasing` logical defaults true and `auto_increasing` overrides it when set.
The default outside-domain isotonic prediction is a quiet NaN; use `clip`
for endpoint predictions or `raise` for an error. See `COVERAGE.md` for the
one-threshold special case and supported subsets. Run the
`monotone_kernel_features` example for monotone fitting and kernel errors.

## Paired predictor/response latent components

```fortran
use sklearn, only: dp, pls_regression, pls_canonical, cca, pls_svd
type(pls_regression) :: model
real(dp), allocatable :: x(:,:), y(:,:), xs(:,:), ys(:,:), prediction(:,:), reconstructed(:,:)
! Populate x(samples, features) and y(samples, targets).
model%n_components=2
model%tol=1.e-8_dp
xs=model%fit_transform(x,y)
ys=model%transform_y(y)
prediction=model%predict(x)
reconstructed=model%inverse_transform(xs)
```

All four cross-decomposition types accept rank-two response matrices, including
one-column responses. `fit_transform` returns the X transform; call
`transform_y` to obtain the Y transform. PLSRegression, PLSCanonical and CCA
provide reconstruction for each side via `inverse_transform` and
`inverse_transform_y`, and predictions with shape (samples, targets).
`coef_` has shape (targets, features), following sklearn 1.6.1.

PLSCanonical additionally accepts `algorithm='svd'`; its default is `nipals`.
PLSRegression and CCA support NIPALS only. PLSSVD computes a full SVD of the
centered/scaled cross-covariance and provides transforms without a regression
prediction or inverse API. See `COVERAGE.md` for component/rank restrictions
and early-stop semantics. Run the `paired_latent_components` example.

## MDS and locally linear embedding

```fortran
use sklearn, only: dp, mds, locally_linear_embedding
type(mds) :: scaling
type(locally_linear_embedding) :: lle
real(dp), allocatable :: x(:,:), query(:,:), coordinates(:,:), projected(:,:)
! Populate x(samples, features) and query(new_samples, features).
scaling%random_state=42
coordinates=scaling%fit_transform(x)
! Optional second argument: init(samples, embedding_dimensions).
! For nonmetric MDS: scaling%metric=.false.
! For distances as input: scaling%dissimilarity='precomputed'.
lle%n_neighbors=8
lle%n_components=2
coordinates=lle%fit_transform(x)
projected=lle%transform(query)
```

`mds%normalized_stress` accepts the strings `auto`, `true`, or `false`.
There is no MDS query transform. All four `standard`, `hessian`, `modified` and `ltsa` LLE methods support
query transforms through local barycentric projection. The implementation
uses a dense eigensolver and one-based fitted neighbor indices. See
`COVERAGE.md` for rank/connectivity limits and the sklearn 1.6.1 MDS stress
convention. Run the `manifold_coordinates` example.

## Kernel regression and one-class SVM

```fortran
use sklearn, only: dp, svr, nu_svr, one_class_svm
type(svr) :: regression
type(nu_svr) :: adaptive
type(one_class_svm) :: novelty
real(dp), allocatable :: x(:,:), y(:), query(:,:), prediction(:), score(:)
integer, allocatable :: labels(:)
! Populate x(samples, features), y(samples) and query(new_samples, features).
regression%c=2
regression%epsilon=0.1_dp
regression%gamma=0.7_dp
call regression%fit(x,y)
prediction=regression%predict(query)
adaptive%nu=0.4_dp
call adaptive%fit(x,y)
! adaptive%epsilon_ records its learned tube width.
novelty%nu=0.2_dp
call novelty%fit(x)
score=novelty%decision_function(query)
labels=novelty%predict(query)
```

The optional last fit argument is `sample_weight(:)` for all three types.
For gamma strings, leave `gamma=-1` and set `gamma_rule='scale'` or `'auto'`.
`support_` uses original one-based training indices; `dual_coef_` is a vector
and `intercept_` is a scalar. For linear kernels, `coef_` is a feature vector.
With `kernel='precomputed'`, fit a PSD Gram matrix and predict from the
query-to-training Gram matrix. Check `converged_` or `fit_status_` after capped
fits. One-class labels are +1/-1, with zero decisions labeled -1.
See `COVERAGE.md` for supported kernels, weights and nu restrictions. Run the
`kernel_regression_outliers` example.

## Power transforms

```fortran
use sklearn_preprocessing, only: power_transformer
use sklearn_common, only: dp
implicit none
type(power_transformer) :: power
real(dp) :: x(6,1)
real(dp), allocatable :: z(:,:),restored(:,:)
x(:,1)=[-3._dp,-1._dp,0._dp,1._dp,4._dp,12._dp]
z=power%fit_transform(x)             ! default Yeo-Johnson, standardized
restored=power%inverse_transform(z)
! power%lambdas_ is one fitted power per feature.
! Set power%method='box-cox' before fitting strictly positive data.
! Set power%standardize=.false. to return unstandardized power values.
```

Fit only on training rows, then call `transform` for new rows. NaNs pass through;
all-missing columns fail. Box-Cox also rejects nonpositive and constant training
columns. `mean_` and `scale_` hold transformed training moments when
standardizing, or zero/one otherwise. See `example/power_transforms.f90`.

## Discretization

```fortran
use sklearn
implicit none
type(kbins_discretizer) :: bins
real(dp) :: x(8,2),weights(8)
real(dp), allocatable :: codes(:,:),approximate(:,:)
x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,6._dp,8._dp,12._dp]
x(:,2)=x(:,1)**2; weights=1
bins%strategy='quantile'
bins%encode='ordinal'                 ! or 'onehot-dense'
codes=bins%fit_transform(x,n_bins=[3,4],sample_weight=weights)
approximate=bins%inverse_transform(codes)
! bins%n_bins_ are actual counts; bins%bin_edges_(j)%values are edges.
! Use bins%transform(x_new) for new samples.
```

The fragment illustrates weighted fitting; `example/discretization.f90` is a
complete executable for queries and approximate inversion. Without a per-feature argument,
`bins%n_bins` supplies the requested count (default 5). Quantile and k-means
accept weights; uniform binning does not. Ordinal codes start at zero and use
real64 storage. The inverse returns bin midpoints; constant columns return NaN.
Dense one-hot outputs concatenate feature blocks in input-feature order.

## Sparse inverse covariance

```fortran
use sklearn
implicit none
type(graphical_lasso) :: graph
real(dp) :: covariance(3,3)
covariance=reshape([1._dp,0.6_dp,0.1_dp,0.6_dp,1._dp,0.4_dp,0.1_dp,0.4_dp,1._dp],[3,3])
graph%covariance='precomputed'
graph%alpha=0.2_dp; graph%tol=1.e-8_dp; graph%enet_tol=1.e-10_dp
call graph%fit(covariance)
! graph%precision_ is the sparse inverse covariance estimate.
! Check graph%converged_, graph%n_iter_ and graph%dual_gap_.
! With sample data use covariance='empirical' (default) and fit(x_train).
! Then graph%mahalanobis(x_new) and graph%score(x_new) use the fitted mean.
```

`alpha` penalizes off-diagonal precision entries. `costs_(:,1)` stores the
shifted objective; `costs_(:,2)` stores dual gaps. Alpha zero uses a direct inverse
and records one diagnostic row with `n_iter_=0`. See `example/sparse_precision.f90`.

## Theil-Sen robust regression

```fortran
use sklearn
implicit none
type(theil_sen_regressor) :: robust
real(dp) :: x(8,1),y(8)
real(dp), allocatable :: prediction(:)
x(:,1)=[1._dp,2._dp,3._dp,4._dp,5._dp,6._dp,7._dp,8._dp]
y=2*x(:,1)+0.5_dp; y(1)=30
robust%tol=1.e-8_dp; robust%max_iter=1000
call robust%fit(x,y)
prediction=robust%predict(x)
! Inspect robust%converged_, robust%n_subpopulation_ and robust%breakdown_.
! Set robust%n_subsamples to control rows per subset (zero selects default).
! Set robust%max_subpopulation and robust%random_state for capped sampling.
```

The exhaustive path enumerates combinations; the sampled path retains its
one-based `subsets_` indices for inspection and reproducibility. See
`example/robust_subsets.f90` for comparison with ordinary least squares.

## Passive-aggressive classification and regression

```fortran
use sklearn
implicit none
type(passive_aggressive_classifier) :: classifier
type(passive_aggressive_regressor) :: reg
real(dp) :: x(4,2),y(4)
integer :: labels(4)
x(:,1)=[-2._dp,-1._dp,1._dp,2._dp]; x(:,2)=[0._dp,1._dp,0._dp,1._dp]
labels=[-3,-3,8,8]; y=2*x(:,1)-0.5_dp*x(:,2)+0.3_dp
classifier%shuffle=.false.; classifier%c=0.5_dp
classifier%loss='squared_hinge'       ! PA-II; default hinge selects PA-I
call classifier%fit(x,labels)
reg%shuffle=.false.; reg%epsilon=0.1_dp; reg%average=1
call reg%fit(x,y)
! Inspect coef_, intercept_, n_iter_, t_ and converged_.
! classifier%decision_function(x) is a rank-two array, including binary fits.
! Set tol=-1 for exactly max_iter passes; average=0 disables averaging.
```

`class_weight_balanced` selects balanced class weights. Repeated fits reset
coefficients and averaging state. See `example/passive_aggressive.f90` for
training accuracy and regression R2 on a complete synthetic fixture.


## OPTICS

```fortran
use sklearn, only: dp,optics,cluster_optics_dbscan
implicit none
type(optics) :: model
real(dp) :: x(8,1)
integer, allocatable :: labels(:)
x(:,1)=[0._dp,.1_dp,.2_dp,.3_dp,5._dp,5.1_dp,5.2_dp,5.3_dp]
model%min_samples=2
model%min_cluster_size=3
labels=model%fit_predict(x)
! Reuse the fitted graph with a different extraction radius.
labels=cluster_optics_dbscan(model%reachability_,model%core_distances_,model%ordering_,.25_dp)
```

`fit` exposes `ordering_`, `predecessor_`, `reachability_`, `core_distances_`,
`labels_`, `cluster_hierarchy_`, resolved `min_samples_`, `min_cluster_size_`,
and `max_eps_`. Use `cluster_method='xi'` (default) or `'dbscan'`; configure
`xi`, `predecessor_correction`, `max_eps`, and `eps`. DBSCAN fits return an
empty `(0,2)` hierarchy. `cluster_optics_xi(reachability, predecessor, ordering,
min_samples, labels, clusters, xi, min_cluster_size, predecessor_correction)`
has allocatable output labels and hierarchy; its final three arguments are
optional. Row IDs and hierarchy endpoints are 1-based; cluster labels are
0-based, with noise -1 and root predecessors -1. See COVERAGE.md for distance,
fractional-size, infinity and complexity conventions.


## BIRCH

```fortran
use sklearn, only: dp,birch
implicit none
type(birch) :: model
real(dp) :: x(8,1)
integer, allocatable :: labels(:)
x(:,1)=[0._dp,.1_dp,.2_dp,.3_dp,5._dp,5.1_dp,5.2_dp,5.3_dp]
model%threshold=.2_dp
model%branching_factor=2
model%n_clusters=0 ! Python n_clusters=None: retain leaf subclusters.
call model%partial_fit(x(:4,:))
call model%partial_fit(x(5:,:))
labels=model%predict(x)
model%n_clusters=2
call model%partial_fit() ! Recluster the existing leaf centroids.
labels=model%predict(x)
```

Methods are `fit`, `partial_fit` (optional X), `fit_predict`, `predict`,
`fit_transform`, and `transform`. The last two return Euclidean distances to
all `subcluster_centers_`, independent of the global cluster count.
`subcluster_labels_` maps each centroid to a zero-based global label;
`labels_` stores the last labeled input batch. `compute_labels=false` skips
that assignment. Global-only updates leave prior batch labels intact;
`predict` returns current labels. `insufficient_subclusters_` reports the
fallback when fewer centroids than requested remain. Tree controls must stay
fixed during incremental insertion; call `fit` to change them. See COVERAGE.md
for complexity, cluster-name and supported-input limits.


## QuantileTransformer

```fortran
use sklearn, only: dp,quantile_transformer
implicit none
type(quantile_transformer) :: model
real(dp) :: x(5,1)
real(dp), allocatable :: z(:,:),recovered(:,:)
x(:,1)=[0._dp,1._dp,2._dp,4._dp,8._dp]
model%n_quantiles=5
model%subsample=0 ! Python subsample=None; use every row.
model%output_distribution='normal' ! Or 'uniform' (default).
z=model%fit_transform(x)
recovered=model%inverse_transform(z)
```

Available methods are `fit`, `fit_transform`, `transform`, and
`inverse_transform`. Fitted arrays are `quantiles_` and `references_`, with
`n_quantiles_` and `n_features_in_` metadata. Subsampling uses `random_state`
and a native random stream. NaNs are supported; infinities are rejected.
All-missing columns map to NaNs, including finite queries. Normal scores
are clipped to finite tails, and out-of-range inverse values saturate at
learned endpoints. `output_distribution` may switch between uniform and
normal after fit because the learned percentile tables are distribution
independent. See COVERAGE.md for boundary and subsampling conventions.


## Chi-squared kernel feature maps

```fortran
use sklearn, only: dp,additive_chi2_sampler,skewed_chi2_sampler
implicit none
type(additive_chi2_sampler) :: additive
type(skewed_chi2_sampler) :: skewed
real(dp) :: x(3,2)
real(dp), allocatable :: z(:,:)
x(1,:)=[0._dp,1._dp]
x(2,:)=[1._dp,2._dp]
x(3,:)=[2._dp,0._dp]
additive%sample_steps=3
z=additive%fit_transform(x) ! 3 rows, 10 columns.
skewed%skewedness=.5_dp
skewed%n_components=100
skewed%random_state=42
z=skewed%fit_transform(x) ! 3 rows, 100 columns.
```

Both provide fit, transform and fit_transform and record n_features_in_.
The additive map also permits transform before fit. Its sample_interval=-1
selects defaults for sample_steps in 1..3; other counts need an explicit
nonnegative interval. Inputs must be nonnegative. The skewed map stores
random_weights_ and random_offset_; transform requires X > -skewedness.
Finite zero/negative skewedness is allowed on that shifted domain. See
COVERAGE.md for output column ordering, kernel definitions and RNG conventions.


## GradientBoostingClassifier

```fortran
use sklearn, only: dp,gradient_boosting_classifier
implicit none
type(gradient_boosting_classifier) :: model
real(dp) :: x(6,1)
integer :: y(6)
real(dp), allocatable :: probability(:,:),stages(:,:,:)
x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp]
y=[-4,-4,-4,7,7,7]
model%n_estimators=20
model%max_depth=1
model%loss='log_loss' ! Also exponential for binary labels.
call model%fit(x,y)
probability=model%predict_proba(x)
stages=model%staged_predict_proba(x)
```

Also available: decision_function, predict_log_proba, predict, score,
staged_decision_function, staged_predict and apply. Binary decisions have
shape (n,1); multiclass decisions (n,K). Staged outputs put stages on the last
axis; staged labels are (n,n_stages). Apply returns (n,n_stages,channels)
1-based native leaf IDs. `estimators_` is (stage,channel), where channel count
is one for binary labels and K for multiclass. Fitted fields include classes_,
class_prior_, init_ (raw initial scores), n_estimators_, n_trees_per_iteration_,
n_features_in_, train_score_ and feature_importances_. Set init_zero=true for
zero initial scores. Tree controls include max_depth, min_samples_leaf,
min_samples_split, max_features (0 means all), and min_impurity_decrease.
Sample weights/subsampling/early stopping are not part of this initial port;
see COVERAGE.md for loss, tree-backend and staged-memory conventions.


## Ridge classification

```fortran
use sklearn, only: dp,ridge_classifier,ridge_classifier_cv
implicit none
type(ridge_classifier) :: classifier
type(ridge_classifier_cv) :: selected
real(dp) :: x(6,1),weights(6)
integer :: y(6)
real(dp), allocatable :: scores(:,:)
x(:,1)=[0._dp,1._dp,2._dp,3._dp,4._dp,5._dp]
y=[-4,-4,-4,7,7,7]
weights=[1._dp,2._dp,1._dp,1._dp,2._dp,1._dp]
classifier%alpha=1
call classifier%fit(x,y,sample_weight=weights)
scores=classifier%decision_function(x)
selected%alphas=[.1_dp,1._dp,10._dp]
selected%store_cv_results=.true.
call selected%fit(x,y,sample_weight=weights)
```

Both types expose fit, decision_function, predict and score, with coef_,
intercept_, classes_, n_features_in_ and native rank_. Coefficients are
(channels,n_features); decisions (n,channels), with one binary channel or one
per class. Optional class_weight in fit is a vector aligned to sorted labels;
alternatively set class_weight_balanced=true. class_weight and sample_weight
multiply. fit_intercept defaults true. RidgeClassifier supports alpha>=0;
RidgeClassifierCV requires positive alphas and implements default leave-one-out
negative-MSE selection. It adds alpha_, best_score_, cv_scores_, and optional
cv_results_(n,channels,n_alphas). It does not provide explicit-fold CV, arbitrary
scorers or probabilities. See COVERAGE.md for weighting and rank conventions.


## Multi-task sparse regression

```fortran
use sklearn, only: dp,multi_task_elastic_net,multi_task_lasso
implicit none
type(multi_task_elastic_net) :: model
real(dp) :: x(6,2),y(6,2)
real(dp), allocatable :: prediction(:,:)
integer :: i
do i=1,6
  x(i,:)=[real(i,dp),real(mod(i,2),dp)]
end do
y(:,1)=2*x(:,1)+1; y(:,2)=-x(:,1)+3
model%alpha=.05_dp; model%l1_ratio=.7_dp
call model%fit(x,y)
prediction=model%predict(x)
```

Use multi_task_lasso for the pure group L1/L2 penalty. Both types accept a
rank-two target matrix and return rank-two predictions. coef_ is
(n_targets,n_features), intercept_ is (n_targets). Set warm_start=true to reuse
coefficients on subsequent fits of the same shape. score returns the uniform
mean of per-target R2. Inspect n_iter_, dual_gap_, eps_ and converged_ for
stopping status. selection accepts cyclic/random with scalar random_state.
Finite alpha>=0 and l1_ratio in [0,1] include unpenalized/pure-L2 boundaries.
See COVERAGE.md for the objective and supported controls.


## Categorical Naive Bayes

```fortran
use sklearn, only: dp,categorical_nb
implicit none
type(categorical_nb) :: model
real(dp) :: x(4,2)
real(dp), allocatable :: probability(:,:)
x(1,:)=[0._dp,0._dp]; x(2,:)=[1._dp,0._dp]
x(3,:)=[1._dp,1._dp]; x(4,:)=[1._dp,1._dp]
call model%partial_fit(x(:2,:),[-3,-3],classes=[-3,9])
call model%partial_fit(x(3:,:),[9,9])
probability=model%predict_proba(x)
```

fit and partial_fit accept optional sample_weight. The first partial_fit must
supply all classes; batch fit infers them. Set alpha>0 for smoothing,
fit_prior=false for uniform priors, or allocate class_prior in sorted class
order. min_categories reserves a scalar width; min_categories_per_feature
reserves feature-specific widths. Feature codes are nonnegative integer values
stored in real(dp) arrays. Counts and feature log probabilities are available
as category_count_(j)%values and feature_log_prob_(j)%values, each with shape
(n_classes,n_categories_(j)). Other methods are joint_log_likelihood,
predict_log_proba, predict and score. See COVERAGE.md for incremental metadata
and out-of-range category rules.


## Neighbor imputation and missingness indicators

```fortran
use sklearn, only: dp,knn_imputer,missing_indicator
use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
implicit none
type(knn_imputer) :: model
type(missing_indicator) :: indicator
real(dp) :: x(3,2)
real(dp), allocatable :: completed(:,:)
logical, allocatable :: missing(:,:)
x(1,:)=[0._dp,10._dp]; x(2,:)=[2._dp,20._dp]
x(3,:)=[1._dp,ieee_value(0._dp,ieee_quiet_nan)]
model%n_neighbors=2; model%weights='distance'; model%add_indicator=.true.
completed=model%fit_transform(x)
missing=indicator%fit_transform(x)
```

KNNImputer's transform returns real(dp) values, optionally followed by fitted
missingness columns. Set keep_empty_features=true to retain all-missing
training columns as zeros. MissingIndicator returns logical values and exposes
1-based features_. Its features control is 'missing-only' or 'all'; set
error_on_new=false to accept new missingness without changing output width.
Both types provide fit, transform and fit_transform, accepting NaN sentinels.
See COVERAGE.md for donor ties, missing-data distances and empty-column rules.


## Gaussian process classification

```fortran
use sklearn, only: dp,gaussian_process_classifier
implicit none
type(gaussian_process_classifier) :: model
real(dp) :: x(6,1)
real(dp), allocatable :: probability(:,:)
integer :: i
do i=1,6
  x(i,1)=real(i-4,dp)
end do
model%amplitude=2; model%length_scale=1
call model%fit(x,[-3,-3,-3,9,9,9])
probability=model%predict_proba(x)
```

The kernel remains fixed during fit. Multiclass labels use one-vs-rest GPs
and normalized probabilities in sorted class order. Methods are fit, predict,
predict_proba, score and log_marginal_likelihood(). Inspect n_iter_ and
converged_ for each binary channel; max_iter_predict controls the iteration cap.
Posterior arrays pi_, w_sqrt_, posterior_mode_ and alpha_ have shape
(n_train,channels), with one binary channel or one per multiclass label.
See COVERAGE.md for the evidence/probability approximation and supported limits.


## B-spline features

```fortran
use sklearn, only: dp,spline_transformer
implicit none
type(spline_transformer) :: model
real(dp) :: x(5,1)
real(dp), allocatable :: basis(:,:)
x(:,1)=[0._dp,.5_dp,1._dp,1.5_dp,2._dp]
model%n_knots=3; model%degree=3; model%include_bias=.false.
basis=model%fit_transform(x)
```

Set knots='quantile' for empirical knot positions, or pass knots=positions to
fit/fit_transform for an explicit (n_base_knots,n_features) matrix.
extrapolation is 'constant', 'linear', 'continue', 'periodic' or 'error'.
Periodic fits require degree<n_base_knots. include_bias=false removes the
last basis per feature. Inspect base_knots_, extended knots_, n_features_in_
and n_features_out_. Transform returns real(dp) columns grouped by input
feature. See COVERAGE.md for knot, endpoint and supported-degree conventions.


## Primal linear support-vector models

```fortran
use sklearn, only: dp,linear_svc,linear_svr
implicit none
type(linear_svc) :: classifier
type(linear_svr) :: regressor
real(dp) :: x(4,1)
real(dp), allocatable :: prediction(:),decision(:,:)
x(:,1)=[-2._dp,-1._dp,1._dp,2._dp]
call classifier%fit(x,[-3,-3,9,9])
decision=classifier%decision_function(x)
regressor%epsilon=.1_dp
call regressor%fit(x,[-4._dp,-2._dp,2._dp,4._dp])
prediction=regressor%predict(x)
```

Both use the primal L2 squared-loss formulation and accept optional sample_weight.
LinearSVC also accepts class_weight (sorted class order) or a
class_weight_balanced flag. C, tol, max_iter, fit_intercept and intercept_scaling
control fitting. The intercept is penalized through its scaled synthetic
feature. Inspect converged_, n_iter_, objective_ and gradient_norm_;
classification diagnostics are per-channel where applicable. LinearSVR uses
squared_epsilon_insensitive loss, including at its default epsilon=0. See
COVERAGE.md for multiclass weighting, stopping and supported solver limits.


## Streaming principal components

```fortran
use sklearn, only: dp, incremental_pca
type(incremental_pca) :: model
real(dp), allocatable :: x(:,:), z(:,:), reconstructed(:,:)
! Populate x with finite data; the initial batch needs at least two rows.
model%n_components = 2
call model%partial_fit(x(:10,:))
call model%partial_fit(x(11:,:))
z = model%transform(x)
reconstructed = model%inverse_transform(z)
! Alternatively: model%batch_size=10; call model%fit(x)
```

`fit` resets previous state. `partial_fit` preserves running mean/variance and
updates the retained basis. Set `whiten=.true.` for variance-scaled coordinates.
See COVERAGE.md for initialization, batch boundaries and numerical limits.


## Clustering feature columns

```fortran
use sklearn, only: dp, feature_agglomeration
type(feature_agglomeration) :: model
real(dp), allocatable :: x(:,:), pooled(:,:), reconstructed(:,:)
! Populate x with finite data and at least two feature columns.
model%n_clusters = 2
model%linkage = 'ward'
model%pooling_func = 'mean'
pooled = model%fit_transform(x)
reconstructed = model%inverse_transform(pooled)
```

The labels_ array assigns each input feature to a zero-based cluster. Columns
of pooled follow these cluster IDs. Reconstruction repeats pooled values into
member columns. Median and maximum pooling use 'median' and 'max'. See
COVERAGE.md for tree attributes, ordering and scope.


## Selecting elastic-net regularization

```fortran
use sklearn, only: dp, elastic_net_cv, lasso_cv
type(elastic_net_cv) :: model
type(lasso_cv) :: sparse
real(dp), allocatable :: x(:,:), y(:), predicted(:)
! Populate finite x and y with at least three samples.
model%cv = 3
model%alphas = [1._dp, .1_dp, .01_dp]
model%l1_ratios = [.5_dp, 1._dp]
call model%fit(x,y)
predicted = model%predict(x)
! model%alpha_, model%l1_ratio_ and model%best_mse_ report the selection.
sparse%cv = 3
sparse%n_alphas = 20
call sparse%fit(x,y) ! Automatic grid, L1 ratio fixed to one.
```

The retained mse_path_ has dimensions (alpha,fold,ratio); alphas_ is
(alpha,ratio). Folds are contiguous without shuffling. See COVERAGE.md for
parameter defaults, stopping differences, convergence flags and scope.


## Boosting weighted classification stumps

```fortran
use sklearn, only: dp, adaboost_classifier
type(adaboost_classifier) :: model
real(dp), allocatable :: x(:,:), probabilities(:,:), stages(:,:,:)
integer, allocatable :: y(:), predicted(:)
! Populate finite x and integer y with at least two classes.
model%n_estimators = 50
model%learning_rate = .5_dp
call model%fit(x,y)
predicted = model%predict(x)
probabilities = model%predict_proba(x)
stages = model%staged_predict_proba(x)
```

Optional sample_weight is passed to fit. The weak learner is fixed to a weighted
Gini stump. n_estimators_ records the retained count after early stopping; see
COVERAGE.md for SAMME conventions, staged layouts, tie behavior and limits.


## Streaming SGD regression

```fortran
use sklearn, only: dp, sgd_regressor
type(sgd_regressor) :: model
real(dp), allocatable :: x(:,:), y(:), predictions(:)
! Populate finite x and y before fitting.
model%learning_rate = 'constant'
model%eta0 = .01_dp
model%average = 1
call model%partial_fit(x(:10,:),y(:10))
call model%partial_fit(x(11:,:),y(11:))
predictions = model%predict(x)
! Alternatively call model%fit(x,y), which resets all accumulated state.
```

Set tol=-1 to disable fit stopping. Ordered updates are the default; native
shuffle can be enabled explicitly. See COVERAGE.md for losses, supported
penalties/schedules, averaging, sample weights and differences from sklearn.


## Streaming SGD classification

```fortran
use sklearn, only: dp, sgd_classifier
type(sgd_classifier) :: model
real(dp), allocatable :: x(:,:), probabilities(:,:)
integer, allocatable :: y(:), predictions(:)
! Populate finite x and labels y; declare all possible classes on the first batch.
model%loss = 'log_loss'
model%learning_rate = 'constant'
model%eta0 = .01_dp
model%average = 1
model%fit_intercept = .false. ! Required for averaged classifier partial_fit.
call model%partial_fit(x(:10,:),y(:10),classes=[-4,7,21])
call model%partial_fit(x(11:,:),y(11:))
predictions = model%predict(x)
probabilities = model%predict_proba(x)
```

fit infers classes and resets state. First streaming batches may contain only
one of the declared classes. Class weights are optional per-call arguments;
balanced weighting is supported by fit. See COVERAGE.md for defaults, layouts,
probability-supported losses and supported SGD controls.


## Group-aware and predefined validation

```fortran
use sklearn, only: group_kfold,leave_one_out,leave_one_group_out,predefined_split,index_split
type(group_kfold) :: grouped
type(leave_one_out) :: loo
type(leave_one_group_out) :: logo
type(predefined_split) :: explicit
type(index_split), allocatable :: folds(:)
grouped%n_splits=3
folds=grouped%split([10,10,20,20,30,30])
folds=loo%split(6)
folds=logo%split([10,10,20,20,30,30])
folds=explicit%split([-1,0,0,4,4,-1])
```

Indices start at 1. Pass fold train/test rows to the estimator manually.
Predefined -1 rows always stay in training. The splitters store all folds in
memory; group-count ties use descending group labels (see COVERAGE.md).


## Sequential feature selection for regression

```fortran
use sklearn, only: dp,ridge,sequential_feature_selector
type(ridge) :: estimator
type(sequential_feature_selector) :: selector
real(dp), allocatable :: selected(:,:)
estimator%alpha=1._dp
selector%n_features_to_select=2
selector%direction='backward'
selector%cv=4
selected=selector%fit_transform(x,y,estimator)
call estimator%fit(selected,y)
```

`y` is a real regression target. The selector scores independent estimator
copies with unshuffled K-fold R2; fit the final estimator separately.
Use `n_features_to_select=0` for auto and `use_tol=.true.` to enable an auto
stopping tolerance. `get_support()` returns a logical mask in original column
order; `inverse_transform` restores omitted columns as zeros. See COVERAGE.md
for the supported estimator family, odd-feature auto behavior and R2 fold sizes.


## Recursive elimination and cross-validated feature counts

```fortran
use sklearn, only: dp,rfe,rfecv,ridge,logistic_regression
type(rfe) :: selector
type(rfecv) :: search
type(ridge) :: regressor
type(logistic_regression) :: classifier
real(dp), allocatable :: reduced(:,:),predictions(:),probabilities(:,:)
integer, allocatable :: labels(:)
selector%n_features_to_select=2
selector%step=.25_dp
call selector%fit(x,y,regressor) ! real y
predictions=selector%predict(query)
search%cv=4
search%min_features_to_select=1
reduced=search%fit_transform(x,integer_targets,classifier)
labels=search%predict_classes(query)
probabilities=search%predict_proba(query)
! search%cv_results_%split_test_score(feature_count_index,fold_index)
```

Both selectors retain a fitted estimator on selected columns. Use real targets
with linear-regression-family estimators and integer targets with logistic
regression. Callers' estimators remain unchanged. Query arrays contain the
original columns; prediction methods select the fitted mask internally.
See COVERAGE.md for counts, step fractions, ranking, CV folds and supported scope.


## Least-angle regression and Lasso coefficient paths

```fortran
use sklearn, only: dp,lars,lasso_lars,lars_path,lars_path_result
type(lars) :: least_angle
type(lasso_lars) :: sparse
type(lars_path_result) :: path
real(dp), allocatable :: predictions(:)
least_angle%n_nonzero_coefs=3
call least_angle%fit(x,y)
sparse%alpha=.05_dp
sparse%positive=.true.
call sparse%fit(x,y)
predictions=sparse%predict(query)
path=lars_path(x,y,method='lasso',alpha_min=.01_dp)
! path%coefficients(feature,knot); path%active uses 1-based feature indices.
```

Estimator fits center inputs for intercepts by default; lars_path uses its
inputs directly. Paths include the initial zero coefficients and all generated
knots through the final endpoint. Set fit_path=false to retain only endpoint
model attributes. See COVERAGE.md for limits and degenerate-column behavior.


## Selecting a least-angle model by CV or information criterion

```fortran
use sklearn, only: dp,lasso_lars_cv,lasso_lars_ic
type(lasso_lars_cv) :: cv_model
type(lasso_lars_ic) :: bic_model
cv_model%cv=4
cv_model%max_n_alphas=100
call cv_model%fit(x,y)
! cv_model%mse_path_(alpha_index,fold), cv_model%cv_alphas_, cv_model%alpha_
bic_model%criterion='bic'
call bic_model%fit(x,y)
! bic_model%criterion_, bic_model%noise_variance_, bic_model%alpha_
```

`lars_cv` provides the LAR alternative to LassoLarsCV. Set positive=true on
LassoLarsCV/IC for nonnegative coefficients. IC noise_variance=-1 estimates
variance from full-feature least squares; provide a positive estimate when
n <= p + intercept. Prediction uses the final selected model. See COVERAGE.md
for grid interpolation, path storage and inherited fields that are ignored.


## Sparse components and dictionary codes

```fortran
use sklearn, only: dp,sparse_pca,dictionary_learning,sparse_coder,sparse_encode
type(sparse_pca) :: projection
type(dictionary_learning) :: learner
type(sparse_coder) :: coder
real(dp), allocatable :: latent(:,:),restored(:,:),codes(:,:)
projection%n_components=3
projection%alpha=.2_dp
latent=projection%fit_transform(x)
restored=projection%inverse_transform(latent)
learner%n_components=6
learner%fit_algorithm='cd'
learner%transform_algorithm='lasso_cd'
codes=learner%fit_transform(x)
coder%dictionary=learner%components_
coder%transform_algorithm='threshold'
coder%transform_alpha=.2_dp
codes=coder%transform(query)
codes=sparse_encode(query,learner%components_,algorithm='omp',n_nonzero_coefs=2)
```

SparsePCA returns ridge projection scores and reconstructs with its mean.
DictionaryLearning fit_transform returns learned sparse codes; its transform
performs fresh coding with the chosen transform algorithm. SparseCoder needs
only a supplied dictionary. See COVERAGE.md for complete initialization pairs,
positive constraints, split-sign output, automatic parameter sentinels and scope.


## AdaBoost regression

`adaboost_regressor` is available from `sklearn` and `sklearn_ensemble`.

```fortran
type(adaboost_regressor) :: model
model%n_estimators=50
model%random_state=42
model%loss='square' ! linear (default), square, exponential
model%max_depth=3
call model%fit(x,y,sample_weight=weights)
prediction=model%predict(query)
stages=model%staged_predict(query) ! (n_query,n_estimators_)
scores=model%staged_score(x,y)
```

`score` returns unweighted R2. Optional fit weights must be finite,
nonnegative and have a positive finite sum. Controls include learning_rate,
min_samples_split, min_samples_leaf, max_features (zero means all), and
min_impurity_decrease. Fitted attributes are n_features_in_, n_estimators_,
estimators_, estimator_weights_, estimator_errors_, and feature_importances_.
The weights/errors retain requested n_estimators slots, with unused slots 0/1;
estimators_ contains only retained trees. Each tree supports predict, apply,
get_depth and get_n_leaves. See COVERAGE.md for the supported base learner.


## QuantileRegressor

Available as quantile_regressor from sklearn, sklearn_linear_model and
sklearn_quantile.

```fortran
type(quantile_regressor) :: model
model%quantile=.9_dp
model%alpha=.02_dp
model%fit_intercept=.true.
call model%fit(x,y,sample_weight=weights) ! weights optional
prediction=model%predict(query)
r2=model%score(x,y)
```

Fitted attributes: coef_, intercept_, n_features_in_, n_iter_ and converged_.
Native LP controls are tol and max_iter; native diagnostics are objective_,
dual_coef_, dual_gap_ and optimality_. dual_coef_ has one entry per training
row, with zero entries for zero-weight rows. For normalized weights w, an
optimal dual has (quantile-1)*w <= dual_coef_ <= quantile*w,
abs(matmul(transpose(x),dual_coef_)) <= alpha and, with an intercept,
sum(dual_coef_)=0. Its objective dot_product(y,dual_coef_) agrees with the
primal objective. See COVERAGE.md for numerical tolerance, scaling, nonunique
solutions, solver differences and convergence behavior. score is unweighted R2
and requires at least two query samples.


## NuSVC

Import nu_svc from sklearn, sklearn_svm or sklearn_nu_classification.

```fortran
type(nu_svc) :: model
model%nu=.35_dp
model%kernel='rbf'
model%gamma_rule='scale' ! gamma=-1 selects this rule; numeric gamma overrides it
call model%fit(x,labels,sample_weight=weights) ! weights optional
prediction=model%predict(query)
scores=model%decision_function(query)
accuracy=model%score(x,labels)
model%decision_function_shape='ovo' ! default ovr; break_ties requires ovr
```

Core fitted attributes are classes_, support_, n_support_, support_vectors_,
dual_coef_, intercept_, n_features_in_ and linear-only coef_. Binary scores
are (n_query,1), multiclass OVR scores are (n_query,n_classes), and OVO scores
are (n_query,n_classes*(n_classes-1)/2). Native n_iter_ and kkt_violation_
are vectors over class pairs, with aggregate converged_ and fit_status_.
pair_alpha_ and pair_scales_ expose native unscaled dual diagnostics.
Nu feasibility uses per-class sample-weight totals. Precomputed queries have
one column per original training row, including zero-weight rows. Support
indices refer to original training rows and start at one. See COVERAGE.md for
fully bound normalizations, solver differences, explicit degenerate errors,
parameter scope and the standard multiclass coefficient layout.


## SGDOneClassSVM

Import sgd_one_class_svm from sklearn, sklearn_linear_model or
sklearn_sgd_one_class.

```fortran
type(sgd_one_class_svm) :: model
model%nu=.1_dp
model%average=1
model%random_state=42
call model%fit(x,sample_weight=weights) ! optional weights
call model%partial_fit(next_batch)     ! one epoch; preserves t_ and averages
scores=model%decision_function(query)
raw_scores=model%score_samples(query)
labels=model%predict(query)            ! +1 inliers, -1 outliers
```

Optional fit arguments coef_init(:) and scalar offset_init initialize the
parameters. warm_start reuses the public fitted parameters on a subsequent
fit, while resetting its sample clock and averaging buffers. fit_predict
returns integer training labels. Coefficients are a feature vector; offset_
is scalar. n_iter_ counts the most recent call's epochs; t_ counts updates
plus one. Native converged_ and loss_ report stopping and last-epoch hinge
loss. average=0 disables averaging, average=1 enables it immediately, and a
larger integer starts it at that update index. eta0 must be positive for
constant, invscaling and adaptive schedules; optimal permits zero. power_t
may be negative. tol=-1 disables stopping. See COVERAGE.md for exact zero-weight,
shuffle/averaging and warm/partial-fit contracts and unsupported options.


## IsolationForest

Import isolation_forest from sklearn, sklearn_ensemble or
sklearn_isolation_forest.

```fortran
type(isolation_forest) :: model
model%random_state=42
model%max_samples=128       ! 0 selects auto; oversized counts are clipped
model%contamination=.1_dp   ! -1 selects auto
call model%fit(x,sample_weight=weights) ! weights optional
scores=model%score_samples(query)
decisions=model%decision_function(query)
labels=model%predict(query) ! +1 inliers, -1 outliers
model%warm_start=.true.
model%n_estimators=150
call model%fit(x)
```

fit_predict returns training labels after fit. To use max_samples_fraction,
leave integer max_samples at zero; to use max_features_fraction, leave integer
max_features at zero. Fraction zero disables that option. Fitted counts and
offset_, estimators_, estimators_features_ and estimators_samples_ are exposed.
Per-tree index metadata uses %values, for example
model%estimators_samples_(1)%values. Indices start at one; row metadata preserves
bootstrap duplicates and all actual warm-start generations. Tree feature
indices are local to estimators_features_(i)%values. See COVERAGE.md for exact
sampling, float32 rounding, weighted counts, empty roots, warm starts and scope.


## MinCovDet and EllipticEnvelope

Import min_cov_det, elliptic_envelope and fast_mcd from sklearn,
sklearn_covariance or sklearn_robust_covariance.

```fortran
type(min_cov_det) :: robust
type(elliptic_envelope) :: detector
robust%random_state=42
robust%support_fraction=.7_dp ! 0 selects the automatic support size
robust%store_precision=.false.
call robust%fit(x)
precision=robust%get_precision()
distances=robust%mahalanobis(query)
likelihood=robust%score(query)
detector%contamination=.1_dp
call detector%fit(x)
labels=detector%predict(query)
accuracy=detector%score(query,true_labels,sample_weight=weights) ! optional weights
call fast_mcd(x,raw_location,raw_covariance,raw_support,raw_dist,random_state=42)
```

fast_mcd outputs are allocatable arrays, with logical raw_support. Both types
expose raw_location_, raw_covariance_, raw_support_, location_, covariance_,
support_ and dist_, plus optional cached precision_. Native diagnostics include
rank_, n_features_in_, correction_ and reweight_threshold_. With
assume_centered=true both locations are zero. error_norm supports norm
frobenius/spectral and optional scaling/squared flags. EllipticEnvelope adds
scalar offset_, score_samples, decision_function and fit_predict; its score
returns accuracy, while MinCovDet.score returns log likelihood. See COVERAGE.md
for full FastMCD stages, raw tie conventions, singular precision, numerical
guards and supported parameter/method scope.


## BisectingKMeans

Import bisecting_kmeans from sklearn, sklearn_cluster or
sklearn_bisecting_kmeans.

```fortran
type(bisecting_kmeans) :: model
model%n_clusters=6
model%init='k-means++'
model%algorithm='elkan'
model%bisecting_strategy='largest_cluster'
model%n_init=3
model%random_state=42
call model%fit(x,sample_weight=weights) ! weights optional
labels=model%predict(query)
distances=model%transform(query)
negative_inertia=model%score(query,sample_weight=query_weights)
```

fit_predict and fit_transform fit before returning hierarchical training labels
or final-center distances. Fitted cluster_centers_, labels_, inertia_ and
n_features_in_ match the sklearn counterpart. Native tree arrays describe
split centers, children, leaf labels, row counts and selection scores;
children_ links start at one and labels start at zero. Native per-split
iterations/inertias/convergence and aggregate converged_ are available.
See COVERAGE.md for seeded weighted initialization, native Elkan bounds,
empty leaves, one-cluster root behavior, tolerance and the distinction between
hierarchical predict and global nearest-center score.


## MiniBatchKMeans

Import mini_batch_kmeans from sklearn, sklearn_cluster or
sklearn_minibatch_kmeans.

```fortran
type(mini_batch_kmeans) :: model
model%n_clusters=4
model%batch_size=64
model%random_state=42
call model%partial_fit(first_batch,sample_weight=first_weights)
call model%partial_fit(next_batch,sample_weight=next_weights)
labels=model%predict(query)
distances=model%transform(query)
negative_inertia=model%score(query,sample_weight=query_weights)
```

fit runs sampled mini-batch optimization on a complete training matrix;
partial_fit applies one update to the entire supplied batch. The first supplied
batch needs at least n_clusters rows; later batches can be smaller. fit_predict
and fit_transform fit before returning labels or final-center distances.
Labels are zero-based. cluster_centers_, labels_, inertia_, n_features_in_,
n_steps_ and (after fit) n_iter_ have sklearn counterpart semantics. Native
counts_ exposes accumulated weights; n_steps_ is a 64-bit integer. converged_ reports early stopping in fit;
has_inertia_ indicates whether inertia_ is meaningful.

Native sentinels are n_init=0 for 'auto' (one k-means++ or three random starts),
init_size=0 for automatic sizing and max_no_improvement=-1 for None. Allocate
initial_centers(n_clusters,n_features) for explicit initialization; deallocate
it to resume the string init method. Supplied centers force one initialization.
With compute_labels=false, labels are omitted and fit reports smoothed inertia
when at least two optimization steps ran. A single-step fit has has_inertia_
false; a fresh partial_fit with labels disabled has no inertia either. See
COVERAGE.md for restart selection, reassignment, stopping, RNG and scope limits.


## Spectral co-clustering and biclustering

Import spectral_coclustering and spectral_biclustering from sklearn,
sklearn_cluster or sklearn_bicluster.

```fortran
type(spectral_coclustering) :: paired
type(spectral_biclustering) :: grid
integer, allocatable :: row_indices(:),column_indices(:)
real(dp), allocatable :: block(:,:)
paired%n_clusters=3
paired%random_state=42
call paired%fit(x)
grid%n_row_clusters=2
grid%n_column_clusters=3
grid%method='log' ! also 'scale' and 'bistochastic' (default)
grid%n_components=6
grid%n_best=3
call grid%fit(x)
call grid%get_indices(0,row_indices,column_indices)
block=grid%get_submatrix(0,x)
```

Both types expose zero-based row_labels_/column_labels_ and logical
rows_/columns_ membership matrices. Paired clustering has K blocks; checkerboard
clustering has K_row*K_column blocks. Leave both grid count overrides zero to
use scalar n_clusters on each side. get_shape(block_id) returns two dimensions;
get_biclusters(rows,columns) copies both indicator matrices. Block IDs start at
zero; extracted row/column indices start at one. Empty blocks are supported.
Submatrix input must have the fitted shape. These estimators provide fit and
membership extraction, with no source predict/transform contract.

The native backend is a full thin dense SVD, with no svd_method/n_svd_vecs tuning.
Both types accept init='k-means++'/'random', positive n_init and scalar
random_state; mini_batch=true uses the native MiniBatchKMeans centroid solver.
See COVERAGE.md for normalization, vector selection, null-space/centroid ties,
cluster-count bounds, diagnostic arrays and supported input/solver scope.


## BayesianGaussianMixture

Import bayesian_gaussian_mixture from sklearn, sklearn_mixture or
sklearn_bayesian_mixture.

```fortran
type(bayesian_gaussian_mixture) :: model
integer, allocatable :: labels(:),sample_labels(:)
real(dp), allocatable :: probabilities(:,:),draws(:,:)
model%n_components=8
model%weight_concentration_prior_type='dirichlet_process'
model%weight_concentration_prior=.1_dp
model%covariance_type='full'
model%max_iter=300
call model%fit(x)
probabilities=model%predict_proba(query)
labels=model%predict(query)
call model%sample(100,draws,sample_labels)
```

fit_predict returns fitted zero-based labels. score_samples returns per-row
variational log scores; score returns their mean. init_params accepts kmeans,
random, random_from_data or k-means++. Both weight prior families and all four
covariance forms are supported. warm_start=true continues a compatible fitted
posterior. max_iter=0 performs initialization only.

Native scalar zero sentinels select default weight concentration 1/K and degrees
of freedom p; mean precision defaults to one. Allocate mean_prior(p) or
covariance_prior(p,p) for explicit priors. Diagonal covariance priors are positive
diagonal matrices, and spherical priors are positive scalar identity matrices.
All fitted covariance/precision arrays have shape (p,p,K), with repeated tied
matrices and expanded diagonal/spherical entries. weight_concentration_ is (K,2)
for Beta parameters or (K,1) for finite-Dirichlet concentrations. Resolved priors
have trailing underscores. lower_bounds_ stores only the selected run's bounds.
See COVERAGE.md for score semantics, warm/refit rules, history, storage cost,
prior validation and native sampling reproducibility versus NumPy.


## HDBSCAN

Import hdbscan from sklearn, sklearn_cluster or sklearn_hdbscan.

```fortran
type(hdbscan) :: model
integer, allocatable :: labels(:),cut_labels(:)
model%min_cluster_size=8
model%min_samples=5
model%cluster_selection_method='eom'
model%store_centers='both'
labels=model%fit_predict(x)
! model%probabilities_ gives membership strengths, not a probability matrix.
! model%centroids_ and model%medoids_ have shape (n_clusters_,n_features_in_).
cut_labels=model%dbscan_clustering(.7_dp,min_cluster_size=5)
```

fit stores labels_ and probabilities_. Labels start at zero; -1 is noise,
-2 marks infinite feature rows and -3 marks missing rows (NaN strength).
min_samples=0 selects min_cluster_size, and max_cluster_size=0 means no cap.
Selection supports eom/leaf, nonnegative cluster_selection_epsilon and
allow_single_cluster. Metrics include euclidean, sqeuclidean, manhattan,
cityblock, chebyshev, minkowski with p>=1, cosine and precomputed. Centers are
optional (none/centroid/medoid/both) and unavailable for precomputed inputs.

The backend always uses dense brute semantics, including alpha scaling before
core distances. DBSCAN cuts use strict height<cut_distance; their default
minimum size is five independently of fit. Raw children_ uses one-based leaf
IDs and raw_N+i merge IDs, whereas condensed/core/stability diagnostics use the
finite subset mapped by finite_indices_. See COVERAGE.md for storage, tie
behavior, diagnostics, nonfinite handling and source corrections.


## BernoulliRBM

Import bernoulli_rbm from sklearn, sklearn_neural_network or sklearn_rbm.

```fortran
type(bernoulli_rbm) :: model
real(dp), allocatable :: features(:,:),draws(:,:),scores(:)
model%n_components=16
model%batch_size=10
model%n_iter=20
model%learning_rate=.05_dp
model%random_state=42
features=model%fit_transform(x)
call model%partial_fit(next_batch)
features=model%transform(query)
draws=model%gibbs(query)
scores=model%score_samples(query)
```

components_ is (n_components,n_features_in_); intercept_hidden_ and
intercept_visible_ are vectors; h_samples_ is the persistent binary hidden
chain. transform returns probabilities, gibbs returns real64 zeros/ones,
and score_samples returns one pseudo-log-likelihood value per row. fit resets
model arrays, and n_iter=0 initializes only. partial_fit performs one update on
the full supplied batch, preserving its chain; it does not use n_iter to repeat
updates. Use fit after changing n_components. A changed batch_size affects
chain size only on initialization/fresh fit.

Integer-seeded fit uses a local RNG; partial_fit and Gibbs share a separate
persistent stream. fit preserves that stream if it already exists. Scoring
restarts its scalar-seeded RNG each call without consuming the sampling stream.
All inputs must be nonempty dense finite arrays with the fitted feature count.
See COVERAGE.md for distribution assumptions, RNG contracts and supported scope.


## LatentDirichletAllocation

Import latent_dirichlet_allocation from sklearn, sklearn_decomposition or
sklearn_lda. It operates on dense nonnegative document/word count arrays.

```fortran
type(latent_dirichlet_allocation) :: model
real(dp), allocatable :: topics(:,:),gamma(:,:)
real(dp) :: evidence,perplexity
model%n_components=5
model%learning_method='batch' ! or 'online'
model%max_iter=20
model%random_state=42
topics=model%fit_transform(counts)
gamma=model%transform(query,normalize=.false.)
evidence=model%score(query)
perplexity=model%perplexity(query)
model%total_samples=10000
call model%partial_fit(next_documents)
```

components_ contains topic-word Dirichlet pseudocounts in (K,vocabulary_size)
layout; exp_dirichlet_component_ stores exponentiated expected log probabilities.
Transform rows normally sum to one. doc_topic_prior/topic_word_prior=-1 choose
default 1/K; explicit priors must be positive and <=1. Zero iteration budgets
are allowed. partial_fit always uses online minibatches and preserves its RNG,
resolved priors and existing fitted state. Fresh fit resets parameters and RNG.
Inference is deterministic and does not advance the stream.

bound_ is training perplexity from the last fit; it is NaN for partial-only
models and remains stale after incremental updates. score is the evidence bound;
perplexity optionally accepts sub_sampling=.true. n_iter_ follows source's
early-stop counter convention; n_batch_iter_ counts EM updates plus one.
See COVERAGE.md for scaling, zero-word/overflow behavior, numerical scoring
corrections, parameter limits and validation against scikit-learn.


## MiniBatchDictionaryLearning and MiniBatchSparsePCA

Import mini_batch_dictionary_learning and mini_batch_sparse_pca from sklearn,
sklearn_decomposition or sklearn_dictionary.

```fortran
type(mini_batch_dictionary_learning) :: dictionary
type(mini_batch_sparse_pca) :: projection
real(dp), allocatable :: code(:,:),scores(:,:),reconstruction(:,:)
dictionary%n_components=8
dictionary%fit_algorithm='cd'
dictionary%transform_algorithm='lasso_cd'
dictionary%alpha=.1_dp
call dictionary%partial_fit(first_batch)
call dictionary%partial_fit(next_batch)
code=dictionary%transform(query)
projection%n_components=3
projection%max_iter=30
scores=projection%fit_transform(x)
reconstruction=projection%inverse_transform(scores)
```

Dictionary n_components=0 selects the feature count. Allocate dict_init(rows,p)
for a supplied initializer; it is copied, truncated or zero-padded. fit uses
randomized SVD otherwise and resets RNG/statistics. partial_fit performs one
whole-array update and preserves the initialized solver, RNG and statistics,
regardless of batch_size/max_iter. Use fit after changing component count.
transform supports all five existing sparse coding methods and split_sign.
transform_alpha=-1 defaults to alpha; transform_n_nonzero_coefs=0 means auto.

max_no_improvement=-1 disables patience, and zero stops at the first eligible
check. tol=0 disables dictionary-motion stopping. n_steps_ is the update count;
n_iter_ is refreshed by fit. a_/b_ and ewa_cost_/ewa_cost_min_/no_improvement_
expose native diagnostics corresponding to source private state.

MiniBatchSparsePCA centers data and trains on X.T, with method='lars'/'cd'.
Its components_ rows are sparse unit vectors or zeros, mean_ restores feature
means and ridge_alpha controls projection regularization. It supports fit,
fit_transform, transform and inverse_transform. See COVERAGE.md for randomized
initialization, streaming/convergence rules, state lifetimes and supported scope.


## MiniBatchNMF

Import mini_batch_nmf from sklearn, sklearn_decomposition or
sklearn_minibatch_nmf.

```fortran
type(mini_batch_nmf) :: model
real(dp), allocatable :: factors(:,:),reconstruction(:,:)
model%n_components=4
model%init='nndsvdar'
model%beta_loss=1._dp ! KL; use 2 for Frobenius or 0 for Itakura-Saito.
model%batch_size=32
model%max_iter=50
factors=model%fit_transform(x)
reconstruction=model%inverse_transform(factors)
call model%partial_fit(next_batch)
factors=model%transform(query)
```

n_components=0 selects source auto rank; init='auto' selects source default
initialization. All random/NNDSVD/custom initializers and finite numeric beta
losses are supported. Pass W=initial_w,H=initial_h to fit/fit_transform or the
first partial_fit with init='custom'. Custom auto rank is inferred from H.
Inputs and custom arrays are preserved. beta<=0 requires strictly positive X.

alpha_w/alpha_h and l1_ratio control regularization; alpha_h=-1 means same as
alpha_w. fresh_restarts optionally re-solves training coefficients, using
fresh_restarts_max_iter. transform_max_iter=0 caches max_iter as its limit.
max_no_improvement=-1 disables patience; tol=0 disables motion stopping.
partial_fit uses one whole-array update with fresh coefficients and preserves
cached rank, beta, forgetting weight and transform limit. Use fit to reset state.

components_ is (K,p); online numerator/denominator arrays share that layout.
n_steps_ counts updates and n_iter_/reconstruction_err_ describe the last fit;
streaming does not refresh the latter fields. Partial-only error is NaN.
See COVERAGE.md for loss conventions, clipping, cached controls, zero handling
and the difference between training factors and a fresh transform.


## Multi-task cross-validation

Import multi_task_elastic_net_cv and multi_task_lasso_cv from sklearn,
sklearn_linear_model or sklearn_multitask.

```fortran
type(multi_task_elastic_net_cv) :: model
real(dp), allocatable :: prediction(:,:)
model%cv=5
model%n_alphas=30
model%l1_ratios=[.2_dp,.7_dp,1._dp]
model%selection='random'
model%random_state=42
call model%fit(x,y) ! y is (n_samples,n_targets), even for one target.
prediction=model%predict(query)
! alpha_, l1_ratio_, best_mse_ report selection.
```

Allocate alphas(:) for explicit nonnegative candidates; otherwise eps/n_alphas
control a full-data alpha grid. Zero mixing requires explicit alphas. LassoCV
fixes mixing to one. Allocate test_fold(N) for predefined validation folds,
with -1 for training-only rows and all other integer labels identifying folds.
A predefined map can contain one holdout; integer cv otherwise needs 2..N splits.

alphas_ is (n_alpha,n_ratio); mse_path_, fold_n_iter_ and fold_dual_gaps_ are
(n_alpha,n_fold,n_ratio). Singleton axes remain present. coef_/intercept_ use
the base multi-task layouts, and predict/score share its two-dimensional targets
and uniform R2. Final refits are cold regardless of inherited warm_start.

Source NumPy/uint32 seeded coordinate selection is supported by base and CV
models. At alpha=0 the mixing ratio is inactive and last-bit validation ties can
select a different equivalent ratio; effective penalties and fit results agree.
See COVERAGE.md for grid ordering, stopping, folds, diagnostics and current scope.


## Cross-validated orthogonal matching pursuit

Import `orthogonal_matching_pursuit_cv` from `sklearn` or
`sklearn_linear_model` (also available directly in `sklearn_omp_cv`).

```fortran
type(orthogonal_matching_pursuit_cv) :: model
! X(N,P), y(N); at least two features.
model%cv=5
model%max_iter=0  ! automatic min(max(P/10,5),P)
call model%fit(x,y)
prediction=model%predict(query)
print *, model%n_nonzero_coefs_,model%best_mse_
! mse_path_(active_step,fold), truncated to the shortest fold path.
```

`fit_intercept` defaults to true. `max_iter` caps active features; zero uses
the source default, and a positive value cannot exceed P. Selected sparsity is
`n_nonzero_coefs_`; `n_iter_` describes the full-data refit. Further diagnostics
are `n_features_in_`, `n_splits_`, `fold_n_iter_`, `fold_train_sizes_`, and
`fold_test_sizes_`. `score(X,y)` returns R2. Inputs are preserved.

For a predefined split, allocate `test_fold(N)` with integer test-fold labels;
-1 means training-only. For arbitrary partitions, allocate both
`train_mask(N,F)` and `test_mask(N,F)` instead, using one true/false membership
column per fold. For example, a rolling window can train on rows 1:20 and test
on 23:27 by marking precisely those rows, leaving a two-row gap. Each fold
needs nonempty train and test sets. Clear allocated controls before switching
partition modes. Masks use input row order and cannot encode permutations or
duplicate rows. Multi-output targets, weights,
parallel fitting and Python splitter objects are not supported.

Path-tail iteration counts and nonunique coefficient representations may differ
at machine precision; see [coverage](COVERAGE.md#orthogonalmatchingpursuitcv-2026-11-20).
No zero-feature candidate is scored; zero-step fold paths raise an explicit
diagnostic. See `example/omp_sparsity_selection.f90` for a complete program.


## Statistical univariate feature selection

`select_percentile`, `select_fpr`, `select_fdr`, `select_fwe` and
`generic_univariate_select` are exported by `sklearn`, `sklearn_feature_selection`
and `sklearn_univariate_selection`.

```fortran
type(select_fdr) :: selector
! Integer y selects classification; real y selects regression.
selector%alpha=.05_dp
call selector%fit(x,y)
mask=selector%get_support()
indices=selector%get_support_indices() ! one-based Fortran indices
selected=selector%transform(x)
restored=selector%inverse_transform(selected) ! removed columns become zero
```

The threshold types use `alpha` (default .05) in [0,1]. FPR uses p<alpha;
FWE uses p<alpha/P; FDR is the inclusive Benjamini-Hochberg step-up rule.
SelectPercentile uses `percentile` (default 10) in [0,100], NumPy's linear
percentile threshold and input-order tie filling. GenericUnivariateSelect uses
`mode='percentile'` (default) or `'k_best'`, `'fpr'`, `'fdr'`, `'fwe'` and real
`param` (default 1.e-5). In k_best mode param must be integral, zero selects
none, k>P selects all and `k_all=.true.` represents the source `'all'` option.
Top-k ties favor later columns, matching source stable sorting. Controls are
read again by get_support/transform/inverse_transform, so changing them after
fitting changes the projection without rescoring.

Default `score_func='auto'` uses f_classif for integer labels and f_regression
for real targets, following the existing Fortran select_k_best convention.
Classification accepts `score_func='chi2'` (also an optional third fit argument),
requiring nonnegative X. Regression accepts `center=.false.`; force_finite=True
is fixed. Fitted `scores_`, `pvalues_` and `n_features_in_` are public. The helper
subroutines `univariate_classification_scores(X,y,scores,pvalues,score_func)` and
`univariate_regression_scores(X,y,scores,pvalues,center)` return allocated arrays.
The existing score-only f_classif/f_regression/chi2 functions remain unchanged.

For a custom scoring routine, call `fit_scores(scores,pvalues)` after computing
its arrays. Percentile and k_best also accept `fit_scores(scores)` alone;
probability-threshold modes require p-values. Score NaNs/infinities are supported
with source selection semantics. Inputs to ordinary fit and transform must be
finite dense real64 arrays. Sparse matrices, object labels, feature names and
Python callback objects are outside this interface.

Transforming to zero columns is supported. The native inverse of an N-by-zero
matrix returns an N-by-P zero matrix, consistent with the other native selectors;
sklearn 1.6.1's dense inverse raises a minimum-feature error on that input.
Near-perfect regression F scores are ill-conditioned; retained probabilities
and masks are source checked, and separately reported score checks certify the
underlying squared correlation within 64 machine epsilons. See
[coverage](COVERAGE.md#statistical-univariate-selectors-2026-11-20) and
`example/statistical_feature_selection.f90`.

## PolynomialCountSketch

Import `polynomial_count_sketch` from `sklearn`,
`sklearn_kernel_approximation` or `sklearn_polynomial_count_sketch`.

```fortran
type(polynomial_count_sketch) :: features
features%degree=3
features%gamma=.5_dp
features%coef0=1
features%n_components=1009
features%random_state=42
z=features%fit_transform(x)
query_features=features%transform(query)
! dot_product(z(i,:),z(j,:)) approximates
! (gamma*dot_product(x(i,:),x(j,:))+coef0)**degree.
```

Defaults are `degree=2`, `gamma=1`, `coef0=0`, `n_components=100` and native
`random_state=0`. `fit(X)` builds seeded hash tables; `fit_transform(X)` fits
and returns an allocated N-by-n_components matrix. Degree and component count
must be positive, gamma finite and nonnegative, and coef0 finite. Negative
coef0 is accepted by sklearn 1.6.1 but produces all-NaN transformed features;
the native implementation returns quiet NaNs explicitly, including with
floating-point traps enabled. Use nonnegative coef0 for a real polynomial map.

Learned `index_hash_` and `bit_hash_` have shape `(degree,augmented_features)`;
the augmented count is P+1 when coef0 is nonzero and P otherwise. Hash buckets
are **zero-based**, matching source `indexHash_`, while signs are -1 or +1.
This is an intentional exception to ordinary one-based native support indices.
Scalar seeds in [0,huge(default integer)] reproduce the source NumPy legacy
RandomState draw order. `n_features_in_` records P; `n_features_out_` records
the fitted component count. A full refit replaces the tables and metadata.

Source live transform controls are retained: gamma and coef0 magnitudes can
change; degree can shrink to a prefix of fitted hash rows; component count can
change when every retained bucket still fits. The output uses the current
component count, while `n_features_out_` remains the fitted value, as in source.
Switching coef0 between zero and nonzero changes the augmented width and
requires refitting. Growing degree beyond fitted rows or excluding a stored
bucket raises an explicit native diagnostic. Native transform revalidates
controls, including invalid post-fit changes that source may reject less
directly. Changing random_state affects the next fit, not existing tables.

Dense finite real64 inputs, scalar seeds and allocated outputs are supported.
Sparse matrices, Python RNG objects/global RNG state, feature-name methods and
metadata routing are outside this interface. FFT work is O(degree*D*log(D))
per sample, plus O(degree*P) hashing, using radix-2 and Bluestein paths; the
plan is reused within each transform call. Native SciFort FFT helpers are
vendored under MIT with attribution, without a runtime library dependency.
See `example/polynomial_kernel_features.f90` for a complete program.

## Label and multilabel binarization

`label_binarizer`, `multi_label_binarizer` and the native ragged container
`label_set` are exported by `sklearn`, `sklearn_preprocessing` and
`sklearn_label_binarization`. The container holds one sample's integer labels;
it is supporting data representation, excluded from core algorithm counts.

```fortran
type(label_binarizer) :: target
integer, allocatable :: indicator(:,:),restored(:)
indicator=target%fit_transform([-3,7,7,-3]) ! shape (4,1), positive class 7
restored=target%inverse_transform(indicator)
```

`fit(y)` and `fit_transform(y)` accept a nonempty rank-one integer target.
Learned `classes_` are sorted; `y_type_` is 'binary' for one/two classes and
'multiclass' otherwise. `neg_label=0` and `pos_label=1` are configurable
integers, with neg_label<pos_label. Encoding returns an allocated integer
matrix. Ordinary binary targets use one positive-class column; multiclass
targets use one column per class. A one-class fit normally encodes all labels
as neg_label. Unknown query labels receive negative entries.

The source determines transform target type from each query: a query with
three or more distinct labels uses all fitted class columns, even after a
binary fit. With a one-class fit, this can encode the known class positively.
Changing pos_label/neg_label affects subsequent transforms; refitting replaces
the classes and target type. Transform rejects empty queries, as in source.

`inverse_transform(scores,threshold)` accepts rank-two integer or real64
scores and returns an allocated integer target. Binary inverse uses strict
score>threshold, defaulting to the midpoint of pos_label and neg_label. With
two score columns it uses column two. Multiclass inverse ignores threshold,
selects the first maximum per row and clips an extra-column winning index to
the last class, following source. Empty inverse row counts are supported.
Native inverse requires finite scores, positive score width and a finite
binary threshold. A one-class two-column inverse selecting the absent positive
class receives an explicit diagnostic where source raises IndexError.
Rank-two multilabel-indicator fitting, string/object labels and sparse outputs
are outside this LabelBinarizer subset.

```fortran
type(multi_label_binarizer) :: tags
type(label_set) :: samples(3)
type(label_set), allocatable :: sets(:)
samples(1)%values=[10,30,10] ! duplicates become one indicator
samples(2)%values=[20,30]   ! sample 3 is empty (values unallocated)
tags%classes=[30,10,20]     ! optional constructor control, in requested order
indicator=tags%fit_transform(samples)
sets=tags%inverse_transform(indicator)
```

Without allocated `classes`, fit discovers a sorted union. With allocated
`classes`, fit preserves its order and requires unique entries. Empty class
lists and empty datasets are supported. Learned `classes_` is a snapshot;
changing constructor classes affects the next fit. Deallocate `classes` to
resume automatic discovery. Repeated labels are idempotent; unknown transform
labels are ignored, matching source values without emitting Python warnings.

The output is an N-by-C integer 0/1 matrix, including N-by-zero when no classes
are fitted. Both integer and real64 inverse indicators are accepted, but every
entry must be exactly zero or one. Inverse returns allocated label_set rows
in fitted class order, including allocated zero-length values for empty sets.
Unallocated input values and allocated zero-length values both represent an
empty set. Inputs are preserved. Generic Python iterables/object labels,
sparse matrices and feature-name integration are outside this interface.
See `example/label_encoding.f90` for a complete program.

## Mutual-information and correlation scores

`mutual_info_classif`, `mutual_info_regression` and `r_regression` are exported
by `sklearn`, `sklearn_feature_selection` and `sklearn_mutual_info`.
`mutual_info_score` is exported by `sklearn`, `sklearn_metrics` and the focused
module. These add scoring algorithms; the core estimator type count is unchanged.

```fortran
! X(N,P): dense real64. y_real(N): real64; y_class(N): integer.
scores=mutual_info_regression(x,y_real,discrete_features=[3],random_state=42)
class_scores=mutual_info_classif(x,y_class,n_neighbors=3,random_state=42)
correlations=r_regression(x,y_real)
partition_information=mutual_info_score(labels_true,labels_pred)
```

The MI scorers return allocated P-element real64 arrays of nonnegative
information estimates in nats. Dense inputs default to all continuous features.
`discrete_features` accepts a scalar logical, a P-element logical mask, or a
nonempty integer list of **one-based** column indices. Repeated indices are
allowed; zero, negative and out-of-range indices are rejected. This follows
native indexing conventions rather than Python's zero-based/negative indexing.
Discrete feature values are finite real64 category codes compared exactly;
fractional numeric category codes are allowed. Classification targets are
integer labels. Regression targets are real64 and treated as continuous.

`n_neighbors=3` is the default. It must be positive. Continuous/continuous
estimation needs n_neighbors<N. Continuous/discrete estimation adapts to each
category's size and removes singleton categories, matching source; at least
one repeated category is required. A requested count above N is valid on paths
that adapt per category. Discrete/discrete estimates use contingency counts.
Negative estimates are clipped to zero. The native search is brute-force with
O(N**2) distance/count work per feature and O(N) scratch storage, including
quickselect for neighbor radii. No KD-tree, SciPy or Python runtime is required.

Native `random_state=0` is deterministic. Nonnegative default-integer scalar
seeds reproduce NumPy legacy RandomState Gaussian draws in source C order.
Continuous columns/targets are variance-scaled without centering and receive
source-sized 1.e-10 Gaussian perturbations. Feature/target near-zero-scale rules
are distinguished. NumPy pairwise sums and source-selected Euclidean distance
arithmetic preserve radius boundaries that can change discrete neighbor counts.
Small tied/near-tied distances remain sensitive to platform floating arithmetic.
Inputs are always preserved; native copy=False, sparse inputs, RNG objects,
joblib parallelism and object/string labels are not exposed.

`r_regression(X,y,center,force_finite)` returns signed Pearson correlations,
with both logical options defaulting to true. center=False computes uncentered
cosines. The source moment formula, division order and unclipped scores are
retained. force_finite replaces NaNs with zero and preserves signed infinities,
including when a nonzero target's squared norm underflows. Explicit IEEE paths
produce these values with floating traps enabled. Exact constant columns can
produce source/native differences among NaN, zero, signed infinity or tiny
roundoff values because BLAS reduction order changes an undefined correlation.
The reference suite reports those checks with an explicit constant-column
certificate; ordinary correlations remain direct source comparisons.

`mutual_info_score(labels_true,labels_pred)` accepts equally sized, nonempty
integer cluster-label arrays and returns discrete MI, invariant under label
renaming. Its zero-entropy case is zero; a precomputed contingency matrix is
not exposed. All score functions reject empty/nonfinite feature/real-target
data and inconsistent lengths. Sparse matrices and estimator callbacks are
outside the interface. To select columns using MI, pass its score vector to
`generic_univariate_select%fit_scores`, using a score-only mode such as k_best
or percentile. See `example/nonlinear_feature_scoring.f90`.

## Clustering information and pair agreement

`sklearn` and `sklearn_metrics` export `normalized_mutual_info_score`,
`adjusted_mutual_info_score`, `fowlkes_mallows_score`, `contingency_matrix`,
`pair_confusion_matrix`, `entropy`, `expected_mutual_information` and
`homogeneity_completeness_v_measure`. These eight functions/subroutines are
also available directly from `sklearn_cluster_information`. Existing
`homogeneity_score`, `completeness_score` and `v_measure_score` use this backend.

```fortran
use, intrinsic :: iso_fortran_env, only: int64
! truth(:), predicted(:): default integers; labels need not be consecutive.
integer(int64), allocatable :: counts(:,:)
integer(int64) :: pairs(2,2)
real(dp) :: normalized,adjusted,h,c,v
counts=contingency_matrix(truth,predicted)
pairs=pair_confusion_matrix(truth,predicted)
normalized=normalized_mutual_info_score(truth,predicted)
adjusted=adjusted_mutual_info_score(truth,predicted,average_method='arithmetic')
call homogeneity_completeness_v_measure(truth,predicted,h,c,v,beta=2._dp)
```

All paired inputs must have equal lengths. Contingency rows follow sorted true
labels and columns sorted predicted labels. Without eps, contingency_matrix
returns an allocated **int64** count matrix. Supplying real64
`eps=...` selects its real64 overload, adding a finite nonnegative value to
every cell, including zeros. Sparse output and dtype overrides are not exposed.
`pair_confusion_matrix` returns an int64 2-by-2 matrix with rows `[TN,FP]` and
`[FN,TP]`; these are **ordered** distinct-sample pairs, summing to N*(N-1).
Use int64 output variables because ordinary datasets can exceed 32-bit pair
counts. Fowlkes-Mallows is sqrt(TP/(TP+FP))*sqrt(TP/(TP+FN)), or zero when TP=0.

MI averages accept `'min'`, `'geometric'`, `'arithmetic'` (default), or `'max'`.
NMI divides observed MI by the chosen entropy average. AMI subtracts the
hypergeometric expectation and retains the source signed-epsilon denominator
floor; negative adjusted scores are valid. Entropy and expected MI use natural
logarithms (nats). `expected_mutual_information(counts,n_samples)` accepts
default-integer or int64 nonnegative matrices, with a nonnegative default-integer
sample count equal to their sum. Zero margins and nonempty all-zero tables are
supported. Empty matrix extents and work sizes outside native indexing limits
raise diagnostics. Smoothed real-valued contingency matrices are not raw counts.

Internal marginal/joint histograms use heap sorting and O(N) storage. Entropy,
observed MI, pair scores and H/C/V use O(N*log(N)) sorting work without a dense
K-by-L intermediate. Only the explicit contingency_matrix function materializes
that matrix. Expected MI additionally sums feasible hypergeometric counts over
all pairs of cluster margins, using cached log factorials; many clusters can
make AMI expensive. All routines preserve input labels. Only dense integer
labels are exposed; sparse matrices, object/string labels and parallel options
are outside this interface.

The source edge conventions are retained: empty NMI/AMI and H/C/V return one;
empty entropy returns one; empty contingency is 0-by-0; empty pair confusion is
all zeros; empty/single-sample Fowlkes-Mallows is zero. Individual H/C/V wrappers
now accept empty input, and `v_measure_score(y,p,beta)` supports the same finite
nonnegative beta as the combined subroutine. Beta=0 with zero completeness and
positive homogeneity raises an explicit diagnostic, corresponding to the
source's ZeroDivisionError; independent partitions with H=C=0 still return V=0.

All-singleton partitions, or min-normalized AMI with one singleton partition,
have a mathematically zero chance-adjustment denominator. Last-bit differences
between intrinsic log_gamma and SciPy/libc can substantially change the adjusted
score there. The native source formula is retained. Validation explicitly
certifies MI/expected-MI/entropy agreement and the epsilon-guarded ratio interval
on those exact degeneracies; other methods and ordinary AMI remain directly
source checked. See `example/clustering_agreement.f90`.

## Randomized, repeated and exhaustive resampling

`sklearn_model_selection` and the `sklearn` facade export seven additional
splitter types. The implementations and shared `index_split` type live in
`sklearn_resampling`, which can also be imported directly. Existing code
importing `index_split` from `sklearn_model_selection` remains valid. Each
`split` returns an allocated array of `index_split` values with allocated,
one-based integer `train` and `test` indices; inputs are unchanged.

| Type | Split call | Count call | Defaults |
| --- | --- | --- | --- |
| `shuffle_split` | `split(n_samples)` | `get_n_splits()` | `n_splits=10`, 10% test samples |
| `stratified_shuffle_split` | `split(labels)` | `get_n_splits()` | `n_splits=10`, 10% test samples |
| `group_shuffle_split` | `split(groups)` | `get_n_splits()` | `n_splits=5`, 20% test groups |
| `repeated_kfold` | `split(n_samples)` | `get_n_splits()` | `n_splits=5`, `n_repeats=10` |
| `repeated_stratified_kfold` | `split(labels)` | `get_n_splits()` | `n_splits=5`, `n_repeats=10` |
| `leave_p_out` | `split(n_samples)` | `get_n_splits(n_samples)` | `p=2` |
| `leave_p_groups_out` | `split(groups)` | `get_n_splits(groups)` | `n_groups=2` |

Labels/groups are rank-one default-integer arrays. Sample-count arguments are
scalar default integers. Randomized/repeated types have `random_state=0` by
default; nonnegative default-integer seeds produce the NumPy legacy
RandomState sequence. Each call starts a local stream, repeats retain that
stream internally, and calls neither seed nor consume Fortran's global RNG.
Python's `None`/RandomState-object inputs and seeds beyond the positive
Fortran default-integer range are not supported.

All three shuffle types support real64 `test_size` and `train_size`, with
`-1` meaning unspecified. Explicit fractions must lie strictly between zero
and one. Integer `test_count`/`train_count` fields, zero meaning unspecified,
provide the corresponding sklearn integer-size behavior; an explicitly
selected count must be positive and smaller than the sample/group population.
A count and fraction for the same side cannot both be selected. Test
fractions round upward; train fractions round downward. An unspecified side
uses the complement; two specified sides can leave samples/groups unused.
Two unspecified sides select the type's default test fraction. Training sets
must be nonempty and combined sizes cannot exceed the population.

ShuffleSplit preserves permutation order. StratifiedShuffleSplit uses sorted
class labels, source approximate hypergeometric quotas with random tie
breaking, per-class permutations and final train/test permutations. It
requires at least two samples per class and train/test sizes at least the
number of classes. These constraints do not guarantee every class occurs in
every holdout when counts are very imbalanced, matching sklearn. Group sizes
refer to distinct sorted group labels; selected groups are expanded to sorted
sample indices, with no group shared between training and testing.

Repeated splitters always shuffle. RepeatedKFold balances fold sizes with
larger folds first. RepeatedStratifiedKFold encodes classes by first occurrence,
allocates counts round-robin and shuffles fold assignments within each class.
A class with fewer than `n_splits` members is allowed when another class has
at least that many; sklearn emits a warning there, while Fortran returns the
same indices without a warning channel. Both return sorted training/testing
indices. Every sample is tested exactly once per repeat.

LeavePOut/LeavePGroupsOut require `1 <= p < n_samples` or
`1 <= n_groups < number_of_unique_groups`. They enumerate lexicographic
combinations of sample indices/sorted group labels, then return sorted sample
indices. `get_n_splits` computes the binomial count without materializing
folds. Counts beyond the native default-integer range are rejected before
allocation. All splitters eagerly materialize folds, unlike Python generators;
exhaustive splitting can consume combinatorial time and memory even when the
count fits an integer. Allocation failure is an error stop. No arbitrary
materialization cap is imposed.

See `example/resampling.f90` and `validation/compare_resampling.py` for usage
and exact source comparisons. These are core sklearn algorithms, with no new
public extensions or dependencies.

## Dense exact t-SNE and trustworthiness

`tsne` and `trustworthiness` are exported by `sklearn_manifold`, the `sklearn`
facade and their implementation module `sklearn_tsne`. `sklearn_tsne_core` is an
internal backend for the source's private affinity/objective/optimizer helpers;
its diagnostic trace interface is not part of the facade API.

```fortran
type(tsne) :: embedding_model
real(dp), allocatable :: embedding(:,:)
embedding_model%perplexity=8
embedding_model%max_iter=400
embedding_model%random_state=42
embedding=embedding_model%fit_transform(x)
! Or: call embedding_model%fit(x, init=initial_embedding)
print *, embedding_model%kl_divergence_, trustworthiness(x,embedding,4)
```

Parameters: `n_components=2`, `perplexity=30`, `early_exaggeration=12`,
`learning_rate=-1` (source `auto`), `max_iter=1000`,
`n_iter_without_progress=300`, `min_grad_norm=1.e-7`, `random_state=0`,
`init='pca'`, `metric='euclidean'`, `p=2` and `method='exact'`.
This port implements the dense exact solver. Python defaults to `barnes_hut`;
that method is rejected here. Exact t-SNE uses quadratic memory and quadratic
pair interactions per iteration. It supports dimensions above three, with
source degrees of freedom `max(n_components-1,1)` and source objective/gradient
formulas. At least two samples, positive dimension and
`0 < perplexity < n_samples` are required. Iterations must be at least 250,
patience positive, exaggeration at least one, and gradient threshold
nonnegative. Numeric controls must be finite; explicit learning rates must
be positive. Perplexities rounding to zero in float32, distances outside the
float32 working range, oversized native workspaces, zero Student normalization
and initial coordinates beyond the objective working range have diagnostics.

Both methods accept an optional real64 `init(:,:)` with exactly
`(n_samples,n_components)` entries. Supplying it overrides the initialization
string and follows the float64 optimization path. Otherwise `init='random'`
uses a local scalar-seeded NumPy-compatible Gaussian stream and `init='pca'`
uses the existing randomized SVD, source loading-based sign choice and PC1
population-standard-deviation scaling to 1.e-4. PCA requires a feature matrix,
a supported component count and nonzero PC1 scale. Random/PCA initialization
and embedding/gradient/gain state follow source float32 rounding, while public
arrays remain real64 storage. Automatic learning rates also preserve NumPy's
mixed float32/float64 update behavior. Seed range is the nonnegative Fortran
default-integer range; Python None/object RNG inputs are not implemented.

Metrics use the package's dense pairwise distances, including Euclidean,
Manhattan/cityblock, cosine, Chebyshev, Minkowski and squared Euclidean.
`p` configures the native Minkowski exponent. Every non-Euclidean metric
result is squared before constructing affinities, as in sklearn; selecting
`sqeuclidean` therefore gives fourth-power Euclidean distances. `metric='precomputed'` accepts
a square finite nonnegative distance matrix; it is squared internally,
matching the source's non-Euclidean path. Inputs and explicit initial arrays
are preserved, including precomputed input, which Python 1.6.1 can square in
place. Euclidean feature distances are computed directly as squared
differences; roundoff can differ from Python's Gram calculation.
Sparse matrices, callable metrics, general metric-parameter dictionaries,
parallel backends and Barnes-Hut/neighbor-graph approximation are unsupported.
There is no out-of-sample transform, inverse transform or prediction method.

Fitted attributes: `embedding_`, `kl_divergence_`, `learning_rate_`,
`n_iter_`, `n_features_in_`. Auto rate is
`max(n_samples/early_exaggeration/4,50)`. Fitting performs 250 early-exaggeration
steps with momentum .5, followed by normal probabilities with momentum .8;
gains and updates restart at the phase boundary. Stopping is checked every
50 steps using gain-scaled gradient norm and progress, retaining source
zero-based `n_iter_`. Early gradient stopping still enters the second phase.
The reported KL is evaluated before the final update, as in sklearn. With
`max_iter=250` and a full first phase, the source's empty second-phase call
reports the largest finite real64 KL value and `n_iter_=250`; this observable
convention is preserved. Select a longer run for a post-exaggeration KL.

Nonconvex trajectories and discrete gain decisions can amplify last-bit
reduction differences. Coordinate-by-coordinate parity is not guaranteed.
The validator compares affinities, objectives, gradients and short steps
directly, then certifies every step of explicit-initialization trajectories
against the source objective and NumPy update formulas. Recorded cancellation
bounds identify near-zero gain-sign ambiguities; replay identity links those
trajectories to the public fitted embedding, stopping and reported KL.

`trustworthiness(x,x_embedded,n_neighbors,metric,p)` returns the source rank
penalty with default five neighbors and Euclidean original-space distances.
It requires equal sample counts and `1 <= n_neighbors < n_samples/2`.
Embedded distances are Euclidean; original input can be features or a square
nonnegative precomputed distance matrix. Original/embedded distance ties use
increasing sample index, consistent with the native neighbor routines.
NumPy sorting and sklearn neighbor backends can resolve ties differently.
The reference suite records both source and native results for tied embedding
neighborhoods, checks the native stable-rank identity and certifies valid rank
bounds where original-space ranks are unique. See `example/nonlinear_embedding.f90`.

## Hessian eigenmapping and modified LLE

`locally_linear_embedding` now supports all four sklearn methods through the
existing `sklearn_manifold` and `sklearn` imports:

```fortran
type(locally_linear_embedding) :: model
model%n_neighbors=11
model%n_components=2
model%method='hessian'             ! Or 'modified'.
embedding=model%fit_transform(x)
query_embedding=model%transform(query)
```

New controls are `hessian_tol=1.e-4` and `modified_tol=1.e-12`, both finite and
nonnegative, including when another method is selected. The existing
`method='standard'`, neighbor/component controls and reg defaults are unchanged.
Hessian eigenmapping requires `n_neighbors > d*(d+3)/2`, where
`d=n_components`; the integer dimension expression is checked in int64 before
workspace allocation. It centers each neighborhood, estimates tangent vectors
using SVD or covariance eigenvectors, forms constant/linear/quadratic features,
uses a full Householder QR basis and applies the source column-sum tolerance
normalization before accumulating the local alignment blocks.

Modified LLE requires `n_neighbors >= n_components`, then computes full local
spectral bases of neighbor-minus-center differences. It uses the source fixed
`1.e-3*local_eigenvalue_sum` regularizer for affine weights, the median
small/large spectral energy ratio, source cumulative-spectrum/searchsorted
selection of each almost-null-space size and the multiple-weight Householder
alignment update. This fit regularizer is independent of the public `reg`
control. The data can still select an empty local nullspace, a zero spectrum
or a zero Householder denominator at tolerance zero; these cases fail with
explicit diagnostics instead of passing invalid arithmetic to the eigensolver.
Zero Hessian normalization sums also have a diagnostic.

All four methods share the existing dense global Jacobi eigensolver, analytic
constant-mode removal with a Helmert basis, canonical signs, fit/fit_transform
and Euclidean barycentric transform. Native neighbor ties use stable increasing
indices and always exclude the training point itself. New-method weights_
remains zero: a single standard-LLE fitting weight matrix does not represent
these objectives. Public `reg` controls their query transforms. Changing it
before a refit does not change their fitted alignment, as in sklearn. Query
transforms use the fitted neighborhood size and training coordinates.

The internal `sklearn_lle_variants` backend provides local spectral and
alignment helpers for implementation/reference validation; these are not
exported by the facade. It uses the existing native Jacobi SVD/eigensolvers,
full orthogonal completion and Householder QR, with no new dependency.
Modified LLE retains the source O(samples*neighbors**2) spectral storage as
well as dense global alignment storage. These are dense small/medium-data
implementations; sparse graphs and ARPACK are not implemented.

Local nullspace frames and repeated/near-zero global alignment eigenvalues
are not unique. The source's median/cumulative spectral cutoff can also change
an integer nullspace size at a rounding-level equality. The validator checks
actual local eigensystems and alignment assembly, compares the source matrix
directly when cutoffs agree, and requires a bounded near-threshold witness and
source-size counterfactual alignment when they differ. Nonunique fitted
subspaces require orthogonality, zero-mean, eigen-residual, lowest restricted
energy and source barycentric-query checks. Ordinary unique embeddings and
transforms are compared directly. Very small Hessian normalization tolerances
or large modified Householder tolerances can make results depend strongly on
local QR/SVD frames; eigenvalue ties and duplicate-distance neighborhoods can
also differ across native and sklearn linear-algebra/neighbor backends.
See `example/lle_variants.f90` and `validation/compare_lle_variants.py`.

## Ranking curves, probability calibration and weighted scores

`sklearn_metrics` and `sklearn` export `roc_curve`, `precision_recall_curve`,
`det_curve`, `auc` and `brier_score_loss`. `calibration_curve` is exported by
`sklearn_calibration` and `sklearn`. Existing `roc_auc_score` and
`average_precision_score` now share the `sklearn_ranking` threshold backend.
Their first three arguments remain compatible; new optional controls are:

```fortran
call roc_curve(y,scores,fpr,tpr,thresholds, &
  pos_label=1,sample_weight=weights,drop_intermediate=.true.)
call precision_recall_curve(y,scores,precision,recall,thresholds, &
  pos_label=1,sample_weight=weights,drop_intermediate=.false.)
call det_curve(y,scores,fpr,fnr,thresholds,pos_label=1,sample_weight=weights)
area=auc(fpr,tpr)
area=roc_auc_score(y,scores,pos_label=1,sample_weight=weights,max_fpr=.5_dp)
ap=average_precision_score(y,scores,pos_label=1,sample_weight=weights)
loss=brier_score_loss(y,prob,pos_label=1,sample_weight=weights)
call calibration_curve(y,prob,prob_true,prob_pred,n_bins=5,strategy='quantile',pos_label=1)
```

Targets are rank-one default integers and scores/probabilities/weights are
rank-one real64 arrays. Inputs must be nonempty, length-matched and finite.
Returned curve arrays are allocatable real64 outputs. Scores can be arbitrary
finite decision values; Brier/calibration probabilities must lie in [0,1].
The shared stable merge sort is ascending-stable then reversed, matching the
source tie order. Zero-weight samples are removed before threshold creation
and positive-label inference. True/false weighted cumulative counts are
sample-by-sample, and all tied scores form one threshold. Sorting is
O(samples*log(samples)) rather than the former quadratic ROC AUC/AP paths.
An explicit nonzero-weight sample is required when weights are used by curves.
Signed weights retain source arithmetic; rates/scores can leave their usual
ranges, and DET can return empty arrays. The sum of absolute weights must
remain within the native working range (at most huge(real64)/8).

ROC thresholds decrease, beginning with positive infinity and the origin.
The default drop_intermediate=true removes source second-difference-collinear
points. Precision-recall thresholds increase; precision/recall have one extra
terminal point (1,0). Their default drop_intermediate=false can optionally
remove interior constant-true-positive points. DET retains the source trimmed
interval from the initial false-positive plateau through the first full
true-positive count, reversed to increasing thresholds.

Without pos_label, ROC/PR/DET/calibration infer 1 only for targets contained in
{0,1} or {-1,1}. Other integer labels require an explicit positive label.
ROC/PR can binarize multiclass targets with an explicit label; DET requires
exactly two original classes. ROC AUC/AP/Brier/calibration accept binary or
single-class targets only. AP defaults to positive label 1; for two classes
that label must occur, while a singleton can omit it. ROC AUC defaults to the
larger label, retaining the existing native pos_label convenience as explicit
binary relabeling. Brier uses source canonical-label inference and falls back
to the largest numeric label for other label sets, including arbitrary
singletons. These defaults intentionally differ among the source functions.

No-positive PR recall is one at each threshold, followed by terminal zero;
AP is zero there. Undefined ROC rates and single-class ROC AUC are quiet IEEE
NaNs, with no Python warning channel. This single-class AUC behavior replaces
the former native error stop. Zero weights can remove a class even when two
original labels occur: full AUC then returns NaN. Partial AUC returns NaN when
positive support is removed, but requires negative-class support, matching
sklearn 1.6.1's otherwise failing one-point partial curve. DET uses IEEE
NaN/infinity for zero weighted denominators. AP is the source step integral,
not trapezoidal PR area.

Optional max_fpr lies in (0,1]. A value below one truncates/interpolates the
ROC curve and applies the source McClish standardized partial-area correction.
Generic auc accepts at least two equal-length finite coordinate vectors;
x must be monotonic in either direction, allowing repeated coordinates.
It preserves signed y-area and corrects direction for descending x. Signed
weights can make ROC x nonmonotonic, which AUC rejects as in sklearn.
Brier returns the weighted mean squared probability error; weight sum zero
(including signed cancellation) is rejected.

Calibration defaults to five uniform bins. Positive n_bins selects bin count;
strategy is uniform or quantile. Quantile edges use source NumPy linear
percentiles; duplicate edges are allowed. Left search assigns values equal
to an interior edge to the lower bin. Empty bins are omitted and outputs are
observed positive fraction and mean predicted probability in increasing bin
order. The calibration_curve function has no sample_weight argument in sklearn
1.6.1. Calibrated estimators are documented below; multilabel/multiclass score
matrices and plot displays remain outside the ranking-functions batch. See example/ranking_curves.f90 and validation/compare_ranking.py.

## Self-training classification

`self_training_classifier` is available from `sklearn_semi_supervised` and
`sklearn`. Pass a configured `logistic_regression`, `gaussian_nb` or
`kneighbors_classifier` as the required estimator argument to fit:

```fortran
type(self_training_classifier) :: model
type(kneighbors_classifier) :: estimator
estimator%n_neighbors=3
model%criterion='threshold'
model%threshold=.75_dp
model%max_iter=10
call model%fit(x,y,estimator)
prob=model%predict_proba(query)
labels=model%predict(query)
```

Dense real64 feature matrices have samples in rows; integer label -1 marks
unlabeled samples. All other integer labels, including other negative labels,
are ordinary classes. The estimator is copied and fitted internally; the input
estimator and arrays are preserved. Its native fitting and probability
semantics apply, including logistic's requirement for two initial classes and
KNN's minimum initial training size. Gaussian NB and KNN also support one
initial class. Other base-estimator types, sparse inputs, metadata routing,
sample-weight routing and verbose output are not implemented.

`criterion='threshold'` selects probabilities strictly greater than threshold
(default .75; finite and in [0,1)). `criterion='k_best'` labels up to k_best
samples per iteration (default 10, positive). Equal confidence selects earlier
input rows first using a stable O(N log N) merge sort. sklearn's argpartition
does not specify this tied-cutoff choice; native/source histories may differ
there, including probabilities saturated near one within rounding error.
The reference suite certifies every such step against source-estimator
probabilities and separately records these cases.

`max_iter` defaults to 10; zero performs only the final supervised fit, and
-1 represents Python None (unlimited). Iteration stops after all samples
are labeled, no new samples are selected, or the iteration limit. As in
sklearn, max_iter takes precedence over no_change and all_labeled takes
precedence over both. Every successful fit performs a final base-estimator
refit using all currently labeled samples. Initially labeled rows have
`labeled_iter_=0`; newly labeled rows record their 1-based iteration and
remaining unlabeled rows retain -1. `transduction_`, `classes_`, `n_iter_`,
`n_features_in_`, `termination_condition_` and polymorphic `estimator_` expose
the fitted result. Refit replaces prior state.

`predict`, `predict_proba` and accuracy `score` delegate to the fitted classifier.
`decision_function` is available for the logistic base and returns a matrix
with one column for binary classification. `predict_log_proba` is supported
for logistic and Gaussian NB; KNN has no source counterpart and is rejected.
Unsupported delegated methods produce explicit diagnostics. A multiclass
logistic decision matrix has the usual unidentifiable common intercept;
reference comparisons center logits per row while comparing probabilities
directly. See `example/self_training.f90` and
`validation/compare_self_training.py`.

## CalibratedClassifierCV

`calibrated_classifier_cv` is exported by `sklearn_calibration` and `sklearn`.
Supply a configured native logistic_regression, gaussian_nb or
kneighbors_classifier as the required estimator argument to fit:

```fortran
type(calibrated_classifier_cv) :: model
type(gaussian_nb) :: estimator
model%method='sigmoid'
model%cv=5
model%ensemble=.true.
call model%fit(x,y,estimator)
prob=model%predict_proba(query)
labels=model%predict(query)
accuracy=model%score(query,query_labels)
```

Inputs are dense finite real64 feature matrices with samples in rows and
rank-one integer labels. At least two estimator classes are required.
method is sigmoid (default) or isotonic. cv defaults to five integer,
unshuffled stratified folds; every class needs at least cv samples. The fold
allocation reproduces sklearn's first-appearance class encoding. Native
fold_id_ holds 1-based fold numbers. Explicit splitters/indices, a default
LinearSVC base, general estimator cloning, FrozenEstimator, sparse data,
metadata routing, parallel fitting and arbitrary fit parameters are outside
this native interface.

ensemble=true fits one base/map pair per fold and averages calibrated query
probabilities. ensemble=false collects out-of-fold responses, fits one set
of calibration maps and refits the base on all input rows. Set prefit=true
for sklearn 1.6.1's cv='prefit' workflow: the supplied estimator must already
be fitted, its classes define the output columns, and it is copied without
refitting. That path uses one pair, regardless of ensemble/cv. The Python
deprecation warning is not emitted. Missing calibration classes are allowed;
unknown calibration labels are binarized to zero, as in the source.

Optional sample_weight must match the calibration rows, be finite and
nonnegative, and have maximum weight at most huge(real64)/(16*n_samples).
Weighted CV fitting is supported for all three bases. Logistic/Gaussian NB
receive training-fold weights and a full-data refit receives all weights.
KNN does not consume training weights, as in sklearn, so its weights affect
calibration maps only. Weighted prefit calibration supports all three bases.
Each native base fit also enforces its documented numerical working range.
Zero-weight rows do not affect isotonic maps; every isotonic fit needs positive
weight support. An all-zero-weight sigmoid fit returns probability .5 before
multiclass normalization. Signed weights are outside this convex-fitting API.

Logistic uses decision_function responses; Gaussian NB/KNN use probabilities.
Binary calibration maps the second sorted class only and obtains the first
column by complement. Multiclass fits one-vs-rest maps and normalizes each
row; all-zero map output becomes uniform. Each ensemble member normalizes
before averaging. Sigmoid uses source weighted Bayesian target smoothing
and the source score rescaling at maximum absolute response >=30. A damped
two-variable Newton optimizer replaces source L-BFGS-B while minimizing the
same objective. Independent reference checks certify its gradient and loss.
Isotonic uses the existing weighted PAV implementation with clipped query
extrapolation and binary-target bounds [0,1] to contain arithmetic roundoff.

Multiclass logistic intercepts now have zero mean, removing a numerical
softmax null mode that otherwise introduces arbitrary fold-dependent offsets
into raw calibration scores. Probabilities retain the same softmax model.
The reference suite checks raw calibration/query responses directly against
source and separately records argmax differences at machine-precision ties.

classes_, n_features_in_ and calibrated_classifiers_ expose fitted state.
Each pair contains a polymorphic estimator_ and calibrators_; sigmoid maps
store a_/b_ and isotonic maps store isotonic_model_. Refit replaces all pairs
and clears obsolete fold metadata. Query features must match fit dimensions.
The input estimator and arrays are preserved. There is no decision_function
or predict_log_proba on the source calibrated wrapper. See
example/calibrated_probabilities.f90 and validation/compare_calibrated.py.

## Weighted and streaming classifiers

`gaussian_nb%fit(x,y,sample_weight)` and
`logistic_regression%fit(x,y,sample_weight)` accept an optional real64 weight
vector. Existing unweighted calls retain their signatures. Weights must be
finite, nonnegative and length-matched. Both fits bound each weight by
sqrt(huge(real64))/(16*n_samples); Gaussian NB also bounds cumulative class
mass by sqrt(huge(real64))/16. Additional Gaussian moment/feature-range and
logistic scaled-penalty checks produce explicit diagnostics before unsafe
arithmetic. Logistic requires positive total weight; Gaussian NB preserves
the source's undefined empirical priors at zero total weight. Signed weights,
scalar weight broadcasting and logistic class_weight configuration remain
outside these native interfaces.

Weighted logistic minimizes the weighted binomial/multinomial log loss plus
the source L2 penalty. Its internal mean-loss scaling uses 1/(C*sum(weights));
the weighted Newton system is diagonally equilibrated. Forming the combined
penalty supports tiny C values whose reciprocal alone would overflow when
large weights keep the scaled problem representable. Native solver stopping
and iteration counts differ from sklearn's solvers. All tested fits have an
independently checked normalized gradient of at most 3e-8. With classes of
zero total weight, an unregularized intercept can have only a limiting optimum;
reference validation checks coefficients, probabilities and gradient while
recording the unmatched finite intercept/decision representation separately.

Logistic predict now follows source raw-score classification: the second
binary class is selected for a strictly positive decision value, and multiclass
prediction uses the largest raw logit. This preserves classification when
probabilities round to .5 or a uniform vector. Self-training predict delegates
to its fitted base's predict method, retaining that distinction. Self-training
selection and probability queries require finite base probabilities; undefined
Gaussian responses have a clear diagnostic before confidence comparisons.

Gaussian NB supports incremental fitting:

```fortran
type(gaussian_nb) :: model
model%priors=[.4_dp,.6_dp] ! Optional; order follows sorted classes.
call model%partial_fit(x_first,y_first,classes=[0,1],sample_weight=w_first)
call model%partial_fit(x_next,y_next,sample_weight=w_next)
prob=model%predict_proba(query)
```

The first unfitted partial_fit requires the complete class set; later calls
can omit classes or repeat the same set. Ordering and duplicate declarations
are normalized as in sklearn. New labels or changed feature counts are rejected.
fit always resets accumulated statistics and re-infers classes, retaining
the configured priors. n_features_in_, class_count_, theta_, var_, class_prior_
and epsilon_ expose the fitted state. Counts are real64 sums of weights.

Optional allocated priors is a nonnegative finite vector whose length matches
the sorted classes and whose sum satisfies source np.isclose tolerances
(1e-8 absolute plus 1e-5 relative at one). Accepted priors are retained without
renormalizing. Leave priors unallocated for empirical class frequencies.
Weighted population moments use Chan/Golub/LeVeque updates. As in sklearn
1.6.1, weighted class batches of mass <=1e-8 skip moment updates while still
increasing counts. Smoothing uses unweighted current-batch feature variance,
and streaming subtracts the CURRENT batch epsilon from old variances before
combining and restoring it. Thus nonzero-smoothing streaming results need
not equal one pooled fit; this pinned-source behavior is explicitly tested.

Declared but unseen classes retain zero mass and initial moments. With positive
variances a zero prior has posterior zero and log posterior negative infinity.
Zero empirical total mass or zero variances can yield quiet IEEE NaN
posteriors, as in source; these cases no longer raise the former zero-variance
fit error. Gaussian predict uses joint likelihoods and source first-NaN argmax
behavior, which differs from applying argmax to normalized NaN posteriors.
Infinite tail likelihoods are handled without a log-zero/divide-zero trap.
The package does not emit Python's warning messages for undefined outputs.

CalibratedClassifierCV now forwards training weights to logistic/Gaussian NB
in every CV fold and the full-data refit; all three supported base types can
use weighted CV. KNN retains source calibration-only weighting. Prefit behavior
is unchanged. Calibration requires finite base responses; a zero-total Gaussian
training fit is therefore diagnosed there, while zero-total sigmoid calibration
weights on an already valid base retain their source behavior.

See example/weighted_streaming_classifiers.f90 and
validation/compare_weighted_classifiers.py. A saved synthetic fixture in
validation/fixtures/logistic_tiny_weight_line_search.json preserves a source
Newton-CG line-search stall. Its recovery uses sklearn Newton-Cholesky only
after independent source/native gradients establish the stall, and checks
loss equality with the recovered source solution. Such checks are recorded
separately from ordinary Newton-CG comparisons.


## Independent multioutput estimators

`multi_output_regressor%fit(x,y,estimator,sample_weight)` takes real64
`y(n_samples,n_outputs)`; the classifier takes integer targets of the same
shape. The required estimator is a native concrete base instance. Each column
gets an independently fitted copy; the supplied instance and input arrays are
preserved, including when the template was already fitted. Normal Fortran
assignment makes independent copies of the fitted wrappers. Refit replaces
the output list. `predict(x)` returns a dense matrix with one column per output.
Fitted `estimators_(j)%estimator_` retains its dynamic base type; use SELECT TYPE
to inspect its parameters. Both wrappers expose `n_outputs_` and `n_features_in_`.

Regression accepts the `linear_regression` hierarchy (including ridge, elastic
net and its other native descendants), `sgd_regressor`, `kneighbors_regressor`
and the `decision_tree_regressor` hierarchy (including ExtraTree). Classification accepts `logistic_regression`,
`gaussian_nb`, `sgd_classifier`, `kneighbors_classifier`,
the `decision_tree_classifier` hierarchy (including ExtraTree), and the `multinomial_nb` hierarchy (including
Bernoulli and Complement NB). Other bases are explicitly rejected. Each base
keeps its existing capabilities, optimization choices and numerical limits.
There is no arbitrary Python-style estimator protocol, sparse support, parallel
backend or metadata routing. Chain interfaces are described below.

Training weights are supported for SGD and tree regression, and logistic, Gaussian NB,
SGD and tree classification. Passing weights to another supported base is diagnosed;
this wrapper does not add weights to an unweighted native implementation.
`partial_fit(x,y,estimator,sample_weight)` supports SGD regression;
classifier `partial_fit(x,y,estimator,classes,sample_weight)` supports Gaussian
NB and SGD. The first call requires an estimator; it clones configuration and
clears pretrained state. Later calls continue the stored models, so estimator
is optional. Classifier classes is a `label_set(:)` whose `values` contain each
output's full class set, required by the base on its first streaming call.
The list must cover all outputs; later calls follow the base's class checks.
Averaged SGD classification streaming inherits the base's requirement for
`fit_intercept=.false.`. Feature counts must stay fixed. Matching sklearn 1.6.1,
a later partial_fit can shrink the output list; increasing it is diagnosed.

Classifier `classes_(j)%values` is populated by fit. The pinned source does not
create top-level classes_ in partial_fit; this quirk is preserved. A subsequent
streaming shrink after fit can leave that top-level list stale. Current labels
are available directly in each fitted base or in `predict_proba(x)`, an array
of `multioutput_probability` with per-output `classes(:)` and
`values(n_query,n_classes)`. Different outputs may have different class counts;
there is no padding. Base-defined quiet NaN probabilities remain quiet NaNs.
A base without probabilities (e.g. hinge SGD) diagnoses predict_proba.
Classification score is exact subset accuracy: every output in a row must
match; like source, this score has no sample_weight argument.

Regression score is the uniform mean of per-output R2, with optional query
sample weights and force_finite constant-target behavior. A singleton returns
quiet NaN. Effective constant targets are detected explicitly, preventing
weighted-mean reduction roundoff from creating a spurious nonzero variance.
This can differ from the pinned source's floating-point artifact; validation
checks the source R2 translation-invariance identity and records that case.
Underlying CART equal-gain split choices and Complement NB near-tied class
scores can select different equally valid outcomes; the reference suite
certifies these cases explicitly rather than widening ordinary tolerances.

See `example/multioutput.f90` and `validation/compare_multioutput.py`.


## RegressorChain

`regressor_chain%fit(x,y,estimator,sample_weight,splits)` fits one independent
native regressor at each chain position. Real64 targets have shape
`(n_samples,n_outputs)`. The fitted model at position j sees the original
features followed by the first j-1 chain targets. Supported bases are the
same linear regression hierarchy, SGD, KNN and tree regressors as
multi_output_regressor; weights require a base with native weighted fitting
(SGD and tree regressors). Fitted templates and input arrays are preserved.

Default order follows target columns. Set allocatable `order(:)` to a
one-based permutation, or set `random_order=.true.` with nonnegative
`random_state` (default zero) for a permutation matching NumPy RandomState.
These two configuration choices are mutually exclusive. Fitted `order_`
records chain positions; `predict(x)` always restores original target-column
order. Each `estimators_(j)%estimator_` keeps its native dynamic type and
augmented feature count. Refitting replaces models and fitted order; Fortran
assignment deep-copies fitted chains.

Default `cv=0` trains each stage on the observed preceding targets. With
integer `cv>=2`, preceding targets are replaced by out-of-fold predictions
from unshuffled KFold. Alternatively pass `type(index_split) :: splits(:)`
with `cv=0`; one-based test indices must partition every row exactly once.
Training indices are used as supplied, including repeated rows or overlap
with test indices, matching source iterable-split behavior. Bounds, missing
indices and empty used folds are diagnosed. An unused CV specification on a
single-output chain is not evaluated, matching source; integer cv must still
be zero or at least two. There is no partial_fit or general metadata routing.

Full stage fits receive sample_weight. Matching sklearn 1.6.1 with metadata
routing disabled, the separate cross_val_predict fits do not receive those
weights. Each full stage uses all training rows and its already constructed
chain features. Query prediction always chains earlier predictions, even
when fitting used observed targets. `score(x,y,sample_weight)` shares the
multioutput uniform-average R2 implementation, including quiet NaN for a
singleton and force_finite constant-target handling.

The shared SGD core now follows source WeightVector64 lazy L2 scaling,
explicit ordered inner products, the 1e-9 scale-reset threshold and final
materialization. This corrects accumulated decay roundoff exposed by a
weighted cross-validated chain with very large downstream coefficients.
Averaging keeps the existing dense mean update; other base capabilities and
limits remain as documented for those native estimators.

See `example/regressor_chains.f90` and `validation/compare_chains.py`.


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


## ClassifierChain

`classifier_chain%fit(x,y,estimator,sample_weight,splits)` accepts dense real64
features and integer `y(n_samples,n_outputs)`. The native base families are
logistic regression, Gaussian NB, SGD classification, KNN, decision trees,
and the multinomial/Bernoulli/Complement NB hierarchy. Each stage is fitted
independently on original features plus earlier chain targets/features;
the supplied template and input arrays are preserved. Weights are supported
by logistic, Gaussian NB, SGD and tree bases (including ExtraTree). This implements the numerical
behavior of sklearn's metadata-routed weighted full fits; it does not expose
a general metadata-routing framework. Internal CV fits remain unweighted,
as in source. Base parameters, numerical limits and available responses
remain those of the corresponding native estimator.

Set one-based allocatable `order(:)` or `random_order=.true.` and a nonnegative
`random_state` (default zero). Fitted `order_` gives chain positions;
`predict`, `predict_proba` and `decision_function` restore original target
column order. `classes_(j)%values` and `estimators_(j)%estimator_` are in
CHAIN order, matching source. Both feature/output counts are exposed.
Refit replaces all fitted metadata/models. Fortran assignment deep-copies
fitted chains.

`chain_method` selects the feature response: predict (default),
predict_proba, predict_log_proba or decision_function. Alternatively allocate
`character(20) :: chain_methods(:)` with ordered preferences; when allocated
this list takes precedence over chain_method. Fitted `chain_method_` records
the first available method. All supported bases predict labels. Probabilities
are available except for unsupported SGD losses; KNN lacks log probabilities.
Decision scores require logistic or SGD classification. Native stable NB log
probabilities are used directly; other available log responses follow source
log(predict_proba), including quiet negative infinity for zero probabilities.

Default `cv=0` fits with observed preceding target values. Integer `cv>=2`
uses unshuffled StratifiedKFold independently at each chain position, with
classes encoded by first appearance, matching source. A minority class may
have fewer rows than the fold count when another class has enough rows;
Python warning messages are not emitted. For probability CV, globally absent
fold classes are padded with zero probability (or the minimum finite real64
log probability), then source's second global class column is selected.
Incomplete decision-function class sets are diagnosed. Full fits always use
all training rows and their already constructed chain features.

Alternatively pass one-based `index_split(:)` with cv=0; test rows must
partition the training set once. Training repeats and overlap are used as
supplied, matching source iterable splits. Used indices must be allocated,
nonempty and within bounds. Unused single-output CV folds are not evaluated.
A chain has no partial_fit interface.

Query features always use the fitted response choice. Positive-class
probabilities/log probabilities select each binary model's highest sorted
class label. `predict_proba(x)` returns that positive-class value per original
target column; `decision_function(x)` returns one binary score per column.
Predict-label chains can handle multiclass and constant-class outputs, but
query probability responses require exactly two classes and query decisions
must be scalar. These checks mirror source's response-shape restrictions.
Probability/log/decision feature methods can therefore fit a multiclass
chain yet fail at query time. Existing base input checks also apply to chain
features: for example, a log-probability feature can violate nonnegative
input requirements of count-based NB, or introduce infinity into a later
stage. Unsupported responses and these invalid paths are explicitly diagnosed.

`score(x,y,sample_weight)` uses exact subset accuracy and accepts finite
signed query weights with nonzero total, within the native working range.
For multiple outputs, each of y and predicted labels must have at most two
distinct integer values over the whole matrix, matching source's multilabel
indicator detection (values need not be 0 and 1). Single-output scoring
accepts multiclass labels. Training weights still follow each base's own
validation rules.

See `example/classifier_chains.f90` and
`validation/compare_classifier_chains.py`. The reference suite checks full
chain outputs and also source fits on native CV-feature matrices. It records
equal-gain CART alternatives using exhaustive Gini scans/source stumps, and
Complement NB label alternatives only when source log scores tie within a
scaled 64-machine-epsilon bound. These certified alternatives can propagate
through earlier predicted-label features; ordinary tolerances are unchanged.


## Multiclass reductions

`one_vs_rest_classifier`, `one_vs_one_classifier` and `output_code_classifier`
are exported by `sklearn_multiclass` and the `sklearn` facade. Call
`fit(x,y,estimator,sample_weight)` with dense real64 features and integer
labels. Each binary head receives a fresh configuration clone of the supplied
native template; fitted template statistics and input arrays are preserved.
Refitting replaces prior heads and metadata; Fortran assignment deep-copies
fitted wrappers.

Supported classifier bases are logistic regression, Gaussian NB, SGD
classification, KNN, decision trees, multinomial/Bernoulli/Complement NB,
Perceptron, passive-aggressive classification, RidgeClassifier (including
RidgeClassifierCV), LinearSVC and SVC. OVR and OVO also support the native
linear-regression hierarchy (including Ridge), SGD regression, KNN regression
and decision-tree regression. Output-code bases must supply a decision
function or probabilities. Training weights are available with logistic,
Gaussian NB, SGD classification/regression, ridge classification, LinearSVC
and both tree families (including ExtraTree). Other bases reject weights; their existing native limits still apply.
This is explicit native dispatch, without an arbitrary Python estimator,
metadata-routing, sparse or precomputed-kernel protocol.

OVR `fit` also accepts an integer matrix of indicator targets. Multiple
columns must collectively contain at most two distinct integers, following
source detection rather than requiring 0/1 values. Use `predict(x)` for label
vectors and `predict_indicator(x)` for multilabel matrices; using the wrong
rank interface is diagnosed. A one-column matrix follows vector semantics.
`classes_` contains sorted labels for vectors or zero-based column numbers
for multilabel targets. `multilabel_`, `n_classes_` and `n_features_in_` expose
fitted metadata. `decision_function(x)` always returns a matrix, including
(n_samples,1) for binary classification. `predict_proba(x)` uses each head's
positive probability, expands one head to [1-p,p], and normalizes rows only
for single-label tasks. The one-class OVR result consequently has two
probability columns despite having one class label, matching sklearn 1.6.1.
Zero or undefined probability mass produces quiet NaNs, including the
source-compatible undefined Gaussian NB posterior path.

OVR prefers decision scores, otherwise positive probabilities or regression
predictions. Binary decision classifiers use threshold zero; other heads use
0.5. A constant head also uses 0.5, reflecting sklearn's constant predictor
estimator tags. Multilabel thresholds follow the first fitted head. Multiclass
OVR ties choose the last class, including source maximum/NaN propagation.

OVO requires at least two classes, fits all sorted class pairs on their own
rows, and maps each pair's labels to 0/1. It combines votes with bounded
confidence corrections: sum/(3*(abs(sum)+1)). Binary decisions return the
second class's column; multiclass decisions return one column per class.
Multiclass ties choose the first class. It provides `predict`,
`decision_function` and `score`, without a probability interface.

OVR and OVO `partial_fit(x,y,estimator,classes,sample_weight)` support Gaussian
NB and SGD classification. On the first call supply the global class set;
subsequent sets must match. Empty OVO pairs are skipped, and querying a pair
that has never received rows is diagnosed. Fresh multilabel partial_fit is
unsupported in sklearn 1.6.1; native multilabel streaming requires an initial
fit and nonconstant heads. Constant heads and SGD regression do not support
these streaming calls. Averaged SGD retains its native partial-fit limits.

Output-code classification sets `code_size` (default 1.5) and nonnegative
`random_state` (default zero). It draws the class-by-bit `code_book_` in
row-major order using the legacy NumPy RandomState generator. There are
int(n_classes*code_size) bits, with at least one required. Codes are -1/+1
for decision-function bases and 0/1 for probability bases. Prediction minimizes
squared Euclidean code distance, choosing the first minimum. It offers
`predict` and `score`, without streaming or wrapper probabilities/decisions.
Query responses must be finite and within the checked distance working range.

All wrappers expose `estimators_(:)%estimator_` for trained models. Constant
heads instead set `constant_` and `constant_value_`, leaving estimator_
unallocated. Slots record `n_features_in_`. Scores use accuracy or exact
multilabel subset accuracy and accept finite signed query weights with a
nonzero sum. Training weights retain the underlying base's validation.

The binary Perceptron now stores one positive-class coefficient row and
returns a single decision column, following source; label predictions are
unchanged. Multiclass Perceptron retains one row per class.

See `example/multiclass_reductions.f90` and
`validation/compare_multiclass.py` for usage and source comparisons.


## Voting ensembles

`voting_classifier`, `voting_regressor` and the configuration container
`voting_estimator` are exported by sklearn_ensemble and the sklearn facade.
Fill an array of voting_estimator slots with distinct nonempty names (up to
64 characters) and allocate each active slot's polymorphic `estimator` from
a native base. Set `drop=.true.` to skip a slot, including an unallocated one.
Call `fit(x,y,estimators,sample_weight)`; supplied fitted templates and input
arrays are preserved. Active models are fitted afresh on all rows, even when
their voting weight is zero. Refit replaces all models; Fortran assignment
deep-copies fitted ensembles. Python constructor parameter names and names
containing a double underscore are rejected as estimator names.

Supported classifier families are logistic regression, Gaussian NB, SGD
classification, KNN, decision trees, multinomial/Bernoulli/Complement NB,
Perceptron, passive-aggressive classification, RidgeClassifier (including
RidgeClassifierCV), LinearSVC, SVC and DummyClassifier. Regressor families
are the native linear-regression hierarchy (OLS, Ridge, ElasticNet, Lasso
and its native descendants), SGD regression, KNN regression, decision trees,
passive-aggressive regression, KernelRidge and DummyRegressor. Every base
retains its existing native controls, numerical limitations and responses.
Training weights are accepted by logistic, Gaussian NB, SGD classification,
ridge classification, LinearSVC, SGD regression and tree bases (including ExtraTree); all active bases
must support them. Dropped bases are not fitted or capability checked.

Classifier `voting='hard'` (default) combines encoded class predictions;
`voting='soft'` averages probabilities. Labels are sorted and encoded as
0..K-1 before base fitting, and restored to original labels for `predict`.
Fitted `classes_`, `n_features_in_`, `active_indices_` (one-based configured
slot numbers), and `estimators_(:)%name/estimator` expose fitted metadata.
Regressor metadata follows the same active-slot layout without classes_.
The native named slots replace Python's named_estimators_ dictionary.

Allocate optional `weights(:)` with one finite value per CONFIGURED slot,
including dropped slots; otherwise all active weights equal one. Signed
voting weights are supported. Soft voting and regression require a nonzero
sum of active weights when predicting. Hard voting allows a zero sum and
all-zero weights. Hard ties choose the first encoded bin; its bincount ends
at the highest label actually predicted in that row, preserving source
behavior even with negative weights. Soft ties choose the first maximum,
including source first-NaN argmax behavior. Undefined Gaussian NB posterior
probabilities remain quiet NaNs in the average.

Classifier `transform(x)` returns real64 encoded labels with hard voting,
or horizontally concatenated per-base probabilities with soft voting.
These are encoded labels, not original integer labels. For soft voting,
`transform_proba(x)` returns (n_active_estimators,n_samples,n_classes),
matching the source unflattened shape. The default flatten_transform is true;
when false, use transform_proba rather than the rank-two transform interface.
Regressor transform returns (n_samples,n_active_estimators) predictions.
Transforms exclude dropped slots and ignore voting weights. Both types offer
rank-two `fit_transform(x,y,estimators,sample_weight)` convenience functions.

`predict_proba` and transform_proba require soft voting and probability
support from every active base. Fitting a soft ensemble with a hard-only
base succeeds; querying its probabilities then fails, as in source. There
is no partial_fit or decision_function interface. The APIs accept dense
single-output targets; sparse inputs, arbitrary Python estimators, general
metadata routing, job scheduling, feature-name generation and persistence
are outside this translation.

`score(x,y,sample_weight)` computes accuracy or source force-finite R2.
Query weights may be finite and signed with nonzero sum. A singleton R2
query returns quiet NaN, as in sklearn; warning messages are not emitted.
See example/voting_ensembles.f90 and validation/compare_voting.py.


## Permutation importance

`sklearn_inspection` and the sklearn facade export
`permutation_importance(x,scorer,n_repeats,random_state,sample_weight,max_samples)`.
It returns permutation_importance_result with source-shaped fields:
`importances(n_features,n_repeats)`, `importances_mean(n_features)` and
`importances_std(n_features)`, using population standard deviation.
The result container is not an additional estimator type.

The native scoring interface replaces Python estimator/scorer dispatch with
an explicit Fortran callback. A scalar callback has this interface:

```fortran
function scorer(features,rows,sample_weight) result(value)
  use sklearn, only: dp
  real(dp), intent(in) :: features(:,:)
  integer, intent(in) :: rows(:)
  real(dp), intent(in), optional :: sample_weight(:)
  real(dp) :: value
end function
```

The callback captures a fitted estimator and targets through host or module
association. Use original targets indexed by the one-based `rows`, e.g.
`model%score(features,y(rows))`; optional scoring weights have already been
subset to the same rows. This works with any native estimator and scalar,
multiple-output, classification, regression or unsupervised scoring contract
that the callback implements. There is no automatic default score or Python
scorer-name registry. The algorithm never fits an estimator. Callback side
effects remain the caller's responsibility.

`permutation_importance_multi` takes a callback with identical arguments
returning an allocated, nonempty real64 vector, of fixed length across calls.
It returns an allocated result array in that vector's metric order. Scoring
is called once per perturbed matrix for all metrics together; the callback
can reuse model predictions. Abstract interfaces permutation_score and
permutation_scores are public for explicit procedure declarations.

Defaults are n_repeats=5 and random_state=0, with at least one repeat and a
nonnegative default-integer seed required. The legacy NumPy-compatible RNG
first draws a shared worker seed, then resets it for every feature. Each
feature consequently uses the same row subset and shuffle sequence.
Subsampling follows sklearn's tracking-selection, permutation or reservoir
algorithm as appropriate. Row-shuffling indices and the permuted column both
persist across repeats, matching source cumulative shuffles.

Omit max_samples for all rows. An integer (default or int64 kind) selects
1..n_samples rows; a real32/real64 fraction in (0,1] is multiplied by the row
count and truncated. Thus integer 1 means one row, while real 1.0 means all
rows. Real32 fractions are promoted to real64 before multiplication, matching
a supported Python float constructed from the same represented value. For
example, at ten rows, .7 (real32) selects six while .7_dp selects seven.
A small valid fraction may round to zero: source allows this, and the
callback then receives empty feature/row/weight arrays. The callback decides
whether such a score is meaningful.

The baseline always uses ALL original rows, even when permutation scoring
uses a subset. Importance is baseline minus permuted score; constant-column
importance can therefore reflect the change of evaluation rows. Features
and supplied weights are preserved, and each feature works on a fresh copy.
Finite signed sample weights are forwarded unchanged; the callback chooses
its normalization and any nonzero-total requirements. Features may contain
NaNs, but not infinities; the scorer decides how to handle missing data.

NaN/infinite scores and overflowing score differences follow quiet IEEE
source reduction behavior. Matching infinities subtract to NaN, and summary
means/deviations propagate nonfinite values. No Python warnings are emitted.
The implementation is dense, sequential and real64. Sparse/pandas paths,
Python metadata routing and joblib execution are outside this interface.
See example/permutation_importance.f90 and validation/compare_inspection.py.


## Partial dependence and individual conditional expectation

`partial_dependence(x,features,predictor,percentiles,grid_resolution,
categorical_features,kind,sample_weight)` is exported by sklearn_inspection,
sklearn_partial_dependence and the sklearn facade. This implements sklearn's
brute numerical method, averaging predictions over the supplied background
rows. Tree-recursion estimation and automatic Python response-method dispatch
are outside this interface. No estimator is fitted by the algorithm.

The predictor is a Fortran callback with this interface:

```fortran
function predictor(features) result(values)
  use sklearn, only: dp
  real(dp), intent(in) :: features(:,:)
  real(dp), allocatable :: values(:,:)
end function
```

Capture a fitted estimator through host/module association. Return a matrix
(n_samples,n_outputs) with an allocated, nonempty output axis and a constant
output count across calls. A regression predict vector can be reshaped to
(n_samples,1); multiple-output regression and multiclass probabilities or
decisions retain their columns. For sklearn-compatible binary probabilities,
select the highest sorted class's probability column; binary decisions use
one column. Response selection belongs to the callback, so its numeric meaning
is explicit. Callback side effects remain the caller's responsibility.

`features(:)` contains one-based original feature indices, in axis order.
Each Cartesian grid point replaces those columns for every original row,
then calls the predictor once. Other columns and the original input are
preserved. Repeated feature indices are accepted; the last substitution wins,
as in source. The result container exposes:

- grid_values(j)%values: a ragged numeric vector for each selected feature;
- average(n_outputs,n_grid_points), when kind is average or both;
- individual(n_outputs,n_samples,n_grid_points), when kind is individual or both.

Default kind is average. Unrequested result fields remain unallocated. The
last selected axis varies fastest in the flattened grid-point dimension.
For two axes of lengths a and b, index (i,j) maps to (i-1)*b+j. This fixed-rank
representation replaces Python arrays with a variable number of grid axes.
Result assignment deep-copies all axes and response arrays.

Default grid_resolution is 100 and must exceed one. Numeric features use
sorted unique values when their count is STRICTLY less than grid_resolution;
otherwise an evenly spaced grid spans the two empirical percentiles.
Percentiles default to [.05,.95]; supply two finite, increasing real64 values
in [0,1]. The interpolation follows SciPy mquantiles with alphap=betap=.4,
matching sklearn 1.6.1. Source's default allclose threshold rejects numerical
percentile ranges that are too close to form a grid.

Categorical features always use all sorted unique values. Supply a logical
mask over all original columns or a default-integer or int64 list of one-based
categorical indices. Entries not among the selected features are ignored.
Numeric categories and NaNs are supported; strings, pandas, sparse arrays
and masked-array metadata are outside this dense real64 interface.

Features can contain NaNs, but not infinities. Missing values are retained
in unique axes; source percentile calculations can produce all-NaN grid axes.
The predictor decides whether it can evaluate those points. Sample weights
must be finite with a nonzero sum; signed weights are supported. They affect
averaged responses and leave individual curves unchanged. Even individual-
only evaluation validates this averaging requirement, following source.
Quiet IEEE arithmetic handles nonfinite predictions, including zero-weight
infinite responses and mixed infinities. Python warnings are not emitted.

Sklearn 1.6.1 has a binary-response reshape failure when the grid contains
two points, or when the background has two rows. Native callbacks have an
explicit output dimension and retain the complete mathematical curve in
these cases. Validation records the source failure and checks the native
means/individual curves against the predictions actually produced by source
at every grid point. These are labeled source_binary_shape_identity rather
than ordinary public-API comparisons.

See example/partial_dependence.f90 and
validation/compare_partial_dependence.py. The helper axis/result containers
are not additional estimator types.


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


## Callable transforms and transformed regression targets

`function_transformer` is exported through `sklearn_preprocessing` and `sklearn`.
Its `func` and `inverse_func` procedure pointers accept a dense real(dp) matrix
with intent(in) and return an allocatable real(dp) matrix. An absent callback
means identity. Procedures and any captured/module state must remain valid for
the lifetime of the transformer and its copies. Callbacks own their numerical
domain and may change output dimensions. No Python kwargs, sparse arrays,
arbitrary object inputs, pandas output containers or metadata routing are provided.

Methods are fit, transform, fit_transform, inverse_transform and
get_feature_names_out. Transform needs no preceding fit. Defaults are
validate=false and check_inverse=true. Fit records n_features_in_ even without
validation. Optional feature_string feature_names(:) records explicit column
names; each fit resets this metadata. With validation enabled, inputs must be
nonempty and finite; forward transforms enforce a known fitted input width.
Inverse transforms validate their own input without enforcing the forward
width. Callback outputs are not separately validated by FunctionTransformer.

For feature names, set feature_names_out='one-to-one' or associate names_func.
The latter receives class(*) transformer and feature_string input_features(:),
returning an allocatable feature_string vector. An empty vector represents
unknown input names for that callback. Names are either explicitly supplied,
stored on fit, or generated as x0, x1, ... from the known input width.
A one-to-one request with no known input width/names is diagnosed rather than
returning Python's zero-dimensional object array containing None. One-to-one
names for zero-column, unvalidated inputs require explicit feature names.

When both callbacks are present, fitting optionally checks a round trip on
rows 1, 1+step, ... with step=max(1,n/100). Tolerances are rtol=1e-7 and
atol=1e-9, matching sklearn 1.6.1's dense inverse helper. Warning conditions are
reported as inverse_checked_ and inverse_consistent_ instead of Python warnings.
A failed numerical inverse check does not prevent fitting. Incompatible check
shapes are diagnosed; rank-two NumPy broadcasting and NaN/Inf comparisons are
preserved for the dense check.

`transformed_target_regressor` is exported by `sklearn_compose` and `sklearn`.
Allocate regressor and transformer from native prototypes, or associate func
and inverse_func together. Combining a transformer with callbacks is rejected.
The default base is LinearRegression; the default transformer is a validated
FunctionTransformer. Supported exact transformer types are function_transformer,
standard_scaler, minmax_scaler, maxabs_scaler, robust_scaler, power_transformer
and quantile_transformer. Fitted copies are held in regressor_ and transformer_;
prototypes are left unchanged. FunctionTransformer cloning copies its controls
and procedure pointers while dropping learned metadata.

fit accepts vector or matrix targets, with optional sample_weight. It fits the
transformer on all unweighted targets, transforms y, then fits the base with
weights where supported. Original targets and transformed training targets
must be finite and nonempty; transformed sample count must equal X's count.
Single transformed outputs accept the regression types supported by the native
base dispatcher: linear_regression and its existing descendants, sgd_regressor,
kneighbors_regressor, decision_tree_regressor/extra_tree_regressor, kernel_ridge
and dummy_regressor. Base-specific controls and limitations continue to apply.
Exact LinearRegression and Ridge additionally use the existing weighted SVD
helper. Other bases inherit the dispatcher's current weighting restrictions.

Multiple transformed outputs currently support exact linear_regression, ridge,
elastic_net, lasso, kneighbors_regressor, kernel_ridge and dummy_regressor.
These models fit outputs independently, preserving the corresponding sklearn
semantics. regressor_ then contains a target_regressor_bundle with one fitted
model_ per estimators_ slot. Joint multi-target trees and CV estimators are
rejected in this path. Storage helper types are not additional algorithms.

predict_matrix always returns a rank-two inverse-transformed result; predict
returns a vector and requires one output column. transformed_predict exposes
base predictions before inversion. training_dim_, n_outputs_,
n_transformed_outputs_ and n_features_in_ record training dimensions. Fortran's
static ranks require separate vector/matrix prediction methods. Inverse output
is otherwise controlled by the callback, without a forced original width or
finite-output restriction. score computes R2 on the original target scale,
with uniform averaging across output columns and optional score weights.

The wrapper's check_inverse defaults true and checks every max(1,n/10)-th row
with NumPy's rtol=1e-5, atol=1e-8. Its inverse flags are separate from an embedded
FunctionTransformer's flags. Original-target constant R2 can change from 1 to
0 after a one-ulp prediction difference; it is not smoothed or clamped beyond
sklearn's force_finite rule. See example/transformed_targets.f90.


## ColumnTransformer and FeatureUnion

`column_transformer` is exported by `sklearn_compose`; `feature_union` by
`sklearn_pipeline`. Both are available through `sklearn`. They combine dense
rank-two real(dp) matrices using the existing native transformer implementations.
The storage helper `transformer_branch` holds an allocatable name, an action
('transform', 'drop' or 'passthrough'), an allocatable class(*) model and a
real(dp) weight (default one). Actions drop/passthrough ignore an allocated model.
Model allocation is explicit, for example:

```fortran
allocate(columns%transformers(1))
columns%transformers(1)%name='numeric'
columns%transformers(1)%columns=[1,3]
allocate(columns%transformers(1)%model,source=scaler)
z=columns%fit_transform(x)
```

Supported exact child types are function_transformer; standard_scaler,
minmax_scaler, maxabs_scaler, robust_scaler; simple_imputer, knn_imputer,
missing_indicator; one_hot_encoder, ordinal_encoder; power_transformer,
quantile_transformer, kbins_discretizer, spline_transformer; pca, truncated_svd;
and variance_threshold. ColumnTransformer and FeatureUnion can recursively
contain either composition type. The underlying model's numerical controls
and restrictions still apply. MissingIndicator logical output is promoted to
real(dp) before weighting/concatenation, as with a numeric sklearn composition.
No supervised y dispatch, sample-weight routing, arbitrary external estimator
objects, sparse outputs, joblib parallel execution or Python parameter-search
framework is provided by this increment.

ColumnTransformer uses transformers(:) as prototypes and transformers_(:) as
fitted branches. Exactly one selector is required on every prototype branch:
columns(:) contains 1-based integer positions (duplicates/order preserved),
mask(:) is a logical vector of input width, column_names(:) is a feature_string
vector, or selector is a procedure pointer. The callback is a subroutine accepting X,
the input feature_string vector and an allocatable integer intent(out) position
vector; it is evaluated only on fit. Allocation is checked after the callback,
so an unallocated selection is diagnosed safely in optimized builds. Generated
x0, x1, ... names are supplied to the callback when explicit names are absent. A zero-length selector is valid and
its transformer is not fitted. Selections may overlap. Negative/zero native
indices are diagnosed; Python negative indices, scalar rank-one selection,
and Python slices are represented using explicit native position vectors.

remainder is another transformer_branch, with default action='drop'. Set its
action to 'passthrough', or to 'transform' and allocate its model. Its selected
columns are the sorted complement of all explicit selections, including dropped
branches. If nonempty, remainder is appended after the explicit fitted branches;
its fitted name is 'remainder'. Its weight is applied like other branches.
Fit clones child prototypes. FunctionTransformer cloning drops learned metadata
and preserves callback pointers/controls. Empty/drop branches are skipped and retain prototype
storage. Native intrinsic allocatable assignment makes independent model copies;
procedure targets and module/captured callback state remain the caller's responsibility.

fit(X,feature_names) and fit_transform(X,feature_names) optionally record explicit
feature_string names. ColumnTransformer fit uses fit_transform internally, like
sklearn. transform(X,feature_names) can then map reordered named query columns,
accept extra columns and omit fitted columns used only by dropped/empty branches.
Active names must be present and unique in the query. Without names on both fit
and query, positional queries require the fitted feature width. These explicit
arrays implement named-column semantics; they are not a pandas container API.
Global finite-value validation is delegated to children, allowing passthrough
and imputation branches to retain the appropriate NaN behavior.

n_features_in_, n_features_out_, feature_names_in_ and sparse_output_ expose
fitted metadata; sparse_output_ is always false in this dense implementation.
named_transformer_index(name) returns the 1-based transformers_ position or
zero. output_indices(name) returns the corresponding 1-based output positions;
empty/drop branches and an absent remainder return an empty vector. Branch
output_start_/output_stop_ are inclusive native bounds, with stop<start for an
empty slice. ColumnTransformer weights are read from prototype branch controls
on every transform, allowing weight changes without refitting.

Native FeatureUnion transform/name calls require a fitted graph, including
stateless-only unions. FeatureUnion uses transformer_list(:), with no column selectors: every branch
receives the complete input. Fitting updates the models in transformer_list,
matching sklearn's fitted-list convention, while Fortran value semantics do not
mutate an independently allocated caller variable. Passthrough becomes a fitted
one-to-one FunctionTransformer. fit invokes child fit; fit_transform invokes child
fit_transform when available (or fit then transform). A fit-only union has
n_features_out_=-1 because no output was evaluated; fit_transform records its
actual width and output slices. These width/slice fields are native bookkeeping.
The native API additionally accepts an empty transformer_list, returning an empty
matrix; sklearn 1.6.1 rejects an empty list during fit. An all-dropped union is
accepted by both. Native unprefixed names for an empty/all-dropped union return
an empty vector; sklearn 1.6.1 raises a NumPy empty-concatenate error on its
unprefixed all-dropped naming path. These are explicit native empty-container
adapters, not additional algorithms.
transform uses the current branch weights. A passthrough-only union need not
enforce fitted input width because its FunctionTransformer has validate=false.

Both compositions concatenate active results in branch order. If no branches
are active, they return shape (input_rows,0). Active zero-width blocks
participate in row consistency checks. A single branch may change the row count,
matching sklearn; multiple branches must have equal result row counts. The
compositions do not force the input row count onto a valid child result.

get_feature_names_out(input_features) uses explicit/stored/generated input names
and delegates child naming. verbose_feature_names_out defaults true, prefixing
branch names with '__'. False returns unprefixed names and diagnoses duplicates.
ColumnTransformer additionally accepts a name_formatter procedure pointer:
character(*) transformer_name and feature_name inputs, returning allocatable
character(:). This represents sklearn's callable formatter; Python format-string
parsing is not provided. Custom formatter output follows sklearn's callable
path without an additional uniqueness constraint. Numeric one-hot names use
shortest round-trip decimal strings with Python float decimal/exponent layout.
KNNImputer names reproduce sklearn 1.6.1's omission of all-missing columns even
when keep_empty_features retains their numeric output, so source names can have
a different length from numeric output. A child's missing name API is diagnosed.
See example/feature_composition.f90 for numeric/categorical branches and nesting.


## Sequential Pipeline

`pipeline` and the storage helper `pipeline_step` are exported by
`sklearn_pipeline` and `sklearn`. Allocate steps(:), set each step's allocatable
name, and allocate its model from a native estimator/transformer. action defaults
to 'model'; 'passthrough' represents sklearn's None/'passthrough'. Drop is not
a pipeline action. Names must be unique, omit '__' and avoid constructor-control
names. At least one step is required. For example:

```fortran
allocate(model%steps(2))
model%steps(1)%name='scale'
allocate(model%steps(1)%model,source=scaler)
model%steps(2)%name='regress'
allocate(model%steps(2)%model,source=regressor)
call model%fit(x,y)
predicted=model%predict(x)
```

Fitting updates the models inside steps, matching sklearn's default fitted-step
convention. Allocatable assignment/copying preserves native value semantics;
independently allocated caller variables and sliced/copied pipelines do not
share mutable learned arrays. Procedure targets and callback module/captured
state remain the caller's responsibility. No joblib memory cache, Python
get_params/set_params framework or metadata routing is provided.

Intermediate transformers always use fit_transform, including TargetEncoder's
cross-fitted training encodings. A final transformer uses fit on Pipeline fit,
and fit_transform on Pipeline fit_transform. Targets are passed unchanged
through intermediate steps. Final estimator type determines classification or
regression independently of target dtype: integer regression targets become
real-valued regression data, and integral real classification labels are checked
and converted to native integer labels. Native labels must fit default integer.
Each child retains its existing data-domain and parameter restrictions.

Supported transformer steps include the composition dispatcher's 17 existing
unsupervised types, column_transformer, feature_union, nested pipeline,
select_k_best, generic_univariate_select and its select_percentile/select_fpr/
select_fdr/select_fwe descendants, and target_encoder. Selectors preserve their
native typed-target conventions: integer targets use classification scoring,
real targets regression scoring unless the native selector's controls override
that choice. For regression scoring this corresponds to explicitly selecting
f_regression in sklearn, rather than its Python SelectKBest default f_classif.
TargetEncoder keeps its own configured/auto target-type rules.

Pipelines can appear inside ColumnTransformer/FeatureUnion branches as well as
contain either graph type. An internal abstract dense_transformer protocol
provides this recursive bridge without a circular module dependency. It is an
implementation interface, not another algorithm. The current parallel graph
fit API does not forward y, so a pipeline embedded in such a branch must be
fittable without a supervised target. An outer sequential pipeline does forward
y to its supervised steps.

Final estimators include the native base dispatcher's classifier and regressor
types, nested pipelines and transformed_target_regressor. Classifiers include
logistic_regression and its native descendants, Gaussian/multinomial-family NB,
SGD, k-neighbors, decision/extra trees, perceptron, passive-aggressive,
ridge-classifier families, LinearSVC/SVC and DummyClassifier. Regressors include
linear_regression and its native descendants, SGD, k-neighbors, decision/extra
trees, kernel_ridge and DummyRegressor. Exact LinearRegression and Ridge use
the existing weighted SVD helper. Other bases retain the dispatcher's weighting
limits. Matrix targets currently require a transformed_target_regressor final
estimator, directly or inside a nested pipeline; other native scalar-target
bases are not silently replaced with independent multioutput models.

fit accepts X alone, a real target vector, an integer target vector, or supported
real matrix targets. Optional sample_weight goes to the final estimator, with
semantics equivalent to final_step__sample_weight in a Python pipeline. It does
not train intermediate scalers/selectors with weights. fit_transform accepts
optional vector targets and requires a transformer/passthrough final step.
transform requires every active step to transform; inverse_transform requires
all active steps to provide a native inverse and applies them in reverse order.
Imputer/one-hot inverse paths absent from the underlying native types remain
unsupported. A FunctionTransformer's default identity inverse remains its own
contract, even when its forward callback changes width.

predict returns a real vector for scalar regression; predict_matrix returns
rank-two regression output, including multiple outputs from a supported final
TTR. predict_class returns integer classifier labels. predict_proba and
decision_function delegate to supported final classifier methods; binary
native decision scores have shape (samples,1). Methods are present as native
bindings and diagnose unavailable final-estimator capabilities. Native
transform/predict/score/name/inverse calls require fitting first, including
stateless-only pipelines. predict_log_proba, clustering-only final estimators,
fit_predict, partial_fit and arbitrary external estimator objects are outside
this increment's dispatch.

score delegates the final estimator's score semantics after preprocessing:
accuracy for classifiers, R2 for ordinary regressors/TTR, and the native GLM's
D2 for Tweedie/Poisson/Gamma descendants. Optional score weights are supported
where the underlying native score permits them; weighted native GLM D2 remains
unsupported. Pipeline scoring transforms each input once before delegating.
Targets and weights are validated by the relevant child/score helpers; the
pipeline does not impose the original X row count on a valid row-changing
transform before final fitting.

get_feature_names_out(input_features) chains supported child name methods.
Optional explicit feature_string names model named dense input; standard child
outputs are numeric matrices. Names remain available to fitting while steps
are passthrough or unvalidated identity FunctionTransformers; other numeric
outputs drop the input-container naming state. The pipeline records original
input metadata separately for chained output-name generation. TargetEncoder
multiclass names preserve integer versus real target-label formatting.

named_step_index(name) returns a 1-based index or zero and works before fitting.
Access fitted models through steps(i)%model. slice(first,last) returns an
independent, contiguous, inclusive 1-based pipeline copy; negative indices,
strides and Python's shared estimator references are not represented by this
helper. Fitted slices can transform/invert or expose names for a preprocessing
prefix. Slice input metadata comes from the first selected step's fitted input.

n_features_in_, feature_names_in_, n_features_out_, estimator_kind_, target_rank_
and classes_ provide native bookkeeping. estimator_kind_ is zero for a
transformer, one for a classifier and two for a regressor; classes_ is copied
on fitting. n_features_out_ is known after fit_transform or a final passthrough,
and otherwise -1. Step input_width_/output_width_/input_names_ record fitted
stage boundaries; integer_target_ records supervised label representation.
These are stored fields rather than Python delegated properties. In particular,
an all-passthrough native pipeline can generate x0, x1, ... names from recorded
width; Python can return None on that empty-transform naming path.
See example/sequential_pipeline.f90 for an imputation/scaling pipeline inside a
column transformer followed by regression.


## SelectFromModel

Import select_from_model from sklearn_feature_selection, sklearn, or the
implementation module sklearn_model_selection_features. Configure a native
base with allocate(selector%estimator,source=base), or pass base as the optional
estimator argument to fit/fit_transform. The selector stores independent copies;
its fitted child is estimator_. Caller objects are not mutated. With prefit=true,
fit copies a supported fitted model; get_support and transform can also use the
configured prefit estimator directly before fit. Native feature names require
fit. Fitted n_features_in_ records the dense input width.

Signatures (optional arguments shown in brackets):

```
call selector%fit(x, y_real_or_integer [,estimator,sample_weight,feature_names])
call selector%fit(x, y_matrix [,estimator,feature_names])
call selector%fit(x [,estimator,feature_names])  ! prefit=true only
z = selector%fit_transform(...)               ! same argument sets
call selector%partial_fit(x,y_real [,sample_weight,feature_names])
call selector%partial_fit(x,y_integer [,classes,sample_weight,feature_names])
z = selector%transform(x)
original_width = selector%inverse_transform(z)
mask = selector%get_support()
indices = selector%get_support_indices()      ! 1-based
scores = selector%feature_importances()
threshold = selector%threshold_value()
names = selector%get_feature_names_out([input_features])
```

During fit/fit_transform, target representation does not choose the estimator
role: a regression child accepts integer targets and a classifier accepts integral,
in-range real targets. Matrix targets require a native multi-task estimator.
fit without targets requires prefit=true; targets/weights are ignored for a
prefit child. Fitting imports optional feature_string names. Explicit names
passed to get_feature_names_out must match fitted metadata; otherwise names
are generated as x0, x1, ... . inverse_transform fills removed columns with zero.
Zero selected columns are allowed for transform; inverse_transform requires
nonempty finite input and the selected width.

threshold='auto' selects 1e-5 for Lasso/LassoCV, L1 ElasticNet/multi-task variants
and prefit L1 SGD flags; otherwise it uses the mean importance. Supported text
controls include 'mean', 'median', '1.25*mean' and '0.75 * median'. To supply a
numeric threshold set threshold='value' and numeric_threshold=<real>, including
positive/negative infinity. This two-field spelling replaces Python's union
of string/float/None. threshold_value() computes the current threshold, including
on a direct prefit model; it replaces Python's threshold_ property. NaN numeric
thresholds, nonfinite scales/importances and invalid expressions are rejected.

norm_order is real storage for a nonzero integer or positive/negative infinity.
Vectors use absolute values; matrices use column norms over response/class
rows. max_features=-1 means no cap, otherwise it is an integer in [0,p]. Caps
are computed into max_features_ on fit/first partial_fit and applied before the
threshold with stable descending order (earlier columns win exact ties). The
support is recomputed when requested from current fitted importances and controls.

importance_getter is 'auto', 'coef_', 'feature_importances_' or 'callback'. The
native coefficient adapters cover linear_regression descendants, logistic
regression descendants, ridge classifiers, perceptron/PA classification, SGD
and multi-task linear models. Tree adapters cover CART/ExtraTree descendants,
random forests/ExtraTrees descendants and gradient boosting. Native fit/weight
restrictions of these children apply. Custom importance callbacks can also
extract from other supported native base-dispatch models. Arbitrary Python
attribute paths are represented through a callback rather than reflection.

For importance_getter='callback', associate importance_func with a subroutine
(model,values,matrix): class(*) intent(in) model, allocatable real(dp)
intent(out) values(:,:), logical intent(out) matrix. A vector is represented
by one row and matrix=false; matrix=true applies norm_order down the rows.
Associate max_features_func with an integer function(x) returning a valid cap;
this is evaluated once per fit or first partial_fit. A callable cap requires
fit even for a prefit estimator. Procedures must outlive the selector.

partial_fit supports the existing SGD regression/classification bases and
prefit copying. Its typed entries use real targets for SGD regression and
integer labels for SGD classification. On the first SGD classification call pass the complete classes
array; later calls retain model state and cap. First cloning from an already
trained SGD base discards its fitted state, matching sklearn clone semantics.
Pipeline supports the selector as a supervised intermediate or final transformer;
its current weight forwarding applies to the final estimator. Parallel column
and feature graphs do not route supervised targets to this selector.


## General parameter search

Import grid_search_cv/randomized_search_cv and parameter helpers from sklearn_search
or sklearn. The existing sklearn_model_selection module remains independent of
these meta-estimators to avoid dependency cycles. Each search stores an allocated
class(*) estimator; configure param_grid(:), with each grid containing axes(:).
An axis has an allocatable name and values(:) of parameter_value, made by param
(real(dp)), param(integer), param(logical), or param(character). Axes are sorted
by name; products use rightmost-fastest order, and arrays of grids concatenate.
A grid with no axes represents one empty parameter dictionary. Axes require
nonempty values and unique names. Grid length()/at(i) and enumerate(), plus
enumerate_grids(grids), expose the same 1-based candidate order.

For randomized_search_cv configure n_iter (default 10) and random_state (default
0, nonnegative native integer). Finite spaces sample without replacement, capped
at the total grid length. To represent a distribution, leave an axis's values
unallocated and associate rvs with a function(legacy_random_state) returning a
parameter_value. If any distribution is present, sampling uses replacement and
chooses each supplied grid uniformly, matching ParameterSampler. A distribution
callback may use the legacy RNG's uniform/normal/integer_index methods and must
outlive the search. Grids cannot be exhaustively enumerated with distributions.
Direct helper sample_parameters(grids,n_iter,random_state) uses the same policy.
Large spaces are sampled by mixed-radix indexing without exhaustive expansion.

Call fit(x,y[,sample_weight]) with real or integer vector y. task='auto' recognizes
supported classifier models (including Pipeline tails); a regressor accepts
integer targets, and a classifier accepts integral in-range real labels.
Regression targets are converted to real before fitting; a Pipeline transformer
whose native scoring choice depends on integer target representation therefore
receives real targets through this search API. Use a custom fitting callback
for a different native target-typing convention.
task='regression'/'classification' is available for custom adapters. cv defaults
to 5 and produces unshuffled KFold/StratifiedKFold; the latter encodes classes by
first appearance, as sklearn. Supply cv_splits(:) of index_split for arbitrary
folds, including precomputed group/time splits. Indices are 1-based, nonempty and
in range; duplicates and train/test overlap are accepted. Fit weights are sliced
for each training fold and used for the final refit; CV scoring is unweighted,
as with disabled source metadata routing. Dense finite X and vector y only.

The default scoring='auto' uses estimator scoring. Built-in classifier names:
accuracy, balanced_accuracy, f1_macro, f1_weighted, neg_log_loss and binary roc_auc.
Regression names: r2, neg_mean_squared_error, neg_root_mean_squared_error and
neg_mean_absolute_error. Regression GLM auto scores retain the native D2 score.
The default neg_log_loss scorer infers test labels and requires the probability
class count to match, as sklearn; a custom scorer can provide explicit label
semantics. Associate scorer_func with a scalar function(model,x,yr,yi) instead
of a name, or allocate scorers(:) for named multiple metrics. search_scorer has
64-character name/scoring fields and an optional func with that same signature.
Only one target optional is present, according to the task. Do not configure
both scorer_func and scorers. Procedures must outlive the fitted search.

refit=true is the default. Single-metric refit chooses the first minimum-rank
candidate; multiple metrics require refit_metric to name the selection metric.
refit=false omits best_estimator_; single-metric searches still store best
metadata, while multi-metric searches have best_index_=0. Associate refit_func
with integer function(search_cv_results) to choose a 1-based candidate, with
refit=true. Callable refitting omits the best_score_ value/availability flag.
best_score_available_ distinguishes absent scores from valid NaN results.
For a multi-metric callable refit, score delegation additionally needs an
explicit refit_metric; the source callable-refit multi-metric score method has
no named selection metric. Refit models and parameter sets are independent
native value copies; scalar callback pointers retain their procedure targets.

cv_results_ stores params(:), metric_names(:), split_test_score(candidate,fold,
metric), mean_test_score(candidate,metric), std_test_score and rank_test_score.
return_train_score=true adds analogous train arrays. Ranks use exact minimum
rank ties; NaNs rank below finite scores, and all-NaN columns have rank 1.
Native timing arrays fit_time/score_time are candidate-by-fold, with mean/std
vectors. n_splits_, n_features_in_, folds_, best_index_, best_params_, best_score_,
classes_ and refit_time_ provide bookkeeping. n_features_in_ records input width
even without refitting. best_index_ is 1-based; zero means unavailable. Standard
deviations use population normalization. Fold means are unweighted across folds.

After refit, predict(x) returns real regression values; predict_class(x) returns
integer labels. predict_proba and decision_function delegate supported native
capabilities, with binary margins stored as n-by-1 matrices. score(x,y) uses the
fitted selected scorer. A later fit clears previous best models and train-score
arrays. Access the fitted child directly through best_estimator_.

Built-in setters cover common scalar controls of Ridge/OLS/ElasticNet and
logistic models, SGD, Gaussian/discrete NB, neighbors, CART/ExtraTree, forests,
gradient boosting, SVC/LinearSVC, scaling, PCA and simple/model-based selectors.
Nested names such as model__alpha address Pipeline steps; selector estimator__
paths and transformed-target regressor__ paths are supported. A Pipeline step
can be set to the text 'passthrough'. Unimplemented parameter paths are errors;
there is no silent acceptance. Native child option restrictions still apply.

For other parameter structures, associate set_params_func with subroutine
(model,parameter_set). For a custom estimator fitter, associate fit_func with
subroutine(model,x,yr,yi,sample_weight). Model arguments are class(*), x is a
real matrix, and targets/weights are optional vectors with the same typing as
the scoring callback. A fitter must reset training state on each call, as a
cloned estimator would. These hooks also support custom native regression
models with task='regression'; default prediction adapters remain restricted
to known native model types. Callbacks can inspect best_estimator_ directly.
The sole supported error_score policy is 'raise', which is the native default.
No parallelism, fit-error recovery, arbitrary Python reflection, sparse arrays,
unsupervised fitting or matrix targets are implemented here.


## General cross-validation

Import cross_validate, cross_val_predict, cross_val_predict_class and option/
result helpers from sklearn_validation or sklearn. The general cross_val_score
is exported only by sklearn_validation; the facade retains its older typed
linear/logistic helper. When using both modules, rename the general function:

```
use sklearn_validation, only: cv_score=>cross_val_score
```

```
results = cross_validate(estimator,x,y[,options,sample_weight])
results = cross_validate(estimator,x[,options,sample_weight]) ! no target
scores = cv_score(estimator,x,y[,options,sample_weight])
scores = cv_score(estimator,x[,options,sample_weight])
predictions = cross_val_predict(estimator,x,y[,options,sample_weight])
labels = cross_val_predict_class(estimator,x,y_integer[,options,sample_weight])
```

x is a finite real(dp) matrix. Supervised y is a real or integer vector. Options
are a cross_validation_options value; default cv=5, task='auto', scoring='auto',
method='predict', error_score='raise'. Task auto recognizes supported native
classifier prototypes and Pipeline tails; explicit 'classification'/'regression'
support custom native adapters. Integral, in-range real class labels are accepted.
An integer regression target fits real-valued bases; its integer representation
is preserved through default native Pipeline fits so supervised selectors retain
their native scoring dispatch. Custom fit callbacks receive real regression or
integer classification targets, according to task.

options%cv_splits(:) overrides cv with index_split values containing allocated,
nonempty 1-based train/test indices in range. Otherwise classification uses
unshuffled StratifiedKFold with first-appearance class encoding; regression and
no-target cases use KFold. cross_validate allows repeated tests and arbitrary
valid folds. cross_val_predict additionally requires every sample exactly once
in held-out indices; missing/repeated entries are errors. Predictions restore
the original sample order even when folds/test indices are nonmonotone.

Scoring names and search_scorer records reuse the parameter-search adapters.
Set options%scoring to a supported single metric, associate scorer_func with a
scalar callback, or allocate scorers(:) for named multiple metrics. Names/scoring
fields have length 64. Scalar callbacks have signature function(model,x,yr,yi),
with class(*) intent(in) model, real(dp) x(:,:), optional real(dp) yr(:), optional
integer yi(:), returning real(dp). At most one target is present. No-target
scoring requires explicit callbacks. Do not set both scorer_func and scorers.
The general scalar wrapper rejects multiple-scorer arrays and suppresses train
scoring/returned models/indices, regardless of the corresponding option flags.

cross_validation_result contains test_score(fold,metric), metric_names(:),
fit_time(:), score_time(:), n_splits and n_features_in. Set return_train_score to
allocate train_score(fold,metric), return_estimator to allocate estimators(:), and
return_indices to allocate indices(:). Each cv_estimator slot has an allocated
class(*) model containing that fold's fitted copy. No final full-data refit is
performed; caller templates remain untouched. Fit weights are sliced by training
indices, including duplicates; scoring is unweighted with disabled metadata
routing. Pipeline weights reach the final estimator. Existing native child
weighting/solver restrictions apply. Times are native wall-clock measurements
and exclude train scoring from score_time, following sklearn's ordering.

cross_val_predict returns an allocated real(dp) matrix: n-by-1 for scalar
regression/class predictions and binary decisions, or n-by-nclass for ordinary
class response matrices. method accepts 'predict', 'predict_proba',
'predict_log_proba' and 'decision_function'; the latter three require a classifier.
Global sorted integer labels are encoded to 0..k-1 before response-method fits,
as sklearn does. Missing fold classes get zero probability columns or
-huge(1._dp) log/margin columns. Missing-class decision outputs must represent
more than two fold classes and have compatible column widths. Built-in NB and
boosting log methods preserve stable log values. Other supported log APIs use
quiet log(probability), including -infinity for zero; KNN has no such API.
The typed class adapter requires method='predict', classifier task and one
integral finite response column, returning an allocated integer vector.

Associate fit_func with subroutine(model,x,yr,yi,sample_weight), using class(*)
intent(inout) model and the optional target/weight vectors above. It must reset
fitted state as a fresh clone would. No-target default fits use supported native
composition transformers; weights there require an explicit fitter. Associate
response_func with function(model,x,method) returning an allocated real(dp)
matrix for custom prediction adapters. For classifier response alignment,
classes_func may provide an allocated integer vector of that fitted model's
encoded class indices. Otherwise native class metadata are used. Encoded classes
must be unique and in global range. Procedure targets must outlive their use.

This API supports serial dense/vector tasks, explicit fail-fast diagnostics,
known native default adapters and custom native callbacks. It has no general
Python cloning/reflection, sparse or precomputed pairwise slicing, metadata
routing, parallelism, fit-error replacement, matrix/multilabel targets or
no-target OOF prediction. The broader sklearn translation remains ongoing.


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


## KernelPCA learned inverse

`kernel_pca` adds `%fit_inverse_transform=.false.` and `%alpha=1`.
Enable inverse fitting before `fit` or `fit_transform` to learn a kernel-ridge
pre-image of the original dense features from their latent coordinates.
`%x_transformed_fit_` stores those training coordinates and `%dual_coef_`
stores the sample-by-feature solution of `(K_latent + alpha I) dual = X`.
The latent kernel is uncentered and uses the same kernel, degree, coef0 and
fitted gamma as the forward model. In particular, automatic gamma remains
`1 / original_feature_count`, rather than being recomputed for latent width.

`inverse_transform(z)` returns a dense sample-by-original-feature matrix for
new latent rows. It is a learned approximation. For the linear kernel it does
not restore the original feature mean, matching sklearn; use ordinary PCA
when mean-restoring linear reconstruction is required. Supported kernels are
linear, RBF, polynomial, cosine and sigmoid. Nonnegative finite alpha is
accepted, including zero when the resulting kernel system is positive definite.
An indefinite or singular system is rejected by Cholesky, as in the source's
positive-definite solve. Inverse fitting with a precomputed kernel is rejected.

Inverse queries must be finite, nonempty and have the fitted latent width.
A refit with inverse fitting disabled clears both inverse-model arrays, so
reenabling the option requires a new fit. Explicit zero-eigenvalue components
remain usable; removing all components leaves no valid inverse feature space.
See `example/kernel_preimage.f90`. This increment retains the existing dense
Jacobi eigensolver and adds no new dependency or public estimator type.


## SVD linear discriminant analysis

`linear_discriminant_analysis` adds `%solver='svd'`, `%tol=1.e-4_dp`,
`%n_components=0` and `%store_covariance=.false.`. The native default solver
remains `'lsqr'` for compatibility; select `'svd'` for sklearn's default solver
algorithm. Native zero components denotes automatic `min(features, classes-1)`
selection (sklearn `None`). Positive requests cannot exceed that limit.
SVD fitting uses empirical class priors, within-class standardization and two
thin SVDs, with absolute within-rank and relative between-rank thresholds.
It supports wide and rank-deficient data and individual singleton classes,
provided there are at least two classes, more samples than classes and at least
one retained within-class direction. Native zero shrinkage denotes unregularized
fitting; nonzero shrinkage is rejected with SVD.

`fit_transform(x,y)` and `transform(x)` return centered discriminant coordinates.
`%scalings_` stores all retained discriminant axes, `%xbar_` the prior-weighted
center and `%explained_variance_ratio_` the between-class singular energy ratios.
`%within_rank_`, `%between_rank_` and `%n_components_` expose actual fitted ranks
and transform width. Component selection caps transform width; classification
still uses all retained discriminant axes, matching sklearn. Axis signs are
nonunique; reference comparisons align signs while checking predictions directly.

By default SVD never allocates a feature-by-feature covariance or precision
matrix. `%store_covariance=.true.` stores the population pooled covariance in
each inherited `%covariance_(:,:,class)` slice. SVD does not fit the inherited
QDA precision/logdet diagnostics. Switching back to LSQR clears SVD-specific
arrays and restores existing covariance-based fitting. Binary `%coef_`,
`%intercept_` and decision scores retain two native class rows; subtract the
first class from the second to obtain sklearn's single binary margin.

Coincident class means yield zero transform columns and NaN explained variance
ratios, matching source. An empty within-class rank receives an explicit native
diagnostic; sklearn 1.6.1 instead fails indexing an empty singular-value array.
LSQR transform is unsupported as in sklearn. Queries must be finite, nonempty
and match training feature count. See `example/discriminant_projection.f90`.
Custom priors and sparse inputs remain outside this interface; the eigen solver
is documented below.


## Eigen linear discriminant analysis

`linear_discriminant_analysis%solver='eigen'` enables sklearn's generalized
within/between covariance eigenproblem, including scalar `%shrinkage` in [0,1].
The native default remains `'lsqr'`. Eigen fitting uses population within-class
covariance and total covariance with the same scalar shrinkage, then solves
`(total - within) v = lambda within v` through Cholesky whitening and the native
symmetric eigensolver. The within covariance must be positive definite; positive
shrinkage can regularize rank-deficient data when its average variance is
positive. Two or more classes, more samples than classes and empirical priors
remain the native contract. Singleton classes are supported.

`%scalings_` stores a full feature-by-feature basis normalized in the within
covariance metric, sorted by decreasing eigenvalue. Native `%eigenvalues_`
exposes those generalized eigenvalues. `%explained_variance_ratio_` follows
sklearn's sorted normalization over all eigenvalues. `%n_components=0` means
automatic `min(features,classes-1)`; positive requests obey that bound.
`%n_components_` is the resulting transform width. Classification uses the
full basis, independently of that cap. The SVD-specific `%within_rank_` and
`%between_rank_` diagnostics do not describe eigen fitting. `%tol` is validated
but does not truncate the eigen basis, as in sklearn.

`transform` and `fit_transform` return `X @ scalings_(:, :n_components_)` without
centering, matching the eigen solver's source convention. This differs from
SVD's centered projection. Eigen fitting always retains within covariance
regardless of `%store_covariance`, as does source; inherited covariance slices
repeat it, precision slices store its inverse and logdet entries its log
determinant. Refitting with another solver clears `%eigenvalues_`.

Repeated eigenvalues allow rotations of their eigenspaces, including the choice
of axes at a component cutoff. Reference certificates check those subspaces,
covariance normalization and the generalized eigen-equation. Coincident class
means with zero between energy preserve source's NaN variance ratios; eigen
transform width is retained even in that case. See
`example/shrunk_discriminant_projection.f90`. Custom priors, automatic shrinkage
and custom covariance-estimator objects remain outside this increment.
