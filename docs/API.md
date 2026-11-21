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

Weighted decision trees and all standalone ExtraTrees use float32 feature
geometry, sklearn's 1e-7 split grouping threshold and seeded legacy NumPy /
xorshift feature selection. Training features must be finite and within
float32 range. Queries may contain NaNs: each missing feature follows the
larger active-row child, with ties going right, matching a source tree trained
without missing values. Query infinities and float32 overflow are rejected.
NaN training, sparse input, pruning, monotonic constraints, fractional feature
counts, alternate regression criteria, and multioutput single-tree fitting
remain outside this subset. Native max_depth=0 retains its existing root-only
meaning; sklearn requires a positive depth.

DecisionTree fitting without weights retains the existing real64 geometry and
native RNG/tie convention. Both CART paths now allow a zero-gain split, so a
later split can learn XOR. Prefix recomputation can select a different optimal
split from sklearn's incremental C criterion at floating-point ties; an exact
source tree topology is not guaranteed in those cases.

Voting, independent multioutput estimators, regressor/classifier chains, and
multiclass reductions accept ExtraTree bases through their decision-tree
interfaces and forward training weights to both tree families. Tree streaming
(`partial_fit`) remains unavailable. The new internal `sklearn_ieee_math`
module handles quiet nonfinite arithmetic in weighted trees and composite
reductions; it adds no public estimator or external runtime dependency.
