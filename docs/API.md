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
There is no MDS query transform. Both `standard` and `ltsa` LLE methods support
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
stopping status. See COVERAGE.md for the objective and supported controls.


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
