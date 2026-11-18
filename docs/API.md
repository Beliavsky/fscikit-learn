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
regression, k-means, NMF, Gaussian mixtures, MLPs, and SVC. A completed `fit` does
not guarantee convergence if `max_iter` was exhausted. MLP convergence means
training loss failed to improve by `tol` for `n_iter_no_change` iterations;
there is no held-out validation-set stopping criterion.

Set `random_state` for repeatable stochastic fits. Seeding uses the Fortran
runtime's global RNG, not NumPy's RNG; matching integer seeds do not reproduce
Python initialization. Stochastic results need not match across compilers.

See [COVERAGE.md](COVERAGE.md) for the complete type list, supported algorithm
variants, resource limits, omitted parameters and remaining translation work.
