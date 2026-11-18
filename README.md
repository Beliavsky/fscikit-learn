# fscikit-learn

A native Fortran fpm package translating a practical dense-array subset of
scikit-learn: **56 estimator, transformer and splitter types**, metrics,
kernels, text features and synthetic dataset generators. This is not a complete
or drop-in translation. Read [coverage and limits](docs/COVERAGE.md) before use.

## Build and test

```
fpm build
fpm test
fpm run --example classification
python validation/compare_sklearn.py
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

SciFort at `../scifort` is available as a future numerical backend; the initial
package intentionally needs no external dependency or network fetch to build.
