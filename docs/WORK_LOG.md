# Translation work log

## Initial package milestone (2026-11-17/18)

The requested target directory was empty. The reference source was recovered
from the locally installed scikit-learn 1.6.1 Python package. A standalone native
Fortran fpm package was built in this directory, with no external runtime or
network dependencies. SciFort remains available for subsequent backend work.

Implemented 56 public estimator/transformer/splitter types across preprocessing,
linear models, Naive Bayes, neighbors, clustering, decomposition, CART trees,
ensembles, model selection, feature selection, text features, dummy estimators,
kernel ridge, discriminant analysis, SVC, Gaussian mixtures, MLPs, covariance and
fixed-kernel Gaussian-process regression, plus metrics, kernels and generators.

Validation includes 14 native test programs, a 192-comparison Python reference
suite, 16 isolated invalid-input tests and four runnable examples. See
`validation/README.md` for the comparison scope and exceptions. Direct Jacobi SVD
was added after initial normal-equation work to preserve nearly collinear
least-squares solutions; tests include rank-deficient, nearly collinear, wide,
and zero matrices.

## Git checkpoint status

`git init` succeeded. The first `git add .` and `git commit` failed with:

```
fatal: Unable to create 'C:/fortran/fscikit-learn/.git/index.lock': Permission denied
```

The session's filesystem policy explicitly grants read-only access to `.git`,
despite permitting writes to the source workspace. A request to enable write
access was sent to the user while implementation continued. No existing user
repository or source work was overwritten. The restriction was not bypassed or
worked around with an alternate repository location.

On 2026-11-18 the user enabled unrestricted filesystem access and requested a
commit. The tested initial package is being recorded as one initial commit,
including the implementation, examples, validation scripts and documentation.

## Graph, neighbor and outlier algorithms (2026-11-18)

Added eight further scikit-learn counterpart types: radius-neighbor
classification/regression, nearest centroids, Local Outlier Factor, Kernel PCA,
Isomap, LabelPropagation and LabelSpreading. Kernel PCA is re-exported from
`sklearn_decomposition` and LOF from `sklearn_neighbors`, with focused modules
also available. All are exported by the core facade; no extension was added.

Added two native test programs, a semi-supervised example, 120 numerical/status
comparisons against sklearn 1.6.1 and ten additional input-error cases. The
existing 192 reference comparisons continue to pass. Scope, API, provenance and
coverage documentation now distinguish the supported dense subsets and their
remaining limitations.
