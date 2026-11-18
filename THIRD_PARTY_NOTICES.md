# Third-party notices and provenance

This package is a native Fortran adaptation of a subset of scikit-learn APIs and
algorithms. The reference baseline is the locally installed scikit-learn 1.6.1,
copyright (c) 2007-2024 The scikit-learn developers, BSD-3-Clause. Its license is
retained in `LICENSES/scikit-learn-COPYING`. No scikit-learn DLL, Cython extension,
Python runtime, NumPy binary, or SciPy binary is distributed by this project.

Numerical algorithms are rewritten for dense real64 Fortran arrays. In
particular the Ledoit-Wolf and OAS shrinkage formulas are adapted from
`sklearn/covariance/_shrunk_covariance.py`. Reference comparisons invoke the
installed Python package only from the validation scripts.

Reference paths by module:

| Fortran module | scikit-learn reference |
| --- | --- |
| preprocessing | preprocessing/_data.py, _label.py, _encoders.py, _polynomial.py; impute/_base.py |
| metrics | metrics/_regression.py, _classification.py, _ranking.py, pairwise.py; metrics/cluster/ |
| linear_model | linear_model/_base.py, _ridge.py, _coordinate_descent.py, _logistic.py, _perceptron.py |
| naive_bayes | naive_bayes.py |
| neighbors | neighbors/_base.py, _classification.py, _regression.py |
| decomposition | decomposition/_pca.py, _truncated_svd.py, _nmf.py |
| cluster | cluster/_kmeans.py, _dbscan.py, _agglomerative.py |
| tree | tree/_classes.py; CART splitting and recursion rewritten in Fortran |
| ensemble | ensemble/_forest.py, _gb.py |
| model_selection | model_selection/_split.py, _validation.py; linear_model/_ridge.py |
| feature_selection | feature_selection/_variance_threshold.py, _univariate_selection.py |
| feature_extraction_text | feature_extraction/text.py (ASCII tokenizer subset) |
| dummy | dummy.py |
| kernel_ridge | kernel_ridge.py |
| discriminant_analysis | discriminant_analysis.py (LDA lsqr covariance convention) |
| svm | svm/_classes.py; deterministic SMO replaces libsvm backend |
| mixture | mixture/_gaussian_mixture.py, _base.py |
| neural_network | neural_network/_multilayer_perceptron.py, _stochastic_optimizers.py |
| gaussian_process | gaussian_process/_gpr.py, kernels.py (fixed kernel subset) |
| covariance | covariance/_empirical_covariance.py, _shrunk_covariance.py |
| datasets | datasets/_samples_generator.py (independent Fortran random stream) |

SciFort is not currently linked. There are no third-party runtime dependencies.
