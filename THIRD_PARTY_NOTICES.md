# Third-party notices and provenance

Original fscikit-learn contributions are licensed under the MIT License in
`LICENSE`. This project license does not replace third-party licenses:
scikit-learn-derived portions retain the BSD-3-Clause terms and copyright
notices below and in `LICENSES/scikit-learn-COPYING`. Distributors must retain
both the project MIT license and the applicable upstream notices.

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
| minibatch_kmeans | cluster/_kmeans.py MiniBatchKMeans initialization subset/restart selection, sampled batches, streaming state, reassignment and smoothed-inertia/center-motion stopping; _k_means_minibatch.pyx weighted dense running means; _k_means_lloyd.pyx and _k_means_common.pyx dense assignment/inertia; native implementation with existing legacy RNG |
| bisecting_kmeans | cluster/_bisect_k_means.py leaf selection, split restarts, hierarchy/labels and prediction routing; _kmeans.py weighted random/greedy k-means++ initialization and Lloyd/Elkan stopping; _k_means_lloyd.pyx, _k_means_elkan.pyx and _k_means_common.pyx dense bounds, centroid updates, empty relocation and inertia; native implementation with existing legacy RNG |
| robust_covariance | covariance/_robust_covariance.py univariate FastMCD, C-steps, random candidate screening, subset/merge/full search, normal consistency correction and reweighting; _empirical_covariance.py covariance/precision and score/error conventions; _elliptic_envelope.py contamination offset and outlier scoring; independent dense eigendecomposition, regularized-gamma/chi-square inversion and native seeded permutation |
| isolation_forest | ensemble/_iforest.py random targets, path-length scoring and contamination; _bagging.py seeded feature/sample selection and warm-start seed skipping; utils/_random.pyx tracking/permutation/reservoir sampling; tree/_splitter.pyx, _partitioner.pyx, _criterion.pyx and _tree.pyx dense weighted random-tree construction; native implementation using existing legacy RNG |
| sgd_one_class | linear_model/_stochastic_gradient.py SGDOneClassSVM fit/partial_fit, initialization, offset conversion and warm/average state; _sgd_fast.pyx.tp hinge boundary, one-class intercept, L2 decay, schedules, stopping and sample clock; utils/_weight_vector.pyx.tp lazy scaling/averaging; utils/_seq_dataset.pyx.tp and _random.pxd deterministic shuffle; native dense implementation and existing legacy RNG |
| nu_classification | svm/_base.py and _classes.py NuSVC kernel/gamma, binary signs and OVO/OVR contracts; utils/multiclass.py vote/confidence transformation; bundled svm/src/libsvm/svm.cpp solve_nu_svc and Solver_NU group constraints, threshold levels and margin scaling; independent real64 gain-based pair optimizer |
| quantile | linear_model/_quantile.py QuantileRegressor weighted pinball/L1 objective, intercept exemption, positive/negative parameter and residual LP formulation, zero-weight row filtering and coefficient reconstruction; independent dense Bland-rule primal simplex, scaling and dual diagnostics; no SciPy/HiGHS code linked or adapted |
| adaboost_regression | ensemble/_weight_boosting.py AdaBoostRegressor/AdaBoost.R2 weighted bootstrap, losses, stopping and weighted median; tree/_splitter.pyx, _partitioner.pyx, _criterion.pyx dense squared-error CART and utils/_random.pxd Xorshift ordering; native dense implementation using existing legacy RNG |
| least_angle | linear_model/_least_angle.py _lars_path_solver and Lars/LassoLars centering, active-set equiangular directions, LAR sign changes, Lasso dropouts, correlation rounding and endpoint interpolation; independent dense Gram/Cholesky implementation; LarsCV/LassoLarsCV fold centering, alpha union/stride, residual interpolation and final refit; LassoLarsIC AIC/BIC and OLS/NNLS noise-variance contracts; native Jacobi SVD/active-set NNLS and piecewise-linear interpolation |
| dictionary | decomposition/_dict_learning.py DictionaryLearning/SparseCoder/sparse_encode, SVD initialization, code updates, constrained atom updates/resampling, objective and stopping; _sparse_pca.py centering, loading normalization and ridge projection; linear_model/_cd_fast.pyx Gram Lasso dual-gap stopping; native solvers and legacy RNG |
| recursive_selection | feature_selection/_rfe.py RFE/RFECV squared coefficient importance, elimination/ranking updates, per-fold paths, feature-count selection and final refit; model_selection/_split.py first-appearance StratifiedKFold allocation; native serial dense implementation |
| sequential_selection | feature_selection/_sequential.py SequentialFeatureSelector greedy candidate scoring, forward/backward masks, auto-count and tolerance stopping contracts; native serial K-fold R2 implementation for the linear regression family |
| model_selection splitters | model_selection/_split.py GroupKFold greedy group balancing, LeaveOneOut, LeaveOneGroupOut and PredefinedSplit contracts; native materialized 1-based index arrays and deterministic group-count tie ordering |
| discretization | preprocessing/_discretization.py; utils/stats.py weighted lower percentiles; cluster/_kmeans.py and _k_means_common.pyx Lloyd initialization, convergence and empty-cluster contracts; native weighted 1D implementation |
| power_transform | preprocessing/_data.py PowerTransformer; Yeo-Johnson forward/inverse formulas, constant-feature handling, likelihood and standardization contracts; native log-scaled objective and golden-section minimizer |
| quantile_transform | preprocessing/_data.py QuantileTransformer: percentile/reference tables, duplicate interpolation, NaN handling, boundary masks, inverse mapping and finite normal clipping; native erfc inversion |
| knn_impute | impute/_knn.py KNNImputer donor selection, weighting, empty features and indicator contracts; impute/_base.py MissingIndicator fitted column selection and new-missingness behavior; metrics/pairwise.py nan_euclidean_distances scaling; native dense search and stable donor ordering |
| spline | preprocessing/_polynomial.py SplineTransformer knot placement, boundary extension, periodic coefficient folding, bias removal and extrapolation contracts; native Cox-de Boor basis and derivative recurrences replace scipy.interpolate.BSpline |
| preprocessing | preprocessing/_data.py, _label.py, _encoders.py, _polynomial.py; impute/_base.py |
| metrics | metrics/_regression.py, _classification.py, _ranking.py, pairwise.py; metrics/cluster/ |
| multitask | linear_model/_coordinate_descent.py MultiTaskElasticNet/MultiTaskLasso centering and scaling; _cd_fast.pyx enet_coordinate_descent_multi_task block shrinkage and dual-gap stopping; native dense cyclic implementation |
| ridge_classification | linear_model/_ridge.py RidgeClassifier encoding, weighting, centering and RidgeClassifierCV default leave-one-out scoring/results; native shared SVD and dual-resolvent implementation |
| coordinate CV | linear_model/_coordinate_descent.py _alpha_grid and LinearModelCV (ElasticNetCV/LassoCV), model_selection/_split.py KFold boundaries; existing native coordinate solver |
| sgd | linear_model/_stochastic_gradient.py and _sgd_fast.pyx.tp: SGDRegressor/SGDClassifier losses, binary/OVR and class-weight contracts, probability maps, update order, L2 decay, learning rates, averaging, sample clock and stopping contracts; native dense update implementation |
| linear_model | linear_model/_base.py, _ridge.py, _coordinate_descent.py, _logistic.py, _perceptron.py, _bayes.py, _omp.py, _huber.py, _ransac.py, _glm/glm.py; _loss/loss.py (Tweedie objectives/deviance); _theil_sen.py subset-fit, breakdown and modified-Weiszfeld contracts; native SVD and independent Fortran subset sampling; _passive_aggressive.py, _stochastic_gradient.py, _sgd_fast.pyx.tp PA-I/II updates, class weighting, averaging and stopping contracts; independent native shuffling |
| naive_bayes | naive_bayes.py |
| categorical_nb | naive_bayes.py CategoricalNB per-feature category counts, smoothing, priors, sample weights, incremental expansion and likelihood contracts; native ragged dense tables and strict integer-code validation |
| neighbors | neighbors/_base.py, _classification.py, _regression.py |
| incremental_pca | decomposition/_incremental_pca.py stacked-SVD update, mean correction, batch and variance conventions; native Jacobi SVD |
| decomposition | decomposition/_pca.py, _truncated_svd.py, _nmf.py |
| optics | cluster/_optics.py: dense reachability graph, DBSCAN extraction, Xi steep regions, hierarchy labels and predecessor correction |
| feature_agglomeration | cluster/_agglomerative.py FeatureAgglomeration fitting and _feature_agglomeration.py pooling/inverse contracts; existing native merge tree |
| cluster | cluster/_birch.py (CF summaries, recursive insertion/splitting, leaf order and global/partial-fit contracts), cluster/_kmeans.py, _dbscan.py, _agglomerative.py, _mean_shift.py, _affinity_propagation.py, _spectral.py (including cluster_qr) |
| tree | tree/_classes.py; CART splitting and recursion rewritten in Fortran |
| adaboost | ensemble/_weight_boosting.py BaseWeightBoosting and AdaBoostClassifier SAMME updates, score/probability normalization and stopping; native weighted Gini stumps follow tree split contracts |
| ensemble | ensemble/_forest.py, _gb.py; GradientBoostingClassifier prior links, simultaneous gradients, Newton leaf updates, multiclass correction, train-score conventions and staged predictions; _loss/loss.py and link.py; tree/_criterion.pyx FriedmanMSE improvement scale |
| model_selection | model_selection/_split.py, _validation.py; linear_model/_ridge.py |
| feature_selection | feature_selection/_variance_threshold.py, _univariate_selection.py |
| feature_extraction_text | feature_extraction/text.py (ASCII tokenizer subset) |
| dummy | dummy.py |
| kernel_ridge | kernel_ridge.py |
| discriminant_analysis | discriminant_analysis.py (LDA lsqr covariance convention) |
| kernel_svm | svm/_base.py, _classes.py, src/libsvm/svm.cpp; SVR/NuSVR/OneClassSVM weighted dual formulations, gamma rules, thresholds and normalization; independent native pairwise optimizer |
| linear_svm | svm/_classes.py LinearSVC/LinearSVR contracts; svm/src/liblinear/linear.cpp L2 squared-loss objectives, synthetic intercept regularization and binary/OVR class costs; tron.cpp stopping behavior used in reference validation; independent native primal Newton/Armijo solver |
| svm | svm/_classes.py; deterministic SMO replaces libsvm backend |
| mixture | mixture/_gaussian_mixture.py, _base.py |
| neural_network | neural_network/_multilayer_perceptron.py, _stochastic_optimizers.py |
| gp_classification | gaussian_process/_gpc.py Laplace posterior mode, evidence stopping, binary prediction and five-erf probability approximation (Williams & Barber); multiclass.py OVR normalization and tie contracts; native dense Cholesky implementation |
| gaussian_process | gaussian_process/_gpr.py, kernels.py (fixed kernel subset) |
| covariance | covariance/_empirical_covariance.py, _shrunk_covariance.py; _graph_lasso.py and linear_model/_cd_fast.pyx block updates, Gram coordinate descent, objective and gap definitions; native Cholesky initialization/diagnostics |
| datasets | datasets/_samples_generator.py (independent Fortran random stream) |
| radius neighbors / nearest centroid | neighbors/_classification.py, _regression.py, _nearest_centroid.py |
| outlier (LOF) | neighbors/_lof.py |
| kernel_pca | decomposition/_kernel_pca.py; kernel centering and PSD eigenvalue rules adapted for Fortran |
| manifold (Isomap) | manifold/_isomap.py; utils/graph.py (connected-component completion) |
| semi_supervised | semi_supervised/_label_propagation.py; RBF/KNN diffusion and clamping |
| mds | manifold/_mds.py; metric/nonmetric SMACOF updates, stress normalization, legacy convergence and multi-start selection; native isotonic fitting and independent Fortran RNG |
| locally_linear | manifold/_locally_linear.py; local barycentric reconstruction, standard/LTSA alignment objectives and query interpolation; native linear solve and dense eigensolver |
| cross_decomposition | cross_decomposition/_pls.py; centering/scaling, NIPALS mode A/B, regression/canonical deflation, rotations, prediction and PLSSVD contracts; native dense SVD/pseudoinverse including historical CCA rank cutoff |
| isotonic | isotonic.py, _isotonic.pyx; weighted monotonic fitting/interpolation contracts, Spearman direction, near-duplicate aggregation and threshold trimming; native PAVA implementation |
| kernel_approximation | kernel_approximation.py (including AdditiveChi2Sampler defaults, harmonic map and layout; SkewedChi2Sampler inverse-CDF weights, offsets, shifted-log map and domain); RBFSampler Fourier distribution/transform and Nystroem landmark normalization contracts; independent Fortran RNG and native PSD eigensolver |
| fast_ica | decomposition/_fastica.py; parallel/deflation fixed-point updates, contrasts, whitening and scaling contract; native dense SVD/eigen/pseudoinverse and independent random stream |
| factor_analysis | decomposition/_factor_analysis.py; full-SVD likelihood iteration, diagonal noise updates and posterior transform |
| random_projection | random_projection.py; JL dimension bound, Gaussian/three-point distributions and pseudoinverse contract; independent Fortran random stream |
| spectral | manifold/_spectral_embedding.py; degree-normalized eigenvectors and affinity construction, with native dense eigensolver |
| density | neighbors/_kde.py, _binary_tree.pxi; kernel normalization, bandwidth rules and weighted mixture contract; native brute-force evaluation and Gaussian/uniform-ball sampling |

The additional SVM models also follow the LIBSVM formulations bundled with
sklearn 1.6.1 in `sklearn/svm/src/libsvm/svm.cpp`, copyright (c) 2000-2009
Chih-Chung Chang and Chih-Jen Lin. Its complete BSD-3-Clause notice is retained
in `LICENSES/libsvm-COPYRIGHT`. The bundled sklearn file credits dense/weighted
support to Ming-Fang Weng, Fabian Pedregosa and the instance-weight contributors
Ming-Wei Chang, Hsuan-Tien Lin, Ming-Hen Tsai, Chia-Hua Ho and Hsiang-Fu Yu.
This adaptation uses a native dense SMO optimizer with gain-based second-coordinate selection and
full real64 Gram matrices; no LIBSVM C++ source or binary is compiled or linked.
Original contributions remain MIT; the retained upstream terms still apply.

The primal linear SVM models follow the objective and weighting conventions
of LIBLINEAR bundled with sklearn 1.6.1, copyright (c) 2007-2014 The LIBLINEAR
Project. Its complete BSD-3-Clause notice is retained in
`LICENSES/liblinear-COPYRIGHT`. The native active-set Newton/Armijo solver is
implemented independently; no LIBLINEAR C/C++ source or binary is compiled
or linked. Original contributions remain MIT; retained upstream terms apply.

SciFort is not currently linked. There are no third-party runtime dependencies.

## Proposed future reuse

The projects assessed in `docs/UPSTREAM_REUSE.md` are candidates, not included
dependencies or vendored source. No code from those candidates is incorporated
by the license/review change. When code is adopted, record the exact repository,
revision, source files, modifications, copyright holders and license here; retain
the complete upstream license under `LICENSES/` and source-level attribution.
Dependency licenses and file-level exceptions must be checked separately.
Each adopted feature must also be classified under `docs/SCOPE.md`. Public
extensions require an entry in `docs/EXTENSIONS.md` and explicit module/API
identification, in addition to the same license and provenance requirements
that apply to core implementations.


## NumPy legacy random stream compatibility

sklearn_legacy_rng.f90 is a native Fortran adaptation of the NumPy 2.2.6
RandomState scalar-seed MT19937, masked interval selection and cached polar
Gaussian contracts, used for dictionary atom resampling, AdaBoost regression
bootstraps, SGDOneClassSVM shuffle seeds, IsolationForest sampling/tree seeds,
FastMCD permutations, BisectingKMeans initialization and MiniBatchKMeans
initialization/batch/reassignment draws. Reference sources:

- numpy/random/src/mt19937/mt19937.c and mt19937.h
- numpy/random/src/legacy/legacy-distributions.c (legacy_gauss)
- numpy/random/src/distributions/distributions.c (random_interval)

Exact upstream tag: https://github.com/numpy/numpy/tree/v2.2.6/numpy/random/src
The implementation uses int64 storage/masks to avoid signed 32-bit overflow,
1-based Fortran indices, and explicit seed-state checks. No NumPy source or
binary is compiled or linked. Array seeds, jumps and other distributions are
not implemented by this helper.

NumPy Developers' BSD notice is retained in LICENSES/numpy-COPYING. Original
MT/RandomKit notices and the MIT-style uniform constants notice, including
Jean-Sebastien Roy (2003-2005), Makoto Matsumoto and Takuji Nishimura (1997-2002),
Richard J. Wagner, Magnus Jonsson and Isaku Wada, are retained in
LICENSES/numpy-MT19937-LICENSE.md. The random module's Kevin Sheppard (2019)
dual NCSA/BSD notice is retained in LICENSES/numpy-RANDOM-LICENSE.md; BSD is the
selected alternative for these portions. The complete distributions component
notice is retained in LICENSES/numpy-DISTRIBUTIONS-LICENSE.md; the Julia-derived
ziggurat implementations described there are not used. Applicable upstream
terms continue to apply to adapted portions; original contributions retain
the project MIT license.
