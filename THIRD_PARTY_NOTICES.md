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

The sklearn adaptations use dense real64 Fortran arrays. Explicitly identified
vendored kernels below retain their upstream implementations. In particular
the Ledoit-Wolf and OAS shrinkage formulas are adapted from
`sklearn/covariance/_shrunk_covariance.py`. Reference comparisons invoke the
installed Python package only from the validation scripts.

Reference paths by module:

| Fortran module | scikit-learn reference |
| --- | --- |
| nnls / linear_model / weighted_linear | sklearn 1.6.1 linear_model/_base.py positive LinearRegression centering/rescaling, _ridge.py positive Ridge objective, _coordinate_descent.py positive ElasticNet/Lasso/CV; SciPy 1.15.3 optimize/_cython_nnls.pyx Lawson-Hanson active sets, QR insertion/removal and iteration accounting (BSD, LICENSES/scipy-COPYING); LAPACK 3.12.0 DLARFGP/DLARTGP reflector/rotation conventions (LICENSES/lapack-LICENSE, University of Tennessee, University of California Berkeley, University of Colorado Denver, NAG Ltd.). Native scaled reflector, checked ranges and correction of the stale last QR-tail entry in SciPy 1.15.3. Research: C. L. Lawson and R. J. Hanson, Solving Least Squares Problems (1974) |
| nmf_mu / decomposition | decomposition/_nmf.py _fit_multiplicative_update, _multiplicative_update_w/h, _beta_divergence and NMF fixed-H initialization; float32-epsilon denominator conventions, float64 factor cutoffs, ten-iteration stopping and sklearn 1.6.1 cached KL/L1 transform behavior. Research: D. D. Lee and H. S. Seung (2001), multiplicative NMF; C. Fevotte and J. Idier (2011), beta-divergence MM. Existing attributed NNDSVD/legacy-RNG initializer; original native checked real64 arithmetic |
| decomposition (coordinate-descent NMF) | decomposition/_nmf.py NMF initialization, regularization scaling, alternating coordinate sweeps, projected-gradient stopping and fixed-component transform; decomposition/_cdnmf_fast.pyx projected coordinate update order and zero-Hessian handling. Reuses the attributed MiniBatchNMF NNDSVD initializer and NumPy-compatible legacy RNG. Native finite working-range diagnostics |
| graphical_lasso_cv (covariance) | covariance/_graph_lasso.py refined CV grids and warm-start paths; covariance/_empirical_covariance.py likelihood and alpha-zero baseline; linear_model/_cd_fast.pyx zero-diagonal CD skip; model_selection/_split.py KFold; shared native Jacobi and IEEE environment helpers |
| ordinal_encoding | preprocessing/_encoders.py numeric OrdinalEncoder, category grouping and inverse index conventions; utils/_encode.py counting contracts; shared native numeric category container from target_encoding |
| target_encoding | preprocessing/_target_encoder.py, preprocessing/_target_encoder_fast.pyx, preprocessing/_encoders.py numeric categories and empirical Bayes encodings; model_selection/_split.py cross-fitting; existing NumPy legacy RNG backend |
| tsne / tsne_core | manifold/_t_sne.py exact TSNE and trustworthiness; manifold/_utils.pyx perplexity search; existing legacy RNG, randomized SVD and NumPy pairwise reduction backends |
| resampling | model_selection/_split.py random/group/stratified holdouts, repeated folds and exhaustive combinations; utils/extmath.py _approximate_mode quotas and random tie breaking; existing NumPy legacy RNG backend |
| cluster_information | metrics/cluster/_supervised.py normalized/adjusted MI averages and signed epsilon denominator, Fowlkes-Mallows, contingency smoothing/order, ordered pair orientation, entropy and homogeneity/completeness/weighted V-measure conventions; _expected_mutual_info_fast.pyx hypergeometric expected MI with log factorials. Native heap-sorted observed cells, int64 counts, cached intrinsic log-gamma and explicit validation; existing H/C/V metric wrappers delegate to this backend |
| mutual_info | feature_selection/_mutual_info.py Kraskov continuous/continuous and Ross continuous/discrete nearest-neighbor MI, singleton removal, adaptive class-wise neighbor counts, scaling and seeded Gaussian jitter; neighbors/_base.py auto backend rule and metrics/_pairwise_distances_reduction/_argkmin.pyx.tp ordered Euclidean Gram terms; _univariate_selection.py r_regression moment and IEEE contracts; metrics/cluster/_supervised.py mutual_info_score contingency/log/epsilon rules. Native pair selection and integer harmonic digamma differences; source-compatible scalar-seeded legacy Gaussian stream |
| label_binarization | preprocessing/_label.py LabelBinarizer sorted target classes, binary/query-dependent output width, configurable positive/negative labels, threshold and clipped-argmax inverse contracts; MultiLabelBinarizer discovered/configured class order, duplicate/unknown label handling, dense indicators and inverse set ordering. Native integer ragged label_set representation; explicit diagnostics for unsupported inverse scores and source single-class positive-index failure |
| polynomial_count_sketch | kernel_approximation.py PolynomialCountSketch index/sign draw order, augmented polynomial features, count sketches, FFT product/inverse and live transform-control semantics; native legacy RandomState helper described below; FFT kernels separately attributed to SciFort below |
| univariate_selection | feature_selection/_univariate_selection.py ANOVA/regression/chi-square score and p-value contracts, percentile/NaN/tie rules, FPR, Benjamini-Hochberg FDR, Bonferroni FWE and generic mode selection; feature_selection/_base.py support, projection and inverse conventions; NumPy 2.2.6 lib/_function_base_impl.py percentile interpolation semantics. Native beta continued fraction (modified Lentz) from [NIST DLMF 8.17.22-23](https://dlmf.nist.gov/8.17#v), symmetry 8.17.4, and integer/half-integer upper-gamma recurrence [DLMF 8.8.6](https://dlmf.nist.gov/8.8#E6). No SciFort code was copied or linked; mathematical specifications were independently implemented. Native zero-column inverse restoration is explicitly documented |
| omp_cv | linear_model/_omp.py OrthogonalMatchingPursuitCV and _omp_path_residues fold centering, minimum-length path truncation, equally weighted fold MSE selection and full-data refit; _cholesky_omp forward paths and _gram_omp auto-precomputed refit with column permutations and incremental triangular solves; model_selection/_split.py KFold/PredefinedSplit partition conventions. Native mask-based arbitrary partitions, path diagnostics and explicit empty-path diagnostic |
| minibatch_nmf | decomposition/_nmf.py MiniBatchNMF beta-divergence/MM updates, regularization scaling, random/NNDSVD initialization and fill conventions, online numerator/denominator forgetting, fresh restarts, cached streaming parameters and EWA/motion/patience stopping; existing utils/extmath.py randomized-SVD backend and NumPy-compatible Gaussian stream. Research: C. Boutsidis and E. Gallopoulos, Pattern Recognition (2008), NNDSVD; C. Fevotte and J. Idier, Neural Computation (2011), beta-divergence algorithms; A. Lefevre, F. Bach and C. Fevotte (2011), online Itakura-Saito NMF. Native explicit null-direction/zero-norm handling and consistent beta-domain checks |
| minibatch dictionary / sparse PCA | decomposition/_dict_learning.py MiniBatchDictionaryLearning randomized initialization, fixed shuffled batches, normalized sufficient statistics/forgetting, constrained atom resampling, smoothed-cost/motion/patience stopping and streaming state; _sparse_pca.py MiniBatchSparsePCA transposed training, recoding and normalization; utils/extmath.py randomized_svd/range_finder Gaussian draw order, oversampling, iteration/transpose defaults and signs. Native pivoted LU/Householder QR/Jacobi SVD; existing batch atom updates and ridge projections share private helpers without changing arithmetic |
| lda | decomposition/_lda.py batch/online variational Bayes, stochastic sufficient statistics, partial-fit scaling, inference, evidence bounds and perplexity; _online_lda_fast.pyx AS 103 digamma/mean-change contracts (J. Bernardo, 1976); NumPy 2.2.6 legacy-distributions.c shape-100 gamma rejection sampler; native stable Dirichlet-bound algebra and Stirling remainder. Research attribution: Matthew D. Hoffman, David M. Blei and Francis Bach (2010); Hoffman, Blei, Chong Wang and John Paisley (2013). Translation references are the BSD-licensed sklearn and NumPy sources, with full notices retained below and in LICENSES/ |
| rbm | neural_network/_rbm.py BernoulliRBM persistent contrastive-divergence updates, hidden/visible conditionals, free energy/pseudo-likelihood, minibatch scheduling, scalar-seeded fit/partial_fit/Gibbs RNG lifetimes and initialization; native stable softplus and existing NumPy-compatible legacy RNG |
| hdbscan | cluster/_hdbscan/hdbscan.py dense brute orchestration, nonfinite markers, weighted centers and DBSCAN cuts; _reachability.pyx mutual reachability; _linkage.pyx Prim frontier/linkage; _tree.pyx condensing, stability, EOM/leaf/epsilon selection and membership strengths; native deterministic tie ordering and nonfinite-row corrections; component BSD notice in [LICENSES/hdbscan-BSD.md](LICENSES/hdbscan-BSD.md), copyright Leland McInnes, with contributors Steve Astels, John Healy, Meekail Zain and Guillaume Lemaitre |
| bayesian_mixture | mixture/_bayesian_mixture.py Normal-Wishart/Beta/Dirichlet updates, normalized covariance conventions, expected log scores and simplified variational bound; _base.py initializer/restart/warm-start/zero-iteration and Gaussian sampling contracts; _gaussian_mixture.py sufficient statistics and precision Cholesky conventions; native positive digamma/log-Gamma, Cholesky solves and legacy seeded initialization |
| centroid_core | extracted unweighted Lloyd helper formerly private to bicluster; cluster/_kmeans.py greedy/random initialization, variance-scaled stopping and same-partition restart selection; _k_means_lloyd.pyx and _k_means_common.pyx updates/relocation; reused by biclustering and Bayesian mixture initialization, including an optional continued RNG state |
| bicluster | cluster/_bicluster.py scale/bistochastic/log normalization, Dhillon row-column embedding, Kluger piecewise-vector selection and original-data projections; base.py BiclusterMixin indicators/getters; _kmeans.py greedy/random centroid seeding, Lloyd convergence and same-partition restarts; _k_means_lloyd.pyx and _k_means_common.pyx dense updates, inertia and relocation; native thin SVD/null-space completion and existing MiniBatchKMeans/legacy RNG |
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
| multitask | linear_model/_coordinate_descent.py MultiTaskElasticNet/MultiTaskLasso centering/scaling, MultiTaskElasticNetCV/MultiTaskLassoCV alpha grids, warm paths, fold residuals, ordering and full-data refits; _cd_fast.pyx enet_coordinate_descent_multi_task block shrinkage and dual-gap stopping; utils/_random.pxd uint32 XorShift and 31-bit feature sampling; model_selection/_split.py KFold/PredefinedSplit partition semantics; existing NumPy-compatible bounded seed draws and native dense cyclic/random implementation |
| ridge_classification | linear_model/_ridge.py RidgeClassifier encoding, weighting, centering and RidgeClassifierCV default leave-one-out scoring/results; native shared SVD and dual-resolvent implementation |
| coordinate CV | linear_model/_coordinate_descent.py _alpha_grid and LinearModelCV (ElasticNetCV/LassoCV), model_selection/_split.py KFold boundaries; existing native coordinate solver |
| sgd | linear_model/_stochastic_gradient.py and _sgd_fast.pyx.tp: SGDRegressor/SGDClassifier losses, binary/OVR and class weights, probability maps, L1/elastic-net truncated gradients, L2 decay, all four schedules, lazy averaging, sample clocks and stopping; utils/_seq_dataset.pyx.tp and _random.pxd seeded permutations; native dense implementation |
| linear_model | linear_model/_base.py, _ridge.py, _coordinate_descent.py, _logistic.py, _perceptron.py, _bayes.py, _omp.py, _huber.py, _ransac.py, _glm/glm.py; _loss/loss.py (Tweedie objectives/deviance); _theil_sen.py subset-fit, breakdown and modified-Weiszfeld contracts; native SVD and independent Fortran subset sampling; _passive_aggressive.py, _stochastic_gradient.py, _sgd_fast.pyx.tp PA-I/II updates, class weighting, averaging and stopping contracts; independent native shuffling |
| naive_bayes | naive_bayes.py |
| categorical_nb | naive_bayes.py CategoricalNB per-feature category counts, smoothing, priors, sample weights, incremental expansion and likelihood contracts; native ragged dense tables and strict integer-code validation |
| neighbors | neighbors/_base.py, _classification.py, _regression.py |
| incremental_pca | decomposition/_incremental_pca.py stacked-SVD update, mean correction, batch and variance conventions; native Jacobi SVD |
| decomposition | decomposition/_pca.py, _truncated_svd.py, _nmf.py |
| optics | cluster/_optics.py: dense reachability graph, DBSCAN extraction, Xi steep regions, hierarchy labels and predecessor correction |
| feature_agglomeration | cluster/_agglomerative.py FeatureAgglomeration fitting and _feature_agglomeration.py pooling/inverse contracts; existing native merge tree |
| cluster | cluster/_birch.py (CF summaries, recursive insertion/splitting, leaf order and global/partial-fit contracts), cluster/_kmeans.py, _dbscan.py, _agglomerative.py, _mean_shift.py, _affinity_propagation.py, _spectral.py (including cluster_qr) |
| stacking / weighted_linear / composite_scores | ensemble/_stacking.py; model_selection/_validation.py cross_val_predict response ordering; model_selection/_split.py first-appearance stratification; linear_model/_ridge.py weighted LOO/GCV; existing native weighted SVD, classifier dispatch and score arithmetic |
| bagging / bagging_sampling / base_dispatch | ensemble/_bagging.py weighted/indexed fit paths, aggregation, OOB and warm starts; ensemble/_base.py derived estimator seeds; utils/_random.pyx index sampling; linear_model/_base.py weighted centering/rescaling and rank cutoff; linear_model/_ridge.py weighted ridge; existing native SVD, voting dispatch and legacy NumPy RNG backend |
| tree | tree/_classes.py decision/standalone ExtraTree interfaces; tree/_tree.pyx weighted node growth/importance; tree/_splitter.pyx and _partitioner.pyx float32 split geometry, random thresholds and feature caches; tree/_criterion.pyx weighted Gini/entropy/squared error; utils/_random.pxd xorshift feature selection; native dense Fortran adaptation |
| adaboost | ensemble/_weight_boosting.py BaseWeightBoosting and AdaBoostClassifier SAMME updates, staged/weighted scores, log probabilities and stopping; ensemble/_base.py seeded nested parameters; tree/_splitter.pyx, _partitioner.pyx and _criterion.pyx dense classification sorting, partitioning and incremental criterion contracts; NumPy pairwise reduction association; native configurable weighted CART implementation |
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
| manifold (Isomap) | manifold/_isomap.py radius/fixed-count neighborhoods, precomputed distances and projection; utils/graph.py connected-component completion; neighbors/_base.py stored sparse neighbor selection; base.py output feature names; SciPy csgraph undirected shortest-path contracts; original native FW and heap Dijkstra implementations |
| semi_supervised | semi_supervised/_label_propagation.py; RBF/KNN diffusion and clamping |
| mds | manifold/_mds.py; metric/nonmetric SMACOF updates, stress normalization, legacy convergence and multi-start selection; native isotonic fitting and independent Fortran RNG |
| ranking / calibration | metrics/_ranking.py binary threshold curves, AUC/AP and partial-area conventions; metrics/_classification.py Brier loss; calibration.py reliability bins; NumPy pairwise sum and percentile interpolation |
| locally_linear / lle_variants | manifold/_locally_linear.py; local barycentric reconstruction, standard/Hessian/modified/LTSA alignment objectives and query interpolation; native QR, orthogonal completion, Jacobi local SVD/eigensolver and dense global eigensolver |
| cross_decomposition | cross_decomposition/_pls.py; centering/scaling, NIPALS mode A/B, regression/canonical deflation, rotations, prediction and PLSSVD contracts; native dense SVD/pseudoinverse including historical CCA rank cutoff |
| isotonic | isotonic.py, _isotonic.pyx; weighted monotonic fitting/interpolation contracts, Spearman direction, near-duplicate aggregation and threshold trimming; native PAVA implementation |
| kernel_approximation | kernel_approximation.py (including AdditiveChi2Sampler defaults, harmonic map and layout; SkewedChi2Sampler inverse-CDF weights, offsets, shifted-log map and domain); RBFSampler Fourier distribution/transform and Nystroem landmark normalization contracts; independent Fortran RNG and native PSD eigensolver |
| fast_ica | decomposition/_fastica.py; parallel/deflation fixed-point updates, contrasts, whitening and scaling contract; native dense SVD/eigen/pseudoinverse and independent random stream |
| factor_analysis | decomposition/_factor_analysis.py; full-SVD likelihood iteration, diagonal noise updates and posterior transform |
| random_projection | random_projection.py; JL dimension bound, Gaussian/three-point distributions and pseudoinverse contract; independent Fortran random stream |
| spectral | manifold/_spectral_embedding.py; degree-normalized eigenvectors and affinity construction, with native dense eigensolver |
| density | neighbors/_kde.py, _binary_tree.pxi.tp, _kd_tree.pyx.tp, _ball_tree.pyx.tp and metrics/_dist_metrics.pyx.tp; kernel normalization, KD/Ball bounds, traversal/stopping, weighted mixtures and metric contracts; NumPy RandomState draw ordering and reductions; scipy.special.gammainc uniform-ball mapping; native trees, heap partitions and incomplete-gamma implementation |

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
initialization/batch/reassignment draws and spectral bicluster centroid
initialization, Bayesian mixture initializations and native mixture Gaussian
sampling, and BernoulliRBM initialization, persistent chains, Gibbs transitions
and pseudo-likelihood corruption. Mini-batch dictionary/SparsePCA initialization,
row shuffling and unused-atom resampling also reuse this stream. MiniBatchNMF
uses its Gaussian draws for random/NNDSVD initialization and NNDSVDAR filling.
Multi-task coordinate paths also reuse the bounded-integer stream to seed
sklearn's separate uint32 XorShift coordinate sampler.

PolynomialCountSketch uses this same legacy stream for its bucket and sign
tables, retaining source zero-based bucket values and C-order draw traversal.

Reference sources:

- numpy/random/src/mt19937/mt19937.c and mt19937.h
- numpy/random/src/legacy/legacy-distributions.c (legacy_gauss)
- numpy/random/src/distributions/distributions.c (random_interval)

Exact upstream tag: https://github.com/numpy/numpy/tree/v2.2.6/numpy/random/src
The implementation uses int64 storage/masks to avoid signed 32-bit overflow,
1-based Fortran indices, and explicit seed-state checks. No NumPy source or
binary is compiled or linked. Array seeds, jumps and other distributions are
not implemented by this helper. LDA adds a private shape-100 Gamma sampler
using the same legacy_standard_gamma contract from
[numpy/random/src/legacy/legacy-distributions.c](https://github.com/numpy/numpy/blob/v2.2.6/numpy/random/src/legacy/legacy-distributions.c);
it does not expand the helper's public distribution API.

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

## Clustering information metrics (BSD)

The clustering-information adaptation was inspected locally on 2026-11-20 in
the installed sklearn 1.6.1 package. Exact source SHA-256 identifiers:

- `metrics/cluster/_supervised.py`:
  `b295e06460372e48fa426308413029796c69241b75412eae77749dacf4de27e4`
- `metrics/cluster/_expected_mutual_info_fast.pyx`:
  `c489de6ee38a9c22578088e73cc51eac9f9e9444ae0cb4f36cd809da733e3e5c`

Native changes use sorted int64 histograms, heap sorting, scalar/vector Fortran
interfaces, cached intrinsic log factorials and explicit validity checks.
The original sklearn BSD notice remains in `LICENSES/scikit-learn-COPYING`.
Research references, following sklearn's source: Vinh, Epps and Bailey (2010),
"Information Theoretic Measures for Clusterings Comparison: Variants,
Properties, Normalization and Correction for Chance," JMLR; E. B. Fowlkes and
C. L. Mallows (1983), "A Method for Comparing Two Hierarchical Clusterings."

## NumPy pairwise summation (BSD)

The mutual-information implementation also adapts NumPy 2.2.6's real
pairwise summation order from
[`numpy/_core/src/umath/loops_utils.h.src`](https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/loops_utils.h.src)
in the private `numpy_sum` helper of `src/sklearn_mutual_info.f90` and
`pairwise_sum` in `src/sklearn_cluster_information.f90` and `pair_sum` in
`src/sklearn_tsne_core.f90` and `src/sklearn_ranking.f90`. The native
adaptation uses real64 array sections, eight partial accumulators, a 128-value
block cutoff and aligned recursive splits, with no compiled NumPy dependency.
The complete NumPy BSD license/copyright is retained in `LICENSES/numpy-COPYING`.
Inspected 2026-11-20; exact upstream source SHA-256:
`ed85d5b99f0b457cb7a316bf8c4acb884034bbbb81a89c5cedb62be80bf7c4f8`.
This helper supports source scaling/jitter boundaries and Pearson centering;
the legacy Gaussian helper retains the additional notices listed above.

Research references for the MI estimators, following sklearn's source:
A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual information,"
Physical Review E 69 (2004); B. C. Ross, "Mutual Information between Discrete
and Continuous Data Sets," PLoS ONE 9(2) (2014).

## SciFort FFT helpers (MIT)

`src/sklearn_count_sketch_fft.f90` adapts the FFT plan, radix-2 transform and
Bluestein convolution paths from the user-provided SciFort translation at
`c:/fortran/scifort/src/signal_spectral.f90`, inspected 2026-11-20. The exact
input file is identified by SHA-256
`77283c0c5c14214e9a9c95c04ac9d737ee5153444a1027d38acd2f2d14f082a6`.
Copyright (c) 2026 SciFort contributors. The full MIT license is retained in
[`LICENSES/scifort-MIT.txt`](LICENSES/scifort-MIT.txt) and the source header
retains the upstream copyright and license identifier.

Adaptation isolates radix-2/Bluestein, uses project real64 kinds and explicit
input/allocation diagnostics, removes unrelated spectral/window/mixed-radix
code, and computes chirp phases with an int64 modular square. The radix-2
stage loop stops before its final integer doubling. The helpers are vendored
implementation support; no SciFort library, Python package or external FFT
runtime is linked. The plan is absent from the core `sklearn` facade and
algorithm counts. SciFort identifies these FFT kernels as original MIT work;
its separately attributed Welch/window analysis code is not included here.

Mathematical reference: Leo I. Bluestein, "A linear filtering approach to the
computation of discrete Fourier transform," IEEE Transactions on Audio and
Electroacoustics 18(4), 1970, pp. 451-455,
[doi:10.1109/TAU.1970.1162132](https://doi.org/10.1109/TAU.1970.1162132).
The estimator's sklearn contracts retain the existing BSD-3-Clause notice;
original project contributions remain MIT.

## Randomized, repeated and exhaustive splitters (BSD)

`src/sklearn_resampling.f90` adapts scikit-learn 1.6.1
`model_selection/_split.py`: ShuffleSplit, StratifiedShuffleSplit,
GroupShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold, LeavePOut and
LeavePGroupsOut; and `utils/extmath.py` `_approximate_mode` class quota and
random tie-breaking logic. Source files under the installed baseline have
SHA-256 hashes:

- `_split.py`: `1dc467b09b1b4903c15db0329ee5431e66a7d67c8e44f61ebe70885f33e5ff1b`
- `extmath.py`: `8b608e1ba07b779dcb9cc063131494c36c64f602e4bbd8ba0dc26a9cad573720`

The existing `sklearn_legacy_rng` backend supplies NumPy-compatible seeded
integer draws; its existing NumPy notices remain applicable. Native
Fisher-Yates permutations reproduce RandomState.permutation and the complete
permutation consumed by RandomState.choice without replacement, including
full-size tie choices. Combinations use a native lexicographic successor and
checked int64 binomial recurrence; no itertools code was copied. The shared
`index_split` declaration moved from `sklearn_model_selection`, which re-exports
it and the new types. No upstream binary or new dependency is distributed.
Original contributions retain MIT; adapted portions retain the full upstream
BSD-3-Clause terms in `LICENSES/scikit-learn-COPYING`.

## Exact/Barnes-Hut t-SNE and trustworthiness (BSD)

`src/sklearn_tsne.f90` and `src/sklearn_tsne_core.f90` adapt scikit-learn 1.6.1
`manifold/_t_sne.py`: TSNE exact-distance handling, PCA/random initialization,
Student objective/gradient, adaptive gains, two momentum/exaggeration phases,
checkpoint stopping, fitted diagnostics and trustworthiness rank penalties.
The conditional-probability binary search follows `manifold/_utils.pyx`,
including float32 input/perplexity conversion, 100 steps, tolerances and mass
fallback. Source SHA-256 hashes under the installed baseline:

- `_t_sne.py`: `207fd179a53a2067990c3411ac0ad49dc70df67a942bd1e91b5b4003ead9e34a`
- `_utils.pyx`: `1bae8cb04ebc6682f2d878103e9821d8fab6399876b87c9de3ca6259700dd1bc`

`src/sklearn_tsne_bh.f90` additionally translates the serial spatial tree and
Barnes-Hut forces from sklearn 1.6.1 `neighbors/_quad_tree.pyx` and
`manifold/_barnes_hut_tsne.pyx`, plus sparse affinity construction from
`_t_sne.py`. The Barnes-Hut source credits Christopher Moody and Nick Travers;
the spatial tree credits Thomas Moreau and Olivier Grisel.
Attribution and BSD-3-Clause terms are retained in
`LICENSES/scikit-learn-COPYING`. Additional source hashes:

- `_barnes_hut_tsne.pyx`: `c2222a881c6a22426ed5da3c40493efd19bb0199f390cdbf7726712f80e8e64e`
- `_quad_tree.pyx`: `81a023897606191ad22c7ed387cc3a4648904101d36568e8e905a9086e0ece72`
- `_quad_tree.pxd`: `3c1c56c3ba2e3a71b5268e9fc3df976ff8c51e6f011e1ac4360222879506d96e`

Research attribution: Laurens van der Maaten, "Accelerating t-SNE using
Tree-Based Algorithms," Journal of Machine Learning Research 15 (2014),
3221--3245. The port follows the installed binary's exported duplicate
threshold `EPSILON=0`; its pxd source initializer is 1.e-6. Skipped-point
handling is tested directly against the compiled backend on buffer-safe
fixtures because the Python wrapper does not forward that argument. Native
tree traversal does not use the upstream fixed summary buffer. NumPy-style
float32 pairwise summation also supplies PCA initialization's PC1 scaling.
The package remains MIT with retained third-party BSD notices.

Research attribution: Laurens van der Maaten and Geoffrey Hinton, "Visualizing
Data using t-SNE," Journal of Machine Learning Research 9 (2008), 2579--2605;
Laurens van der Maaten, "Learning a Parametric Embedding by Preserving Local
Structure," AISTATS/PMLR 5 (2009), 384--391; Jarkko Venna and Samuel Kaski,
"Neighborhood Preservation in Nonlinear Projection Methods: An Experimental
Study," ICANN (2001), 485--491. The translation references the BSD sklearn
implementation; no code from other t-SNE implementations was imported.

`pair_sum` follows the NumPy pairwise summation reference recorded above.
The existing NumPy-compatible legacy RNG and randomized-SVD backend are reused
with their retained provenance. The optional internal optimizer trace only
records native states for independent source-step validation; normal fitting
does not allocate traces. New original contributions remain MIT, adapted
portions retain BSD-3-Clause and the complete existing upstream notices in
`LICENSES/scikit-learn-COPYING` and the applicable NumPy license files.
No new dependency or Python/SciPy runtime is required by the Fortran package.

## Hessian eigenmapping and modified LLE (BSD)

`src/sklearn_lle_variants.f90` and the extended
`src/sklearn_locally_linear.f90` adapt sklearn 1.6.1
`manifold/_locally_linear.py`: Hessian tangent/monomial/full-QR construction,
column-sum normalization, modified-LLE local eigensystems, fixed trace
regularization, median spectral ratio and cumulative-spectrum search,
Householder multiple weights and dense alignment accumulation. Installed
source SHA-256:
`1c4e17b21891c57d802492f627a7c65e0af0cc1347bdcd40a1e486b5b12fbe29`.
The existing native Jacobi SVD and eigensolver are reused; new full Householder
QR and orthogonal-completion routines implement the required matrix operations.

Research attribution follows the source references: D. Donoho and C.
Grimes, "Hessian eigenmaps: Locally linear embedding techniques for
high-dimensional data," PNAS 100:5591 (2003); Z. Zhang and J. Wang,
"MLLE: Modified Locally Linear Embedding Using Multiple Weights." Existing
standard-LLE/LTSA attribution remains applicable to shared fitting and query
interpolation. Original contributions retain MIT; adapted sklearn portions
retain BSD-3-Clause and the complete notices in LICENSES/scikit-learn-COPYING.
No new dependency is added.

## Ranking curves, calibration and Brier loss (BSD)

`src/sklearn_ranking.f90` adapts sklearn 1.6.1 `metrics/_ranking.py`:
_binary_clf_curve zero-weight removal, stable score order and tied cumulative
thresholds, ROC/PR thinning and endpoints, DET trimming, trapezoidal AUC,
weighted binary ROC AUC/AP and McClish partial-area correction. Binary Brier
label inference and weighted squared error follow `metrics/_classification.py`;
uniform/quantile calibration bins, left-edge assignment and empty-bin removal
follow `calibration.py`. `src/sklearn_calibration.f90` re-exports the source
calibration counterpart, and sklearn_metrics re-exports the ranking metrics.
Installed baseline SHA-256 hashes:

- `metrics/_ranking.py`: `1857a13da466d965db1ac18c9c383bc06eeb56aec146f4dcf3f0abe6bb9cdc21`
- `metrics/_classification.py`: `569fff857c6a27025ad6f4fb61a6ebc30cca07b35c786c8ce026af1fc0b7f7dd`
- `calibration.py`: `d7a4a6c186bdf5c84ee359ee89cc841d3180659eafbe99012b468fce945425c9`

NumPy 2.2.6 `pair_sum` reduction order uses the pairwise-summation source
recorded above; linear percentile interpolation/linspace conventions follow
`numpy/lib/_function_base_impl.py`, SHA-256
`dd3d59d7025ab419fc7d5ee890df66a0ba222f98d288d30646f7b1d2345f86f8`.
The stable bottom-up merge sort and safe IEEE denominator handling are native
implementations. The trapezoid formula is implemented directly; no SciPy
runtime or new dependency is required. Original contributions remain MIT;
adapted portions retain the existing complete sklearn/NumPy BSD notices in
LICENSES/. The source's partial-AUC and probability-calibration references
remain attributed to the sklearn implementation.

## Self-training classification (BSD)

`src/sklearn_self_training.f90` adapts scikit-learn 1.6.1
`semi_supervised/_self_training.py`: labeled/unlabeled masks, iterative
base fitting, strict confidence thresholds, k-best selection, transduction
and labeling-iteration histories, stopping-rule precedence, final refitting
and delegated prediction. Installed source SHA-256:
`12bdc9c504e27e18c20210736123ecdf07361eb8df48f32e1a639fe50b5de96a`.
The stable merge sort gives native tied-cutoff ordering; NumPy's partition
implementation is not copied. Existing logistic, Gaussian NB and KNN
implementations retain their existing provenance. The source research
reference is D. Yarowsky, "Unsupervised word sense disambiguation rivaling
supervised methods," ACL (1995).

Original contributions remain MIT. Adapted sklearn portions retain the
complete BSD-3-Clause notice in `LICENSES/scikit-learn-COPYING`.
No dependency is added.

## Cross-validated probability calibration (BSD)

`src/sklearn_calibrated_classifier.f90` adapts scikit-learn 1.6.1
`calibration.py`: classifier/calibrator fold pairs, out-of-fold responses and
full refitting, prefit calibration, binary/one-vs-rest targets, sigmoid target
smoothing and score scaling, isotonic calibration, probability normalization
and ensemble averaging. Installed source SHA-256:
`d7a4a6c186bdf5c84ee359ee89cc841d3180659eafbe99012b468fce945425c9`.
Stratified fold allocation follows `model_selection/_split.py`'s
first-appearance class encoding, with its existing retained attribution.
Installed source SHA-256:
`1dc467b09b1b4903c15db0329ee5431e66a7d67c8e44f61ebe70885f33e5ff1b`.

Sigmoid fitting uses a native two-variable damped Newton optimizer for the
same Platt objective; no SciPy optimizer code is copied. Research attribution
follows the source reference to J. Platt, "Probabilistic Outputs for Support
Vector Machines." Isotonic fitting reuses the attributed weighted PAV and
interpolation implementation. Native logistic fitting now removes the common
softmax-intercept null mode before downstream raw-score calibration. Its
existing objective and solver provenance remain applicable.

Original contributions remain MIT; adapted portions retain BSD-3-Clause and
the complete notice in `LICENSES/scikit-learn-COPYING`. No new dependency or
Python runtime is needed by the Fortran package.

## Weighted and streaming Gaussian/logistic classification (BSD)

`src/sklearn_naive_bayes.f90` adapts sklearn 1.6.1 `naive_bayes.py`:
Gaussian weighted population moments, partial-fit class/state checks,
Chan/Golub/LeVeque updates, current-batch unweighted smoothing and source
epsilon subtraction, tiny-mass skipping with count accumulation, optional
fixed priors and joint-likelihood prediction. Explicit quiet IEEE handling
retains undefined posteriors and first-NaN prediction behavior without
raising floating-point traps for those source-defined cases.

Weighted logistic objective/regularization conventions in
`src/sklearn_linear_model.f90` follow `linear_model/_logistic.py` and
`_linear_loss.py`. Raw-score classifier prediction follows
`linear_model/_base.py`. The native Newton solver and its new diagonal
equilibration remain original implementations; no SciPy solver code is copied.
Training-weight routing in sklearn_calibrated_classifier and base prediction
delegation in sklearn_self_training retain their existing source attribution.

Installed source SHA-256:

- `naive_bayes.py`: `4126fb342ef030d92e95f418d168e3af1ca5e22278435d792512b84d51e0208d`
- `linear_model/_logistic.py`: `14b72d51a007c2c1c2af7ab1824783818c058f3f611584eb7e7432a1b4e0426a`
- `linear_model/_linear_loss.py`: `8792fb74ec8ab80219b2918dd9be88075de5c8bf817fc69335672215e94d381b`
- `linear_model/_base.py`: `d9a38e2aea003351917b744f85fcb7d0a5f849b55b3d12480fbae560ebc38bd0`

The moment-update research attribution follows source: Chan, Golub and
LeVeque, Stanford technical report STAN-CS-79-773. The regression fixture is
synthetic native-project validation data. Original contributions remain MIT;
adapted portions retain BSD-3-Clause and the complete sklearn notice in
`LICENSES/scikit-learn-COPYING`. No dependency is added.


### Independent multioutput wrappers

`src/sklearn_multioutput.f90` adapts independent-column fitting, cloning,
streaming, class metadata, prediction and subset-score semantics from
scikit-learn 1.6.1 `multioutput.py`, with regression scoring following the
uniform-average R2 convention. Installed `multioutput.py` SHA-256:
`164702b2de983ba8c5a8bbfd93842f9651f5b01dd7f4a4dd3541becf220a273e`.
Fortran dispatch and ragged probability containers are original work.
Original contributions remain MIT; adapted portions retain BSD-3-Clause
attribution under `LICENSES/scikit-learn-COPYING`. No additional dependency
or public extension is introduced.


### RegressorChain and lazy SGD weight scaling

`src/sklearn_multioutput.f90` additionally adapts RegressorChain ordering,
observed-target and cross_val_predict feature construction, fitting and
prediction from the same sklearn 1.6.1 `multioutput.py` identified above.
Its random ordering uses the previously attributed NumPy-compatible legacy
RNG. `src/sklearn_sgd.f90` now follows WeightVector64's lazy L2 scale,
ordered inner product, scale reset and materialization from
`utils/_weight_vector.pyx.tp` and `_sgd_fast.pyx.tp`.
Installed source SHA-256:

- `linear_model/_sgd_fast.pyx.tp`: `dcaf04170f5454a410a486704ad8bb32cd7ba00ca878ee8c857a4cb7a29c90a8`
- `utils/_weight_vector.pyx.tp`: `f57b27c21d8d5b7ab08536d6c3677093a6deab21181ffff099e17f3850a1f6c6`

Adapted portions retain sklearn BSD-3-Clause attribution and the complete
notice in `LICENSES/scikit-learn-COPYING`. Original contributions remain
MIT. No runtime dependency or public extension is introduced.


### ClassifierChain response and CV semantics

`src/sklearn_multioutput.f90` additionally adapts ClassifierChain fitting,
response-method selection, positive-class response compression, class
metadata and ordered prediction from sklearn 1.6.1 `multioutput.py` and
`utils/_response.py`. Its unshuffled stratification follows first-appearance
class encoding and allocation in `model_selection/_split.py`, including
minority classes smaller than the fold count. Existing estimator code and
NumPy-compatible random ordering retain their prior notices.
Installed source SHA-256:

- `multioutput.py`: `164702b2de983ba8c5a8bbfd93842f9651f5b01dd7f4a4dd3541becf220a273e`
- `utils/_response.py`: `ba600bd75b63c3c0e787965b082ba36147524db824c6d809a270b20a92d6a2a6`
- `model_selection/_split.py`: `1dc467b09b1b4903c15db0329ee5431e66a7d67c8e44f61ebe70885f33e5ff1b`

Adapted portions retain BSD-3-Clause and the complete scikit-learn notice in
`LICENSES/scikit-learn-COPYING`; original contributions remain MIT. No
additional runtime dependency or public extension is introduced.


### Multiclass reductions and binary Perceptron layout

`src/sklearn_multiclass.f90` adapts OVR, OVO, output-code, constant predictor,
streaming and response-combination semantics from sklearn 1.6.1
`multiclass.py` and `utils/multiclass.py`. Output-code sampling uses the
previously attributed NumPy-compatible legacy RNG. The binary Perceptron
layout in `src/sklearn_linear_model.f90` follows the single positive-class
head in `linear_model/_stochastic_gradient.py`.
Installed source SHA-256:

- `multiclass.py`: `b54b232c9f22c725a8a7f665f7341d07e53a5f5ea9a4c600c8cd702f025c12b1`
- `utils/multiclass.py`: `55589d96c1ac5f6544630f53b68cbfb10c9b49446eef23f646c7e282af9e71ab`
- `linear_model/_stochastic_gradient.py`: `794fb96ab273ba06aad41e7470e05bc47422e8e047d07a99703d0dbb8447aa0b`

Adapted portions retain BSD-3-Clause attribution and the complete notice in
`LICENSES/scikit-learn-COPYING`. Original contributions remain MIT. No runtime
dependency or public extension is introduced.


### Voting ensembles

`src/sklearn_voting.f90` adapts fitting, dropped-estimator filtering, class
encoding, hard/soft voting, averaging and transform layouts from sklearn
1.6.1 `ensemble/_voting.py`; name validation follows `ensemble/_base.py`.
Scoring follows the accuracy/R2 mixin contracts and force-finite R2 behavior
in `metrics/_regression.py`. Native configuration containers and dispatch
are original work. Installed source SHA-256:

- `ensemble/_voting.py`: `8f08b8c94dc77e372dc4e8a8366e339e62d117f04a067e8ce94afba0a989d5aa`
- `ensemble/_base.py`: `a1d7f71030dd358bbae7b7a782e9a4410e9785976c16a82d12b16a319b705790`
- `metrics/_regression.py`: `60453695ff142bd8242274f894ad141fbc025814996f51b4507660bf6688dfd1`

Adapted portions retain BSD-3-Clause and the complete notice in
`LICENSES/scikit-learn-COPYING`. Original contributions remain MIT. No
additional runtime dependency or public extension is introduced.


### Permutation feature importance

`src/sklearn_inspection.f90` adapts baseline-minus-permuted scoring, seeded
worker setup, repeated shuffles, row subsampling and multiple-score reduction
from sklearn 1.6.1 `inspection/_permutation_importance.py`. Subsampling follows
`ensemble/_bagging.py` and `utils/_random.pyx`; the native implementation shares
the already attributed sampling/RNG contracts used by IsolationForest.
Fortran callback interfaces and guarded quiet-IEEE arithmetic are original
work; reductions follow NumPy mean/std semantics. Installed source SHA-256:

- `inspection/_permutation_importance.py`: `70b1969b1feeb6c601b807f8102fc465d556240fbc3e687891b4595a1b983d26`
- `ensemble/_bagging.py`: `fd07bd726787e8c1d4c0d02518a7763ce9f38f017b5c867db2955869e31f5b86`
- `utils/_random.pyx`: `d5e999626dd7af7200ff531f35e00f4d5f0748fcca2e4325a960bd17dec35321`

Adapted portions retain BSD-3-Clause and the complete sklearn notice in
`LICENSES/scikit-learn-COPYING`; prior NumPy notices are retained. Original
contributions remain MIT. No runtime dependency or public extension is added.


### Brute partial dependence and individual conditional expectation

`src/sklearn_partial_dependence.f90` adapts grids, categorical/unique selection,
Cartesian evaluation, brute averaging and individual-response semantics from
sklearn 1.6.1 `inspection/_partial_dependence.py`. Empirical grid endpoints
follow SciPy 1.15.3 `stats/_mstats_basic.py` mquantiles with alphap=betap=.4.
The SciPy source module credits original author Pierre GF Gerard-Marchant
(2007). SciPy copyright (c) 2001-2002 Enthought, Inc. 2003-2024 SciPy Developers;
its source BSD-3-Clause notice is retained in `LICENSES/scipy-COPYING`.
No bundled SciPy binary component or runtime is used. Native callbacks,
flattened grid-axis storage and guarded IEEE arithmetic are original work.
Installed source SHA-256:

- `sklearn/inspection/_partial_dependence.py`: `82ecaf365bb7734ce9745799f1da8fa9cb1d899236088583f65aadaf52b3b80b`
- `scipy/stats/_mstats_basic.py`: `d6822a0a1e7d62e05b9f39425e145102861fb79a1fcc7be4da0e77cffb766d68`

Adapted portions retain their sklearn/SciPy BSD-3-Clause attribution and
complete retained notices. Original contributions remain MIT. No runtime
dependency or public extension is introduced.

Weighted CART and standalone ExtraTrees were adapted against the installed
scikit-learn 1.6.1 source files listed above. Their BSD-3-Clause attribution
is retained in `LICENSES/scikit-learn-COPYING`; original Fortran contributions
remain under MIT. The existing legacy NumPy RNG attribution also applies to
tree seed initialization. No additional repository or runtime was imported.

### LogisticRegressionCV

The dense L2 CV grid, global class balancing, stratified folds, warm-started
paths, score selection and refit/averaging contracts adapt scikit-learn 1.6.1
`linear_model/_logistic.py`, `model_selection/_split.py`,
`utils/class_weight.py` and `metrics/_classification.py`/`_scorer.py`.
Scikit-learn's BSD-3-Clause notice is retained in
`LICENSES/scikit-learn-COPYING`. Native full Newton optimization and Fortran
array/wrapper integration are original contributions under MIT. No new
repository or runtime dependency was imported. Reference reports record
SHA-256 hashes of the consulted installed source files.

### Binary threshold classifiers and unweighted CART

`src/sklearn_threshold.f90` adapts FixedThresholdClassifier and
TunedThresholdClassifierCV behavior from scikit-learn 1.6.1
`model_selection/_classification_threshold.py`, `metrics/_scorer.py`,
`utils/_response.py`, and the stratified splitters in `model_selection/_split.py`.
Their BSD-3-Clause notice is retained in `LICENSES/scikit-learn-COPYING`.
The existing sklearn-derived CART kernel now also handles omitted weights;
its prior retained attribution continues to apply. Reference reports record
hashes of consulted installed source files. Original Fortran interfaces,
callbacks and validation code remain MIT. No external repository or runtime
dependency was imported.

### RandomTreesEmbedding and KernelCenterer

`src/sklearn_tree_embedding.f90` adapts sklearn 1.6.1 `ensemble/_forest.py`,
`ensemble/_base.py`, and `preprocessing/_encoders.py`, reusing the retained
sklearn-derived ExtraTree kernel and NumPy-compatible RNG. The source BSD notice
remains in `LICENSES/scikit-learn-COPYING`.

`src/sklearn_kernel_centering.f90` adapts sklearn `preprocessing/_data.py` and
NumPy 2.2.6's pairwise summation ordering from
[numpy/_core/src/umath/loops_utils.h.src](https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/loops_utils.h.src).
NumPy copyright (c) 2005-2024 NumPy Developers; its BSD notice is retained in
`LICENSES/numpy-COPYING`. Consulted file SHA-256:
`ed85d5b99f0b457cb7a316bf8c4acb884034bbbb81a89c5cedb62be80bf7c4f8`.

The tested sklearn tree binary exports FEATURE_THRESHOLD=0 despite its .pxd
initializer of 1e-7. The native grouping now follows the observed runtime;
validation records source and binary hashes and reads the exported capsule
without modifying it. Original interfaces and validation remain MIT.
No external runtime dependency was added.


### Neighbor graph transformers and CSR storage

`src/sklearn_neighbor_graph.f90` adapts the behavior of scikit-learn 1.6.1
`sklearn/neighbors/_graph.py` and `_base.py` (The scikit-learn developers,
BSD-3-Clause; retained in `LICENSES/scikit-learn-COPYING`). The native brute
implementation uses existing package metrics and stable ordering rather than
copying upstream Cython heap/partition code. Dense versus CSR stored-edge
semantics and k+1 distance-mode selection follow these sources.
`src/sklearn_sparse.f90` is original MIT Fortran storage code using the standard
compressed-row representation; no SciPy implementation code is copied.
Reference checks use installed sklearn, SciPy and NumPy only during validation.


### Iterative multivariate imputation

`src/sklearn_iterative_impute.f90` adapts scikit-learn 1.6.1
`sklearn/impute/_iterative.py` (The scikit-learn developers, BSD-3-Clause).
The retained license is `LICENSES/scikit-learn-COPYING`. Round-robin fitting,
feature ordering, correlation-weighted neighborhoods, convergence, empty-feature
handling, prediction clipping and posterior mean guards follow that source.
Bayesian ridge fitting and NumPy-compatible MT19937 use existing package code
with its existing notices. The truncated-Gaussian inverse-CDF implementation
is original Fortran using intrinsic erf/erfc, centered Newton iteration and
monotone bisection; no SciPy implementation or approximation coefficients are
copied or distributed. SciPy is used only by the installed validation reference.


### Histogram gradient boosting and numeric tree grower

`src/sklearn_hist_gradient_boosting.f90`, `sklearn_histogram_tree.f90` and
`sklearn_histogram_binning.f90` adapt scikit-learn 1.6.1
`ensemble/_hist_gradient_boosting/{gradient_boosting.py,binning.py,grower.py,
splitting.pyx,histogram.pyx}`, `_loss/{loss.py,_loss.pyx.tp}`, and the lower
weighted percentile contract in `utils/stats.py`. Copyright The scikit-learn
developers, BSD-3-Clause; retained in `LICENSES/scikit-learn-COPYING`.
Numeric binning, best-first Newton splits, missing routing, monotonic bounds,
loss derivatives, baseline/link functions, nonsmooth terminal refits and
training-loss stopping follow those contracts. Mergesort and histogram storage
are native Fortran implementations. The existing NumPy-derived legacy RNG
notices also cover its full uint32 seeding helper; no new runtime or external
implementation is linked or distributed. Original contributions retain MIT.


### Image patch extraction and reconstruction

`src/sklearn_image.f90` adapts the enumeration, with-replacement sampling,
overlap accumulation and reconstruction denominator in scikit-learn 1.6.1
`feature_extraction/image.py` (`PatchExtractor`, `extract_patches_2d`,
`reconstruct_from_patches_2d`, `_compute_n_patches`). Copyright The scikit-learn
developers; BSD-3-Clause terms are retained in `LICENSES/scikit-learn-COPYING`.
The existing NumPy-derived RandomState helper supplies sampling. Original
Fortran storage, overloads and quiet IEEE arithmetic retain the project MIT
license. No new external implementation or runtime dependency is linked.


### Neighborhood components analysis and L-BFGS-B

`src/sklearn_nca.f90` adapts scikit-learn 1.6.1 `neighbors/_nca.py`, its
soft leave-one-out same-class objective and analytic gradient, initialization,
warm starts and optimizer/callback contracts. LDA initialization follows the
SVD scaling in `discriminant_analysis.py`; PCA uses the existing native Jacobi
SVD. Copyright The scikit-learn developers, BSD-3-Clause; terms are retained
in `LICENSES/scikit-learn-COPYING`. Original Fortran adapters retain MIT.

`vendor/lbfgsb` contains unmodified numerical source, License.txt and fpm.toml
from Jacob Williams's lbfgsb 1.0.0, commit
`bcb0ddc71ae1afdf5715ededaf9453aa704c2384`:
https://github.com/jacobwilliams/lbfgsb/tree/1.0.0 . It is the modern Fortran
refactoring of L-BFGS-B 3.0 by J. Nocedal and J. L. Morales, originally Ciyou Zhu,
R. H. Byrd, P. Lu-Chen and J. Nocedal. Upstream identifies copyright Regents of
the University of California and licenses it under BSD-3-Clause. Its complete
upstream License.txt, authorship headers and manifest are retained in place;
PROVENANCE.json records normalized SHA256 hashes. All four source files and
the license/manifest were compared against the upstream tag, differing only
in Windows line endings. The files were sourced from SciFort's dependency
cache; no SciFort optimizer adapter is copied. The numerical kernel is an
internal implementation dependency of a core sklearn algorithm, rather than
an additional public ML extension. No Python, SciPy or external BLAS binary is linked.


### DictVectorizer, FeatureHasher and MurmurHash3

`src/sklearn_feature_extraction.f90` adapts scikit-learn 1.6.1
`feature_extraction/_dict_vectorizer.py`, `_hash.py` and `_hashing_fast.pyx`:
numeric/categorical and iterable-string expansion, vocabulary ordering,
unknown-name handling, restriction, inverse mapping, signed hash columns and
sparse collision accumulation. Copyright The scikit-learn developers;
BSD-3-Clause terms are retained in `LICENSES/scikit-learn-COPYING`.
Native ordered records, hash-table storage and stable mergesort are original
project contributions under MIT; adapted numerical reductions retain their
upstream attribution below. No SciPy sparse
implementation is copied or linked.

`src/sklearn_murmurhash.f90` is an original Fortran implementation of Austin
Appleby's public-domain MurmurHash3 x86_32 algorithm, as used by sklearn's
`utils/murmurhash.pyx` wrapper. Attribution and the public-domain provenance are
stated in that installed wrapper. The native implementation uses explicit
little-endian byte assembly and bounded int64 products to implement uint32
modular arithmetic. Signed output and UTF-8/raw-byte behavior are checked
against the installed sklearn helper. No external hash library, C++ source or
binary is distributed or linked by this increment. Original Fortran code
retains MIT; sklearn-derived contracts retain their BSD notices.

Sparse inverse duplicate reduction follows SciPy 1.15.3
`sparse/_compressed.py::_get_intXint` and NumPy 2.2.6
`numpy/_core/src/umath/loops_utils.h.src` contiguous binary64 pairwise summation.
The native quiet-IEEE reduction is reused from this project's attributed
KernelCenterer port. SciPy and NumPy BSD terms are retained in
`LICENSES/scipy-COPYING` and `LICENSES/numpy-COPYING`. The NumPy upstream source
SHA256 was reverified as ed85d5b99f0b457cb7a316bf8c4acb884034bbbb81a89c5cedb62be80bf7c4f8.

### FunctionTransformer and TransformedTargetRegressor

`src/sklearn_function_transformer.f90` and `src/sklearn_compose.f90` adapt the
contracts and fitting/check order of scikit-learn 1.6.1
`preprocessing/_function_transformer.py` and `compose/_target.py`, including
feature metadata, cloned transformers, target reshaping and sampled inverse
checks. Copyright The scikit-learn developers; BSD-3-Clause terms remain in
`LICENSES/scikit-learn-COPYING`. Dense inverse tolerance follows sklearn's
`utils/validation.py::_allclose_dense_sparse`; wrapper checks follow NumPy
allclose defaults. Native callback interfaces, allocatable storage adapters
and static-rank prediction methods are original MIT contributions. Existing
weighted SVD, quiet arithmetic and source-conditioned CART validation helpers
retain their earlier attribution. No additional dependency is introduced.

Reference API documentation:
https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.FunctionTransformer.html
https://scikit-learn.org/1.6/modules/generated/sklearn.compose.TransformedTargetRegressor.html

### ColumnTransformer and FeatureUnion

`src/sklearn_feature_composition.f90` adapts dense orchestration contracts from
scikit-learn 1.6.1 `compose/_column_transformer.py` and `pipeline.py`: branch
validation, cloned column transforms, fitted union lists, dropped/empty branches,
remainder selection, array column indexing, weighted concatenation,
feature names and output slices. Copyright The scikit-learn developers;
BSD-3-Clause terms remain in `LICENSES/scikit-learn-COPYING`. Native polymorphic
storage, recursive dispatch, explicit named matrix selection and shortest
round-trip numeric name formatting are original MIT contributions. Existing
child model implementations retain their earlier notices. No additional
source dependency or external Fortran repository is imported by this increment.

Reference APIs:
https://scikit-learn.org/1.6/modules/generated/sklearn.compose.ColumnTransformer.html
https://scikit-learn.org/1.6/modules/generated/sklearn.pipeline.FeatureUnion.html

### Sequential Pipeline and recursive transformer protocol

`src/sklearn_pipeline.f90` adapts scikit-learn 1.6.1 `pipeline.py` contracts for
step validation, intermediate fit_transform ordering, final fit/transform,
method delegation, scoring, inverse order, feature names and slicing.
Copyright The scikit-learn developers; BSD-3-Clause terms are retained in
`LICENSES/scikit-learn-COPYING`. Native typed prediction/target adapters,
allocatable step bookkeeping and the abstract dense_transformer bridge in
`src/sklearn_transformer_protocol.f90` are original MIT contributions. Existing
child implementations and weighted SVD retain their earlier notices. There is
no new external dependency or imported Fortran repository in this increment.

The validation-only KNN certificate uses sklearn neighbor APIs and precomputed
metric aggregation to distinguish admissible distance ties/cancellation from
pipeline errors. It checks actual fitted training arrays, target order,
neighbor geometry, measured feature perturbations and explicit floating-point
bounds. It does not copy/link a new neighbor implementation into the library.
Reference API: https://scikit-learn.org/1.6/modules/generated/sklearn.pipeline.Pipeline.html


### Model-based feature selection (SelectFromModel)

`src/sklearn_model_selection_features.f90` adapts scikit-learn 1.6.1
`feature_selection/_from_model.py` and `_base.py` contracts for coefficient
norms, automatic/scaled thresholds, stable caps, prefit copying, incremental
fitting, support masks, feature names and transforms. Copyright The scikit-learn
developers; BSD-3-Clause terms remain in `LICENSES/scikit-learn-COPYING`.
Native typed dispatch, callback interfaces and clone bookkeeping are original
MIT contributions. Existing child models retain their earlier notices.

The validation-only tree certificate independently reconstructs weighted CART
impurities, optimal split losses and feature contributions to handle the
previously documented native/source tie traversal difference; sklearn remains
the feature-selection oracle on these certified importances. No external
repository, dependency or extension algorithm is imported in this increment.
Reference API: https://scikit-learn.org/1.6/modules/generated/sklearn.feature_selection.SelectFromModel.html


### General parameter grids and searches

`src/sklearn_parameter_space.f90` and `src/sklearn_search.f90` adapt scikit-learn
1.6.1 `model_selection/_search.py`, `_validation.py`, `_split.py` and
`utils/_random.pyx` contracts for Cartesian grids, parameter sampling, default
folds, split/aggregate/rank tables, cloning/refitting and delegation. Copyright
The scikit-learn developers; BSD-3-Clause terms remain in
`LICENSES/scikit-learn-COPYING`. The existing NumPy legacy RNG retains its earlier
notices. Typed parameter/callback storage, mixed-radix indexing and estimator
adapters (`src/sklearn_search_adapters.f90`) are original MIT contributions.
No external repository/dependency or public extension algorithm is imported.

The validation-only rank certificate calls SciPy's public rankdata API on
native mean scores and bounds source/native ordering changes due to roundoff;
it does not copy/link a SciPy ranking implementation into the Fortran library.
Reference APIs:
https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html
https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.RandomizedSearchCV.html


### General cross-validation and out-of-fold prediction

`src/sklearn_validation.f90` adapts scikit-learn 1.6.1
`model_selection/_validation.py` and `_split.py` contracts for fold fitting,
scoring, result artifacts, class encoding, partition checks, response alignment
and inverse prediction ordering. Copyright The scikit-learn developers;
BSD-3-Clause terms remain in `LICENSES/scikit-learn-COPYING`. Typed options,
result/model slots and callback adapters are original MIT contributions.
Existing native estimator/search/composition implementations retain their earlier
notices. No external repository, dependency or public extension algorithm is
imported by this increment.
Reference APIs:
https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.cross_validate.html
https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.cross_val_predict.html
https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.cross_val_score.html

Learning curves, validation curves and permutation-test significance in
`src/sklearn_validation.f90` additionally adapt sklearn 1.6.1
`model_selection/_validation.py` and `_split.py`: prefix sizing, clone/update
order, incremental fit parameters, fold scheduling, grouped target permutations,
legacy RNG consumption and corrected empirical p-values. These portions retain
the sklearn BSD-3-Clause notice above and in LICENSES/scikit-learn-COPYING.
Typed result/options containers, native adapters and tests are original MIT
contributions. No external repository was imported for this increment.

`src/sklearn_sample_generators.f90`, re-exported by `sklearn_datasets`, adapts
sklearn 1.6.1 `datasets/_samples_generator.py`: make_friedman1/2/3,
make_sparse_uncorrelated, make_s_curve, make_swiss_roll (including hole sampling),
and make_gaussian_quantiles. Feature/target formulas, discrete sample ordering,
Gaussian draw orientation, radial quantile labels and seeded shuffles follow that
BSD-3-Clause source. The sklearn notice and LICENSES/scikit-learn-COPYING remain
applicable. Native interfaces, checks and tests are original MIT contributions.
These generators reuse the existing separately attributed NumPy-compatible RNG;
no additional external code or dependency was imported.

`src/sklearn_classification_datasets.f90`, re-exported by `sklearn_datasets`,
adapts sklearn 1.6.1 `datasets/_samples_generator.py` make_classification,
_generate_hypercube and make_hastie_10_2: weighted cluster allocation, binary
vertices, random covariance and feature blocks, flip masks, affine transforms,
seeded permutations and the Hastie radial boundary. Existing
`sklearn_bagging_sampling` follows `utils/_random.pyx` sample_without_replacement;
its pool allocation is now deferred to the permutation branch. Upstream
BSD-3-Clause terms and LICENSES/scikit-learn-COPYING remain applicable. Typed
options, native interfaces and diagnostics are original MIT contributions.
The existing separately attributed legacy NumPy RNG is reused unchanged.

## Gaussian-process kernel expressions and SciFort Bessel support

`src/sklearn_gp_kernels.f90` adapts sklearn 1.6.1 gaussian_process/kernels.py:
RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel,
WhiteKernel, Sum, Product and Exponentiation covariance/diagonal/log-gradient
contracts and hyperparameter ordering. GP integration adapts _gpr.py and _gpc.py
fixed-kernel training, prior/posterior uncertainty and fitted kernel cloning.
The existing sklearn BSD-3-Clause terms and LICENSES/scikit-learn-COPYING apply.
Flat typed expressions, native interfaces and diagnostics are original MIT work.

`src/sklearn_bessel_k.f90` adapts the MIT SciFort helper
`c:/fortran/scifort/src/special/bessel_k.f90`, inspected 2026-11-22, SHA256
b03ec89063b912c46e2d0899d79f10eb0375b81cc978ffb2808fc82798d0a53d.
Copyright (c) 2026 SciFort contributors. Its MIT header is retained and the full
license is retained in LICENSES/scifort-MIT.txt. Changes rename the module, use
this package's real64 kind and intrinsic IEEE value helpers, and keep only
log_besselk public for internal kernel support. SciFort's defining-integral,
log-scaling, Gauss-Legendre quadrature and large-argument asymptotic routines
supply the general Matern calculation. No SciFort runtime library is linked.
The general Matern log-length derivative uses the standard K-order recurrence;
independent sklearn central differences and SciPy Bessel values validate it.
This numerical support is excluded from the public extension inventory.

The GP regressor further adapts sklearn 1.6.1 gaussian_process/_gpr.py matrix
outputs, scalar/vector alpha, target normalization, joint log likelihood and
trace gradients, non-PD likelihood fallback, cloning semantics and bounded
optimizer/restart scheduling. `sklearn_gp_kernels` adds ordered log-bound storage
and composition. Scalar reduction order follows NumPy 2.2.6
_core/src/umath/loops_utils.h.src, also used by the existing mutual-info backend;
LICENSES/numpy-COPYING and the existing NumPy notices apply. The previously
vendored L-BFGS-B 3.0 solver is reused unchanged under its retained notices.
Native multi-rank interfaces, diagnostic storage and optimizer controls are
original MIT contributions. No additional third-party code or runtime dependency
is imported for this increment.

`src/sklearn_gp_classification.f90` further adapts sklearn 1.6.1
`gaussian_process/_gpc.py`: GPML Algorithm 5.1 likelihood derivatives, binary
posterior-mode caching, bounded optimizer/restart scheduling (including the
pinned classifier's exponentiated restart-theta rule), cloned/mutated evaluation
and independent OVR head kernels. `multiclass.py` supplies the head ordering and
mean likelihood contract. NumPy 2.2.6 pairwise reduction semantics reuse the
existing translated helper under LICENSES/numpy-COPYING; the standard runtime
log1p function is called through ISO_C_BINDING. Existing sklearn BSD and vendor
L-BFGS-B notices remain applicable. Typed per-head storage, cache diagnostics and
native interfaces are original MIT contributions. No additional source package
or third-party runtime dependency is imported.


The one-vs-one path in `src/sklearn_gp_classification.f90` adapts scikit-learn
1.6.1 `gaussian_process/_gpc.py` and `multiclass.py`: lexicographic class pairs,
row subset selection, independent binary fits, mean likelihood and the bounded
confidence voting correction. The existing BSD-3-Clause attribution in
LICENSES/scikit-learn-COPYING applies. Native padded subset maps, state layout,
validation drivers and interfaces are original MIT contributions. No additional
third-party source package or dependency is imported.


`src/sklearn_gaussian_process.f90` adds the `sample_y` contract from scikit-learn
1.6.1 `gaussian_process/_gpr.py`: prior/posterior moments, independent target
sampling and target-major RNG scheduling. The covariance-factor sampling rule
follows NumPy 2.2.6 RandomState's multivariate-normal SVD contract. Existing
LICENSES/scikit-learn-COPYING, LICENSES/numpy-COPYING and NumPy RNG notices apply.
The native scaled Jacobi factorization, explicit-rank Fortran interfaces and
validation certificates are original MIT contributions. No additional source
package, dependency or runtime library is imported.


`src/sklearn_factor_analysis.f90` extends the existing adaptation of scikit-learn
1.6.1 `decomposition/_factor_analysis.py` with randomized-SVD fitting and the
`_ortho_rotation` varimax/quartimax iteration. `src/sklearn_randomized_svd_core.f90`
adds explicit power counts and the automatic no-normalizer/LU rule from
`utils/extmath.py`, preserving its previous default path. Existing sklearn BSD
notices and LICENSES/scikit-learn-COPYING apply. Native null-space completion,
diagnostic storage and Fortran interfaces are original MIT contributions. The
existing NumPy-compatible RNG is reused under its retained notices. No additional
source package, runtime or dependency is imported.


`src/sklearn_decomposition.f90` further adapts scikit-learn 1.6.1
`decomposition/_pca.py` (`_assess_dimension`, `_infer_dimension`, component selection)
and `decomposition/_base.py` (probabilistic covariance and precision contracts),
plus PCA sample likelihoods. Minka's automatic PCA dimensionality method is
credited in the source and sklearn's original documentation. Existing
LICENSES/scikit-learn-COPYING and BSD notices apply. Typed dimension-mode
selection, fitted diagnostic storage, Cholesky scoring interfaces and validation
certificates are original MIT contributions. No additional source package,
third-party runtime or dependency is imported.


`src/sklearn_decomposition.f90` adds solver routing, uncentered-Gram covariance
PCA and randomized training-coordinate contracts from scikit-learn 1.6.1
`decomposition/_pca.py`. `src/sklearn_randomized_svd_core.f90` extends its existing
adaptation of `utils/extmath.py` with oversampling, power normalizers, sign control
and output rank truncation. Existing sklearn BSD and NumPy RNG notices apply;
LICENSES/scikit-learn-COPYING remains the upstream license. Native orthogonal
null completion, Fortran controls and diagnostic storage are original MIT
contributions. No additional source package, runtime or dependency is imported.


`src/sklearn_decomposition.f90` extends TruncatedSVD using the contracts in
scikit-learn 1.6.1 `decomposition/_truncated_svd.py`: randomized factorization,
right-vector sign selection, actual training projections and population variance
ratios. `src/sklearn_randomized_svd_core.f90` generalizes its existing adaptation
of `utils/extmath.py` to dense/CSR products. Existing sklearn BSD and NumPy RNG
notices apply; LICENSES/scikit-learn-COPYING is retained. Native CSR transpose
products, duplicate-aware variance, generic Fortran interfaces and diagnostic
storage are original MIT contributions. No external source package, runtime or
dependency is imported.


`src/sklearn_kernel_pca.f90` adds learned pre-images following scikit-learn
1.6.1 `decomposition/_kernel_pca.py` `_fit_inverse_transform` and
`inverse_transform`: the uncentered latent kernel, alpha-regularized positive
definite solve, stored training coordinates and kernel reconstruction. Existing
sklearn BSD attribution and LICENSES/scikit-learn-COPYING remain retained.
Native Fortran bindings, input contracts, Cholesky calls and refit-state clearing
are original MIT contributions. No new external code or dependency is imported.


`src/sklearn_discriminant_analysis.f90` adapts scikit-learn 1.6.1
`discriminant_analysis.py` `LinearDiscriminantAnalysis._solve_svd` and
`transform`: within-class standardization, absolute/relative rank thresholds,
weighted between-class SVD, discriminant coefficients and projection.
The existing sklearn BSD notice and LICENSES/scikit-learn-COPYING are retained.
Native Fortran interfaces, thin-SVD calls, rank diagnostics, optional covariance
storage and refit clearing are original MIT contributions. No new external
source package or dependency is imported.


`src/sklearn_discriminant_analysis.f90` adds the generalized eigen formulation
from scikit-learn 1.6.1 `discriminant_analysis.py`
`LinearDiscriminantAnalysis._solve_eigen`: within/total covariance shrinkage,
generalized eigenvector normalization, full-basis classifier coefficients and
uncentered projection. Existing sklearn BSD notices and
LICENSES/scikit-learn-COPYING are retained. Native Fortran bindings, Cholesky
whitening, eigen diagnostics, refit-state clearing and numerical certificates
are original MIT contributions. No external source package or dependency is
imported.


`src/sklearn_discriminant_analysis.f90` adds scikit-learn 1.6.1
`discriminant_analysis.py` `_cov` / `_class_cov` automatic shrinkage semantics:
classwise standardization, Ledoit-Wolf estimation and covariance rescaling.
Corrected variance and constant-feature bounds follow `utils/extmath.py`
`_incremental_mean_and_var` and `preprocessing/_data.py` `_is_constant_feature`.
The existing native Ledoit-Wolf estimator and retained sklearn BSD notice in
LICENSES/scikit-learn-COPYING apply. Fortran mode/diagnostic interfaces, centered
margin evaluation and numerical certificates are original MIT contributions.
No additional source package or external Fortran runtime is imported.


`src/sklearn_discriminant_analysis.f90` adds class-prior handling from
scikit-learn 1.6.1 `discriminant_analysis.py` LDA/QDA `fit`: LDA prior
renormalization, QDA retained prior weights, weighted covariance/geometry and
log-prior intercepts. Its wide-QDA path follows the source's thin singular-value
regularization and projected quadratic scoring. Existing sklearn BSD notice
and LICENSES/scikit-learn-COPYING remain retained. Native input/diagnostic
interfaces, safe zero-prior logarithms, positive-reference margins and orthogonal
null completion are original MIT contributions. No external source dependency
is imported.


`src/sklearn_naive_bayes.f90` extends discrete NB following scikit-learn 1.6.1
`naive_bayes.py` `_BaseDiscreteNB.partial_fit`, class-prior updates and
MultinomialNB/BernoulliNB/ComplementNB count and smoothed-feature updates.
Existing sklearn BSD attribution and LICENSES/scikit-learn-COPYING are retained.
Native Fortran streaming/weight interfaces, fitted diagnostics, safe absent-class
and undefined-value handling and input contracts are original MIT contributions.
No external source package or runtime dependency is imported.


CategoricalNB zero smoothing, force_alpha flooring, unnormalized class priors,
zero-mass behavior and latest-batch category metadata follow scikit-learn 1.6.1
`naive_bayes.py` CategoricalNB and _BaseDiscreteNB. Existing sklearn BSD notices
and LICENSES/scikit-learn-COPYING remain retained. Native safe zero logarithms,
explicit undefined-value handling, range contracts and validation identities
are original MIT contributions. No external source package is imported.


NearestCentroid priors, standardized discriminants, softmax probabilities and
uniform-distance prediction routing follow scikit-learn 1.6.1
`neighbors/_nearest_centroid.py` and DiscriminantAnalysisPredictionMixin in
`discriminant_analysis.py`. Existing sklearn BSD notices and
LICENSES/scikit-learn-COPYING remain retained. Native explicit undefined-value
handling, range checks, interfaces and validation certificates are original
MIT contributions. No external source package is imported.


Discrete NB nonnegative/feature-wise smoothing, force_alpha and Complement
addition-before-subtraction order follow sklearn 1.6.1 `naive_bayes.py`
_BaseDiscreteNB, MultinomialNB, BernoulliNB and ComplementNB. Existing sklearn
BSD notices and LICENSES/scikit-learn-COPYING remain retained. Native safe
IEEE likelihood evaluation, explicit undefined-value handling, range contracts,
interfaces and independent validation identities are original MIT contributions.
No external source package is imported.


Discrete NB CSR fitting, sparse likelihoods and Bernoulli per-entry thresholding
follow sklearn 1.6.1 `naive_bayes.py` and `preprocessing/_data.py` binarize.
Existing sklearn BSD notices and LICENSES/scikit-learn-COPYING remain retained.
Native CSR overloads, sparse event accumulation, quiet IEEE integration,
structure/range contracts and validation identities are original MIT
contributions. Existing native CSR/IEEE helpers are reused; no external source
package or Fortran dependency is imported.

Coordinate-descent NMF adapts sklearn 1.6.1 `decomposition/_nmf.py` and
`decomposition/_cdnmf_fast.pyx`, retaining the scikit-learn BSD notices above and
in `LICENSES/scikit-learn-COPYING`. Local reference SHA256 values:

- `_nmf.py`: `6f0ed846e1531f773eb82de20dfa4f07e57f53fa472f3f8c576e8ee3d7007369`
- `_cdnmf_fast.pyx`: `bb378d434b38e28c07c04f0cc9afbf5d996a3ceedbd5f56322dbda36fbd9b4fc`

The adapter reuses the existing attributed MiniBatchNMF NNDSVD initializer and
NumPy-compatible legacy RNG; no upstream source package is imported. Native
Fortran interfaces, working-range diagnostics and independent reconstruction
and objective validation are original MIT contributions. No additional runtime
dependency or public algorithm extension is introduced.


Positive linear regression uses the sklearn 1.6.1 `_base.py`, `_ridge.py` and
`_coordinate_descent.py` contracts. The NNLS backend adapts SciPy 1.15.3's
Lawson-Hanson QR active set and LAPACK 3.12.0 positive-diagonal reflector/rotation
conventions. SciPy BSD terms are retained in `LICENSES/scipy-COPYING`; LAPACK
BSD terms and university copyright notices are retained in
`LICENSES/lapack-LICENSE`. The native implementation corrects the stale final
QR-tail entry observed in SciPy 1.15.3. It does not import a SciPy/LAPACK binary
or require a new runtime dependency. Original Fortran adapters, checked ranges,
certificates and tests remain MIT contributions.

Pinned primary reference files (copies archived in ignored validation results):

- [SciPy 1.15.3 `_cython_nnls.pyx`](https://github.com/scipy/scipy/blob/v1.15.3/scipy/optimize/_cython_nnls.pyx): `eaa84be52696f2ba5b87295ee56895b7e89bb31e0e83378c43a628de7d1a47cd`
- [LAPACK 3.12.0 `dlarfgp.f`](https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/SRC/dlarfgp.f): `92be2c8b8483629c586b0ca43438299c64a7c5f000e7b843412a75c5df30f16c`
- [LAPACK 3.12.0 `dlartgp.f`](https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/SRC/dlartgp.f): `883db0660892cc772839e624243194f73ec8991858e528016071744a0a350dec`

The dense free `non_negative_factorization` adapter follows sklearn 1.6.1
`decomposition/_nmf.py`, including automatic rank inference and fixed-H
initialization. It reuses the attributed CD and MU kernels. Original Fortran
interfaces, validation and tests are MIT contributions; retained sklearn BSD
terms continue to apply to the translated algorithm contracts.

### HashingVectorizer and text analyzers

`src/sklearn_hashing_text.f90` translates the scikit-learn 1.6.1
`feature_extraction/text.py` HashingVectorizer contract and its word/character/
character-within-word analyzer algorithms. The 318-entry English stop-word table
is copied from that release's `feature_extraction/_stop_words.py`. Copyright
The scikit-learn developers; retained BSD-3-Clause terms are in
`LICENSES/scikit-learn-COPYING`. Source SHA256 digests are pinned in the reference
reports. Hashing reuses the already attributed native MurmurHash3 and
FeatureHasher implementation. No additional upstream repository or runtime
library is imported. Original Fortran interfaces, checked storage and tests
remain MIT contributions. ASCII analysis is an explicit native limitation.
The sklearn stop-list source credits the Glasgow Information Retrieval Group
for the original English list, with source location
`http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words`.

### Normalizer, Binarizer and sparse preprocessing

`src/sklearn_stateless_preprocessing.f90` adapts the scikit-learn 1.6.1
`preprocessing/_data.py` Normalizer/Binarizer/free-function contracts,
`utils/sparsefuncs.py`, `utils/sparsefuncs_fast.pyx` and `utils/extmath.py` norm
semantics. Copyright The scikit-learn developers; BSD-3-Clause terms remain in
`LICENSES/scikit-learn-COPYING`. Sparse max canonicalization follows SciPy 1.15.3
`sparse/_compressed.py` and `sparse/_data.py`; retained SciPy BSD terms are in
`LICENSES/scipy-COPYING`. Reference reports pin these source digests and versions.
The native stable sorting, owned storage, feature-name/protocol adapters and
quiet-arithmetic integration are original MIT contributions. Shared NB dispatch
now forwards the existing underlying sample-weight argument. No C++ sorting
implementation, new numerical binary or external runtime is imported.

### PolynomialFeatures

`src/sklearn_polynomial.f90` adapts scikit-learn 1.6.1
`preprocessing/_polynomial.py` and `_csr_polynomial_expansion.pyx`: feature
ordering, degree controls, dense multiplication association, sparse storage
behavior and metadata contracts. Copyright The scikit-learn developers; retained
BSD-3-Clause terms are in `LICENSES/scikit-learn-COPYING`. The higher-degree
sparse multiplication behavior follows SciPy 1.15.3 `sparse/_compressed.py`;
retained SciPy BSD terms remain in `LICENSES/scipy-COPYING`. Reference reports
pin the source files/versions. The native implementation corrects noncanonical
CSR duplicate expansion and empty-output metadata/stacking defects observed in
sklearn 1.6.1; raw source outcomes remain in validation reports. Exact checked
counting, streaming buffers, column ranking, lazy metadata and protocol adapters
are original MIT contributions. No additional runtime library is imported.

### Image/grid graphs and quiet subtraction

`src/sklearn_image_graph.f90` adapts scikit-learn 1.6.1
`feature_extraction/image.py` (`_make_edges_3d`, `_compute_gradient_3d`, mask
compaction, `grid_to_graph`, `img_to_graph`). Copyright The scikit-learn
developers; retained BSD-3-Clause terms are in `LICENSES/scikit-learn-COPYING`.
The row-sorted, owned native CSR builder, range checks and shape adapters are
original MIT contributions. The implementation corrects the source's unsqueezed
singleton-axis mask selection; source exceptions remain in reference reports.
The shared quiet-subtraction boundary correction is original MIT arithmetic,
verified independently against NumPy float64 subtraction; the existing NumPy
BSD notice remains retained. No new external runtime code or dependency is added.


## SGD truncated gradients, schedules and shuffling (BSD)

`src/sklearn_sgd.f90` additionally translates L1/elastic-net cumulative
truncation (`l1penalty`), optimal-rate initialization, adaptive-rate reduction
and WeightVector64 lazy averaging from the sklearn 1.6.1 sources identified
above. L1 follows the method credited upstream to Y. Tsuruoka, J. Tsujii and
S. Ananiadou (2009). The update order preserves the source's lazy averaging
before L1 truncation, including its non-arithmetic-average behavior.
Scalar seed draws and per-class OVR streams follow `_stochastic_gradient.py`;
forward Fisher-Yates and unsigned xorshift follow the following sources:

- `utils/_seq_dataset.pyx.tp`: `90a4faefb8dbb83c55266db5f0710366b01a54d902dfa89ab303a2a089ca2d5f`
- `utils/_random.pxd`: `d25e4ddf7935cbf0a6b3e41a387d76c96ca64fffb6148ee7ab5a1b082a5e287d`

The existing attributed NumPy-compatible MT19937 backend is reused. No
dependency was added; the project remains MIT with retained BSD notices.


## SGD warm fits, initialization and additional losses (BSD)

`src/sklearn_sgd.f90` further adapts sklearn 1.6.1
`linear_model/_stochastic_gradient.py`: BaseSGD parameter allocation,
SGDRegressor/SGDClassifier repeated-fit initialization, warm-start distinctions,
clock resets, binary initializer flattening, all nine classifier loss mappings
and `predict_log_proba`. Regression warm fits preserve the source's existing
standard/average buffers and ignore newly supplied initializers once fitted;
classifier warm fits allocate from the published coefficients, with explicit
initializers taking precedence. Huber and epsilon-insensitive classifier losses
reuse the already attributed `_sgd_fast.pyx.tp` loss formulas. The source hashes
are unchanged from the retained SGD notices above. Original contributions
remain MIT, with complete sklearn BSD terms in `LICENSES/scikit-learn-COPYING`.


`src/sklearn_sgd_clone.f90` is an original MIT implementation of SGD parameter-only
copying, following the `sklearn.base.clone` contract. Clone sites in native
multioutput, multiclass, voting/bagging/stacking, thresholding, feature selection,
validation and search reuse it; prefit branches preserve trained state. The
parameter adapters and recursive pipeline traversal are original typed Fortran
code. No additional third-party code or dependency was imported.


## CSR stochastic gradient descent (BSD)

The CSR paths in `src/sklearn_sgd.f90` further adapt sklearn 1.6.1
`linear_model/_base.py` (`make_dataset` and SPARSE_INTERCEPT_DECAY),
`utils/_seq_dataset.pyx.tp` (CSRDataset stored-row traversal),
`linear_model/_sgd_fast.pyx.tp` and `utils/_weight_vector.pyx.tp`
(stored-index updates, L1 truncation, lazy scaling and averaging).
The existing recorded source hashes apply. Sparse intercepts use 0.01 scaling;
explicit zeros and duplicate indices follow the stored-entry source contract.
The native CSR storage and typed overloads are original MIT contributions.
No additional dependency or public extension estimator was introduced.


## Stochastic multilayer perceptrons (BSD)

`src/sklearn_neural_network.f90` adapts sklearn 1.6.1
`neural_network/_multilayer_perceptron.py` (Glorot initialization, minibatches,
backpropagation, epoch losses and stopping), `_stochastic_optimizers.py`
(Adam and momentum/Nesterov SGD, inverse/adaptive learning rates) and
`_base.py` (activation derivatives and clipped logistic losses).
The full upstream BSD-3-Clause notice is retained in
`LICENSES/scikit-learn-COPYING`. NumPy-compatible scalar draws and permutations
reuse the previously attributed `sklearn_legacy_rng` implementation.
The typed interfaces, diagnostics, examples and validation harness are original
MIT contributions. Validation records hashes of the installed source modules,
and separately identifies the measured one-ULP rounding-sensitivity probes.

The MLP logistic activation also follows scipy.special.expit's negative
exponential-overflow saturation boundary, using an explicit guard to remain
safe with floating-point traps enabled. This mathematical behavior was checked
against SciPy 1.15.3; its BSD notice is retained in LICENSES/scipy-COPYING.


## L-BFGS multilayer perceptrons (BSD)

The L-BFGS paths in `src/sklearn_neural_network.f90` adapt sklearn 1.6.1
`neural_network/_multilayer_perceptron.py` (_fit_lbfgs, _loss_grad_lbfgs,
_pack and _unpack) and the already attributed `_base.py` loss/gradient
contracts. The optimizer adapter follows SciPy 1.15.3
`optimize/_lbfgsb_py.py` defaults and work-limit behavior, using the existing
vendored L-BFGS-B dependency. Upstream notices remain in
LICENSES/scikit-learn-COPYING, LICENSES/scipy-COPYING and the vendor directory.
The typed packing, native state diagnostics, tests, example and validation
harness are original MIT contributions. Validation retains installed-source
and vendored-optimizer hashes and explicitly records differing long-fit
endpoints. No additional dependency was imported.


## Warm-start multilayer perceptrons (BSD)

The warm-fit paths in `src/sklearn_neural_network.f90` further adapt sklearn
1.6.1 `neural_network/_multilayer_perceptron.py` (_fit, _validate_input and
_fit_stochastic): parameter reuse, feature/class checks, optimizer recreation,
seed handling and retained loss/patience/sample state. Existing initialization,
Adam/SGD and L-BFGS notices apply, with full BSD terms retained in LICENSES.
Native shape adapters, 64-bit counters, diagnostics, examples and validation
are original MIT contributions. Validation records unchanged installed-source
hashes and explicitly identifies the isolated roundoff-propagation check.


## MLP matrix targets (BSD)

The matrix-target paths in `src/sklearn_neural_network.f90` further follow
sklearn 1.6.1 `neural_network/_multilayer_perceptron.py` and `_base.py`:
shared multiresponse networks, sample/output loss normalization, supplied
backpropagation Jacobians, multilabel logistic outputs and binary losses.
Indicator prediction follows LabelBinarizer's threshold convention; scoring
follows sklearn uniform-average R2 and exact-match multilabel accuracy.
The complete sklearn BSD terms remain in LICENSES/scikit-learn-COPYING.
Typed rank overloads, matrix-returning prediction adapters, diagnostics,
analytic tests, example and validation harness are original MIT contributions.
No new dependency or extension estimator was added.


## Incremental multilayer perceptrons (BSD)

The incremental MLP paths further adapt sklearn 1.6.1
`neural_network/_multilayer_perceptron.py` (_fit, _fit_stochastic, partial_fit
and _validate_input), `_stochastic_optimizers.py` (cached optimizer fields and
iteration updates) and `utils/multiclass.py` (_check_partial_fit_first_call).
Existing full sklearn/SciPy BSD notices apply. The typed rank adapters, native
optimizer diagnostics and snapshot storage, tests, example and validation
harness are original MIT contributions. Installed-source hashes are retained
with validation results. No new dependency or public extension estimator was
introduced.

## MLP validation stopping (BSD)

Validation stopping further adapts scikit-learn 1.6.1
`neural_network/_multilayer_perceptron.py` (holdout scoring, patience, checkpoint
copies/restoration and warm-fit alias behavior), `model_selection/_split.py`
(ShuffleSplit/StratifiedShuffleSplit partition ordering), and
`utils/extmath.py` (`_approximate_mode` class allocation). Complete sklearn BSD
terms remain in LICENSES/scikit-learn-COPYING. Existing NumPy RNG attribution
also applies. Typed split/checkpoint diagnostics, native NaN handling under
floating-point traps, tests, example and reference harness are original MIT
contributions. No dependency or public extension estimator was added.

## CSR multilayer perceptrons (BSD contracts; MIT kernels)

The CSR overloads in `src/sklearn_neural_network.f90` retain scikit-learn 1.6.1
`neural_network/_multilayer_perceptron.py` sparse-input contracts: first-layer
affine products, transposed feature-gradient products, minibatch/holdout row
selection, solver controls and dense parameter/optimizer state. Existing
sklearn BSD terms remain in LICENSES/scikit-learn-COPYING. Typed adapters,
one-based CSR traversal kernels, nonmutation/shape checks, wide-input native
test, example and reference harness are original MIT contributions. Validation
records sklearn and SciPy sparse source hashes. No dependency or public
extension estimator was added.

## Frozen estimator adapter (BSD contracts; MIT implementation)

`src/sklearn_frozen.f90` follows scikit-learn 1.6.1 `frozen/_frozen.py` for
no-op fit, fitted-state checks, conditional method delegation, clone identity
and estimator-only parameters. Pipeline and calibration integration follow
`pipeline.py`, `base.py` and the existing native adapters. Scikit-learn BSD
terms remain in `LICENSES/scikit-learn-COPYING`. The borrowed Fortran pointers,
typed bridge protocol, tests, example and reference harness are original MIT
contributions. No external repository or dependency was imported.

## Single-class MLP encodings (BSD)

Single-class MLP fitting and prediction further follow scikit-learn 1.6.1
`neural_network/_multilayer_perceptron.py`, `preprocessing/_label.py`
(LabelBinarizer's one-class zero encoding and inverse thresholding), and
`model_selection/_split.py` (one-class stratified holdout ordering). Existing
sklearn BSD terms remain in LICENSES/scikit-learn-COPYING. Native encoding
diagnostics, allocation-after-validation adapters, typed tests, example and
reference harness are original MIT contributions. No dependency or public
extension estimator was added.


## MeanShift grid seeds and bandwidth sampling (BSD)

The `get_bin_seeds` helper and MeanShift/estimate_bandwidth additions in
`src/sklearn_cluster.f90` follow scikit-learn 1.6.1 `cluster/_mean_shift.py`:
frequency-filtered insertion-ordered grid seeds, nearest-even rounding,
float32 seed storage/multiplication, singleton-bin fallback and legacy-RNG
sampling without replacement. Scikit-learn BSD terms remain in
`LICENSES/scikit-learn-COPYING`; the existing NumPy RNG notices continue to
apply. The native hash table, diagnostics, tests, example and reference harness
are original MIT contributions. No dependency or extension algorithm was added.


## Low-rank matrices and regression samples (BSD contracts; MIT QR)

`src/sklearn_regression_datasets.f90` follows scikit-learn 1.6.1
`datasets/_samples_generator.py` make_low_rank_matrix and make_regression:
Gaussian economic-QR bases, singular profiles, informative coefficient draws,
conditional noise, target ranks and sample/feature permutations. Existing sklearn
BSD terms remain in `LICENSES/scikit-learn-COPYING`; existing NumPy RNG notices
continue to apply. The independent native Householder implementation follows
LAPACK's zero-tail reflector convention; no SciPy/LAPACK routine is linked or
copied. Typed adapters, quiet IEEE noise composition, tests, example and reference
harness are original MIT contributions. No dependency or extension was added.


## Seeded blob and geometric generators (BSD)

The make_blobs, make_moons and make_circles adapters in `src/sklearn_datasets.f90`
follow scikit-learn 1.6.1 `datasets/_samples_generator.py`: center/sample/scale
shapes, legacy NumPy draw order, remainder allocation, linspace endpoints,
shuffle/noise ordering and optional center output. Existing sklearn BSD terms
remain in `LICENSES/scikit-learn-COPYING`; existing NumPy RNG notices continue
to apply. Native rank adapters, count guards, quiet IEEE composition, tests,
example and reference harness are original MIT contributions. No dependency or
extension algorithm was added.


## Multilabel sample generation and Poisson draws (BSD)

`src/sklearn_multilabel_datasets.f90` follows scikit-learn 1.6.1
`datasets/_samples_generator.py` make_multilabel_classification: class/feature
probabilities, bounded label and positive length rejection, weighted distinct
class selection, unlabeled noise words, bag-of-words counts and indicators.
Sklearn BSD terms remain in `LICENSES/scikit-learn-COPYING`.

The private multiplicative/PTRS Poisson samplers and log-gamma approximation
adapt NumPy 2.2.6
[random/src/distributions/distributions.c](https://github.com/numpy/numpy/blob/v2.2.6/numpy/random/src/distributions/distributions.c).
Existing terms in `LICENSES/numpy-DISTRIBUTIONS-LICENSE.md`,
`LICENSES/numpy-RANDOM-LICENSE.md` and `LICENSES/numpy-COPYING` apply. The PTRS
method is attributed there to W. Hoermann, Insurance: Mathematics and Economics
12, 39-45 (1993). The implementation uses the existing legacy NumPy RNG and
reduction conventions. Native CSR growth, output overloads, canonical label sets,
diagnostics, tests, example and reference harness are original MIT contributions.
No dependency or public extension estimator was added.

## Bicluster generators and binomial draws (BSD)

`src/sklearn_bicluster_datasets.f90` follows scikit-learn 1.6.1
`datasets/_samples_generator.py` make_biclusters and make_checkerboard.
Sklearn BSD terms remain in `LICENSES/scikit-learn-COPYING`.
The private binomial inversion/BTPE and multinomial partition sampling adapt
[NumPy 2.2.6 distributions.c](https://github.com/numpy/numpy/blob/v2.2.6/numpy/random/src/distributions/distributions.c)
and its legacy random conventions. Existing terms in
`LICENSES/numpy-DISTRIBUTIONS-LICENSE.md`, `LICENSES/numpy-RANDOM-LICENSE.md`
and `LICENSES/numpy-COPYING` apply. Native interfaces, guards, logical masks,
tests, example and reference harness are original MIT contributions.
No dependency or extension algorithm was added.

## Dictionary signals and sparse SPD generation (BSD)

`src/sklearn_structured_datasets.f90` follows sklearn 1.6.1
`datasets/_samples_generator.py` make_sparse_coded_signal and
make_sparse_spd_matrix, including source draw/transposition ordering. It follows
SciPy 1.15.3 `sparse/_construct.py` _random for population selection,
round-half-to-even counts and Fortran-order sparse coordinates, and the existing
NumPy RandomState permutation/uniform/normal contracts. Terms remain in
`LICENSES/scikit-learn-COPYING`, `LICENSES/scipy-COPYING` and the retained
`LICENSES/numpy-*` files. Native interfaces, guards, dense/CSR adapters, tests,
example and reference harness are original MIT contributions. No new dependency
or public extension was introduced.

## Multilabel ranking metrics (BSD)

`src/sklearn_multilabel_ranking.f90` adapts sklearn 1.6.1
`metrics/_ranking.py` coverage_error, label_ranking_average_precision_score and
label_ranking_loss, including maximal-rank ties, degenerate-row conventions
and signed weighted averages. Terms remain in
`LICENSES/scikit-learn-COPYING`. Native grouped merge sorting, Fortran
interfaces, canonical CSR guards, quiet IEEE integration, tests, example and
reference harness are original MIT contributions. No dependency or public
extension algorithm was added.

## Quantile and deviance regression measures (BSD)

`src/sklearn_regression_metrics.f90` adapts sklearn 1.6.1
`metrics/_regression.py` mean_pinball_loss, mean_tweedie_deviance,
d2_tweedie_score, d2_pinball_score, d2_absolute_error_score and
root_mean_squared_log_error, plus `utils/stats.py` _weighted_percentile.
Sklearn terms remain in `LICENSES/scikit-learn-COPYING`. Linear percentile
interpolation follows NumPy 2.2.6 `_function_base_impl.py` _lerp conventions;
retained NumPy terms remain in `LICENSES/numpy-COPYING`. Native merge sorting,
rank/aggregation adapters, quiet IEEE helpers, diagnostics, tests, example and
reference harness are original MIT contributions. No dependency or public
extension algorithm was added.

## Hinge, top-k and discounted gain measures (BSD)

`src/sklearn_score_ranking.f90` adapts sklearn 1.6.1
`metrics/_classification.py` hinge_loss and `metrics/_ranking.py`
top_k_accuracy_score, dcg_score, ndcg_score and private tie-averaged gain
conventions. Sklearn BSD terms remain in `LICENSES/scikit-learn-COPYING`.
Native grouped merge sorting, linear true-class rank counting, label lookup,
rank/type adapters, quiet IEEE composition, tests, example and reference harness
are original MIT contributions. No dependency or public extension was added.

## Additional pairwise measures and kernels (BSD)

`src/sklearn_pairwise_measures.f90` follows sklearn 1.6.1
`metrics/pairwise.py` cosine, missing-aware and paired distances, Laplacian and
chi-squared kernels, plus `metrics/_pairwise_fast.pyx` chi-squared accumulation
and `metrics/_dist_metrics.pyx.tp` haversine formulas. Dense and CSR normalization
conventions follow sklearn preprocessing's normalize and sparse row-normalization
paths. Terms remain in `LICENSES/scikit-learn-COPYING`. Native interfaces,
canonical CSR adapters, callback adaptation, quiet IEEE composition, tests,
example and reference harness are original MIT contributions. No dependency or
public extension algorithm was added.

## Weighted classification and multilabel summaries (BSD)

`src/sklearn_classification_metrics.f90` adapts sklearn 1.6.1
`metrics/_classification.py` precision_score, recall_score, f1_score,
fbeta_score, jaccard_score, precision_recall_fscore_support and
multilabel_confusion_matrix, plus `utils/extmath.py` _nanaverage conventions.
Terms remain in `LICENSES/scikit-learn-COPYING`. Private pairwise summation
follows the existing NumPy 2.2.6 reduction contract, with retained terms in
`LICENSES/numpy-COPYING`. Native sparse intersections, label lookup, typed
adapters, quiet IEEE composition, tests, example and reference harness are
original MIT contributions. No dependency or public extension algorithm was
added.

## Bicluster consensus metric (BSD)

`src/sklearn_bicluster_metrics.f90` follows sklearn 1.6.1
`metrics/cluster/_bicluster.py` consensus_score and Jaccard conventions.
The scikit-learn BSD terms remain in `LICENSES/scikit-learn-COPYING`.
The rectangular Hungarian shortest-augmenting-path implementation, Fortran
indicator and callback adapters, tests, example and reference harness are
original MIT contributions. No SciPy assignment implementation was copied,
and no dependency or public extension algorithm was added.
