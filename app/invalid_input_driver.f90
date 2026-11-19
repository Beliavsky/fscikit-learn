! Validation helper: each selected case must terminate with an input diagnostic.
program invalid_input_driver
  use sklearn
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(fast_ica) :: ica
  type(isotonic_regression) :: monotone
  type(rbf_sampler) :: fourier
  type(nystroem) :: landmarks
  type(pls_regression) :: partial
  type(pls_canonical) :: canonical
  type(cca) :: correlation
  type(pls_svd) :: cross_svd
  type(mds) :: scaling
  type(locally_linear_embedding) :: lle
  type(svr) :: eps_reg
  type(nu_svr) :: nu_reg
  type(one_class_svm) :: novelty_svm
  type(graphical_lasso) :: graphical
  type(kbins_discretizer) :: bins
  type(power_transformer) :: power
  type(standard_scaler) :: scaler
  type(logistic_regression) :: logistic
  type(kneighbors_regressor) :: knn
  type(kmeans) :: km
  type(pca) :: pc
  type(svc) :: svm
  type(decision_tree_classifier) :: tree
  type(gaussian_mixture) :: gm
  type(gaussian_process_regressor) :: gp
  type(variance_threshold) :: vt
  type(simple_imputer) :: im
  type(count_vectorizer) :: text
  type(kfold) :: splitter
  type(radius_neighbors_classifier) :: radius
  type(nearest_centroid) :: centroid
  type(local_outlier_factor) :: lof
  type(kernel_pca) :: kp
  type(isomap) :: iso
  type(label_propagation) :: lp
  type(label_spreading) :: ls
  type(bayesian_ridge) :: br
  type(ard_regression) :: ar
  type(orthogonal_matching_pursuit) :: omp
  type(factor_analysis) :: fa
  type(gaussian_random_projection) :: projection
  type(sparse_random_projection) :: sparse_projection
  type(huber_regressor) :: huber
  type(ransac_regressor) :: ransac
  type(poisson_regressor) :: poisson
  type(gamma_regressor) :: gamma
  type(tweedie_regressor) :: tweedie
  type(mean_shift) :: ms
  type(affinity_propagation) :: ap
  type(spectral_clustering) :: spectral
  type(spectral_embedding) :: embedding
  type(kernel_density) :: density
  type(index_split), allocatable :: folds(:)
  real(dp) :: x(4,2),y(4)
  real(dp), allocatable :: z(:,:),pred(:)
  integer, allocatable :: labels(:)
  character(32) :: mode
  x=1; y=1; call get_command_argument(1,mode)
  select case(trim(mode))
  case('graphical_alpha'); graphical%alpha=-1; call graphical%fit(x)
  case('graphical_nan'); graphical%alpha=ieee_value(0._dp,ieee_quiet_nan); call graphical%fit(x)
  case('graphical_tol'); graphical%tol=0; call graphical%fit(x)
  case('graphical_enet_tol'); graphical%enet_tol=0; call graphical%fit(x)
  case('graphical_eps'); graphical%eps=0; call graphical%fit(x)
  case('graphical_iterations'); graphical%max_iter=0; call graphical%fit(x)
  case('graphical_mode'); graphical%mode='lars'; call graphical%fit(x)
  case('graphical_cov_mode'); graphical%covariance='invalid'; call graphical%fit(x)
  case('graphical_samples'); call graphical%fit(x(:1,:))
  case('graphical_feature_count'); call graphical%fit(x(:,1:1))
  case('graphical_square'); graphical%covariance='precomputed'; call graphical%fit(x)
  case('graphical_asymmetric')
    graphical%covariance='precomputed'; x(1,2)=0.5_dp; call graphical%fit(x(:2,:))
  case('graphical_indefinite')
    graphical%covariance='precomputed'; x(1,2)=2; x(2,1)=2; call graphical%fit(x(:2,:))
  case('graphical_zero_variance'); call graphical%fit(x)
  case('graphical_zero_diagonal')
    graphical%covariance='precomputed'; x(1,:)=[0._dp,0._dp]; x(2,:)=[0._dp,1._dp]
    call graphical%fit(x(:2,:))
  case('graphical_singular')
    graphical%alpha=0; graphical%covariance='precomputed'; call graphical%fit(x(:2,:))
  case('graphical_unfitted'); pred=graphical%mahalanobis(x)
  case('graphical_score_unfitted'); y(1)=graphical%score(x)
  case('graphical_query_width')
    graphical%covariance='precomputed'; x(1,:)=[1._dp,0._dp]; x(2,:)=[0._dp,1._dp]
    call graphical%fit(x(:2,:)); pred=graphical%mahalanobis(x(:,1:1))
  case('graphical_empty'); call graphical%fit(x(:0,:))
  case('bins_strategy'); bins%strategy='invalid'; call bins%fit(x)
  case('bins_encode'); bins%encode='onehot'; call bins%fit(x)
  case('bins_count'); bins%n_bins=1; call bins%fit(x)
  case('bins_count_shape'); call bins%fit(x,n_bins=[3])
  case('bins_count_vector'); call bins%fit(x,n_bins=[3,1])
  case('bins_weight_uniform'); bins%strategy='uniform'; call bins%fit(x,sample_weight=y)
  case('bins_weight_shape'); call bins%fit(x,sample_weight=y(:3))
  case('bins_weight_negative'); y(1)=-1; call bins%fit(x,sample_weight=y)
  case('bins_weight_zero'); y=0; call bins%fit(x,sample_weight=y)
  case('bins_weight_nan'); y(1)=ieee_value(0._dp,ieee_quiet_nan); call bins%fit(x,sample_weight=y)
  case('bins_unfitted'); z=bins%transform(x)
  case('bins_inverse_unfitted'); z=bins%inverse_transform(x)
  case('bins_features'); call bins%fit(x); z=bins%transform(x(:,1:1))
  case('bins_inverse_width'); call bins%fit(x); z=bins%inverse_transform(x(:,1:1))
  case('bins_inverse_range'); bins%encode='ordinal'; call bins%fit(x); z=bins%inverse_transform(x)
  case('bins_inverse_fraction'); bins%encode='ordinal'; call bins%fit(x); x=0.5_dp; z=bins%inverse_transform(x)
  case('bins_inverse_zero'); call bins%fit(x); x=0; z=bins%inverse_transform(x)
  case('bins_inverse_dense_fraction'); call bins%fit(x); x=0.5_dp; z=bins%inverse_transform(x)
  case('bins_no_width'); x(:,1)=[0._dp,1.e-10_dp,2.e-10_dp,3.e-10_dp]; call bins%fit(x)
  case('bins_kmeans_size'); bins%strategy='kmeans'; x(:,1)=[0._dp,1._dp,2._dp,3._dp]; call bins%fit(x)
  case('power_method'); power%method='invalid'; call power%fit(x)
  case('power_positive'); power%method='box-cox'; x(1,1)=0; call power%fit(x)
  case('power_constant'); power%method='box-cox'; call power%fit(x)
  case('power_missing'); x(:,1)=ieee_value(0._dp,ieee_quiet_nan); call power%fit(x)
  case('power_unfitted'); z=power%transform(x)
  case('power_inverse_unfitted'); z=power%inverse_transform(x)
  case('power_features'); call power%fit(x); z=power%transform(x(:,1:1))
  case('power_inverse_features'); call power%fit(x); z=power%inverse_transform(x(:,1:1))
  case('power_domain')
    power%standardize=.false.; call power%fit(x); power%lambdas_=-1; z=power%inverse_transform(x)
  case('power_query_positive')
    power%method='box-cox'; x(:,1)=[1._dp,2._dp,3._dp,4._dp]; x(:,2)=x(:,1)
    call power%fit(x); x(1,1)=-1; z=power%transform(x)
  case('power_empty'); call power%fit(x(:0,:))
  case('ksvm_c'); eps_reg%c=0; call eps_reg%fit(x,y)
  case('svr_epsilon'); eps_reg%epsilon=-1; call eps_reg%fit(x,y)
  case('ksvm_tolerance'); eps_reg%tol=0; call eps_reg%fit(x,y)
  case('ksvm_iterations'); eps_reg%max_iter=0; call eps_reg%fit(x,y)
  case('ksvm_kernel'); eps_reg%kernel='unknown'; call eps_reg%fit(x,y)
  case('ksvm_gamma'); eps_reg%gamma=-2; call eps_reg%fit(x,y)
  case('ksvm_gamma_rule'); eps_reg%gamma_rule='unknown'; call eps_reg%fit(x,y)
  case('ksvm_poly'); eps_reg%coef0=-1; call eps_reg%fit(x,y)
  case('ksvm_square'); eps_reg%kernel='precomputed'; call eps_reg%fit(x,y)
  case('ksvm_symmetry')
    eps_reg%kernel='precomputed'; z=spread(y,2,4); z(1,2)=2; call eps_reg%fit(z,y)
  case('ksvm_indefinite'); eps_reg%kernel='precomputed'; z=-spread(y,2,4); call eps_reg%fit(z,y)
  case('ksvm_features'); call eps_reg%fit(x,y); pred=eps_reg%predict(x(:,:1))
  case('ksvm_unfitted'); pred=eps_reg%predict(x)
  case('ksvm_targets'); call eps_reg%fit(x,y(:3))
  case('ksvm_weights'); call eps_reg%fit(x,y,-y)
  case('ksvm_zero_weights'); call eps_reg%fit(x,y,0*y)
  case('ksvm_weight_shape'); call eps_reg%fit(x,y,y(:3))
  case('nusvr_nu0'); nu_reg%nu=0; call nu_reg%fit(x,y)
  case('nusvr_nu_high'); nu_reg%nu=1.01_dp; call nu_reg%fit(x,y)
  case('oneclass_nu1'); novelty_svm%nu=1; call novelty_svm%fit(x)
  case('oneclass_nu0'); novelty_svm%nu=0; call novelty_svm%fit(x)
  case('oneclass_unfitted'); pred=novelty_svm%score_samples(x)
  case('oneclass_weight_shape'); call novelty_svm%fit(x,y(:3))
  case('ksvm_nan_c'); eps_reg%c=ieee_value(0._dp,ieee_quiet_nan); call eps_reg%fit(x,y)
  case('mds_components'); scaling%n_components=0; call scaling%fit(x)
  case('mds_restarts'); scaling%n_init=0; call scaling%fit(x)
  case('mds_iterations'); scaling%max_iter=0; call scaling%fit(x)
  case('mds_tolerance'); scaling%eps=-1; call scaling%fit(x)
  case('mds_stress'); scaling%normalized_stress='unknown'; call scaling%fit(x)
  case('mds_metric_stress'); scaling%normalized_stress='true'; call scaling%fit(x)
  case('mds_metric'); scaling%dissimilarity='unknown'; call scaling%fit(x)
  case('mds_square'); scaling%dissimilarity='precomputed'; call scaling%fit(x)
  case('mds_symmetry')
    scaling%dissimilarity='precomputed'; z=spread(y,2,4); z(1,2)=2; call scaling%fit(z)
  case('mds_negative'); scaling%dissimilarity='precomputed'; z=-spread(y,2,4); call scaling%fit(z)
  case('mds_diagonal'); scaling%dissimilarity='precomputed'; z=spread(y,2,4); call scaling%fit(z)
  case('mds_init'); call scaling%fit(x,x(:3,:))
  case('mds_collapsed')
    x(:,1)=[1._dp,2._dp,3._dp,4._dp]; z=0*x; call scaling%fit(x,z)
  case('mds_nonmetric_zero'); scaling%metric=.false.; call scaling%fit(x)
  case('lle_neighbors'); lle%n_neighbors=0; call lle%fit(x)
  case('lle_size'); lle%n_neighbors=4; call lle%fit(x)
  case('lle_components'); lle%n_neighbors=2; lle%n_components=3; call lle%fit(x)
  case('lle_reg'); lle%n_neighbors=2; lle%reg=-1; call lle%fit(x)
  case('lle_method'); lle%n_neighbors=2; lle%method='unknown'; call lle%fit(x)
  case('lle_ltsa_neighbors'); lle%n_neighbors=2; lle%method='ltsa'; call lle%fit(x)
  case('lle_unfitted'); z=lle%transform(x)
  case('lle_features'); lle%n_neighbors=2; call lle%fit(x); z=lle%transform(x(:,:1))
  case('lle_singular'); lle%n_neighbors=2; lle%reg=0; call lle%fit(x)
  case('cross_components'); partial%n_components=0; call partial%fit(x,x)
  case('canonical_components'); call canonical%fit(x,reshape(y,[4,1]))
  case('cca_components'); correlation%n_components=3; call correlation%fit(x,x)
  case('svd_components'); cross_svd%n_components=3; call cross_svd%fit(x,x)
  case('cross_iterations'); partial%max_iter=0; call partial%fit(x,x)
  case('cross_tolerance'); partial%tol=-1; call partial%fit(x,x)
  case('cross_algorithm'); partial%algorithm='svd'; call partial%fit(x,x)
  case('cca_algorithm'); correlation%algorithm='svd'; call correlation%fit(x,x)
  case('cross_counts'); call partial%fit(x,x(:3,:))
  case('cross_samples'); call partial%fit(x(:1,:),x(:1,:))
  case('cross_unfitted'); z=partial%predict(x)
  case('cross_features'); call partial%fit(x,x); z=partial%transform(x(:,:1))
  case('cross_targets'); call partial%fit(x,x); z=partial%transform_y(x(:,:1))
  case('cross_inverse'); call partial%fit(x,x); z=partial%inverse_transform(x(:,:1))
  case('cross_score'); call partial%fit(x,reshape(y,[4,1])); pred=[partial%score(x,x)]
  case('svd_rank'); call cross_svd%fit(x,x)
  case('cross_degenerate')
    y=[1._dp,2._dp,3._dp,4._dp]; call partial%fit(x,reshape(y,[4,1]))
  case('iso_unfitted'); pred=monotone%predict(y)
  case('iso_bounds'); monotone%y_min=2; monotone%y_max=1; call monotone%fit(y,y)
  case('iso_weight_count'); call monotone%fit(y,y,y(:2))
  case('iso_weights'); call monotone%fit(y,y,-y)
  case('iso_zero_weights'); call monotone%fit(y,y,0*y)
  case('iso_length'); call monotone%fit(y,y(:2))
  case('iso_out_of_bounds'); monotone%out_of_bounds='unknown'; call monotone%fit(y,y)
  case('iso_query')
    y=[1._dp,2._dp,3._dp,4._dp]; monotone%out_of_bounds='raise'
    call monotone%fit(y,y); pred=monotone%predict([0._dp])
  case('iso_pava_weights'); pred=isotonic_fit(y,0*y)
  case('iso_nan'); y(1)=ieee_value(0._dp,ieee_quiet_nan); call monotone%fit(y,y)
  case('fourier_components'); fourier%n_components=0; call fourier%fit(x)
  case('fourier_gamma'); fourier%gamma=-1; call fourier%fit(x)
  case('fourier_unfitted'); z=fourier%transform(x)
  case('fourier_features'); call fourier%fit(x); z=fourier%transform(x(:,:1))
  case('landmark_components'); landmarks%n_components=0; call landmarks%fit(x)
  case('landmark_kernel'); landmarks%kernel='unknown'; call landmarks%fit(x)
  case('landmark_gamma'); landmarks%gamma=-2; call landmarks%fit(x)
  case('landmark_poly'); landmarks%coef0=-1; call landmarks%fit(x)
  case('landmark_unfitted'); z=landmarks%transform(x)
  case('landmark_features'); call landmarks%fit(x); z=landmarks%transform(x(:,:1))
  case('unfitted'); z=scaler%transform(x)
  case('nan'); x(1,1)=ieee_value(0._dp,ieee_quiet_nan); call scaler%fit(x)
  case('features'); call scaler%fit(x); z=scaler%transform(x(:,:1))
  case('target_size'); call logistic%fit(x,[0,1])
  case('single_class'); call logistic%fit(x,[7,7,7,7])
  case('neighbors'); knn%n_neighbors=10; call knn%fit(x,y); pred=knn%predict(x)
  case('clusters'); km%n_clusters=5; call km%fit(x)
  case('components'); pc%n_components=3; call pc%fit(x)
  case('svm_c'); svm%c=-1; call svm%fit(x,[0,0,1,1])
  case('tree_depth'); tree%max_depth=-1; call tree%fit(x,[0,0,1,1])
  case('mixture_reg'); gm%reg_covar=-1; call gm%fit(x)
  case('gp_length'); gp%length_scale=0; call gp%fit(x,y)
  case('variance'); call vt%fit(x)
  case('missing'); x=ieee_value(0._dp,ieee_quiet_nan); call im%fit(x)
  case('empty_text'); call text%fit([character(4)::'a !','x ?'])
  case('folds'); splitter%n_splits=5; folds=splitter%split(4)
  case('radius_empty'); call radius%fit(x,[0,0,1,1]); labels=radius%predict(x+100)
  case('centroid_metric'); centroid%metric='cosine'; call centroid%fit(x,[0,0,1,1])
  case('lof_mode'); call lof%fit(x); pred=lof%score_samples(x)
  case('lof_contamination'); lof%contamination=0.8_dp; call lof%fit(x)
  case('kpca_symmetry')
    kp%kernel='precomputed'; z=reshape([1._dp,0._dp,1._dp,1._dp],[2,2]); call kp%fit(z)
  case('kpca_indefinite')
    kp%kernel='precomputed'; z=reshape([1._dp,2._dp,2._dp,1._dp],[2,2]); call kp%fit(z)
  case('kpca_width')
    kp%kernel='precomputed'; z=matmul(x,transpose(x)); call kp%fit(z); z=kp%transform(x)
  case('isomap_neighbors'); iso%n_neighbors=4; call iso%fit(x)
  case('unlabeled'); call lp%fit(x,[-1,-1,-1,-1])
  case('spreading_alpha'); ls%alpha=1; call ls%fit(x,[0,-1,-1,1])
  case('bayes_prior'); br%alpha_1=0; call br%fit(x,y)
  case('bayes_unfitted'); pred=br%predict_std(x)
  case('ard_threshold'); ar%threshold_lambda=0; call ar%fit(x,y)
  case('omp_sparsity'); omp%n_nonzero_coefs=3; call omp%fit(x,y)
  case('factor_components'); fa%n_components=3; call fa%fit(x)
  case('factor_noise'); fa%noise_variance_init=[1._dp,0._dp]; call fa%fit(x)
  case('projection_auto'); call projection%fit(x)
  case('projection_density'); sparse_projection%n_components=1; sparse_projection%density=0
    call sparse_projection%fit(x)
  case('projection_unfitted'); z=projection%inverse_transform(x)
  case('huber_epsilon'); huber%epsilon=0.9_dp; call huber%fit(x,y)
  case('ransac_samples'); ransac%min_samples=5; call ransac%fit(x,y)
  case('ransac_threshold'); ransac%residual_threshold=-2; call ransac%fit(x,y)
  case('ransac_loss'); ransac%loss='invalid'; call ransac%fit(x,y)
  case('ransac_probability'); ransac%stop_probability=1.1_dp; call ransac%fit(x,y)
  case('poisson_targets'); y(1)=-1; call poisson%fit(x,y)
  case('gamma_targets'); y(1)=0; call gamma%fit(x,y)
  case('tweedie_power'); tweedie%power=0.5_dp; call tweedie%fit(x,y)
  case('tweedie_link'); tweedie%link='log'; call tweedie%fit(x,y)
  case('glm_alpha'); poisson%alpha=-1; call poisson%fit(x,y)
  case('glm_unfitted'); pred=poisson%predict(x)
  case('glm_constant_score'); call poisson%fit(x,y); pred=[poisson%score(x,y)]
  case('meanshift_bandwidth'); ms%bandwidth=0; call ms%fit(x)
  case('meanshift_seeds'); ms%bandwidth=0.1_dp; call ms%fit(x,seeds=x+100)
  case('affinity_damping'); ap%damping=0.2_dp; call ap%fit(x)
  case('affinity_square'); ap%affinity='precomputed'; call ap%fit(x)
  case('affinity_predict'); ap%affinity='precomputed'; call ap%fit(matmul(x,transpose(x))); labels=ap%predict(x)
  case('spectral_clusters'); spectral%n_clusters=5; call spectral%fit(x)
  case('spectral_neighbors'); embedding%n_neighbors=5; call embedding%fit(x)
  case('spectral_symmetry')
    embedding%affinity='precomputed'; z=reshape([1.e100_dp,0._dp,1._dp,1.e100_dp],[2,2]); call embedding%fit(z)
  case('spectral_negative')
    embedding%affinity='precomputed'; z=reshape([1._dp,-1._dp,-1._dp,1._dp],[2,2]); call embedding%fit(z)
  case('spectral_components'); embedding%n_components=4; call embedding%fit(x)
  case('density_bandwidth'); density%bandwidth=0; call density%fit(x)
  case('density_weights'); call density%fit(x,sample_weight=[0._dp,0._dp,0._dp,0._dp])
  case('density_kernel'); density%kernel='invalid'; call density%fit(x)
  case('density_unfitted'); pred=density%score_samples(x)
  case('density_sampling'); density%kernel='linear'; call density%fit(x); z=density%sample(2)
  case('ica_unfitted'); z=ica%transform(x)
  case('ica_algorithm'); ica%algorithm='unknown'; call ica%fit(x)
  case('ica_fun'); ica%fun='unknown'; call ica%fit(x)
  case('ica_whiten'); ica%whiten='unknown'; call ica%fit(x)
  case('ica_alpha'); ica%alpha=0.5_dp; call ica%fit(x)
  case('ica_iterations'); ica%max_iter=0; call ica%fit(x)
  case('ica_tolerance'); ica%tol=-1; call ica%fit(x)
  case('ica_rank'); call ica%fit(x)
  case('ica_init_shape'); ica%whiten='none'; ica%w_init=reshape([1._dp],[1,1]); call ica%fit(x)
  case('ica_init_singular'); ica%whiten='none'; ica%w_init=x(:2,:); call ica%fit(x)
  case default; error stop 'unknown validation mode'
  end select
  print *, 'ERROR: invalid input was accepted'
end program
