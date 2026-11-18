! Validation helper: each selected case must terminate with an input diagnostic.
program invalid_input_driver
  use sklearn
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
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
  type(index_split), allocatable :: folds(:)
  real(dp) :: x(4,2),y(4)
  real(dp), allocatable :: z(:,:),pred(:)
  integer, allocatable :: labels(:)
  character(32) :: mode
  x=1; y=1; call get_command_argument(1,mode)
  select case(trim(mode))
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
  case default; error stop 'unknown validation mode'
  end select
  print *, 'ERROR: invalid input was accepted'
end program
