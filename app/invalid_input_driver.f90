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
  case default; error stop 'unknown validation mode'
  end select
  print *, 'ERROR: invalid input was accepted'
end program
