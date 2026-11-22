program curve_invalid_driver
  use sklearn_validation
  use sklearn_common, only: dp
  use sklearn_parameter_space, only: param,parameter_value
  use sklearn_linear_model, only: ridge
  use sklearn_sgd, only: sgd_regressor,sgd_classifier
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(ridge) :: base
  type(sgd_regressor) :: sgd
  type(sgd_classifier) :: classifier
  type(learning_curve_options) :: options
  type(cross_validation_options) :: cv
  type(learning_curve_result) :: learning
  type(validation_curve_result) :: validation
  type(permutation_score_result) :: permutation
  real(dp) :: x(24,3),y(24),weights(24),nan,inf
  integer :: i
  character(64) :: test_case
  call get_command_argument(1,test_case)
  x=reshape([(real(i,dp)/20,i=1,72)],[24,3]);y=x(:,1);weights=1
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  options%cv=3;cv%cv=3
  select case(trim(test_case))
  case('fractions_empty');options%train_sizes=[real(dp)::]
  case('fractions_nan');options%train_sizes=[nan]
  case('fractions_inf');options%train_sizes=[inf]
  case('fractions_zero');options%train_sizes=[0._dp]
  case('fractions_negative');options%train_sizes=[-.1_dp]
  case('fractions_large');options%train_sizes=[1.1_dp]
  case('counts_empty');options%train_counts=[integer::]
  case('counts_zero');options%train_counts=[0]
  case('counts_negative');options%train_counts=[-1]
  case('counts_large');options%train_counts=[17]
  case('sizes_both');options%train_counts=[4];options%train_sizes=[.5_dp]
  case('learning_seed');options%random_state=-1
  case('learning_multiple');allocate(options%scorers(1))
  case('incremental_unsupported');options%exploit_incremental_learning=.true.
  case('incremental_empty')
    options%exploit_incremental_learning=.true.;options%train_counts=[3,8,16]
    allocate(options%cv_splits(2));options%cv_splits(1)%train=[(i,i=1,16)]
    options%cv_splits(1)%test=[(i,i=17,24)];options%cv_splits(2)%train=[1,2,3,4]
    options%cv_splits(2)%test=[(i,i=5,24)]
    learning=learning_curve(sgd,x,y,options)
  case('incremental_weights')
    options%exploit_incremental_learning=.true.;options%train_counts=[4,8]
    learning=learning_curve(sgd,x,y,options,weights)
  case('incremental_no_target')
    options%exploit_incremental_learning=.true.;options%scorer_func=>dummy_score
    learning=learning_curve(sgd,x,options)
  case('incremental_wrong_target')
    options%exploit_incremental_learning=.true.;options%task='regression'
    learning=learning_curve(classifier,x,y,options)
  case('validation_empty')
    validation=validation_curve(base,x,y,'alpha',[parameter_value::],cv)
  case('validation_unknown')
    validation=validation_curve(base,x,y,'unknown',[param(1._dp)],cv)
  case('validation_type')
    validation=validation_curve(base,x,y,'alpha',[param('bad')],cv)
  case('validation_nested')
    validation=validation_curve(base,x,y,'model__alpha',[param(.1_dp)],cv)
  case('validation_multiple')
    allocate(cv%scorers(1));validation=validation_curve(base,x,y,'alpha',[param(.1_dp)],cv)
  case('permutation_zero');permutation=permutation_test_score(base,x,y,cv,n_permutations=0)
  case('permutation_negative');permutation=permutation_test_score(base,x,y,cv,n_permutations=-1)
  case('permutation_seed');permutation=permutation_test_score(base,x,y,cv,n_permutations=1,random_state=-1)
  case('permutation_groups');permutation=permutation_test_score(base,x,y,cv,n_permutations=1,groups=[1,2])
  case('permutation_multiple')
    allocate(cv%scorers(1));permutation=permutation_test_score(base,x,y,cv,n_permutations=1)
  case('permutation_target_count');permutation=permutation_test_score(base,x,y(:23),cv,n_permutations=1)
  case('curve_weights')
    learning=learning_curve(base,x,y,options,weights(:23))
  case('curve_nonfinite_target')
    y(1)=nan;learning=learning_curve(base,x,y,options)
  case('validation_error_score')
    cv%error_score='nan';validation=validation_curve(base,x,y,'alpha',[param(.1_dp)],cv)
  case('learning_error_score')
    options%error_score='nan'
  case('permutation_error_score')
    cv%error_score='nan';permutation=permutation_test_score(base,x,y,cv,n_permutations=1)
  case default
    error stop 'unknown curve invalid case'
  end select
  learning=learning_curve(base,x,y,options)
  error stop 'curve invalid case unexpectedly accepted'
contains
  real(dp) function dummy_score(model,x,yr,yi) result(value)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    value=0
  end function
end program
