program test_learning_curves
  use sklearn_validation
  use sklearn_common, only: dp,require
  use sklearn_parameter_space, only: param,parameter_set
  use sklearn_linear_model, only: ridge
  implicit none
  type(ridge) :: model
  type(learning_curve_options) :: options
  type(cross_validation_options) :: cv
  type(learning_curve_result) :: learning,shuffled
  type(validation_curve_result) :: validation
  type(permutation_score_result) :: permutation
  real(dp) :: x(36,3),y(36)
  integer :: i,j
  type :: counter_model
    integer :: fits=0
  end type
  type(counter_model) :: counter
  do i=1,36
    x(i,:)=[sin(real(i,dp)),cos(real(2*i,dp)),real(i,dp)/30]
  end do
  y=2*x(:,1)-x(:,2)+.2_dp*x(:,3);model%alpha=.1_dp
  options%cv=3;options%train_sizes=[1._dp,.5_dp,.5_dp,.1_dp];options%return_times=.true.
  learning=learning_curve(model,x,y,options)
  call require(all(learning%train_sizes==[2,12,24]),'sorted and deduplicated fractional sizes')
  call require(all(shape(learning%test_score)==[3,3]),'learning-curve matrix shape')
  call require(all(learning%fit_time>=0).and.all(learning%score_time>=0),'nonnegative curve timings')
  call require(.not.allocated(model%coef_),'learning curve leaves caller unfitted')
  options%shuffle=.true.;options%random_state=42;shuffled=learning_curve(model,x,y,options)
  call require(any(shuffled%folds(1)%train/=learning%folds(1)%train),'seeded train-index shuffle')
  cv%cv=3
  validation=validation_curve(model,x,y,'alpha',[param(.1_dp),param(10._dp),param(.1_dp)],cv)
  call require(all(validation%test_score(1,:)==validation%test_score(3,:)),'parameter duplicates preserve results/order')
  call require(maxval(abs(validation%test_score(1,:)-learning%test_score(3,:)))<1.e-12_dp, &
    'full-size learning curve agrees with validation curve')
  permutation=permutation_test_score(model,x,y,cv,n_permutations=9,random_state=42)
  call require(size(permutation%permutation_scores)==9.and.permutation%pvalue==.1_dp,'permutation significance')
  permutation=permutation_test_score(model,x,y,cv,n_permutations=1)
  call require(size(permutation%permutation_scores)==1.and.permutation%pvalue==.5_dp,'single permutation')
  cv=cross_validation_options();cv%cv=3;cv%task='regression';cv%fit_func=>count_fit;cv%scorer_func=>count_score
  permutation=permutation_test_score(counter,x,y,cv,n_permutations=2)
  call require(permutation%score==2.and.all(permutation%permutation_scores==2), &
    'permutation test reuses one model across folds')
  validation=validation_curve(counter,x,y,'unused',[param(1),param(2)],cv,set_params_func=ignore_parameters)
  call require(all(validation%train_score==1).and.all(validation%test_score==1), &
    'validation curve clones for every parameter/fold')
  options=learning_curve_options();options%cross_validation_options=cv;options%train_counts=[8,16]
  learning=learning_curve(counter,x,y,options)
  call require(all(learning%train_score==1),'batch learning curve clones for every size/fold')
  print *, 'Learning/validation curves and permutation tests: PASS'
contains
  subroutine count_fit(model,x,yr,yi,sample_weight)
    class(*), intent(inout) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:),sample_weight(:)
    integer, optional, intent(in) :: yi(:)
    select type(model)
    type is(counter_model);model%fits=model%fits+1
    end select
  end subroutine
  real(dp) function count_score(model,x,yr,yi) result(value)
    class(*), intent(in) :: model
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: yr(:)
    integer, optional, intent(in) :: yi(:)
    value=0
    select type(model)
    type is(counter_model);value=model%fits
    end select
  end function
  subroutine ignore_parameters(model,parameters)
    class(*), intent(inout) :: model
    type(parameter_set), intent(in) :: parameters
  end subroutine
end program
