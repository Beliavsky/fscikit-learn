program test_positive_composition
  use sklearn_common, only: dp,require
  use sklearn_linear_model, only: linear_regression
  use sklearn_pipeline, only: pipeline
  use sklearn_compose, only: transformed_target_regressor
  use sklearn_bagging, only: bagging_regressor
  use sklearn_multioutput, only: multi_output_regressor,regressor_chain
  implicit none
  type(linear_regression) :: base
  type(pipeline) :: pipe
  type(transformed_target_regressor) :: target
  type(bagging_regressor) :: bag
  type(multi_output_regressor) :: multi
  type(regressor_chain) :: chain
  real(dp) :: x(12,3),y(12),ys(12,2),w(12)
  integer :: i
  do i=1,12
    x(i,:)=[real(i,dp),sin(real(i,dp)),cos(real(i,dp))];y(i)=3-2*x(i,1)+4*x(i,2)
  end do
  ys(:,1)=y;ys(:,2)=-y;w=1;w(:2)=0;base%positive=.true.
  allocate(pipe%steps(1));pipe%steps(1)%name='positive';allocate(pipe%steps(1)%model,source=base)
  call pipe%fit(x,y,w);call assert_positive(pipe%steps(1)%model)
  allocate(target%regressor,source=base);call target%fit(x,y,w);call assert_positive(target%regressor_)
  bag%n_estimators=3;bag%random_state=42;call bag%fit(x,y,base,w)
  do i=1,3
    call assert_positive(bag%estimators_(i)%estimator)
  end do
  call multi%fit(x,ys,base);call chain%fit(x,ys,base)
  do i=1,2
    call assert_positive(multi%estimators_(i)%estimator_);call assert_positive(chain%estimators_(i)%estimator_)
  end do
  call require(base%positive.and..not.allocated(base%coef_),'composition preserves unfitted positive template')
  print *, 'Positive linear-model composition: PASS'
contains
  subroutine assert_positive(model)
    class(*), intent(in) :: model
    select type(model)
    class is(linear_regression)
      call require(model%positive.and.all(model%coef_>=0),'positive option survives fitted cloning')
    class default
      call require(.false.,'positive composition model type')
    end select
  end subroutine
end program
