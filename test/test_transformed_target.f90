program test_transformed_target
  use sklearn_compose
  use sklearn_preprocessing, only: standard_scaler
  use sklearn_linear_model, only: ridge
  use sklearn_common, only: dp,require
  implicit none
  type(transformed_target_regressor) :: model,copy
  type(standard_scaler) :: scaler
  type(ridge) :: base
  real(dp) :: x(20,2),y(20),multi(20,2),w(20)
  real(dp), allocatable :: predicted(:),matrix(:,:),prior(:)
  integer :: i
  do i=1,20
    x(i,:)=[real(i,dp)/20,sin(real(i,dp))];y(i)=exp(1+2*x(i,1)-.3_dp*x(i,2))
  end do
  model%func=>logarithm;model%inverse_func=>exponential
  call model%fit(x,y);predicted=model%predict(x)
  call require(maxval(abs(predicted-y))<1.e-10_dp,'log target regression recovers exponential fixture')
  call require(model%training_dim_==1.and.model%inverse_consistent_,'target dimension and inverse metadata')
  copy=model;prior=predicted
  call copy%fit(x,y+1)
  call require(all(model%predict(x)==prior),'target model copy refit is independent')
  nullify(model%func,model%inverse_func);allocate(model%transformer,source=scaler)
  base%alpha=.2_dp;allocate(model%regressor,source=base)
  multi(:,1)=y;multi(:,2)=3*y+5;w=1;w(:3)=0
  call model%fit(x,multi,w);matrix=model%predict_matrix(x)
  call require(all(shape(matrix)==[20,2]),'multioutput target prediction shape')
  call require(model%score(x,multi,w)>.8_dp,'multioutput target regression score')
  select type(prototype=>model%regressor)
  type is(ridge)
    call require(.not.allocated(prototype%coef_),'target fit leaves base prototype unfitted')
  end select
  select type(prototype=>model%transformer)
  type is(standard_scaler)
    call require(.not.allocated(prototype%mean_),'target fit leaves transformer prototype unfitted')
  end select
  print *, 'TransformedTargetRegressor: PASS'
contains
  function logarithm(a) result(b)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: b(:,:)
    b=log(a)
  end function
  function exponential(a) result(b)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: b(:,:)
    b=exp(a)
  end function
end program
