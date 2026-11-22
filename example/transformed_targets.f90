program transformed_targets
  use sklearn, only: dp,transformed_target_regressor,function_transformer
  implicit none
  type(transformed_target_regressor) :: model
  type(function_transformer) :: transform
  real(dp) :: x(12,1),y(12)
  real(dp), allocatable :: predicted(:),mapped(:,:)
  integer :: i
  do i=1,12
    x(i,1)=real(i,dp)/12;y(i)=exp(1+2*x(i,1))
  end do
  model%func=>logarithm;model%inverse_func=>exponential
  call model%fit(x,y);predicted=model%predict(x)
  transform%func=>logarithm;mapped=transform%transform(reshape(y,[12,1]))
  print '(a,f10.7)', 'Original-scale R2: ',model%score(x,y)
  print '(a,es12.4)', 'Maximum prediction error: ',maxval(abs(predicted-y))
  print '(a,f10.7)', 'First transformed target: ',mapped(1,1)
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
