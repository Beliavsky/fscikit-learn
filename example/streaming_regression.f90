program streaming_regression
  use sklearn, only: dp,sgd_regressor
  implicit none
  type(sgd_regressor) :: model
  real(dp) :: x(6,1),y(6)
  integer :: i
  x(:,1)=[-1._dp,-.5_dp,0._dp,.5_dp,1._dp,1.5_dp]; y=2*x(:,1)+1
  model%learning_rate='constant'; model%eta0=.05_dp
  do i=1,100
    call model%partial_fit(x(:3,:),y(:3)); call model%partial_fit(x(4:,:),y(4:))
  end do
  print *, 'Samples processed:',model%t_-1
  print *, 'Streaming R2:',model%score(x,y)
end program
