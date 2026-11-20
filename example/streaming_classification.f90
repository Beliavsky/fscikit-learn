program streaming_classification
  use sklearn, only: dp,sgd_classifier
  implicit none
  type(sgd_classifier) :: model
  real(dp) :: x(6,1)
  integer :: y(6),i
  x(:,1)=[-2._dp,-1._dp,-.5_dp,.5_dp,1._dp,2._dp]; y=[-4,-4,-4,7,7,7]
  model%loss='log_loss'; model%learning_rate='constant'; model%eta0=.05_dp; model%average=1; model%fit_intercept=.false.
  call model%partial_fit(x(:3,:),y(:3),classes=[-4,7])
  call model%partial_fit(x(4:,:),y(4:))
  do i=1,99
    call model%partial_fit(x,y)
  end do
  print *, 'Samples processed:',model%t_-1
  print *, 'Training accuracy:',model%score(x,y)
end program
