program weighted_streaming_classifiers
  use sklearn, only: dp,gaussian_nb,logistic_regression
  implicit none
  type(gaussian_nb) :: nb
  type(logistic_regression) :: lr
  real(dp) :: x(8,1),weights(8)=[1._dp,2._dp,2._dp,1._dp,2._dp,1._dp,1._dp,2._dp],query(3,1)
  real(dp), allocatable :: probability(:,:)
  integer :: y(8)=[0,0,0,0,1,1,1,1],i
  x(:,1)=[-4._dp,-3._dp,-2._dp,-1._dp,1._dp,2._dp,3._dp,4._dp]
  query(:,1)=[-2._dp,0._dp,2._dp]
  call nb%partial_fit(x(:4,:),y(:4),classes=[0,1],sample_weight=weights(:4))
  print *, 'First-batch class masses:',nb%class_count_
  call nb%partial_fit(x(5:,:),y(5:),sample_weight=weights(5:))
  print *, 'Streaming class masses:',nb%class_count_
  print *, 'Streaming Gaussian labels:',nb%predict(query)
  call lr%fit(x,y,sample_weight=weights)
  probability=lr%predict_proba(query)
  print *, 'Weighted logistic positive probabilities:'
  do i=1,3
    print *, query(i,1),probability(i,2)
  end do
end program
