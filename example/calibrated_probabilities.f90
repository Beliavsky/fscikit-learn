program calibrated_probabilities
  use sklearn, only: dp,gaussian_nb,calibrated_classifier_cv
  implicit none
  type(gaussian_nb) :: estimator
  type(calibrated_classifier_cv) :: model
  real(dp) :: x(12,1),query(3,1)
  real(dp), allocatable :: prob(:,:)
  integer :: y(12)=[0,0,0,0,0,0,1,1,1,1,1,1],i
  x(:,1)=[-4._dp,-3._dp,-2._dp,-1._dp,-.5_dp,.2_dp,.8_dp,1.2_dp,2._dp,3._dp,4._dp,5._dp]
  query(:,1)=[-2._dp,0._dp,2._dp]; model%cv=3
  call model%fit(x,y,estimator)
  prob=model%predict_proba(query)
  print *, 'Fold ensemble positive probabilities:'
  do i=1,3
    print *, query(i,1),prob(i,2)
  end do
  model%ensemble=.false.; model%method='isotonic'
  call model%fit(x,y,estimator)
  print *, 'Out-of-fold isotonic labels:',model%predict(query)
end program
