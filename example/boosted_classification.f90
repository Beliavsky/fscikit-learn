program boosted_classification
  use sklearn
  implicit none
  type(gradient_boosting_classifier) :: model
  real(dp) :: x(12,1)
  integer :: labels(12),i
  real(dp), allocatable :: probability(:,:)
  do i=1,12
    x(i,1)=real(i,dp)
    labels(i)=merge(-4,7,i<=5)
  end do
  model%n_estimators=20; model%max_depth=1
  call model%fit(x,labels)
  print *, 'Training accuracy:',model%score(x,labels)
  probability=model%predict_proba(x)
  print *, 'First/last positive probabilities:',probability(1,2),probability(12,2)
end program
