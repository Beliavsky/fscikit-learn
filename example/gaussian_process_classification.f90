program gaussian_process_classification
  use sklearn, only: dp,gaussian_process_classifier
  implicit none
  type(gaussian_process_classifier) :: model
  real(dp) :: x(6,1)
  real(dp), allocatable :: probability(:,:)
  integer :: i
  do i=1,6
    x(i,1)=real(i-4,dp)
  end do
  call model%fit(x,[-3,-3,-3,9,9,9]); probability=model%predict_proba(x)
  print *, 'Training accuracy:',model%score(x,[-3,-3,-3,9,9,9])
  print *, 'Positive-class probabilities:',probability(:,2)
end program
