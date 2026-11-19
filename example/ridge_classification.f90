program ridge_classification_demo
  use sklearn
  implicit none
  type(ridge_classifier_cv) :: model
  real(dp) :: x(8,1)
  integer :: y(8),i
  do i=1,8
    x(i,1)=real(i,dp)
  end do
  y=[-4,-4,-4,7,-4,7,7,7]
  model%alphas=[.1_dp,1._dp,10._dp]
  call model%fit(x,y)
  print *, 'Selected alpha:',model%alpha_
  print *, 'Training accuracy:',model%score(x,y)
end program
