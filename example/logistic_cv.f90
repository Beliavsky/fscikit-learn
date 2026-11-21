program example_logistic_cv
  use sklearn, only: dp,logistic_regression_cv
  implicit none
  type(logistic_regression_cv) :: model
  real(dp) :: x(60,2)
  integer :: y(60),i
  do i=1,60
    x(i,:)=[sin(real(i,dp)),cos(.37_dp*real(i,dp))]
    y(i)=merge(7,-3,x(i,1)+.4_dp*x(i,2)>0)
  end do
  model%cs=[.01_dp,.1_dp,1._dp,10._dp]
  model%cv=3
  call model%fit(x,y)
  print '(a,f9.4)', 'Selected C: ',model%c_(1)
  print '(a,f9.6)', 'Training accuracy: ',model%score(x,y)
end program
