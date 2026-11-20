program nu_classification
  use sklearn, only: dp,nu_svc
  implicit none
  type(nu_svc) :: model
  real(dp) :: x(40,2)
  integer :: labels(40),i
  do i=1,40
    labels(i)=merge(-5,7,i<=20)
    x(i,:)=[real(labels(i),dp)/4+.2_dp*sin(real(i,dp)),cos(real(i,dp))]
  end do
  model%nu=.4_dp; model%kernel='rbf'; call model%fit(x,labels)
  print *, 'Converged:',model%converged_
  print *, 'Support counts by class:',model%n_support_
  print *, 'Training accuracy:',model%score(x,labels)
  print *, 'Query labels:',model%predict(reshape([-1._dp,2._dp,0._dp,0._dp],[2,2]))
end program
