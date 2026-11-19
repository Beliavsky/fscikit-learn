program regularization_cv
  use sklearn, only: dp,elastic_net_cv
  implicit none
  type(elastic_net_cv) :: model
  real(dp) :: x(12,2),y(12)
  integer :: i
  do i=1,12
    x(i,:)=[real(i,dp)/5,real(mod(i,3),dp)]; y(i)=2*x(i,1)-x(i,2)+.7_dp
  end do
  model%alphas=[1._dp,.1_dp,.01_dp]; model%l1_ratios=[.5_dp,1._dp]; model%cv=3
  call model%fit(x,y)
  print *, 'Selected alpha and l1 ratio:',model%alpha_,model%l1_ratio_
  print *, 'Training R2:',model%score(x,y)
end program
