program multitask_regression
  use sklearn
  implicit none
  type(multi_task_lasso) :: model
  real(dp) :: x(6,2),y(6,2)
  integer :: i
  do i=1,6
    x(i,:)=[real(i,dp),real(mod(i,2),dp)]
  end do
  y(:,1)=2*x(:,1)+1; y(:,2)=-x(:,1)+3
  model%alpha=.05_dp; call model%fit(x,y)
  print *, 'Joint regression R2:',model%score(x,y)
  print *, 'Shared feature coefficients:',model%coef_(:,1)
end program
