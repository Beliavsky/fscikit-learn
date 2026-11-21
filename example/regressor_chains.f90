program regressor_chains
  use sklearn, only: dp,regressor_chain,ridge
  implicit none
  type(regressor_chain) :: model
  type(ridge) :: base
  real(dp) :: x(12,1),y(12,2),query(2,1)
  integer :: j
  x(:,1)=[(real(j-6,dp),j=1,12)]
  y(:,1)=2*x(:,1)+1; y(:,2)=x(:,1)**2+.2_dp*y(:,1)
  query(:,1)=[-2._dp,2._dp]
  base%alpha=.1_dp; model%order=[2,1]; model%cv=3
  call model%fit(x,y,base)
  print *, 'Regressor chain order:',model%order_
  print *, 'Regressor chain predictions:',model%predict(query)
  print *, 'Regressor chain training R2:',model%score(x,y)
end program
