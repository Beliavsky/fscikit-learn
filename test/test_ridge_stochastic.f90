program test_ridge_stochastic
  use sklearn,only:dp,ridge_regression,ridge_solution,ridge_solver_options,csr_from_dense,csr_from_arrays,csr_matrix
  use sklearn_common,only:require
  implicit none
  type(ridge_solver_options)::options
  type(ridge_solution)::fit,other
  real(dp)::x(4,2),y(4),w(4),expected(2)
  type(csr_matrix)::large
  integer::j
  x(:,1)=[1._dp,0._dp,0._dp,1._dp];x(:,2)=[0._dp,1._dp,1._dp,0._dp]
  y=[2._dp,-1._dp,-1._dp,2._dp];w=[1._dp,.5_dp,.5_dp,1._dp]
  options%solver='sag';options%tol=1.e-11_dp;options%max_iter=10000;options%random_state=42
  expected=[4._dp/3,-.5_dp]
  do j=1,2
    if(j==2)options%solver='saga'
    fit=ridge_regression(x,y,1._dp,options,w)
    call require(maxval(abs(fit%coefficients(1,:)-expected))<1.e-9_dp,'SAG/SAGA weighted analytic solution')
    call require(fit%converged(1),'SAG/SAGA convergence diagnostic')
    other=ridge_regression(csr_from_dense(x),y,1._dp,options,w)
    call require(maxval(abs(fit%coefficients-other%coefficients))<1.e-10_dp,'SAG/SAGA CSR analytical solution')
  end do
  options%solver='auto';options%return_intercept=.true.
  x=0
  fit=ridge_regression(x,[3._dp,3._dp,3._dp,3._dp],1._dp,options)
  call require(fit%solver=='sag'.and.fit%n_iter(1)==1,'auto intercept and coefficient-only stopping')
  call require(fit%intercept(1)/=0,'direct fitted intercept')
  options%return_intercept=.false.;options%solver='sag';options%max_iter=0
  x(1,1)=1
  fit=ridge_regression(x,y,1._dp,options)
  call require(fit%n_iter(1)==1.and.all(fit%coefficients==0),'source zero-epoch count')
  call require(.not.fit%converged(1),'zero-epoch native status')
  options%max_iter=1
  large=csr_from_arrays(2,100000,[1,2,3],[100000,1],[1._dp,2._dp])
  fit=ridge_regression(large,[1._dp,2._dp],1._dp,options)
  call require(size(fit%coefficients,2)==100000,'large CSR stochastic ridge')
  print *,'stochastic ridge tests passed'
end program
