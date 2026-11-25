program ridge_solvers
  use sklearn,only:dp,ridge_regression,ridge_solver_options,ridge_solution
  implicit none
  type(ridge_solution)::fit
  type(ridge_solver_options)::options
  real(dp)::x(3,2),y(3,2)
  x(:,1)=[1._dp,2._dp,3._dp];x(:,2)=[0._dp,1._dp,0._dp]
  y(:,1)=[2._dp,4._dp,6._dp];y(:,2)=[1._dp,0._dp,1._dp]
  options%solver='lsqr';options%tol=1.e-10_dp
  fit=ridge_regression(x,y,[.1_dp,1._dp],options)
  print *,'Target coefficients:',fit%coefficients
  print *,'LSQR iterations:',fit%n_iter
end program
