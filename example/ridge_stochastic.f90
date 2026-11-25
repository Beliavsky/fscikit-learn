program ridge_stochastic
  use sklearn,only:dp,ridge_regression,ridge_solution,ridge_solver_options
  implicit none
  type(ridge_solver_options)::options
  type(ridge_solution)::fit
  real(dp)::x(4,2)
  x(:,1)=[1._dp,2._dp,3._dp,4._dp];x(:,2)=[0._dp,1._dp,0._dp,1._dp]
  options%solver='sag';options%return_intercept=.true.;options%random_state=42
  options%tol=1.e-9_dp;options%max_iter=10000
  fit=ridge_regression(x,[3._dp,5._dp,7._dp,9._dp],.1_dp,options)
  print *,'SAG coefficients:',fit%coefficients
  print *,'Intercept:',fit%intercept
  print *,'Epochs:',fit%n_iter
end program
