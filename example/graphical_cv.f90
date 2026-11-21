program graphical_cv_example
  use sklearn, only: dp,graphical_lasso_cv
  implicit none
  type(graphical_lasso_cv) :: model
  real(dp) :: x(80,3),t
  integer :: i
  do i=1,80
    t=real(i,dp)/80
    x(i,:)=[sin(17*t),.6_dp*sin(17*t)+cos(23*t),sin(31*t)+.2_dp*cos(23*t)]
  end do
  model%cv=4; model%n_refinements=3
  model%max_iter=500; model%tol=1.e-9_dp; model%enet_tol=1.e-11_dp
  call model%fit(x)
  print *, 'Selected alpha:',model%alpha_
  print *, 'Training log likelihood:',model%score(x)
  print *, 'Reported grid size:',size(model%cv_results_%alphas)
  print *, 'Final inverse covariance row:',model%precision_(1,:)
end program
