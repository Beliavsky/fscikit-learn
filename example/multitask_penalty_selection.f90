program multitask_penalty_selection
  use sklearn, only: dp,multi_task_elastic_net_cv
  implicit none
  type(multi_task_elastic_net_cv) :: model
  real(dp) :: x(30,4),y(30,2)
  real(dp), allocatable :: prediction(:,:)
  integer :: i
  do i=1,30
    x(i,:)=[sin(.3_dp*i),cos(.5_dp*i),sin(.7_dp*i),cos(.9_dp*i)]
    y(i,1)=1+2*x(i,1)-x(i,2)+.05_dp*sin(real(i,dp))
    y(i,2)=-2+3*x(i,1)+.5_dp*x(i,2)+.05_dp*cos(real(i,dp))
  end do
  model%l1_ratios=[.2_dp,.7_dp,1._dp]; model%n_alphas=15; model%cv=3
  model%selection='random'; model%random_state=42; model%tol=1.e-7_dp; model%max_iter=3000
  call model%fit(x,y); prediction=model%predict(x)
  print *, 'Selected alpha/l1 ratio: ',model%alpha_,model%l1_ratio_
  print *, 'Mean validation MSE: ',model%best_mse_
  print *, 'Training mean R2: ',model%score(x,y)
  print *, 'Prediction shape: ',shape(prediction)
  print *, 'Refit iterations/dual gap: ',model%n_iter_,model%dual_gap_
end program
