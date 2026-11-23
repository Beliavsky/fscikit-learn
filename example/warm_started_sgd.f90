program warm_started_sgd
  use sklearn, only: dp,sgd_regressor,sgd_classifier
  implicit none
  type(sgd_regressor) :: regression
  type(sgd_classifier) :: classification
  real(dp) :: x(30,2),y(30)
  real(dp), allocatable :: logs(:,:)
  integer :: i,labels(30)
  do i=1,30
    x(i,:)=[sin(real(i,dp)),cos(.3_dp*i)]
    y(i)=2*x(i,1)-x(i,2)+.3_dp;labels(i)=merge(1,0,y(i)>.3_dp)
  end do
  regression%max_iter=5;regression%tol=-1;regression%warm_start=.true.
  call regression%fit(x,y,coef_init=[1._dp,-.5_dp],intercept_init=.1_dp)
  print *, 'First regression R2:',regression%score(x,y)
  call regression%fit(x,y)
  print *, 'Warm regression R2:',regression%score(x,y)
  print *, 'Restarted sample clock:',regression%t_
  classification%loss='huber';classification%epsilon=.2_dp;classification%warm_start=.true.
  call classification%fit(x,labels,coef_init=reshape([.1_dp,-.1_dp],[1,2]),intercept_init=[0._dp])
  call classification%fit(x,labels)
  print *, 'Huber classification accuracy:',classification%score(x,labels)
  classification%warm_start=.false.;classification%loss='log_loss';call classification%fit(x,labels)
  logs=classification%predict_log_proba(x)
  print *, 'Log probability shape:',shape(logs)
end program
