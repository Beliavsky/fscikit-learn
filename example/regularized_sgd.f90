program regularized_sgd
  use sklearn, only: dp,sgd_regressor,sgd_classifier
  implicit none
  type(sgd_regressor) :: regression
  type(sgd_classifier) :: classification
  real(dp) :: x(40,3),y(40)
  integer :: i,labels(40)
  do i=1,40
    x(i,:)=[sin(real(i,dp)),cos(.3_dp*i),sin(.7_dp*i)]
    y(i)=2*x(i,1)-x(i,2)+.3_dp;labels(i)=merge(1,0,y(i)>.3_dp)
  end do
  regression%penalty='elasticnet';regression%l1_ratio=.4_dp;regression%alpha=.01_dp
  regression%learning_rate='adaptive';regression%eta0=.05_dp;regression%shuffle=.true.
  call regression%fit(x,y)
  print *, 'Elastic-net regression R2:',regression%score(x,y)
  classification%loss='log_loss';classification%penalty='l1';classification%alpha=.01_dp
  classification%learning_rate='optimal';classification%eta0=0;classification%shuffle=.true.
  call classification%fit(x,labels)
  print *, 'L1 classification accuracy:',classification%score(x,labels)
end program
