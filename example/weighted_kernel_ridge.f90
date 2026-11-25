program weighted_kernel_ridge
  use sklearn,only:dp,kernel_ridge,csr_from_dense
  implicit none
  type(kernel_ridge)::model
  real(dp)::x(3,2),y(3,2),weights(3)
  real(dp),allocatable::prediction(:,:)
  x(:,1)=[1._dp,0._dp,1._dp];x(:,2)=[0._dp,1._dp,1._dp]
  y(:,1)=[1._dp,2._dp,3._dp];y(:,2)=2*y(:,1);weights=[1._dp,0._dp,4._dp]
  model%kernel='rbf';model%gamma=.3_dp;model%alpha_values=[.1_dp,.5_dp]
  call model%fit(csr_from_dense(x),y,weights)
  prediction=model%predict_multi(x)
  print *,'Weighted multi-target kernel predictions:'
  print *,prediction
end program
