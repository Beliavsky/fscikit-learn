program dummy_likelihood
  use sklearn,only:dp,add_dummy_feature,log_likelihood,empirical_covariance
  implicit none
  real(dp)::x(3,2),precision(2,2)
  real(dp),allocatable::design(:,:),covariance(:,:)
  x(:,1)=[1._dp,2._dp,3._dp];x(:,2)=[0._dp,1._dp,1._dp]
  design=add_dummy_feature(x)
  covariance=empirical_covariance(x);precision=0;precision(1,1)=1;precision(2,2)=1
  print *,'Design columns:',size(design,2)
  print *,'Gaussian mean log-likelihood:',log_likelihood(covariance,precision)
end program
