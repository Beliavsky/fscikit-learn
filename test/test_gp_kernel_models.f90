program test_gp_kernel_models
  use sklearn_gp_kernels
  use sklearn_gaussian_process, only: gaussian_process_regressor,gaussian_process_classifier
  use sklearn_common, only: dp,require
  implicit none
  type(gaussian_process_regressor) :: reg
  type(gaussian_process_classifier) :: classifier
  real(dp) :: x(20,1),y(20),q(4,1),before(4),std(4)
  real(dp), allocatable :: probability(:,:),snapshot(:,:),cov(:,:)
  integer :: i,labels(20)
  do i=1,20
    x(i,1)=-2+4*real(i-1,dp)/19
  end do
  y=sin(x(:,1));labels=merge(9,-4,x(:,1)>0);q(:,1)=[-1.5_dp,-.3_dp,.3_dp,1.5_dp]
  reg%kernel=1.3_dp*matern_covariance(.8_dp,1.5_dp)+white_covariance(.05_dp);reg%alpha=.01_dp
  call require(all(reg%predict(q)==0),'unfitted GPR predicts zero prior means')
  std=reg%predict_std(q);call require(maxval(abs(std-sqrt(1.35_dp)))<1.e-14_dp,'GPR kernel prior standard deviation')
  call reg%fit(x,y);before=reg%predict(q);std=reg%predict_std(q);cov=reg%predict_cov(q)
  do i=1,4
    call require(abs(std(i)**2-cov(i,i))<1.e-12_dp,'GPR posterior standard deviation and covariance agree')
  end do
  reg%kernel=dot_product_covariance();reg%amplitude=99;reg%noise_level=99;reg%length_scale=99
  call require(maxval(abs(reg%predict(q)-before))==0,'GPR fitted kernel snapshot is independent of input edits')
  call reg%fit(x,y)
  call require(maxval(abs(reg%predict(q)-before))>1.e-4_dp,'GPR refit uses changed input kernel')
  classifier%kernel=1.3_dp*matern_covariance(.8_dp,1.5_dp)+white_covariance(.05_dp)
  call classifier%fit(x,labels);probability=classifier%predict_proba(q)
  call require(all(classifier%predict(q)==[-4,-4,9,9]),'custom-kernel GPC predictions')
  classifier%kernel=constant_covariance();classifier%amplitude=99;classifier%length_scale=99
  snapshot=classifier%predict_proba(q)
  call require(maxval(abs(snapshot-probability))==0,'GPC fitted kernel snapshot is independent of input edits')
  call classifier%fit(x,labels);probability=classifier%predict_proba(q)
  call require(maxval(abs(probability-.5_dp))<1.e-7_dp,'GPC refit clears the previous kernel fit')
  print *, 'Gaussian process kernel integration: PASS'
end program
