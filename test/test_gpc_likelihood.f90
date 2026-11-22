program test_gpc_likelihood
  use sklearn_gp_classification, only: gaussian_process_classifier
  use sklearn_gp_kernels
  use sklearn_common, only: dp,require
  implicit none
  type(gaussian_process_classifier) :: model
  real(dp) :: x(18,1),value,up,down,cached
  real(dp), allocatable :: theta(:),point(:),gradient(:),probability(:,:),after(:,:)
  integer :: i,j,labels(18)
  do i=1,18
    x(i,1)=-1.5_dp+3*real(i-1,dp)/17
  end do
  labels=merge(9,-4,x(:,1)>0);model%kernel=1.2_dp*rbf_covariance(.8_dp)+white_covariance(.05_dp)
  call model%fit(x,labels);cached=read_cached(model);theta=model%kernels_(1)%theta()
  probability=model%predict_proba(x);value=model%log_marginal_likelihood(theta,gradient)
  call require(abs(value-cached)<1.e-12_dp,'GPC explicit and cached likelihood')
  do j=1,size(theta)
    point=theta;point(j)=point(j)+1.e-5_dp;up=model%log_marginal_likelihood(point)
    point=theta;point(j)=point(j)-1.e-5_dp;down=model%log_marginal_likelihood(point)
    call require(abs((up-down)/2.e-5_dp-gradient(j))<2.e-5_dp,'Laplace likelihood analytic gradient')
  end do
  after=model%predict_proba(x)
  call require(all(after==probability),'GPC cloned likelihood calls preserve fitted predictions')
  call require(all(model%kernels_(1)%theta()==theta),'GPC cloned likelihood calls preserve fitted parameters')
  model%max_iter_predict=1;model%warm_start=.true.
  value=model%log_marginal_likelihood(theta)
  call require(abs(value-cached)<1.e-12_dp,'GPC fitted mode controls are snapshots of fit settings')
  value=model%log_marginal_likelihood(theta+.02_dp,clone_kernel=.false.)
  call require(maxval(abs(model%kernels_(1)%theta()-theta-.02_dp))<1.e-14_dp,'GPC mutable kernel evaluation')
  call require(read_cached(model)==cached,'GPC likelihood evaluations preserve cached fit likelihood')
  print *, 'GPC Laplace likelihood and gradients: PASS'
contains
  real(dp) function read_cached(estimator) result(value)
    type(gaussian_process_classifier), intent(in) :: estimator
    value=estimator%log_marginal_likelihood()
  end function
end program
