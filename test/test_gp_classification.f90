program test_gp_classification
  use sklearn, only: dp,gaussian_process_classifier
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(gaussian_process_classifier) :: model
  real(dp) :: pair(2,1),triple(3,1),x(6,1),far(1,1),expected
  real(dp), allocatable :: p(:,:),first(:,:)
  integer :: i
  pair=0; model%max_iter_predict=1; call model%fit(pair,[-3,9])
  expected=-2*log(2._dp)-.5_dp*log(1.5_dp)
  call require(abs(model%log_marginal_likelihood()-expected)<1.e-13_dp,'analytic one-step Laplace evidence')
  call require(all(model%pi_==.5_dp).and.all(model%posterior_mode_==0),'symmetric duplicate-point posterior')
  call require(model%n_iter_(1)==1.and..not.model%converged_(1),'iteration cap exposes incomplete mode fit')
  p=model%predict_proba(pair)
  call require(maxval(abs(p-.5_dp))<1.e-7_dp.and.all(model%predict(pair)==-3),'binary zero-mean tie chooses lower label')
  model%max_iter_predict=100; call model%fit(pair,[-3,9])
  call require(model%converged_(1).and.model%n_iter_(1)==2,'evidence convergence for stationary mode')
  triple=0; far=1000; call model%fit(triple,[-4,7,21]); p=model%predict_proba(triple)
  call require(maxval(abs(p-1._dp/3))<1.e-11_dp.and.all(model%predict(far)==21),'OVR normalization and last-class tie')
  call require(abs(model%log_marginal_likelihood()-sum(model%log_marginal_likelihoods_)/3)<1.e-14_dp, &
    'multiclass evidence is the mean binary evidence')
  do i=1,6
    x(i,1)=real(i-4,dp)
  end do
  model%amplitude=2; call model%fit(x,[-3,-3,-3,9,9,9]); p=model%predict_proba(x); first=p
  call require(model%score(x,[-3,-3,-3,9,9,9])==1,'separable training predictions')
  call require(all(ieee_is_finite(p)).and.all(p>=0).and.all(p<=1),'finite bounded probabilities')
  call require(maxval(abs(sum(p,dim=2)-1))<1.e-14_dp,'binary probabilities sum to one')
  model%amplitude=0; model%length_scale=0; p=model%predict_proba(x)
  call require(maxval(abs(first-p))<1.e-14_dp,'predictions retain fitted kernel parameters')
  model%amplitude=2; model%length_scale=1; call model%fit(x,[-3,-3,-3,9,9,9]); p=model%predict_proba(x)
  call require(maxval(abs(first-p))<1.e-14_dp,'cold refit is deterministic')
  far=1000; p=model%predict_proba(far)
  call require(all(ieee_is_finite(p)).and.maxval(abs(p-.5_dp))<1.e-7_dp,'far-field reversion to prior probability')
  print *, 'Gaussian process classification tests passed'
end program
