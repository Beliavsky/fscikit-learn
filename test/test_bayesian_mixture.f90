program test_bayesian_mixture
  use sklearn, only: dp,bayesian_gaussian_mixture
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(bayesian_gaussian_mixture) :: model
  real(dp) :: x(4,2),identity(2,2),expected,previous,mu(2),cov(2,2)
  real(dp), allocatable :: prob(:,:),draws(:,:),again(:,:),centered(:,:)
  integer, allocatable :: labels(:),draw_labels(:),again_labels(:)
  integer :: mode,j
  character(16) :: modes(4)=[character(16)::'full','tied','diag','spherical']
  x(:,1)=[-1._dp,-1._dp,1._dp,1._dp]; x(:,2)=[-1._dp,1._dp,-1._dp,1._dp]
  identity=0; identity(1,1)=1; identity(2,2)=1
  expected=-.2_dp-.5_dp*(2*log(2*acos(-1._dp))+2.25_dp)+log(9._dp/8)+ &
    .5_dp*(25._dp/6-2*.5772156649015328606_dp-.4_dp-2*log(6._dp))
  do mode=1,4
    model=bayesian_gaussian_mixture(); model%covariance_type=modes(mode); model%reg_covar=0; model%tol=1.e-10_dp
    call model%fit(x)
    call require(maxval(abs(model%means_))<1.e-13_dp,'Bayesian mixture analytic posterior mean')
    call require(maxval(abs(model%covariances_(:,:,1)-identity*8/9))<1.e-12_dp,'Bayesian mixture analytic covariance')
    call require(abs(model%mean_precision_(1)-5)<1.e-12_dp,'Bayesian mixture analytic mean precision')
    call require(abs(model%degrees_of_freedom_(1)-6)<1.e-12_dp,'Bayesian mixture analytic degrees of freedom')
    call require(abs(model%weight_concentration_(1,1)-5)<1.e-12_dp,'Bayesian mixture analytic stick parameter')
    call require(abs(model%score(x)-expected)<1.e-12_dp,'Bayesian mixture expected-log variational score')
    prob=model%predict_proba(x); labels=model%predict(x)
    call require(all(prob==1).and.all(labels==0),'Bayesian mixture one-component responsibilities')
    call require(model%converged_.and.model%n_iter_==2,'Bayesian mixture analytic convergence')
    call require(all(model%lower_bounds_(2:)>=model%lower_bounds_(:size(model%lower_bounds_)-1)-1.e-11_dp), &
      'Bayesian mixture nondecreasing variational bounds')
    call require(maxval(abs(matmul(model%covariances_(:,:,1),model%precisions_(:,:,1))-identity))<1.e-12_dp, &
      'Bayesian mixture inverse covariance certificate')
    previous=model%lower_bound_; model%warm_start=.true.; call model%fit(x)
    call require(model%n_iter_==1.and.abs(model%lower_bound_-previous)<1.e-12_dp,'Bayesian mixture warm-start continuation')
    model%warm_start=.false.; model%weight_concentration_prior_type='dirichlet_distribution'; call model%fit(x)
    call require(abs(model%score(x)-expected-.2_dp)<1.e-12_dp,'Bayesian mixture finite-Dirichlet expected weight')
    model%mean_prior=[2._dp,-1._dp]; model%mean_precision_prior=2; model%degrees_of_freedom_prior=3
    model%covariance_prior=2*identity; call model%fit(x)
    call require(maxval(abs(model%means_(1,:)-[2._dp/3,-1._dp/3]))<1.e-12_dp,'Bayesian mixture custom-prior mean')
    cov=0; cov(1,1)=34._dp/21; cov(2,2)=22._dp/21
    if(mode<=2) then
      cov(1,2)=-8._dp/21; cov(2,1)=-8._dp/21
    else if(mode==4) then
      cov=identity*4/3
    end if
    call require(maxval(abs(model%covariances_(:,:,1)-cov))<1.e-12_dp,'Bayesian mixture custom-prior covariance')
  end do
  model=bayesian_gaussian_mixture(); model%reg_covar=0; call model%fit(x)
  call model%sample(20000,draws,draw_labels); call model%sample(20000,again,again_labels)
  call require(all(draws==again).and.all(draw_labels==again_labels),'Bayesian mixture reproducible native samples')
  call require(all(draw_labels==0),'Bayesian mixture sample component IDs')
  mu=sum(draws,dim=1)/size(draws,1); centered=draws-spread(mu,1,size(draws,1))
  cov=matmul(transpose(centered),centered)/size(draws,1)
  call require(maxval(abs(mu))<.04_dp.and.maxval(abs(cov-identity*8/9))<.05_dp,'Bayesian mixture sample moments')
  model=bayesian_gaussian_mixture(); model%max_iter=0; call model%fit(x)
  call require(model%n_iter_==0.and..not.model%converged_.and..not.ieee_is_finite(model%lower_bound_), &
    'Bayesian mixture initialization-only state')
  call require(size(model%lower_bounds_)==0,'Bayesian mixture empty initialization-only history')
  model%max_iter=huge(1); model%tol=1.e9_dp; call model%fit(x)
  call require(model%n_iter_==2.and.size(model%lower_bounds_)==2,'Bayesian mixture bounded history for huge budgets')
  model=bayesian_gaussian_mixture(); call model%fit(x)
  model%n_components=2; model%covariance_type='diag'; labels=model%fit_predict(x(:,:1))
  call require(model%n_features_in_==1.and.all(shape(model%means_)==[2,1]),'Bayesian mixture refit dimensions')
  call require(all(labels>=0).and.all(labels<2),'Bayesian mixture fit_predict labels')
  print *, 'BayesianGaussianMixture tests passed'
end program
