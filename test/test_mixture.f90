program test_mixture
  use sklearn_common
  use sklearn_mixture
  use sklearn_metrics, only: adjusted_rand_score
  implicit none
  type(gaussian_mixture) :: gm
  real(dp) :: x(40,2)
  integer :: i,y(40),kind
  character(9) :: types(4)=['full     ','tied     ','diag     ','spherical']
  real(dp), allocatable :: prob(:,:)
  do i=1,40
    y(i)=merge(0,1,i<=20)
    x(i,:)=[sin(real(i,dp)),cos(2._dp*i)]+10*y(i)
  end do
  gm%n_components=2
  do kind=1,4
    gm%covariance_type=types(kind); call gm%fit(x)
    call require(gm%converged_,'Gaussian mixture convergence')
    call require(adjusted_rand_score(y,gm%predict(x))==1,'Gaussian mixture partition')
    prob=gm%predict_proba(x)
    call require(maxval(abs(sum(prob,dim=2)-1))<1.e-12_dp,'Gaussian mixture probabilities')
    call require(gm%bic(x)>gm%aic(x),'mixture information criteria')
  end do
  gm%n_components=1; call gm%fit(x)
  call require(abs(sum(gm%weights_)-1)<1.e-12_dp,'mixture refit')
  print *, 'Gaussian mixtures: PASS'
end program
