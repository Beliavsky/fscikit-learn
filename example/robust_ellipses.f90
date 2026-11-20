program robust_ellipses
  use sklearn, only: dp,min_cov_det,elliptic_envelope
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  type(min_cov_det) :: covariance
  type(elliptic_envelope) :: detector
  type(legacy_random_state) :: rng
  real(dp) :: x(120,2),query(2,2)
  integer :: i
  call rng%seed(42)
  do i=1,120
    x(i,:)=[rng%normal(),rng%normal()]
    x(i,2)=.4_dp*x(i,1)+x(i,2)
  end do
  x(:12,:)=x(:12,:)+8
  call covariance%fit(x); call detector%fit(x)
  query(1,:)=0; query(2,:)=10
  print *, 'Robust location:',covariance%location_
  print *, 'Retained observations:',count(covariance%support_)
  print *, 'Query distances:',covariance%mahalanobis(query)
  print *, 'Inlier (+1) / outlier (-1) labels:',detector%predict(query)
end program
