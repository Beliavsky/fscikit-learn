program test_discriminant_auto
  use sklearn_common, only: dp,require
  use sklearn_discriminant_analysis, only: linear_discriminant_analysis
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(linear_discriminant_analysis) :: model,rescaled
  real(dp) :: x(24,3),scale(3),zero(24,3)
  integer :: labels(24),i,j
  real(dp), allocatable :: expected(:,:),prob(:,:)
  do i=1,24
    labels(i)=mod(i,3)
    do j=1,3
      x(i,j)=sin(.37_dp*i*j)+labels(i)*.4_dp*j
    end do
  end do
  model%shrinkage_mode='auto';call model%fit(x,labels)
  scale=[.1_dp,2._dp,10._dp];rescaled=model;call rescaled%fit(x*spread(scale,1,24),labels)
  expected=model%covariance_(:,:,1)*spread(scale,1,3)*spread(scale,2,3)
  call require(maxval(abs(expected-rescaled%covariance_(:,:,1)))<1.e-10_dp,'auto LDA rescales class covariances')
  call require(maxval(abs(model%class_shrinkage_-rescaled%class_shrinkage_))<1.e-12_dp, &
    'auto LDA shrinkage is invariant to nonconstant feature units')
  model%solver='eigen';call model%fit(x,labels)
  call require(model%total_shrinkage_>=0.and.model%total_shrinkage_<=1,'auto eigen LDA estimates total shrinkage')
  model%shrinkage_mode='fixed';model%shrinkage=.2_dp;call model%fit(x,labels)
  call require(.not.allocated(model%class_shrinkage_).and.model%total_shrinkage_==-1,'fixed LDA clears auto diagnostics')
  model%solver='lsqr';model%shrinkage_mode='auto';zero=0;call model%fit(zero,labels)
  call require(all(model%coef_==0).and.all(model%precision_==0),'auto LSQR accepts zero covariance by least squares')
  prob=model%predict_proba(zero)
  call require(maxval(abs(prob-1._dp/3))<1.e-12_dp,'zero covariance auto LDA uses class priors')
  call require(all(.not.ieee_is_finite(model%logdet_)),'zero covariance auto LDA has negative infinite log determinant')
  print *, 'Automatic discriminant covariance shrinkage: PASS'
end program
