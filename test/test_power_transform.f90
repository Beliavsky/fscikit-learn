program test_power_transform
  use sklearn
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_nan
  implicit none
  type(power_transformer) :: model
  real(dp) :: x(80,3),positive(80,2),missing(80,3),t
  real(dp), allocatable :: z(:,:),back(:,:),again(:,:)
  integer :: i,j
  do i=1,80
    t=real(i,dp)/80
    x(i,:)=[exp(4*t)-5,sin(13*t)*t,3._dp]
    positive(i,:)=[exp(4*t),0.1_dp+t**3]
  end do
  z=model%fit_transform(x); back=model%inverse_transform(z)
  call require(maxval(abs(x-back))<1.e-10_dp,'Yeo-Johnson inverse roundtrip')
  call require(maxval(abs(sum(z,dim=1)))<1.e-10_dp,'power standardized means')
  call require(maxval(abs(sum(z(:,1:2)**2,dim=1)/80-1))<1.e-12_dp,'power standardized variances')
  call require(model%lambdas_(3)==1.and.maxval(abs(z(:,3)))<1.e-12_dp,'constant Yeo-Johnson identity')
  model%standardize=.false.; z=model%fit_transform(x)
  again=model%transform(x); call require(maxval(abs(z-again))==0,'fit_transform consistency')
  do j=0,2
    model%lambdas_=real(j,dp)
    z=model%transform(x); back=model%inverse_transform(z)
    call require(maxval(abs(x-back))<1.e-10_dp,'Yeo-Johnson limiting powers')
  end do
  model%method='box-cox'; model%standardize=.true.
  z=model%fit_transform(positive); back=model%inverse_transform(z)
  call require(maxval(abs(positive-back))<1.e-10_dp,'Box-Cox inverse roundtrip')
  call require(maxval(abs(sum(z,dim=1)))<1.e-10_dp,'Box-Cox standardized mean')
  model%standardize=.false.; call model%fit(positive); model%lambdas_=0
  z=model%transform(positive)
  call require(maxval(abs(z-log(positive)))<1.e-12_dp,'Box-Cox logarithm at zero power')
  back=model%inverse_transform(z)
  call require(maxval(abs(back-positive))<1.e-10_dp,'Box-Cox zero power inverse')
  model%method='yeo-johnson'; model%standardize=.true.
  missing=x; missing(4,1)=ieee_value(0._dp,ieee_quiet_nan)
  missing(7,2)=missing(4,1)
  z=model%fit_transform(missing); back=model%inverse_transform(z)
  call require(ieee_is_nan(z(4,1)).and.ieee_is_nan(back(7,2)),'power preserves missing values')
  do j=1,3
    do i=1,80
      if(ieee_is_nan(missing(i,j))) cycle
      call require(abs(missing(i,j)-back(i,j))<1.e-10_dp,'observed value roundtrip')
    end do
  end do
  ! Fitted settings stay attached to the learned state until another fit.
  model%method='box-cox'; model%standardize=.false.; again=model%transform(missing)
  do j=1,3
    do i=1,80
      if(ieee_is_nan(z(i,j))) cycle
      call require(z(i,j)==again(i,j),'power fitted state retained')
    end do
  end do
  print *, 'PowerTransformer tests passed'
end program
