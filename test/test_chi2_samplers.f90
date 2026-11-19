program test_chi2_samplers
  use sklearn
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(additive_chi2_sampler) :: additive
  type(skewed_chi2_sampler) :: skewed,again
  real(dp) :: x(6,2),target(6,6),a,b
  real(dp), allocatable :: z(:,:),other(:,:),gram(:,:),weights(:,:)
  integer :: i,j,k
  x(1,:)=[0._dp,0._dp]; x(2,:)=[1._dp,2._dp]; x(3,:)=[.2_dp,.5_dp]
  x(4,:)=[3._dp,1._dp]; x(5,:)=[.7_dp,4._dp]; x(6,:)=[1.5_dp,.8_dp]
  additive%sample_steps=1; z=additive%transform(x)
  call require(maxval(abs(z-sqrt(.8_dp*x)))<1.e-14_dp,'stateless additive first component')
  additive%sample_steps=8; additive%sample_interval=.25_dp; z=additive%fit_transform(x)
  call require(size(z,2)==30.and.all(z(1,:)==0),'additive width and zero row')
  gram=matmul(z,transpose(z)); target=0
  do i=1,6
    do j=1,6
      do k=1,2
        if(x(i,k)+x(j,k)>0) target(i,j)=target(i,j)+2*x(i,k)*x(j,k)/(x(i,k)+x(j,k))
      end do
    end do
  end do
  call require(maxval(abs(gram-target))<.03_dp,'additive positive chi2 kernel approximation')
  additive%sample_interval=0; z=additive%transform(x)
  call require(all(z==0),'zero interval feature map')
  additive%sample_interval=1000; z=additive%transform(x)
  call require(all(ieee_is_finite(z)).and.all(z(:,3:)==0),'large interval avoids cosh overflow')
  skewed%n_components=20000; skewed%random_state=42; skewed%skewedness=.7_dp
  z=skewed%fit_transform(x); gram=matmul(z,transpose(z)); target=1
  do i=1,6
    do j=1,6
      do k=1,2
        a=x(i,k)+.7_dp; b=x(j,k)+.7_dp
        target(i,j)=target(i,j)*2*sqrt(a*b)/(a+b)
      end do
    end do
  end do
  call require(maxval(abs(gram-target))<.04_dp,'skewed multiplicative kernel approximation')
  weights=skewed%random_weights_; again=skewed; call again%fit(x+10)
  call require(all(again%random_weights_==weights),'seeded fit depends on dimensions, not training values')
  call require(maxval(abs(z))<=sqrt(2._dp/20000),'random feature scaling bound')
  skewed%n_components=2; call skewed%fit(x-100)
  skewed%random_weights_=0; skewed%random_offset_=[0._dp,acos(-1._dp)]
  z=skewed%transform(x)
  call require(maxval(abs(z(:,1)-1))<1.e-14_dp.and.maxval(abs(z(:,2)+1))<1.e-14_dp,'known weights and offsets')
  skewed%skewedness=-.5_dp; other=skewed%fit_transform(x+1)
  call require(all(ieee_is_finite(other)),'negative skew accepted on valid shifted domain')
  print *, 'Chi2 sampler tests passed'
end program
