program test_polynomial_count_sketch
  use sklearn_polynomial_count_sketch, only: polynomial_count_sketch
  use sklearn_count_sketch_fft, only: count_sketch_fft_plan
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(polynomial_count_sketch) :: model,copy
  type(count_sketch_fft_plan) :: plan
  integer, parameter :: lengths(*)=[1,2,3,5,7,8,15,16,17,31,32,63,64,100,127,128,257,1009]
  complex(dp), allocatable :: values(:),original(:),expected(:)
  real(dp) :: x(2,2),before(2,2),angle,pi
  real(dp), allocatable :: z(:,:),again(:,:)
  integer :: n,j,k,t
  pi=acos(-1._dp)
  do t=1,size(lengths)
    n=lengths(t); allocate(values(n),original(n),expected(n))
    original=[(cmplx(sin(real(j,dp)*.3_dp),cos(real(j,dp)*.7_dp),dp),j=1,n)]
    values=original; call plan%initialize(n); call plan%forward(values)
    if(n<=32) then
      expected=cmplx(0._dp,0._dp,dp)
      do k=0,n-1
        do j=0,n-1
          angle=-2*pi*real(j*k,dp)/n
          expected(k+1)=expected(k+1)+original(j+1)*cmplx(cos(angle),sin(angle),dp)
        end do
      end do
      call require(maxval(abs(values-expected))<5.e-11_dp,'CountSketch FFT independent direct DFT')
    end if
    call plan%inverse(values)
    call require(maxval(abs(values-original))<5.e-11_dp,'CountSketch FFT arbitrary-length round trip')
    deallocate(values,original,expected)
  end do
  x(1,:)=[2._dp,3._dp]; x(2,:)=[0._dp,0._dp]; before=x
  model%n_components=5; model%degree=2; call model%fit(x)
  model%index_hash_(1,:)=[0,1]; model%index_hash_(2,:)=[1,2]
  model%bit_hash_(1,:)=[1,-1]; model%bit_hash_(2,:)=[-1,1]
  z=model%transform(x)
  call require(maxval(abs(z(1,:)-[0._dp,-4._dp,12._dp,-9._dp,0._dp]))<1.e-12_dp, &
    'PolynomialCountSketch exact tensor enumeration')
  call require(all(z(2,:)==0),'PolynomialCountSketch zero row')
  call require(all(x==before),'PolynomialCountSketch preserves X')
  model%gamma=0; model%coef0=1; model%n_components=1
  z=model%fit_transform(x)
  call require(all(abs(z(:,1))==1),'PolynomialCountSketch constant kernel one-bin signed map')
  copy=model; again=copy%fit_transform(x)
  call require(all(z==again).and.all(model%bit_hash_==copy%bit_hash_),'PolynomialCountSketch scalar-seed repeat')
  model%gamma=1; model%coef0=-1; z=model%fit_transform(x)
  call require(all(ieee_is_nan(z)),'PolynomialCountSketch source negative-offset NaNs')
  model%coef0=0; model%degree=1; model%n_components=8; z=model%fit_transform(x)
  call require(all(shape(z)==[2,8]).and.all(shape(model%index_hash_)==[1,2]),'PolynomialCountSketch repeated shape change')
  print *, 'PolynomialCountSketch tests passed'
end program
