! PolynomialCountSketch TensorSketch contracts: sklearn 1.6.1
! kernel_approximation.py, BSD-3-Clause. Retained notices in LICENSES/.
module sklearn_polynomial_count_sketch
  use sklearn_common, only: dp,require,check_x
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_count_sketch_fft, only: count_sketch_fft_plan
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: polynomial_count_sketch
  type :: polynomial_count_sketch
    real(dp) :: gamma=1,coef0=0
    integer :: degree=2,n_components=100,random_state=0,n_features_in_=0,n_features_out_=0
    integer, allocatable :: index_hash_(:,:),bit_hash_(:,:)
  contains
    procedure :: fit=>sketch_fit
    procedure :: transform=>sketch_transform
    procedure :: fit_transform=>sketch_fit_transform
  end type
contains
  subroutine check_parameters(self)
    class(polynomial_count_sketch), intent(in) :: self
    call require(ieee_is_finite(self%gamma),'PolynomialCountSketch gamma must be finite')
    call require(self%gamma>=0,'PolynomialCountSketch gamma must be nonnegative')
    call require(ieee_is_finite(self%coef0),'PolynomialCountSketch coef0 must be finite')
    call require(self%degree>0,'PolynomialCountSketch degree must be positive')
    call require(self%n_components>0,'PolynomialCountSketch n_components must be positive')
  end subroutine
  subroutine sketch_fit(self,x)
    class(polynomial_count_sketch), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(legacy_random_state) :: rng
    integer :: p,d,j,ierr
    call check_x(x); call check_parameters(self)
    call require(self%random_state>=0,'PolynomialCountSketch seed must be nonnegative')
    p=size(x,2)
    if(self%coef0/=0) then
      call require(p<huge(p),'PolynomialCountSketch augmented feature count overflow'); p=p+1
    end if
    call require(int(p,int64)*self%degree<=huge(p),'PolynomialCountSketch hash array size overflow')
    if(allocated(self%index_hash_)) deallocate(self%index_hash_,self%bit_hash_)
    allocate(self%index_hash_(self%degree,p),self%bit_hash_(self%degree,p),stat=ierr)
    call require(ierr==0,'PolynomialCountSketch hash allocation failed')
    call rng%seed(self%random_state)
    do d=1,self%degree
      do j=1,p
        self%index_hash_(d,j)=rng%integer_index(self%n_components)-1
      end do
    end do
    do d=1,self%degree
      do j=1,p
        self%bit_hash_(d,j)=2*(rng%integer_index(2)-1)-1
      end do
    end do
    self%n_features_in_=size(x,2); self%n_features_out_=self%n_components
  end subroutine
  function sketch_transform(self,x) result(z)
    class(polynomial_count_sketch), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    type(count_sketch_fft_plan) :: plan
    complex(dp), allocatable :: spectrum(:),product(:)
    real(dp), allocatable :: sketches(:,:)
    real(dp) :: scale,value,constant
    integer :: n,p,augmented,i,j,d,bucket
    call require(allocated(self%index_hash_).and.allocated(self%bit_hash_),'PolynomialCountSketch is not fitted')
    call check_x(x); call check_parameters(self)
    call require(size(x,2)==self%n_features_in_,'PolynomialCountSketch feature count mismatch')
    n=size(x,1); p=size(x,2); augmented=p+merge(1,0,self%coef0/=0)
    call require(augmented==size(self%index_hash_,2),'PolynomialCountSketch augmented feature count mismatch')
    call require(all(shape(self%bit_hash_)==shape(self%index_hash_)),'PolynomialCountSketch hash shape mismatch')
    call require(self%degree<=size(self%index_hash_,1),'PolynomialCountSketch degree exceeds fitted hash rows')
    call require(all(self%index_hash_(:self%degree,:)>=0).and. &
      all(self%index_hash_(:self%degree,:)<self%n_components),'PolynomialCountSketch hash bucket outside output range')
    if(self%coef0<0) then
      allocate(z(n,self%n_components))
      ! Source accepts negative coef0 but sqrt(coef0) makes every FFT result NaN.
      z=ieee_value(0._dp,ieee_quiet_nan); return
    end if
    scale=sqrt(self%gamma); constant=sqrt(self%coef0)
    call plan%initialize(self%n_components)
    allocate(z(n,self%n_components))
    allocate(sketches(self%n_components,self%degree),spectrum(self%n_components),product(self%n_components))
    do i=1,n
      sketches=0
      do j=1,augmented
        value=constant
        if(j<=p) value=scale*x(i,j)
        do d=1,self%degree
          bucket=self%index_hash_(d,j)+1
          sketches(bucket,d)=sketches(bucket,d)+real(self%bit_hash_(d,j),dp)*value
        end do
      end do
      product=cmplx(1._dp,0._dp,dp)
      do d=1,self%degree
        spectrum=cmplx(sketches(:,d),0._dp,dp); call plan%forward(spectrum); product=product*spectrum
      end do
      call plan%inverse(product); z(i,:)=real(product,dp)
    end do
  end function
  function sketch_fit_transform(self,x) result(z)
    class(polynomial_count_sketch), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
end module
