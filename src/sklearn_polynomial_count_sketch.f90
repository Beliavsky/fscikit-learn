! PolynomialCountSketch TensorSketch contracts: sklearn 1.6.1
! kernel_approximation.py, BSD-3-Clause. Retained notices in LICENSES/.
module sklearn_polynomial_count_sketch
  use sklearn_common, only: dp,require,check_x
  use sklearn_sparse, only: csr_matrix
  use sklearn_stateless_preprocessing, only: sorted_sparse
  use sklearn_ieee_math, only: ieee_difference,ieee_product
  use, intrinsic :: ieee_exceptions, only: ieee_status_type,ieee_get_status,ieee_set_status, &
    ieee_set_halting_mode,ieee_invalid,ieee_divide_by_zero,ieee_overflow
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_count_sketch_fft, only: count_sketch_fft_plan
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan,ieee_fma
  implicit none
  private
  public :: polynomial_count_sketch
  type :: polynomial_count_sketch
    real(dp) :: gamma=1,coef0=0
    integer :: degree=2,n_components=100,random_state=0,n_features_in_=0,n_features_out_=0
    integer, allocatable :: index_hash_(:,:),bit_hash_(:,:)
  contains
    procedure,private :: fit_dense=>sketch_fit
    procedure,private :: fit_csr=>sketch_fit_csr
    generic :: fit=>fit_dense,fit_csr
    procedure,private :: transform_dense=>sketch_transform
    procedure,private :: transform_csr=>sketch_transform_csr
    generic :: transform=>transform_dense,transform_csr
    procedure,private :: fit_transform_dense=>sketch_fit_transform
    procedure,private :: fit_transform_csr=>sketch_fit_transform_csr
    generic :: fit_transform=>fit_transform_dense,fit_transform_csr
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
    call check_x(x); call fit_dimensions(self,size(x,2))
  end subroutine
  subroutine check_csr(x)
    type(csr_matrix),intent(in)::x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'PolynomialCountSketch empty CSR input')
    call require(all(ieee_is_finite(x%data)),'PolynomialCountSketch nonfinite CSR data')
  end subroutine
  subroutine sketch_fit_csr(self,x)
    class(polynomial_count_sketch),intent(inout)::self
    type(csr_matrix),intent(in)::x
    call check_csr(x);call fit_dimensions(self,x%n_cols)
  end subroutine
  subroutine fit_dimensions(self,width)
    class(polynomial_count_sketch),intent(inout)::self
    integer,intent(in)::width
    type(legacy_random_state) :: rng
    integer :: p,d,j,ierr
    call check_parameters(self)
    call require(self%random_state>=0,'PolynomialCountSketch seed must be nonnegative')
    p=width
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
    self%n_features_in_=width; self%n_features_out_=self%n_components
  end subroutine
  subroutine check_fitted(self)
    class(polynomial_count_sketch),intent(in)::self
    call require(allocated(self%index_hash_).and.allocated(self%bit_hash_),'PolynomialCountSketch is not fitted')
  end subroutine
  function sketch_transform(self,x) result(z)
    class(polynomial_count_sketch), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call check_fitted(self);call check_x(x);z=sketch_values(self,size(x,1),size(x,2),x=x)
  end function
  function sketch_transform_csr(self,x) result(z)
    class(polynomial_count_sketch),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:)
    call check_fitted(self);call check_csr(x);z=sketch_values(self,x%n_rows,x%n_cols,sparse_x=x)
  end function
  function sketch_values(self,n,p,x,sparse_x) result(z)
    class(polynomial_count_sketch),intent(in)::self
    integer,intent(in)::n,p
    real(dp),intent(in),optional::x(:,:)
    type(csr_matrix),intent(in),optional::sparse_x
    real(dp),allocatable::z(:,:)
    type(csr_matrix)::ordered
    type(count_sketch_fft_plan) :: plan
    type(ieee_status_type)::saved_status
    complex(dp), allocatable :: spectrum(:),product(:)
    real(dp), allocatable :: sketches(:,:)
    real(dp) :: scale,value,constant,term
    integer :: augmented,i,j,d,bucket,entry,last,next
    call check_parameters(self)
    call require(p==self%n_features_in_,'PolynomialCountSketch feature count mismatch')
    call require(p<=huge(p)-merge(1,0,self%coef0/=0),'PolynomialCountSketch augmented feature count overflow')
    augmented=p+merge(1,0,self%coef0/=0)
    call require(augmented==size(self%index_hash_,2),'PolynomialCountSketch augmented feature count mismatch')
    call require(all(shape(self%bit_hash_)==shape(self%index_hash_)),'PolynomialCountSketch hash shape mismatch')
    call require(self%degree<=size(self%index_hash_,1),'PolynomialCountSketch degree exceeds fitted hash rows')
    call require(all(self%index_hash_(:self%degree,:)>=0).and. &
      all(self%index_hash_(:self%degree,:)<self%n_components),'PolynomialCountSketch hash bucket outside output range')
    call require(int(n,int64)*self%n_components<=huge(1),'PolynomialCountSketch output size overflow')
    call require(int(self%n_components,int64)*self%degree<=huge(1),'PolynomialCountSketch workspace size overflow')
    if(self%coef0<0) then
      allocate(z(n,self%n_components))
      ! Source accepts negative coef0 but sqrt(coef0) makes every FFT result NaN.
      z=ieee_value(0._dp,ieee_quiet_nan); return
    end if
    scale=sqrt(self%gamma); constant=sqrt(self%coef0)
    call plan%initialize(self%n_components)
    allocate(z(n,self%n_components))
    allocate(sketches(self%n_components,self%degree),spectrum(self%n_components),product(self%n_components))
    if(present(sparse_x)) ordered=sorted_sparse(sparse_x,.false.)
    do i=1,n
      sketches=0
      if(present(sparse_x)) then
        entry=ordered%indptr(i);last=ordered%indptr(i+1)
        do while(entry<last)
          j=ordered%indices(entry);next=entry+1
          do while(next<last)
            if(ordered%indices(next)/=j) exit
            next=next+1
          end do
          do d=1,self%degree
            ! Source scales/signs stored CSC entries before summing duplicate
            ! feature coordinates; only then does it add each column to a bin.
            value=0
            do bucket=entry,next-1
              term=ieee_product(scale,ordered%data(bucket))
              term=ieee_product(real(self%bit_hash_(d,j),dp),term)
              value=ieee_difference(value,-term)
            end do
            bucket=self%index_hash_(d,j)+1
            sketches(bucket,d)=ieee_difference(sketches(bucket,d),-value)
          end do
          entry=next
        end do
      else
        do j=1,p
          value=ieee_product(scale,x(i,j))
          do d=1,self%degree
            bucket=self%index_hash_(d,j)+1
            term=ieee_product(real(self%bit_hash_(d,j),dp),value)
            sketches(bucket,d)=ieee_difference(sketches(bucket,d),-term)
          end do
        end do
      end if
      if(augmented>p) then
        do d=1,self%degree
          bucket=self%index_hash_(d,augmented)+1
          term=ieee_product(real(self%bit_hash_(d,augmented),dp),constant)
          sketches(bucket,d)=ieee_difference(sketches(bucket,d),-term)
        end do
      end if
      ! Like scipy.fft, this boundary returns IEEE results for finite inputs
      ! whose count sketches or Fourier products overflow. Restore the caller's
      ! exception flags and halting modes after each complete transform row.
      call ieee_get_status(saved_status)
      call ieee_set_halting_mode(ieee_invalid,.false.)
      call ieee_set_halting_mode(ieee_divide_by_zero,.false.)
      call ieee_set_halting_mode(ieee_overflow,.false.)
      product=cmplx(1._dp,0._dp,dp)
      do d=1,self%degree
        spectrum=cmplx(sketches(:,d),0._dp,dp); call plan%forward(spectrum)
        ! NumPy's contiguous complex product uses fused real/imaginary updates.
        product=cmplx(ieee_fma(real(product,dp),real(spectrum,dp),-aimag(product)*aimag(spectrum)), &
          ieee_fma(real(product,dp),aimag(spectrum),aimag(product)*real(spectrum,dp)),dp)
      end do
      call plan%inverse(product); z(i,:)=real(product,dp)
      call ieee_set_status(saved_status)
    end do
  end function
  function sketch_fit_transform(self,x) result(z)
    class(polynomial_count_sketch), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
  function sketch_fit_transform_csr(self,x) result(z)
    class(polynomial_count_sketch),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:)
    call self%fit(x);z=self%transform(x)
  end function
end module
