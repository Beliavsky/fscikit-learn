! KernelCenterer: sklearn 1.6.1 preprocessing/_data.py (BSD-3-Clause).
! Contiguous reduction ordering follows NumPy 2.2.6 loops_utils.h.src.
! Both upstream notices are retained in THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_kernel_centering
  use sklearn_common, only: dp,require,check_x
  use sklearn_ieee_math, only: ieee_difference,ieee_quotient,ieee_total
  implicit none
  private
  public :: kernel_centerer
  type :: kernel_centerer
    real(dp), allocatable :: k_fit_rows_(:)
    real(dp) :: k_fit_all_=0
    integer :: n_features_in_=0
  contains
    procedure :: fit=>centerer_fit
    procedure :: transform=>centerer_transform
    procedure :: fit_transform=>centerer_fit_transform
    procedure :: get_feature_names_out=>centerer_names
  end type
contains
  ! NumPy's contiguous binary64 pairwise reduction uses eight accumulators,
  ! blocks of at most 128 and recursively aligned halves. The outer +0 is
  ! the default sum reduction identity. IEEE helpers preserve quiet overflow.
  recursive real(dp) function pairwise(values) result(total)
    real(dp), intent(in) :: values(:)
    real(dp) :: accum(8),left,right
    integer :: n,i,j,half,limit
    n=size(values)
    if(n<8) then
      total=-0._dp
      do i=1,n
        total=ieee_difference(total,-values(i))
      end do
    else if(n<=128) then
      accum=values(:8); limit=n-mod(n,8)
      do i=9,limit,8
        do j=1,8
          accum(j)=ieee_difference(accum(j),-values(i+j-1))
        end do
      end do
      left=ieee_difference(ieee_difference(accum(1),-accum(2)),-ieee_difference(accum(3),-accum(4)))
      right=ieee_difference(ieee_difference(accum(5),-accum(6)),-ieee_difference(accum(7),-accum(8)))
      total=ieee_difference(left,-right)
      do i=limit+1,n
        total=ieee_difference(total,-values(i))
      end do
    else
      half=n/2; half=half-mod(half,8)
      left=pairwise(values(:half)); right=pairwise(values(half+1:)); total=ieee_difference(left,-right)
    end if
  end function
  real(dp) function contiguous_sum(values) result(total)
    real(dp), intent(in) :: values(:)
    total=ieee_difference(0._dp,-pairwise(values))
  end function
  subroutine centerer_fit(self,kernel)
    class(kernel_centerer), intent(inout) :: self
    real(dp), intent(in) :: kernel(:,:)
    integer :: n,j
    call check_x(kernel); n=size(kernel,1)
    call require(size(kernel,2)==n,'kernel centerer fit requires a square matrix')
    self%n_features_in_=n
    if(allocated(self%k_fit_rows_)) deallocate(self%k_fit_rows_)
    allocate(self%k_fit_rows_(n))
    do j=1,n
      ! Match a C-contiguous sklearn reference's noncontiguous axis-0 sum.
      self%k_fit_rows_(j)=ieee_quotient(ieee_total(kernel(:,j)),real(n,dp))
    end do
    self%k_fit_all_=ieee_quotient(contiguous_sum(self%k_fit_rows_),real(n,dp))
  end subroutine
  function centerer_transform(self,kernel) result(centered)
    class(kernel_centerer), intent(in) :: self
    real(dp), intent(in) :: kernel(:,:)
    real(dp), allocatable :: centered(:,:)
    real(dp) :: row_mean,value
    integer :: i,j,n
    call require(allocated(self%k_fit_rows_),'kernel centerer is not fitted')
    call check_x(kernel); n=size(self%k_fit_rows_)
    call require(size(kernel,2)==n,'kernel centerer query width mismatch')
    allocate(centered(size(kernel,1),n))
    do i=1,size(kernel,1)
      row_mean=ieee_quotient(contiguous_sum(kernel(i,:)),real(n,dp))
      do j=1,n
        value=ieee_difference(kernel(i,j),self%k_fit_rows_(j))
        value=ieee_difference(value,row_mean)
        centered(i,j)=ieee_difference(value,-self%k_fit_all_)
      end do
    end do
  end function
  function centerer_fit_transform(self,kernel) result(centered)
    class(kernel_centerer), intent(inout) :: self
    real(dp), intent(in) :: kernel(:,:)
    real(dp), allocatable :: centered(:,:)
    call self%fit(kernel); centered=self%transform(kernel)
  end function
  function centerer_names(self,input_features) result(names)
    class(kernel_centerer), intent(in) :: self
    character(*), intent(in), optional :: input_features(:)
    character(32), allocatable :: names(:)
    integer :: i
    call require(allocated(self%k_fit_rows_),'kernel centerer is not fitted')
    if(present(input_features)) call require(size(input_features)==self%n_features_in_, &
      'kernel centerer input feature name count mismatch')
    allocate(names(self%n_features_in_))
    do i=1,size(names)
      write(names(i),'(a,i0)') 'kernelcenterer',i-1
    end do
  end function
end module
