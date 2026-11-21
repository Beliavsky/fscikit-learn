! Minimal real64 CSR storage for native graph transformers.
! Pointer/column arrays are 1-based; explicit zeros and duplicates are retained.
module sklearn_sparse
  use sklearn_common, only: dp,require
  use sklearn_ieee_math, only: ieee_difference,ieee_product
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  private
  public :: csr_matrix,csr_from_dense,csr_from_arrays
  type :: csr_matrix
    integer :: n_rows=0,n_cols=0
    integer, allocatable :: indptr(:),indices(:)
    real(dp), allocatable :: data(:)
  contains
    procedure :: validate=>csr_validate
    procedure :: nnz=>csr_nnz
    procedure :: to_dense=>csr_dense
    procedure :: stored_mask=>csr_mask
    procedure :: matmul=>csr_matmul
  end type
contains
  subroutine csr_validate(self)
    class(csr_matrix), intent(in) :: self
    integer :: count
    call require(self%n_rows>=0.and.self%n_cols>=0.and.self%n_rows<huge(1),'invalid CSR shape')
    call require(allocated(self%indptr).and.allocated(self%indices).and.allocated(self%data),'CSR arrays are unallocated')
    call require(size(self%data,kind=int64)<int(huge(1),int64),'CSR stored count exceeds integer range')
    count=size(self%data)
    call require(size(self%indices)==count,'CSR data/index lengths differ')
    call require(size(self%indptr)==self%n_rows+1,'CSR pointer length mismatch')
    call require(self%indptr(1)==1.and.self%indptr(self%n_rows+1)==count+1,'CSR pointer endpoints are invalid')
    call require(all(self%indptr>=1).and.all(self%indptr<=count+1),'CSR pointer is out of bounds')
    call require(all(self%indptr(2:)>=self%indptr(:self%n_rows)),'CSR pointers must be nondecreasing')
    call require(all(self%indices>=1).and.all(self%indices<=self%n_cols),'CSR column index is out of bounds')
  end subroutine
  integer function csr_nnz(self) result(count)
    class(csr_matrix), intent(in) :: self
    count=0; if(allocated(self%data)) count=size(self%data)
  end function
  function csr_from_arrays(n_rows,n_cols,indptr,indices,data) result(matrix)
    integer, intent(in) :: n_rows,n_cols,indptr(:),indices(:)
    real(dp), intent(in) :: data(:)
    type(csr_matrix) :: matrix
    matrix%n_rows=n_rows; matrix%n_cols=n_cols
    matrix%indptr=indptr; matrix%indices=indices; matrix%data=data
    call matrix%validate()
  end function
  function csr_from_dense(values,store_zeros) result(matrix)
    real(dp), intent(in) :: values(:,:)
    logical, intent(in), optional :: store_zeros
    type(csr_matrix) :: matrix
    logical :: keep,present_value
    integer :: i,j,k,count
    integer(int64) :: capacity
    keep=.false.; if(present(store_zeros)) keep=store_zeros
    capacity=int(size(values,1),int64)*size(values,2)
    call require(capacity<int(huge(1),int64),'CSR dense conversion exceeds integer range')
    matrix%n_rows=size(values,1); matrix%n_cols=size(values,2); count=0
    do i=1,matrix%n_rows
      do j=1,matrix%n_cols
        present_value=ieee_is_nan(values(i,j))
        if(.not.present_value) present_value=values(i,j)/=0
        if(keep.or.present_value) count=count+1
      end do
    end do
    allocate(matrix%indptr(matrix%n_rows+1),matrix%indices(count),matrix%data(count)); k=1; matrix%indptr(1)=1
    do i=1,matrix%n_rows
      do j=1,matrix%n_cols
        present_value=ieee_is_nan(values(i,j))
        if(.not.present_value) present_value=values(i,j)/=0
        if(.not.keep.and..not.present_value) cycle
        matrix%indices(k)=j; matrix%data(k)=values(i,j); k=k+1
      end do
      matrix%indptr(i+1)=k
    end do
  end function
  function csr_dense(self) result(values)
    class(csr_matrix), intent(in) :: self
    real(dp), allocatable :: values(:,:)
    integer :: i,k
    call self%validate(); allocate(values(self%n_rows,self%n_cols)); values=0
    do i=1,self%n_rows
      do k=self%indptr(i),self%indptr(i+1)-1
        values(i,self%indices(k))=ieee_difference(values(i,self%indices(k)),-self%data(k))
      end do
    end do
  end function
  function csr_mask(self) result(mask)
    class(csr_matrix), intent(in) :: self
    logical, allocatable :: mask(:,:)
    integer :: i,k
    call self%validate(); allocate(mask(self%n_rows,self%n_cols)); mask=.false.
    do i=1,self%n_rows
      do k=self%indptr(i),self%indptr(i+1)-1
        mask(i,self%indices(k))=.true.
      end do
    end do
  end function
  function csr_matmul(self,right) result(values)
    class(csr_matrix), intent(in) :: self
    real(dp), intent(in) :: right(:,:)
    real(dp), allocatable :: values(:,:)
    integer :: i,j,k
    call self%validate(); call require(size(right,1)==self%n_cols,'CSR matmul dimension mismatch')
    allocate(values(self%n_rows,size(right,2))); values=0
    do j=1,size(right,2)
      do i=1,self%n_rows
        do k=self%indptr(i),self%indptr(i+1)-1
          values(i,j)=ieee_difference(values(i,j),-ieee_product(self%data(k),right(self%indices(k),j)))
        end do
      end do
    end do
  end function
end module
