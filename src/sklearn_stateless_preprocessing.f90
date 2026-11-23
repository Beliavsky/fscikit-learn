! sklearn 1.6.1 preprocessing/_data.py Normalizer, Binarizer and free functions.
! CSR semantics also follow sklearn utils/sparsefuncs*. BSD attribution:
! THIRD_PARTY_NOTICES.md. Original typed adapters are MIT contributions.
module sklearn_stateless_preprocessing
  use sklearn_common, only: dp,require,check_x
  use sklearn_sparse, only: csr_matrix
  use sklearn_feature_extraction, only: feature_string
  use sklearn_transformer_protocol, only: dense_transformer
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  private
  public :: normalizer,binarizer,normalize,binarize
  public :: sorted_sparse ! Internal shared CSR helper; not re-exported by facade.
  interface normalize
    module procedure normalize_dense,normalize_sparse
  end interface
  interface binarize
    module procedure binarize_dense,binarize_sparse
  end interface
  type, abstract, extends(dense_transformer) :: stateless_transformer
    integer :: n_features_in_=0
    type(feature_string), allocatable :: feature_names_in_(:)
  contains
    procedure :: is_fitted=>bridge_fitted
    procedure :: bridge_features=>bridge_features
    procedure :: feature_names=>stateless_names
    procedure :: get_feature_names_out=>stateless_names
  end type
  type, extends(stateless_transformer) :: normalizer
    character(8) :: norm='l2'
  contains
    procedure :: fit_data=>normalizer_fit
    procedure, private :: fit_sparse=>normalizer_fit_sparse
    generic :: fit=>fit_data,fit_sparse
    procedure :: map_data=>normalizer_transform
    procedure, private :: transform_sparse=>normalizer_transform_sparse
    generic :: transform=>map_data,transform_sparse
    procedure, private :: fit_transform_dense=>normalizer_fit_transform,fit_transform_sparse=>normalizer_fit_transform_sparse
    generic :: fit_transform=>fit_transform_dense,fit_transform_sparse
  end type
  type, extends(stateless_transformer) :: binarizer
    real(dp) :: threshold=0
  contains
    procedure :: fit_data=>binarizer_fit
    procedure, private :: fit_sparse=>binarizer_fit_sparse
    generic :: fit=>fit_data,fit_sparse
    procedure :: map_data=>binarizer_transform
    procedure, private :: transform_sparse=>binarizer_transform_sparse
    generic :: transform=>map_data,transform_sparse
    procedure, private :: fit_transform_dense=>binarizer_fit_transform,fit_transform_sparse=>binarizer_fit_transform_sparse
    generic :: fit_transform=>fit_transform_dense,fit_transform_sparse
  end type
contains
  subroutine norm_controls(norm,axis)
    character(*), intent(in) :: norm
    integer, intent(in) :: axis
    call require(any(norm==[character(3)::'l1','l2','max']),'unknown normalization norm')
    call require(axis==0.or.axis==1,'normalization axis must be zero or one')
  end subroutine
  subroutine sparse_data(x)
    type(csr_matrix), intent(in) :: x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'X must be nonempty')
    call require(all(ieee_is_finite(x%data)),'X contains NaN or infinity')
  end subroutine
  real(dp) function vector_norm(values,norm) result(scale)
    real(dp), intent(in) :: values(:)
    character(*), intent(in) :: norm
    integer :: i
    scale=0
    do i=1,size(values)
      select case(norm)
      case('l1');scale=ieee_difference(scale,-abs(values(i)))
      case('l2');scale=ieee_difference(scale,-ieee_product(values(i),values(i)))
      case('max');scale=max(scale,abs(values(i)))
      end select
    end do
    if(norm=='l2') scale=sqrt(scale)
  end function
  function normalize_dense(x,norm,axis,norms) result(z)
    real(dp), intent(in) :: x(:,:)
    character(*), optional, intent(in) :: norm
    integer, optional, intent(in) :: axis
    real(dp), allocatable, optional, intent(out) :: norms(:)
    real(dp), allocatable :: z(:,:),scales(:)
    character(:), allocatable :: kind
    integer :: along,i,j,n
    call check_x(x)
    kind='l2';if(present(norm)) kind=norm
    along=1;if(present(axis)) along=axis
    call norm_controls(kind,along)
    n=size(x,1);if(along==0) n=size(x,2)
    allocate(scales(n));z=x
    do i=1,n
      if(along==1) then
        scales(i)=vector_norm(x(i,:),kind)
      else
        scales(i)=vector_norm(x(:,i),kind)
      end if
      ! sklearn _handle_zeros_in_scale: dense near-zero norms become ONE.
      if(scales(i)<10*epsilon(1._dp)) scales(i)=1
    end do
    do j=1,size(x,2)
      do i=1,size(x,1)
        if(along==1) then
          z(i,j)=ieee_quotient(x(i,j),scales(i))
        else
          z(i,j)=ieee_quotient(x(i,j),scales(j))
        end if
      end do
    end do
    if(present(norms)) norms=scales
  end function
  function sorted_sparse(x,coalesce) result(z)
    type(csr_matrix), intent(in) :: x
    logical, intent(in) :: coalesce
    type(csr_matrix) :: z
    integer, allocatable :: order(:),work(:)
    integer :: i,j,k,n,left,right,width,start,mid,last,a,b,t,dest,row_start
    z=x;dest=1
    do i=1,x%n_rows
      left=x%indptr(i);right=x%indptr(i+1)-1;n=right-left+1
      allocate(order(n),work(n));order=[(j,j=1,n)];width=1
      do while(width<n)
        start=1
        do while(start<=n)
          mid=start+min(width,n-start+1);last=mid+min(width,n-mid+1)-1
          a=start;b=mid
          do t=start,last
            if(a>=mid) then
              work(t)=order(b);b=b+1
            else if(b>last) then
              work(t)=order(a);a=a+1
            else if(x%indices(left+order(a)-1)<=x%indices(left+order(b)-1)) then
              work(t)=order(a);a=a+1
            else
              work(t)=order(b);b=b+1
            end if
          end do
          start=last+1
        end do
        order=work
        if(width>n/2) exit
        width=2*width
      end do
      z%indptr(i)=dest;row_start=dest
      do j=1,n
        k=left+order(j)-1
        if(coalesce.and.dest>row_start) then
          if(z%indices(dest-1)==x%indices(k)) then
            z%data(dest-1)=ieee_difference(z%data(dest-1),-x%data(k));cycle
          end if
        end if
        z%indices(dest)=x%indices(k);z%data(dest)=x%data(k);dest=dest+1
      end do
      deallocate(order,work)
    end do
    z%indptr(x%n_rows+1)=dest;z%indices=z%indices(:dest-1);z%data=z%data(:dest-1)
  end function
  function normalize_sparse(x,norm,axis,norms) result(z)
    type(csr_matrix), intent(in) :: x
    character(*), optional, intent(in) :: norm
    integer, optional, intent(in) :: axis
    real(dp), allocatable, optional, intent(out) :: norms(:)
    type(csr_matrix) :: z
    real(dp), allocatable :: scales(:)
    character(:), allocatable :: kind
    integer :: along,n,i,k,index
    logical :: divide
    call sparse_data(x)
    kind='l2';if(present(norm)) kind=norm
    along=1;if(present(axis)) along=axis
    call norm_controls(kind,along)
    if(present(norms)) call require(kind=='max','sparse return norms is implemented only for max normalization')
    z=x
    if(kind=='max') then
      ! scipy's min/max reductions canonicalize duplicates, retaining zeros.
      z=sorted_sparse(x,.true.)
    end if
    n=x%n_rows;if(along==0) n=x%n_cols
    allocate(scales(n));scales=0
    do i=1,z%n_rows
      do k=z%indptr(i),z%indptr(i+1)-1
        index=i;if(along==0) index=z%indices(k)
        select case(kind)
        case('l1');scales(index)=ieee_difference(scales(index),-abs(z%data(k)))
        case('l2');scales(index)=ieee_difference(scales(index),-ieee_product(z%data(k),z%data(k)))
        case('max')
          if(ieee_is_nan(z%data(k))) then
            scales(index)=z%data(k)
          else if(.not.ieee_is_nan(scales(index))) then
            scales(index)=max(scales(index),abs(z%data(k)))
          end if
        end select
      end do
    end do
    if(kind=='l2') scales=sqrt(scales)
    do i=1,z%n_rows
      do k=z%indptr(i),z%indptr(i+1)-1
        index=i;if(along==0) index=z%indices(k)
        divide=.true.
        if(.not.ieee_is_nan(scales(index))) divide=scales(index)/=0
        if(divide) z%data(k)=ieee_quotient(z%data(k),scales(index))
      end do
    end do
    ! Source converts axis-zero CSR through CSC; return its CSR ordering.
    if(along==0.and.kind/='max') z=sorted_sparse(z,.false.)
    if(present(norms)) norms=scales
  end function
  function binarize_dense(x,threshold) result(z)
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: threshold
    real(dp), allocatable :: z(:,:)
    real(dp) :: t
    call check_x(x);t=0;if(present(threshold)) t=threshold
    call require(ieee_is_finite(t),'binarization threshold must be finite')
    z=merge(1._dp,0._dp,x>t)
  end function
  function binarize_sparse(x,threshold) result(z)
    type(csr_matrix), intent(in) :: x
    real(dp), optional, intent(in) :: threshold
    type(csr_matrix) :: z
    real(dp) :: t
    integer :: i,k,dest
    call sparse_data(x);t=0;if(present(threshold)) t=threshold
    call require(ieee_is_finite(t),'binarization threshold must be finite')
    call require(t>=0,'cannot binarize a sparse matrix with threshold below zero')
    z=x;dest=1
    do i=1,x%n_rows
      z%indptr(i)=dest
      do k=x%indptr(i),x%indptr(i+1)-1
        if(x%data(k)<=t) cycle
        z%indices(dest)=x%indices(k);z%data(dest)=1;dest=dest+1
      end do
    end do
    z%indptr(x%n_rows+1)=dest;z%indices=z%indices(:dest-1);z%data=z%data(:dest-1)
  end function
  subroutine record_metadata(self,width,names)
    class(stateless_transformer), intent(inout) :: self
    integer, intent(in) :: width
    type(feature_string), optional, intent(in) :: names(:)
    integer :: i
    self%n_features_in_=width
    if(allocated(self%feature_names_in_)) deallocate(self%feature_names_in_)
    if(present(names)) then
      call require(size(names)==width,'stateless transformer fit name count mismatch')
      do i=1,width
        call require(allocated(names(i)%text),'stateless transformer feature name is unallocated')
      end do
      self%feature_names_in_=names
    end if
  end subroutine
  subroutine query_width(self,width)
    class(stateless_transformer), intent(in) :: self
    integer, intent(in) :: width
    if(self%n_features_in_>0) call require(width==self%n_features_in_,'stateless transformer feature count mismatch')
  end subroutine
  function stateless_names(self,input_features) result(names)
    class(stateless_transformer), intent(in) :: self
    type(feature_string), optional, intent(in) :: input_features(:)
    type(feature_string), allocatable :: names(:)
    character(32) :: buffer
    integer :: i
    call require(self%n_features_in_>0,'stateless transformer feature names require fit')
    if(present(input_features)) then
      call query_width(self,size(input_features))
      do i=1,size(input_features)
        call require(allocated(input_features(i)%text),'stateless transformer feature name is unallocated')
      end do
      if(allocated(self%feature_names_in_)) then
        call require(size(input_features)==size(self%feature_names_in_), &
          'stateless transformer input feature name count mismatch')
        do i=1,size(input_features)
          call require(len(input_features(i)%text)==len(self%feature_names_in_(i)%text), &
            'stateless transformer input feature names differ')
          call require(input_features(i)%text==self%feature_names_in_(i)%text, &
            'stateless transformer input feature names differ')
        end do
      end if
      names=input_features
    else if(allocated(self%feature_names_in_)) then
      names=self%feature_names_in_
    else
      call require(self%n_features_in_>0,'stateless transformer feature names require fit or input names')
      allocate(names(self%n_features_in_))
      do i=1,size(names)
        write(buffer,'(i0)') i-1;names(i)%text='x'//trim(buffer)
      end do
    end if
  end function
  subroutine normalizer_fit(self,x,feature_names)
    class(normalizer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    call norm_controls(self%norm,1);call check_x(x);call record_metadata(self,size(x,2),feature_names)
  end subroutine
  subroutine normalizer_fit_sparse(self,x,feature_names)
    class(normalizer), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    type(feature_string), optional, intent(in) :: feature_names(:)
    call norm_controls(self%norm,1);call sparse_data(x);call record_metadata(self,x%n_cols,feature_names)
  end subroutine
  function normalizer_transform(self,x) result(z)
    class(normalizer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call query_width(self,size(x,2));z=normalize(x,self%norm)
  end function
  function normalizer_transform_sparse(self,x) result(z)
    class(normalizer), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    type(csr_matrix) :: z
    call query_width(self,x%n_cols);z=normalize(x,self%norm)
  end function
  function normalizer_fit_transform(self,x,feature_names) result(z)
    class(normalizer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x,feature_names);z=self%transform(x)
  end function
  function normalizer_fit_transform_sparse(self,x,feature_names) result(z)
    class(normalizer), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    type(feature_string), optional, intent(in) :: feature_names(:)
    type(csr_matrix) :: z
    call self%fit(x,feature_names);z=self%transform(x)
  end function
  subroutine binarizer_fit(self,x,feature_names)
    class(binarizer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    call require(ieee_is_finite(self%threshold),'binarization threshold must be finite')
    call check_x(x);call record_metadata(self,size(x,2),feature_names)
  end subroutine
  subroutine binarizer_fit_sparse(self,x,feature_names)
    class(binarizer), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    type(feature_string), optional, intent(in) :: feature_names(:)
    call require(ieee_is_finite(self%threshold),'binarization threshold must be finite')
    call sparse_data(x);call record_metadata(self,x%n_cols,feature_names)
  end subroutine
  function binarizer_transform(self,x) result(z)
    class(binarizer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call query_width(self,size(x,2));z=binarize(x,self%threshold)
  end function
  function binarizer_transform_sparse(self,x) result(z)
    class(binarizer), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    type(csr_matrix) :: z
    call query_width(self,x%n_cols);z=binarize(x,self%threshold)
  end function
  function binarizer_fit_transform(self,x,feature_names) result(z)
    class(binarizer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_string), optional, intent(in) :: feature_names(:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x,feature_names);z=self%transform(x)
  end function
  function binarizer_fit_transform_sparse(self,x,feature_names) result(z)
    class(binarizer), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    type(feature_string), optional, intent(in) :: feature_names(:)
    type(csr_matrix) :: z
    call self%fit(x,feature_names);z=self%transform(x)
  end function
  logical function bridge_fitted(self) result(ok)
    class(stateless_transformer), intent(in) :: self
    ok=.true.
  end function
  integer function bridge_features(self) result(value)
    class(stateless_transformer), intent(in) :: self
    value=self%n_features_in_
  end function
end module
