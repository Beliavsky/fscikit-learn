! Structured feature mappings: sklearn 1.6.1 _dict_vectorizer.py and _hash.py.
! BSD attribution: THIRD_PARTY_NOTICES.md. Original native storage uses MIT.
module sklearn_feature_extraction
  use sklearn_common, only: dp,require
  use sklearn_sparse, only: csr_matrix
  use sklearn_murmurhash, only: murmurhash3_32
  use sklearn_ieee_math, only: ieee_difference
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: feature_string,feature_item,feature_row,dict_vectorizer,feature_hasher
  type :: feature_string
    character(:), allocatable :: text
  end type
  type :: feature_item
    character(:), allocatable :: key,category
    type(feature_string), allocatable :: categories(:)
    real(dp) :: value=1
    logical :: is_none=.false.
  end type
  type :: feature_row
    type(feature_item), allocatable :: items(:)
    type(feature_string), allocatable :: tokens(:)
  end type
  type :: dict_vectorizer
    logical :: sort=.true.
    character(:), allocatable :: separator
    type(feature_string), allocatable :: feature_names_(:)
    integer :: n_features_out_=0
    integer, allocatable, private :: slots_(:)
  contains
    procedure :: fit=>dict_fit
    procedure :: transform=>dict_transform
    procedure :: fit_transform=>dict_fit_transform
    procedure :: transform_dense=>dict_dense
    procedure :: fit_transform_dense=>dict_fit_dense
    procedure :: get_feature_names_out=>dict_names
    procedure :: vocabulary_index=>dict_index
    procedure, private :: inverse_dense,inverse_sparse,restrict_mask,restrict_indices
    generic :: inverse_transform=>inverse_dense,inverse_sparse
    generic :: restrict=>restrict_mask,restrict_indices
  end type
  type :: feature_hasher
    integer :: n_features=1048576
    character(8) :: input_type='dict'
    logical :: alternate_sign=.true.
  contains
    procedure :: fit=>hasher_fit
    procedure :: transform=>hasher_transform
    procedure :: fit_transform=>hasher_fit_transform
    procedure :: transform_dense=>hasher_dense
    procedure :: fit_transform_dense=>hasher_fit_dense
  end type
  type :: vocabulary_builder
    type(feature_string), allocatable :: names(:)
    integer, allocatable :: slots(:)
    integer :: n=0
  end type
contains
  logical function equal_text(a,b) result(equal)
    character(*), intent(in) :: a,b
    equal=len(a)==len(b)
    if(equal) equal=a==b
  end function
  logical function less_equal_text(a,b) result(less)
    character(*), intent(in) :: a,b
    integer :: i
    do i=1,min(len(a),len(b))
      if(iachar(a(i:i))==iachar(b(i:i))) cycle
      less=iachar(a(i:i))<iachar(b(i:i));return
    end do
    less=len(a)<=len(b)
  end function
  function ordered(names,indices) result(order)
    type(feature_string), optional, intent(in) :: names(:)
    integer, optional, intent(in) :: indices(:)
    integer, allocatable :: order(:),work(:)
    integer :: n,width,start,mid,last,left,right,k
    logical :: take_left
    if(present(names)) then
      n=size(names)
    else
      n=size(indices)
    end if
    order=[(k,k=1,n)];allocate(work(n));width=1
    do while(width<n)
      start=1
      do while(start<=n)
        mid=start+min(width,n-start+1)-1;last=mid+min(width,n-mid);left=start;right=mid+1
        do k=start,last
          if(left>mid) then
            take_left=.false.
          else if(right>last) then
            take_left=.true.
          else if(present(names)) then
            take_left=less_equal_text(names(order(left))%text,names(order(right))%text)
          else
            take_left=indices(order(left))<=indices(order(right))
          end if
          if(take_left) then
            work(k)=order(left);left=left+1
          else
            work(k)=order(right);right=right+1
          end if
        end do
        if(last==n) exit
        start=last+1
      end do
      order=work
      if(width>=n-width) exit
      width=2*width
    end do
  end function
  integer function slot_for(key,slots) result(slot)
    character(*), intent(in) :: key
    integer, intent(in) :: slots(:)
    slot=1+int(modulo(murmurhash3_32(key),int(size(slots),int64)))
  end function
  integer function lookup(names,slots,key) result(index)
    type(feature_string), intent(in) :: names(:)
    integer, intent(in) :: slots(:)
    character(*), intent(in) :: key
    integer :: slot,tries
    index=0;slot=slot_for(key,slots)
    do tries=1,size(slots)
      if(slots(slot)==0) return
      if(equal_text(names(slots(slot))%text,key)) then
        index=slots(slot);return
      end if
      slot=mod(slot,size(slots))+1
    end do
    call require(.false.,'feature vocabulary table is full')
  end function
  subroutine rehash(names,n,slots,capacity)
    type(feature_string), intent(in) :: names(:)
    integer, intent(in) :: n,capacity
    integer, allocatable, intent(inout) :: slots(:)
    integer :: i,slot
    if(allocated(slots)) deallocate(slots)
    allocate(slots(capacity));slots=0
    do i=1,n
      slot=slot_for(names(i)%text,slots)
      do while(slots(slot)/=0)
        slot=mod(slot,capacity)+1
      end do
      slots(slot)=i
    end do
  end subroutine
  subroutine add_name(builder,key)
    type(vocabulary_builder), intent(inout) :: builder
    character(*), intent(in) :: key
    type(feature_string), allocatable :: grown(:)
    integer :: slot,capacity
    if(.not.allocated(builder%names)) then
      allocate(builder%names(16),builder%slots(32));builder%slots=0
    end if
    if(lookup(builder%names,builder%slots,key)>0) return
    if(builder%n+1>size(builder%slots)/2) then
      capacity=size(builder%slots)
      call require(capacity<=huge(1)/2,'feature vocabulary exceeds integer capacity')
      call rehash(builder%names,builder%n,builder%slots,2*capacity)
    end if
    if(builder%n==size(builder%names)) then
      capacity=size(builder%names)
      call require(capacity<=huge(1)/2,'feature vocabulary exceeds integer capacity')
      allocate(grown(2*capacity));grown(:builder%n)=builder%names;call move_alloc(grown,builder%names)
    end if
    builder%n=builder%n+1;builder%names(builder%n)%text=key
    slot=slot_for(key,builder%slots)
    do while(builder%slots(slot)/=0)
      slot=mod(slot,size(builder%slots))+1
    end do
    builder%slots(slot)=builder%n
  end subroutine
  subroutine expand(row,mode,separator,names,values)
    type(feature_row), intent(in) :: row
    character(*), intent(in) :: mode,separator
    type(feature_string), allocatable, intent(out) :: names(:)
    real(dp), allocatable, intent(out) :: values(:)
    integer :: n,i,j,k
    integer(int64) :: count
    if(mode=='string') then
      call require(.not.allocated(row%items),'string feature input must use tokens, not items')
      n=0;if(allocated(row%tokens)) n=size(row%tokens)
      allocate(names(n),values(n));values=1
      do i=1,n
        call require(allocated(row%tokens(i)%text),'feature token is unallocated')
        names(i)%text=row%tokens(i)%text
      end do
      return
    end if
    call require(.not.allocated(row%tokens),'mapping feature input must use items, not tokens')
    n=0;if(allocated(row%items)) n=size(row%items)
    count=0
    do i=1,n
      call require(allocated(row%items(i)%key),'feature key is unallocated')
      call require(.not.(allocated(row%items(i)%category).and.allocated(row%items(i)%categories)), &
        'choose a category or category list')
      if(row%items(i)%is_none) then
        call require(.not.allocated(row%items(i)%category).and..not.allocated(row%items(i)%categories), &
          'None feature value cannot also be categorical')
      end if
      if(mode/='dictionary') then
        call require(.not.allocated(row%items(i)%categories),'FeatureHasher does not accept category lists')
        call require(.not.row%items(i)%is_none,'FeatureHasher does not accept None values')
      end if
      if(allocated(row%items(i)%categories)) then
        count=count+size(row%items(i)%categories,kind=int64)
      else
        count=count+1
      end if
    end do
    call require(count<int(huge(1),int64),'expanded feature count exceeds integer capacity')
    allocate(names(int(count)),values(int(count)));k=0
    do i=1,n
      if(allocated(row%items(i)%categories)) then
        do j=1,size(row%items(i)%categories)
          call require(allocated(row%items(i)%categories(j)%text),'feature category is unallocated')
          k=k+1;names(k)%text=row%items(i)%key//separator//row%items(i)%categories(j)%text;values(k)=1
        end do
      else
        k=k+1;values(k)=row%items(i)%value
        if(allocated(row%items(i)%category)) then
          names(k)%text=row%items(i)%key//separator//row%items(i)%category;values(k)=1
        else
          names(k)%text=row%items(i)%key
          if(row%items(i)%is_none) values(k)=ieee_value(0._dp,ieee_quiet_nan)
        end if
      end if
    end do
  end subroutine
  function separator_for(self) result(separator)
    class(dict_vectorizer), intent(in) :: self
    character(:), allocatable :: separator
    separator='=';if(allocated(self%separator)) separator=self%separator
  end function
  subroutine dict_fit(self,x)
    class(dict_vectorizer), intent(inout) :: self
    type(feature_row), intent(in) :: x(:)
    type(vocabulary_builder) :: builder
    type(feature_string), allocatable :: names(:)
    real(dp), allocatable :: values(:)
    integer, allocatable :: order(:)
    integer :: i,j,capacity
    do i=1,size(x)
      call expand(x(i),'dictionary',separator_for(self),names,values)
      do j=1,size(names)
        call add_name(builder,names(j)%text)
      end do
    end do
    if(allocated(self%feature_names_)) deallocate(self%feature_names_)
    allocate(self%feature_names_(builder%n))
    if(builder%n>0) self%feature_names_=builder%names(:builder%n)
    if(self%sort) then
      order=ordered(names=self%feature_names_);self%feature_names_=self%feature_names_(order)
    end if
    capacity=32
    if(allocated(builder%slots)) capacity=size(builder%slots)
    call rehash(self%feature_names_,builder%n,self%slots_,capacity)
    self%n_features_out_=builder%n
  end subroutine
  integer function dict_index(self,key) result(index)
    class(dict_vectorizer), intent(in) :: self
    character(*), intent(in) :: key
    call require(allocated(self%feature_names_).and.allocated(self%slots_),'DictVectorizer is not fitted')
    index=lookup(self%feature_names_,self%slots_,key)
  end function
  subroutine append_entry(cols,values,used,column,value)
    integer, allocatable, intent(inout) :: cols(:)
    real(dp), allocatable, intent(inout) :: values(:)
    integer, intent(inout) :: used
    integer, intent(in) :: column
    real(dp), intent(in) :: value
    integer, allocatable :: larger_cols(:)
    real(dp), allocatable :: larger_values(:)
    integer :: capacity
    if(used==size(cols)) then
      capacity=size(cols);call require(capacity<=huge(1)/2,'feature CSR count exceeds integer capacity')
      allocate(larger_cols(2*capacity),larger_values(2*capacity))
      larger_cols(:used)=cols;larger_values(:used)=values
      call move_alloc(larger_cols,cols);call move_alloc(larger_values,values)
    end if
    used=used+1;cols(used)=column;values(used)=value
  end subroutine
  subroutine sort_rows(matrix,coalesce)
    type(csr_matrix), intent(inout) :: matrix
    logical, intent(in) :: coalesce
    integer, allocatable :: order(:)
    integer :: i,k,left,right,dest,start
    dest=1
    do i=1,matrix%n_rows
      left=matrix%indptr(i);right=matrix%indptr(i+1)-1
      order=ordered(indices=matrix%indices(left:right))
      matrix%indices(left:right)=matrix%indices(left-1+order)
      matrix%data(left:right)=matrix%data(left-1+order)
      matrix%indptr(i)=dest;start=dest
      do k=left,right
        if(coalesce.and.dest>start) then
          if(matrix%indices(dest-1)==matrix%indices(k)) then
            matrix%data(dest-1)=ieee_difference(matrix%data(dest-1),-matrix%data(k));cycle
          end if
        end if
        matrix%indices(dest)=matrix%indices(k);matrix%data(dest)=matrix%data(k);dest=dest+1
      end do
    end do
    matrix%indptr(matrix%n_rows+1)=dest
    matrix%indices=matrix%indices(:dest-1);matrix%data=matrix%data(:dest-1)
    call matrix%validate()
  end subroutine
  function dict_transform(self,x) result(matrix)
    class(dict_vectorizer), intent(in) :: self
    type(feature_row), intent(in) :: x(:)
    type(csr_matrix) :: matrix
    type(feature_string), allocatable :: names(:)
    real(dp), allocatable :: values(:),data(:)
    integer, allocatable :: cols(:)
    integer :: i,j,column,used
    call require(allocated(self%feature_names_).and.allocated(self%slots_),'DictVectorizer is not fitted')
    call require(size(x,kind=int64)>0.and.size(x,kind=int64)<huge(1),'feature sample sequence must be nonempty')
    matrix%n_rows=size(x);matrix%n_cols=size(self%feature_names_)
    allocate(matrix%indptr(size(x)+1),cols(16),data(16));used=0;matrix%indptr(1)=1
    do i=1,size(x)
      call expand(x(i),'dictionary',separator_for(self),names,values)
      do j=1,size(names)
        column=self%vocabulary_index(names(j)%text)
        if(column>0) call append_entry(cols,data,used,column,values(j))
      end do
      matrix%indptr(i+1)=used+1
    end do
    matrix%indices=cols(:used);matrix%data=data(:used);call sort_rows(matrix,.false.)
  end function
  function dict_fit_transform(self,x) result(matrix)
    class(dict_vectorizer), intent(inout) :: self
    type(feature_row), intent(in) :: x(:)
    type(csr_matrix) :: matrix
    call require(size(x)>0,'feature sample sequence must be nonempty')
    call self%fit(x);matrix=self%transform(x)
  end function
  function dict_dense(self,x) result(values)
    class(dict_vectorizer), intent(in) :: self
    type(feature_row), intent(in) :: x(:)
    real(dp), allocatable :: values(:,:)
    type(csr_matrix) :: matrix
    matrix=self%transform(x);values=matrix%to_dense()
  end function
  function dict_fit_dense(self,x) result(values)
    class(dict_vectorizer), intent(inout) :: self
    type(feature_row), intent(in) :: x(:)
    real(dp), allocatable :: values(:,:)
    type(csr_matrix) :: matrix
    matrix=self%fit_transform(x);values=matrix%to_dense()
  end function
  function dict_names(self) result(names)
    class(dict_vectorizer), intent(in) :: self
    type(feature_string), allocatable :: names(:)
    call require(allocated(self%feature_names_),'DictVectorizer is not fitted')
    names=self%feature_names_
  end function
  subroutine restrict_mask(self,support)
    class(dict_vectorizer), intent(inout) :: self
    logical, intent(in) :: support(:)
    integer :: i
    integer, allocatable :: indices(:)
    call require(allocated(self%feature_names_),'DictVectorizer is not fitted')
    call require(size(support)==size(self%feature_names_),'DictVectorizer support mask length mismatch')
    indices=pack([(i,i=1,size(support))],support);call self%restrict(indices)
  end subroutine
  subroutine restrict_indices(self,indices)
    class(dict_vectorizer), intent(inout) :: self
    integer, intent(in) :: indices(:)
    logical, allocatable :: seen(:)
    integer :: i,capacity
    call require(allocated(self%feature_names_),'DictVectorizer is not fitted')
    call require(all(indices>=1).and.all(indices<=size(self%feature_names_)),'DictVectorizer support index out of bounds')
    allocate(seen(size(self%feature_names_)));seen=.false.
    do i=1,size(indices)
      call require(.not.seen(indices(i)),'DictVectorizer support indices must be unique');seen(indices(i))=.true.
    end do
    capacity=size(self%slots_);self%feature_names_=self%feature_names_(indices)
    self%n_features_out_=size(indices);call rehash(self%feature_names_,size(indices),self%slots_,capacity)
  end subroutine
  function inverse_dense(self,x) result(rows)
    class(dict_vectorizer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    type(feature_row), allocatable :: rows(:)
    integer :: i,j,k
    call require(allocated(self%feature_names_),'DictVectorizer is not fitted')
    call require(size(x,1)>0.and.size(x,2)>0,'DictVectorizer inverse matrix must be nonempty')
    call require(size(x,2)==size(self%feature_names_),'DictVectorizer inverse feature width mismatch')
    call require(all(ieee_is_finite(x)),'DictVectorizer inverse values must be finite')
    allocate(rows(size(x,1)))
    do i=1,size(x,1)
      allocate(rows(i)%items(count(x(i,:)/=0)));k=0
      do j=1,size(x,2)
        if(x(i,j)==0) cycle
        k=k+1;rows(i)%items(k)%key=self%feature_names_(j)%text;rows(i)%items(k)%value=x(i,j)
      end do
    end do
  end function
  ! NumPy 2.2.6 contiguous sum order, reused from the attributed KernelCenterer port.
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
  function inverse_sparse(self,x) result(rows)
    class(dict_vectorizer), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    type(feature_row), allocatable :: rows(:)
    type(feature_item), allocatable :: items(:)
    integer, allocatable :: order(:)
    integer :: i,j,k,left,right,column,used,start
    real(dp) :: value
    logical :: nonzero
    call require(allocated(self%feature_names_),'DictVectorizer is not fitted');call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'DictVectorizer inverse matrix must be nonempty')
    call require(x%n_cols==size(self%feature_names_),'DictVectorizer inverse feature width mismatch')
    call require(all(ieee_is_finite(x%data)),'DictVectorizer inverse values must be finite')
    allocate(rows(x%n_rows))
    do i=1,x%n_rows
      left=x%indptr(i);right=x%indptr(i+1)-1
      order=ordered(indices=x%indices(left:right));allocate(items(size(order)));used=0;j=1
      do while(j<=size(order))
        column=x%indices(left+order(j)-1);start=j;nonzero=.false.
        do while(j<=size(order))
          k=left+order(j)-1
          if(x%indices(k)/=column) exit
          nonzero=nonzero.or.x%data(k)/=0;j=j+1
        end do
        if(.not.nonzero) cycle
        value=contiguous_sum(x%data(left+order(start:j-1)-1))
        used=used+1;items(used)%key=self%feature_names_(column)%text;items(used)%value=value
      end do
      rows(i)%items=items(:used);deallocate(items)
    end do
  end function
  subroutine hasher_controls(self)
    class(feature_hasher), intent(in) :: self
    call require(self%n_features>0,'FeatureHasher n_features must be positive')
    call require(any(self%input_type==['dict    ','pair    ','string  ']),'FeatureHasher unknown input_type')
  end subroutine
  subroutine hasher_fit(self,x)
    class(feature_hasher), intent(inout) :: self
    type(feature_row), optional, intent(in) :: x(:)
    call hasher_controls(self)
  end subroutine
  function hasher_transform(self,x) result(matrix)
    class(feature_hasher), intent(in) :: self
    type(feature_row), intent(in) :: x(:)
    type(csr_matrix) :: matrix
    type(feature_string), allocatable :: names(:)
    real(dp), allocatable :: values(:),data(:)
    integer, allocatable :: cols(:)
    integer(int64) :: hash
    integer :: i,j,column,used
    real(dp) :: value
    call hasher_controls(self)
    call require(size(x,kind=int64)>0.and.size(x,kind=int64)<huge(1),'feature sample sequence must be nonempty')
    matrix%n_rows=size(x);matrix%n_cols=self%n_features
    allocate(matrix%indptr(size(x)+1),cols(16),data(16));matrix%indptr(1)=1;used=0
    do i=1,size(x)
      call expand(x(i),trim(self%input_type),'=',names,values)
      do j=1,size(names)
        value=values(j)
        if(.not.ieee_is_nan(value)) then
          if(value==0) cycle
        end if
        hash=murmurhash3_32(names(j)%text)
        column=1+int(modulo(abs(hash),int(self%n_features,int64)))
        if(self%alternate_sign.and.hash<0) value=-value
        call append_entry(cols,data,used,column,value)
      end do
      matrix%indptr(i+1)=used+1
    end do
    matrix%indices=cols(:used);matrix%data=data(:used);call sort_rows(matrix,.true.)
  end function
  function hasher_fit_transform(self,x) result(matrix)
    class(feature_hasher), intent(inout) :: self
    type(feature_row), intent(in) :: x(:)
    type(csr_matrix) :: matrix
    call self%fit(x);matrix=self%transform(x)
  end function
  function hasher_dense(self,x) result(values)
    class(feature_hasher), intent(in) :: self
    type(feature_row), intent(in) :: x(:)
    real(dp), allocatable :: values(:,:)
    type(csr_matrix) :: matrix
    matrix=self%transform(x);values=matrix%to_dense()
  end function
  function hasher_fit_dense(self,x) result(values)
    class(feature_hasher), intent(inout) :: self
    type(feature_row), intent(in) :: x(:)
    real(dp), allocatable :: values(:,:)
    type(csr_matrix) :: matrix
    matrix=self%fit_transform(x);values=matrix%to_dense()
  end function
end module
