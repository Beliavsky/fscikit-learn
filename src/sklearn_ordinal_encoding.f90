! Numeric OrdinalEncoder adapted from sklearn 1.6.1 preprocessing/_encoders.py.
! BSD-3-Clause attribution: THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_ordinal_encoding
  use sklearn_common, only: dp,require,argsort
  use sklearn_target_encoding, only: ordinal_category=>target_category
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: ordinal_encoder,ordinal_category
  type :: category_map
    integer, allocatable :: codes(:),counts(:),infrequent(:)
    integer :: missing_index=0,cardinality=0
  end type
  type :: ordinal_encoder
    character(20) :: handle_unknown='error'
    ! Unallocated unknown_value means None; unallocated missing value means NaN.
    real(dp), allocatable :: unknown_value,encoded_missing_value
    integer :: min_frequency_count=0,max_categories=0,n_features_in_=0
    real(dp) :: min_frequency_fraction=0
    type(ordinal_category), allocatable :: categories(:),categories_(:),infrequent_categories_(:)
    type(category_map), allocatable :: mappings_(:)
  contains
    procedure :: fit,transform,fit_transform,inverse_transform
  end type
contains
  real(dp) function missing_value(self) result(value)
    class(ordinal_encoder), intent(in) :: self
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(allocated(self%encoded_missing_value)) value=self%encoded_missing_value
  end function

  logical function equal_value(a,b) result(equal)
    real(dp), intent(in) :: a,b
    equal=.false.
    if(ieee_is_nan(a).or.ieee_is_nan(b)) then
      equal=ieee_is_nan(a).and.ieee_is_nan(b)
    else
      equal=a==b
    end if
  end function

  function unique_values(x) result(values)
    real(dp), intent(in) :: x(:)
    real(dp), allocatable :: values(:),sorted(:),work(:)
    integer, allocatable :: order(:)
    integer :: i,n
    sorted=pack(x,.not.ieee_is_nan(x)); order=argsort(sorted); sorted=sorted(order)
    allocate(work(size(x))); n=0
    do i=1,size(sorted)
      if(i>1) then
        if(sorted(i)==sorted(i-1)) cycle
      end if
      n=n+1; work(n)=sorted(i)
    end do
    if(any(ieee_is_nan(x))) then
      n=n+1; work(n)=ieee_value(0._dp,ieee_quiet_nan)
    end if
    values=work(:n)
  end function

  integer function lookup(values,x) result(index)
    real(dp), intent(in) :: values(:),x
    integer :: low,high,middle
    index=0; high=size(values)
    if(high==0) return
    if(ieee_is_nan(x)) then
      if(ieee_is_nan(values(high))) index=high
      return
    end if
    if(ieee_is_nan(values(high))) high=high-1
    low=1
    do while(low<=high)
      middle=low+(high-low)/2
      if(values(middle)==x) then
        index=middle; return
      else if(values(middle)<x) then
        low=middle+1
      else
        high=middle-1
      end if
    end do
  end function

  subroutine check_data(x)
    real(dp), intent(in) :: x(:,:)
    call require(size(x,1)>0.and.size(x,2)>0,'OrdinalEncoder data must be nonempty')
    call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'OrdinalEncoder data contains infinity')
  end subroutine

  subroutine check_code(value,message)
    real(dp), intent(in) :: value
    character(*), intent(in) :: message
    if(ieee_is_nan(value)) return
    call require(ieee_is_finite(value),message)
    call require(modulo(value,1._dp)==0,message)
  end subroutine

  subroutine check_unknown(self)
    class(ordinal_encoder), intent(in) :: self
    call require(self%handle_unknown=='error'.or.self%handle_unknown=='use_encoded_value', &
      'unknown OrdinalEncoder handle_unknown mode')
    if(self%handle_unknown=='use_encoded_value') then
      call require(allocated(self%unknown_value),'OrdinalEncoder unknown_value is required')
    else
      call require(.not.allocated(self%unknown_value),'OrdinalEncoder unknown_value requires use_encoded_value')
    end if
  end subroutine

  subroutine fit(self,x)
    class(ordinal_encoder), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: unique(:),values(:)
    logical, allocatable :: rare(:)
    integer, allocatable :: order(:)
    real(dp) :: missing
    integer :: i,j,ncat,nfinite,nrare,nfrequent,index,code,p
    call check_data(x); call check_unknown(self)
    if(allocated(self%unknown_value)) call check_code(self%unknown_value,'OrdinalEncoder unknown_value must be integral or NaN')
    missing=missing_value(self)
    call check_code(missing,'OrdinalEncoder encoded_missing_value must be integral or NaN')
    call require(self%min_frequency_count>=0,'OrdinalEncoder min_frequency_count must be nonnegative')
    call require(ieee_is_finite(self%min_frequency_fraction),'OrdinalEncoder frequency fraction must be finite')
    call require(self%min_frequency_fraction>=0.and.self%min_frequency_fraction<1, &
      'OrdinalEncoder frequency fraction must lie in [0,1)')
    call require(self%min_frequency_count==0.or.self%min_frequency_fraction==0, &
      'OrdinalEncoder frequency count and fraction are mutually exclusive')
    call require(self%max_categories>=0,'OrdinalEncoder max_categories must be nonnegative')
    p=size(x,2)
    if(allocated(self%categories_)) deallocate(self%categories_,self%infrequent_categories_,self%mappings_)
    allocate(self%categories_(p),self%infrequent_categories_(p),self%mappings_(p))
    if(allocated(self%categories)) call require(size(self%categories)==p,'OrdinalEncoder category feature count mismatch')
    do j=1,p
      if(allocated(self%categories)) then
        call require(allocated(self%categories(j)%values),'OrdinalEncoder category values are unallocated')
        values=self%categories(j)%values
        call require(size(values)>0,'OrdinalEncoder categories must be nonempty')
        call require(all(ieee_is_finite(values).or.ieee_is_nan(values)),'OrdinalEncoder categories contain infinity')
        unique=unique_values(values)
        call require(size(unique)==size(values),'OrdinalEncoder categories must be unique')
        do i=1,size(values)
          if(ieee_is_nan(values(i))) then
            call require(i==size(values),'OrdinalEncoder NaN category must be last')
          else
            call require(equal_value(values(i),unique(i)),'OrdinalEncoder numeric categories must be sorted')
          end if
        end do
      else
        values=unique_values(x(:,j))
      end if
      self%categories_(j)%values=values; ncat=size(values); nfinite=ncat
      if(ieee_is_nan(values(ncat))) then
        self%mappings_(j)%missing_index=ncat; nfinite=ncat-1
      end if
      allocate(self%mappings_(j)%counts(ncat)); self%mappings_(j)%counts=0
      do i=1,size(x,1)
        index=lookup(values,x(i,j))
        if(index==0) then
          call require(self%handle_unknown/='error','OrdinalEncoder unknown category during fit')
        else
          self%mappings_(j)%counts(index)=self%mappings_(j)%counts(index)+1
        end if
      end do
      rare=self%mappings_(j)%counts(:nfinite)<self%min_frequency_count
      if(self%min_frequency_fraction>0) rare=real(self%mappings_(j)%counts(:nfinite),dp)< &
        self%min_frequency_fraction*real(size(x,1),dp)
      if(self%max_categories>0) then
        if(self%max_categories<nfinite-count(rare)+1) then
          order=argsort(real(self%mappings_(j)%counts(:nfinite),dp))
          rare(order(:nfinite-self%max_categories+1))=.true.
        end if
      end if
      nrare=count(rare); nfrequent=nfinite-nrare
      self%mappings_(j)%infrequent=pack([(i,i=1,nfinite)],rare)
      self%infrequent_categories_(j)%values=values(self%mappings_(j)%infrequent)
      self%mappings_(j)%cardinality=nfrequent
      allocate(self%mappings_(j)%codes(ncat)); code=0
      do i=1,nfinite
        if(rare(i)) then
          self%mappings_(j)%codes(i)=nfrequent
        else
          self%mappings_(j)%codes(i)=code; code=code+1
        end if
      end do
      if(nfinite<ncat) self%mappings_(j)%codes(ncat)=ncat-1
      if(allocated(self%unknown_value)) then
        if(.not.ieee_is_nan(self%unknown_value)) call require(.not.(self%unknown_value>=0.and. &
          self%unknown_value<real(nfrequent,dp)),'OrdinalEncoder unknown_value collides with a known code')
      end if
      if(nfinite<ncat.and..not.ieee_is_nan(missing)) call require(.not.(missing>=0.and. &
        missing<real(nfrequent,dp)),'OrdinalEncoder encoded_missing_value collides with a known code')
    end do
    self%n_features_in_=p
  end subroutine

  function transform(self,x) result(z)
    class(ordinal_encoder), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    integer :: i,j,index
    call require(allocated(self%categories_),'OrdinalEncoder is not fitted')
    call check_data(x); call check_unknown(self)
    call require(size(x,2)==self%n_features_in_,'OrdinalEncoder query feature count mismatch')
    allocate(z(size(x,1),size(x,2)))
    do j=1,size(x,2)
      do i=1,size(x,1)
        index=lookup(self%categories_(j)%values,x(i,j))
        if(index==0) then
          call require(self%handle_unknown=='use_encoded_value','OrdinalEncoder unknown category during transform')
          z(i,j)=self%unknown_value
        else if(index==self%mappings_(j)%missing_index) then
          z(i,j)=missing_value(self)
        else
          z(i,j)=real(self%mappings_(j)%codes(index),dp)
        end if
      end do
    end do
  end function

  function fit_transform(self,x) result(z)
    class(ordinal_encoder), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function

  function inverse_transform(self,z,unknown_mask,infrequent_mask) result(x)
    class(ordinal_encoder), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    logical, allocatable, intent(out), optional :: unknown_mask(:,:),infrequent_mask(:,:)
    real(dp), allocatable :: x(:,:),values(:)
    logical :: unknown(size(z,1),size(z,2)),infrequent(size(z,1),size(z,2))
    logical, allocatable :: keep(:)
    real(dp) :: label
    integer :: i,j,index,ncat,group_code
    call require(allocated(self%categories_),'OrdinalEncoder is not fitted')
    call check_data(z); call check_unknown(self)
    call require(size(z,2)==self%n_features_in_,'OrdinalEncoder inverse feature count mismatch')
    allocate(x(size(z,1),size(z,2))); x=ieee_value(0._dp,ieee_quiet_nan)
    unknown=.false.; infrequent=.false.
    do j=1,size(z,2)
      values=self%categories_(j)%values; ncat=size(values); group_code=-1
      if(size(self%mappings_(j)%infrequent)>0) then
        group_code=ncat-size(self%mappings_(j)%infrequent)
        keep=spread(.true.,1,ncat); keep(self%mappings_(j)%infrequent)=.false.
        values=pack(values,keep)
      end if
      do i=1,size(z,1)
        label=z(i,j)
        ! Source replaces the missing sentinel before detecting unknown/groups.
        if(self%mappings_(j)%missing_index>0) then
          if(equal_value(label,missing_value(self))) label=real(self%mappings_(j)%missing_index-1,dp)
        end if
        if(group_code>=0) infrequent(i,j)=equal_value(label,real(group_code,dp))
        if(self%handle_unknown=='use_encoded_value') unknown(i,j)=equal_value(label,self%unknown_value)
        if(unknown(i,j).or.infrequent(i,j)) cycle
        call require(ieee_is_finite(label),'OrdinalEncoder inverse code is not a finite index')
        ! NumPy astype(int64) truncates fractions, and array indices wrap negatives.
        call require(label>=-real(size(values),dp)-1.and.label<real(size(values),dp), &
          'OrdinalEncoder inverse index is out of bounds')
        index=int(label); if(index<0) index=index+size(values)
        call require(index>=0.and.index<size(values),'OrdinalEncoder inverse index is out of bounds')
        x(i,j)=values(index+1)
      end do
    end do
    ! Source inserts None first, then its infrequent string, so group wins ties.
    where(infrequent) unknown=.false.
    if(present(unknown_mask)) unknown_mask=unknown
    if(present(infrequent_mask)) infrequent_mask=infrequent
  end function
end module
