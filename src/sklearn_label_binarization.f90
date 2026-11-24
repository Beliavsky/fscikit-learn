! Dense integer-label contracts from sklearn 1.6.1 preprocessing/_label.py.
! Adapted under BSD-3-Clause; see THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_label_binarization
  use sklearn_common, only: dp,require,unique_labels,label_index
  use sklearn_sparse,only:csr_matrix
  use sklearn_numpy_ordering,only:numpy_argsort
  use sklearn_integer_sets,only:integer_set_order
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: label_binarizer,multi_label_binarizer,label_set
  ! A native representation of one sample's iterable of integer labels.
  ! Unallocated values represent an empty set; duplicates are permitted.
  type :: label_set
    integer, allocatable :: values(:)
  end type
  type :: label_binarizer
    integer :: neg_label=0,pos_label=1
    integer, allocatable :: classes_(:)
    character(24) :: y_type_=''
  contains
    procedure :: fit=>lb_fit
    procedure :: transform=>lb_transform
    procedure :: fit_transform=>lb_fit_transform
    procedure, private :: lb_inverse_real,lb_inverse_integer
    generic :: inverse_transform=>lb_inverse_real,lb_inverse_integer
  end type
  type :: multi_label_binarizer
    integer, allocatable :: classes(:),classes_(:)
  contains
    procedure :: fit=>mlb_fit
    procedure :: transform=>mlb_transform
    procedure :: fit_transform=>mlb_fit_transform
    procedure :: transform_csr=>mlb_transform_csr
    procedure :: fit_transform_csr=>mlb_fit_transform_csr
    procedure, private :: mlb_inverse_real,mlb_inverse_integer
    procedure,private::mlb_inverse_csr
    generic :: inverse_transform=>mlb_inverse_real,mlb_inverse_integer,mlb_inverse_csr
  end type
contains
  subroutine lb_parameters(self)
    class(label_binarizer), intent(in) :: self
    call require(self%neg_label<self%pos_label,'LabelBinarizer neg_label must be less than pos_label')
  end subroutine
  subroutine lb_fit(self,y)
    class(label_binarizer), intent(inout) :: self
    integer, intent(in) :: y(:)
    call lb_parameters(self)
    call require(size(y)>0,'LabelBinarizer labels must be nonempty')
    self%classes_=unique_labels(y)
    self%y_type_='binary'
    if(size(self%classes_)>2) self%y_type_='multiclass'
  end subroutine
  function lb_transform(self,y) result(z)
    class(label_binarizer), intent(in) :: self
    integer, intent(in) :: y(:)
    integer, allocatable :: z(:,:),query_classes(:)
    integer :: c,width,i,j
    logical :: binary_query
    call require(allocated(self%classes_),'LabelBinarizer is not fitted')
    call lb_parameters(self)
    call require(size(y)>0,'LabelBinarizer labels must be nonempty')
    c=size(self%classes_)
    call require(c>0,'LabelBinarizer fitted classes must be nonempty')
    query_classes=unique_labels(y); binary_query=size(query_classes)<=2.and.c<=2
    width=c
    if(binary_query) width=1
    allocate(z(size(y),width)); z=self%neg_label
    do i=1,size(y)
      j=label_index(self%classes_,y(i))
      if(binary_query) then
        if(c==2.and.j==2) z(i,1)=self%pos_label
      else
        if(j>0) z(i,j)=self%pos_label
      end if
    end do
  end function
  function lb_fit_transform(self,y) result(z)
    class(label_binarizer), intent(inout) :: self
    integer, intent(in) :: y(:)
    integer, allocatable :: z(:,:)
    call self%fit(y); z=self%transform(y)
  end function
  function lb_inverse_real(self,z,threshold) result(y)
    class(label_binarizer), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), intent(in), optional :: threshold
    integer, allocatable :: y(:)
    real(dp) :: cut
    integer :: c,i,j,column
    call require(allocated(self%classes_),'LabelBinarizer is not fitted')
    c=size(self%classes_)
    call require(c>0,'LabelBinarizer fitted classes must be nonempty')
    call require(size(z,2)>0,'LabelBinarizer inverse needs at least one column')
    call require(all(ieee_is_finite(z)),'LabelBinarizer inverse scores must be finite')
    allocate(y(size(z,1)))
    if(trim(self%y_type_)=='multiclass') then
      do i=1,size(z,1)
        ! Source classes.take(...,mode="clip") also accepts a changed score width.
        j=min(maxloc(z(i,:),dim=1),c); y(i)=self%classes_(j)
      end do
    else
      call require(size(z,2)<=2,'LabelBinarizer binary inverse needs one or two columns')
      cut=.5_dp*real(self%pos_label,dp)+.5_dp*real(self%neg_label,dp)
      if(present(threshold)) cut=threshold
      call require(ieee_is_finite(cut),'LabelBinarizer inverse threshold must be finite')
      column=size(z,2)
      if(c==1.and.column==2) then
        call require(.not.any(z(:,2)>cut),'LabelBinarizer single-class inverse selected missing positive class')
      end if
      y=self%classes_(1)
      if(c==2) then
        do i=1,size(z,1)
          if(z(i,column)>cut) y(i)=self%classes_(2)
        end do
      end if
    end if
  end function
  function lb_inverse_integer(self,z,threshold) result(y)
    class(label_binarizer), intent(in) :: self
    integer, intent(in) :: z(:,:)
    real(dp), intent(in), optional :: threshold
    integer, allocatable :: y(:)
    y=lb_inverse_real(self,real(z,dp),threshold)
  end function
  subroutine mlb_fit(self,y)
    class(multi_label_binarizer), intent(inout) :: self
    type(label_set), intent(in) :: y(:)
    integer, allocatable :: flat(:),distinct(:)
    integer :: i,n
    integer(int64) :: total,first
    if(allocated(self%classes)) then
      if(size(self%classes)>0) then
        distinct=sorted_unique(self%classes)
        call require(size(distinct)==size(self%classes),'MultiLabelBinarizer classes contain duplicates')
      end if
      self%classes_=self%classes
    else
      total=0
      do i=1,size(y)
        if(allocated(y(i)%values)) total=total+int(size(y(i)%values),int64)
      end do
      call require(total<=huge(n),'MultiLabelBinarizer label count overflow')
      allocate(flat(int(total))); first=1
      do i=1,size(y)
        if(.not.allocated(y(i)%values)) cycle
        n=size(y(i)%values); flat(first:first+n-1)=y(i)%values; first=first+n
      end do
      if(total>0) then
        self%classes_=sorted_unique(flat)
      else
        self%classes_=[integer ::]
      end if
    end if
  end subroutine
  function mlb_transform(self,y) result(z)
    class(multi_label_binarizer), intent(in) :: self
    type(label_set), intent(in) :: y(:)
    integer, allocatable :: z(:,:)
    integer :: i,j,column
    call require(allocated(self%classes_),'MultiLabelBinarizer is not fitted')
    allocate(z(size(y),size(self%classes_))); z=0
    do i=1,size(y)
      if(.not.allocated(y(i)%values)) cycle
      do j=1,size(y(i)%values)
        column=label_index(self%classes_,y(i)%values(j))
        ! Source ignores unknown labels (and emits a Python warning).
        if(column>0) z(i,column)=1
      end do
    end do
  end function
  function mlb_fit_transform(self,y) result(z)
    class(multi_label_binarizer), intent(inout) :: self
    type(label_set), intent(in) :: y(:)
    integer, allocatable :: z(:,:)
    call self%fit(y); z=self%transform(y)
  end function
  function mlb_inverse_real(self,z) result(y)
    class(multi_label_binarizer), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    type(label_set), allocatable :: y(:)
    integer :: i
    call require(allocated(self%classes_),'MultiLabelBinarizer is not fitted')
    call require(size(z,2)==size(self%classes_),'MultiLabelBinarizer inverse class count mismatch')
    call require(all(ieee_is_finite(z)),'MultiLabelBinarizer indicators must be finite')
    call require(all(z==0.or.z==1),'MultiLabelBinarizer indicators must be zero or one')
    allocate(y(size(z,1)))
    do i=1,size(z,1)
      y(i)%values=pack(self%classes_,z(i,:)==1)
    end do
  end function
  function mlb_inverse_integer(self,z) result(y)
    class(multi_label_binarizer), intent(in) :: self
    integer, intent(in) :: z(:,:)
    type(label_set), allocatable :: y(:)
    y=mlb_inverse_real(self,real(z,dp))
  end function
  function sorted_unique(labels) result(classes)
    integer,intent(in)::labels(:)
    integer,allocatable::classes(:),order(:),work(:)
    integer::i,n
    order=numpy_argsort(real(labels,dp));allocate(work(size(labels)));n=0
    do i=1,size(order)
      if(i>1) then
        if(labels(order(i))==labels(order(i-1))) cycle
      end if
      n=n+1;work(n)=labels(order(i))
    end do
    classes=work(:n)
  end function
  integer function mapped_column(sorted,order,label) result(column)
    integer,intent(in)::sorted(:),order(:),label
    integer::left,right,middle
    column=0;left=1;right=size(sorted)
    do while(left<=right)
      middle=left+(right-left)/2
      if(sorted(middle)==label) then
        column=order(middle);return
      else if(sorted(middle)<label) then
        left=middle+1
      else
        right=middle-1
      end if
    end do
  end function
  function sparse_map(self,y,discover) result(z)
    class(multi_label_binarizer),intent(in)::self
    type(label_set),intent(in)::y(:)
    logical,intent(in)::discover
    type(csr_matrix)::z
    integer,allocatable::order(:),sorted(:),work(:),row(:),keys(:),encounter(:),reverse(:)
    integer::i,j,n,column,seen,pos
    integer(int64)::total
    call require(allocated(self%classes_),'MultiLabelBinarizer is not fitted')
    order=numpy_argsort(real(self%classes_,dp));sorted=self%classes_(order)
    total=0
    do i=1,size(y)
      if(allocated(y(i)%values)) total=total+int(size(y(i)%values),int64)
    end do
    call require(total<huge(1),'MultiLabelBinarizer sparse stored count overflow')
    allocate(work(int(total)),encounter(size(sorted)),reverse(size(sorted)));encounter=0;seen=0;pos=1
    z%n_rows=size(y);z%n_cols=size(sorted);allocate(z%indptr(size(y)+1))
    do i=1,size(y)
      z%indptr(i)=pos;n=0
      if(allocated(y(i)%values)) then
        allocate(row(size(y(i)%values)))
        do j=1,size(y(i)%values)
          column=mapped_column(sorted,order,y(i)%values(j))
          if(column==0) cycle
          n=n+1
          if(discover) then
            if(encounter(column)==0) then
              seen=seen+1;encounter(column)=seen;reverse(seen)=column
            end if
            row(n)=encounter(column)-1
          else
            row(n)=column-1
          end if
        end do
        keys=integer_set_order(row(:n));deallocate(row)
        if(discover) then
          work(pos:pos+size(keys)-1)=reverse(keys+1)
        else
          work(pos:pos+size(keys)-1)=keys+1
        end if
        pos=pos+size(keys)
      end if
    end do
    z%indptr(size(y)+1)=pos;z%indices=work(:pos-1);allocate(z%data(pos-1));z%data=1
    call z%validate()
  end function
  function mlb_transform_csr(self,y) result(z)
    class(multi_label_binarizer),intent(in)::self
    type(label_set),intent(in)::y(:)
    type(csr_matrix)::z
    z=sparse_map(self,y,.false.)
  end function
  function mlb_fit_transform_csr(self,y) result(z)
    class(multi_label_binarizer),intent(inout)::self
    type(label_set),intent(in)::y(:)
    type(csr_matrix)::z
    call self%fit(y);z=sparse_map(self,y,.not.allocated(self%classes))
  end function
  function mlb_inverse_csr(self,z) result(y)
    class(multi_label_binarizer),intent(in)::self
    type(csr_matrix),intent(in)::z
    type(label_set),allocatable::y(:)
    integer::i
    call require(allocated(self%classes_),'MultiLabelBinarizer is not fitted')
    call z%validate()
    call require(z%n_cols==size(self%classes_),'MultiLabelBinarizer inverse class count mismatch')
    call require(all(ieee_is_finite(z%data)),'MultiLabelBinarizer indicators must be finite')
    call require(all(z%data==0.or.z%data==1),'MultiLabelBinarizer indicators must be zero or one')
    allocate(y(z%n_rows))
    do i=1,z%n_rows
      ! Source sparse inverse includes stored zeros and duplicate indices.
      y(i)%values=self%classes_(z%indices(z%indptr(i):z%indptr(i+1)-1))
    end do
  end function
end module
