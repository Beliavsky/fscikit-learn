! sklearn 1.6.1 utils/class_weight.py contracts (retained BSD attribution).
module sklearn_class_weight
  use sklearn_common, only: dp,require
  use sklearn_sparse, only: csr_matrix
  use sklearn_ieee_math, only: ieee_product,ieee_difference
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: class_weight_map,compute_class_weight,compute_sample_weight
  type :: class_weight_map
    integer, allocatable :: labels(:)
    real(dp), allocatable :: weights(:)
  end type
  interface compute_class_weight
    module procedure class_weights_integer,class_weights_logical
  end interface
  interface compute_sample_weight
    module procedure sample_vector,sample_matrix,sample_logical_vector,sample_logical_matrix,sample_sparse
  end interface
contains
  function unique_sorted(x) result(values)
    integer, intent(in) :: x(:)
    integer, allocatable :: values(:)
    integer :: a(size(x)),b(size(x)),n,width,left,middle,right,i,j,k,count_
    n=size(x);a=x;width=1
    do while(width<n)
      left=1
      do while(left<=n)
        middle=left+min(width,n-left+1)-1;right=middle+min(width,n-middle)
        i=left;j=middle+1
        do k=left,right
          if(i>middle) then
            b(k)=a(j);j=j+1
          else if(j>right) then
            b(k)=a(i);i=i+1
          else if(a(i)<=a(j)) then
            b(k)=a(i);i=i+1
          else
            b(k)=a(j);j=j+1
          end if
        end do
        left=right+1
      end do
      a=b
      if(width>=n-width) exit
      width=width*2
    end do
    count_=0
    do i=1,n
      if(count_>0) then
        if(a(i)==a(count_)) cycle
      end if
      count_=count_+1;a(count_)=a(i)
    end do
    values=a(:count_)
  end function
  integer function find_label(values,label) result(index_)
    integer, intent(in) :: values(:),label
    integer :: left,right,middle
    left=1;right=size(values);index_=0
    do while(left<=right)
      middle=left+(right-left)/2
      if(values(middle)==label) then
        index_=middle;return
      else if(values(middle)<label) then
        left=middle+1
      else
        right=middle-1
      end if
    end do
  end function
  function mode_value(mode,has_map) result(value)
    character(*), optional, intent(in) :: mode
    logical, intent(in) :: has_map
    character(8) :: value
    value='none';if(present(mode)) then
      call require(mode=='none'.or.mode=='balanced','class weight mode must be none or balanced')
      value=mode
    end if
    call require(.not.(value=='balanced'.and.has_map),'class weight map cannot accompany balanced mode')
  end function
  function class_weights_integer(classes,y,mode,class_weight) result(weights)
    integer, intent(in) :: classes(:),y(:)
    character(*), optional, intent(in) :: mode
    type(class_weight_map), optional, intent(in) :: class_weight
    real(dp), allocatable :: weights(:),ordered_weights(:)
    integer, allocatable :: classes_sorted(:),observed(:),keys(:),counts(:)
    integer :: i,index_,weighted_count,nkeys
    character(8) :: choice
    choice=mode_value(mode,present(class_weight));classes_sorted=unique_sorted(classes)
    do i=1,size(y)
      call require(find_label(classes_sorted,y(i))>0,'class weights classes must include every target label')
    end do
    allocate(weights(size(classes)));weights=1
    if(choice=='balanced') then
      observed=unique_sorted(y)
      call require(size(observed)>0,'class weights balanced targets must be nonempty')
      allocate(counts(size(observed)));counts=0
      do i=1,size(y)
        index_=find_label(observed,y(i));counts(index_)=counts(index_)+1
      end do
      do i=1,size(classes)
        index_=find_label(observed,classes(i))
        call require(index_>0,'class weights balanced classes must occur in targets')
        weights(i)=real(size(y),dp)/(real(size(observed),dp)*real(counts(index_),dp))
      end do
    else if(present(class_weight)) then
      call require(allocated(class_weight%labels).eqv.allocated(class_weight%weights), &
        'class weight map storage is incomplete')
      if(.not.allocated(class_weight%labels)) return
      nkeys=size(class_weight%labels)
      call require(nkeys==size(class_weight%weights),'class weight map label/value lengths differ')
      keys=unique_sorted(class_weight%labels)
      call require(size(keys)==nkeys,'class weight map labels must be unique')
      allocate(ordered_weights(nkeys))
      do i=1,nkeys
        ordered_weights(find_label(keys,class_weight%labels(i)))=class_weight%weights(i)
      end do
      weighted_count=0
      do i=1,size(classes)
        index_=find_label(keys,classes(i))
        if(index_==0) cycle
        weights(i)=ordered_weights(index_);weighted_count=weighted_count+1
      end do
      if(weighted_count<size(classes)) call require(weighted_count==nkeys, &
        'class weight map omits classes while containing unknown labels')
    end if
  end function
  function class_weights_logical(classes,y,mode,class_weight) result(weights)
    logical, intent(in) :: classes(:),y(:)
    character(*), optional, intent(in) :: mode
    type(class_weight_map), optional, intent(in) :: class_weight
    real(dp), allocatable :: weights(:)
    weights=class_weights_integer(merge(1,0,classes),merge(1,0,y),mode,class_weight)
  end function
  function selected_indices(n,indices) result(selected)
    integer, intent(in) :: n
    integer, optional, intent(in) :: indices(:)
    integer, allocatable :: selected(:)
    integer :: i
    if(present(indices)) then
      selected=indices
      do i=1,size(selected)
        call require(selected(i)/=0.and.selected(i)<=n.and.selected(i)>=-n, &
          'class sample weight index is out of range')
        if(selected(i)<0) selected(i)=n+selected(i)+1
      end do
    else
      selected=[(i,i=1,n)]
    end if
  end function
  function expanded(y,choice,class_weight,selected,subsample) result(weights)
    integer, intent(in) :: y(:),selected(:)
    character(*), intent(in) :: choice
    type(class_weight_map), optional, intent(in) :: class_weight
    logical, intent(in) :: subsample
    real(dp), allocatable :: weights(:),per_class(:)
    integer, allocatable :: classes(:),target(:)
    integer :: i,index_
    if(subsample) then
      target=y(selected)
    else
      target=y
    end if
    classes=unique_sorted(target);per_class=class_weights_integer(classes,target,choice,class_weight)
    allocate(weights(size(y)));weights=0
    do i=1,size(y)
      index_=find_label(classes,y(i))
      if(index_>0) weights(i)=per_class(index_)
    end do
  end function
  function sample_vector(y,mode,class_weight,indices) result(weights)
    integer, intent(in) :: y(:)
    character(*), optional, intent(in) :: mode
    type(class_weight_map), optional, intent(in) :: class_weight
    integer, optional, intent(in) :: indices(:)
    real(dp), allocatable :: weights(:)
    integer, allocatable :: selected(:)
    character(8) :: choice
    choice=mode_value(mode,present(class_weight))
    if(present(indices)) call require(choice=='balanced','class sample weights subsampling requires balanced mode')
    selected=selected_indices(size(y),indices)
    weights=expanded(y,choice,class_weight,selected,present(indices))
  end function
  subroutine sample_controls(n,q,mode,class_weight,indices,choice,selected)
    integer, intent(in) :: n,q
    character(*), optional, intent(in) :: mode
    type(class_weight_map), optional, intent(in) :: class_weight(:)
    integer, optional, intent(in) :: indices(:)
    character(8), intent(out) :: choice
    integer, allocatable, intent(out) :: selected(:)
    choice=mode_value(mode,present(class_weight))
    if(present(indices)) call require(choice=='balanced','class sample weights subsampling requires balanced mode')
    if(q>1) call require(choice=='balanced'.or.present(class_weight), &
      'class sample weights multioutput needs balanced mode or per-output maps')
    if(present(class_weight)) call require(size(class_weight)==q,'class sample weight map count differs from outputs')
    if(q==0) then
      selected=[integer ::]
    else
      selected=selected_indices(n,indices)
    end if
  end subroutine
  function sample_matrix(y,mode,class_weight,indices) result(weights)
    integer, intent(in) :: y(:,:)
    character(*), optional, intent(in) :: mode
    type(class_weight_map), optional, intent(in) :: class_weight(:)
    integer, optional, intent(in) :: indices(:)
    real(dp), allocatable :: weights(:),column_weights(:)
    integer, allocatable :: selected(:)
    integer :: i,j
    character(8) :: choice
    call sample_controls(size(y,1),size(y,2),mode,class_weight,indices,choice,selected)
    allocate(weights(size(y,1)));weights=1
    if(size(y,2)==0) then
      weights=[1._dp];return
    end if
    do j=1,size(y,2)
      if(present(class_weight)) then
        column_weights=expanded(y(:,j),choice,class_weight(j),selected,present(indices))
      else
        column_weights=expanded(y(:,j),choice,selected=selected,subsample=present(indices))
      end if
      do i=1,size(weights)
        weights(i)=ieee_product(weights(i),column_weights(i))
      end do
    end do
  end function
  function sample_logical_vector(y,mode,class_weight,indices) result(weights)
    logical, intent(in) :: y(:)
    character(*), optional, intent(in) :: mode
    type(class_weight_map), optional, intent(in) :: class_weight
    integer, optional, intent(in) :: indices(:)
    real(dp), allocatable :: weights(:)
    weights=sample_vector(merge(1,0,y),mode,class_weight,indices)
  end function
  function sample_logical_matrix(y,mode,class_weight,indices) result(weights)
    logical, intent(in) :: y(:,:)
    character(*), optional, intent(in) :: mode
    type(class_weight_map), optional, intent(in) :: class_weight(:)
    integer, optional, intent(in) :: indices(:)
    real(dp), allocatable :: weights(:)
    weights=sample_matrix(merge(1,0,y),mode,class_weight,indices)
  end function
  function sample_sparse(y,mode,class_weight,indices) result(weights)
    type(csr_matrix), intent(in) :: y
    character(*), optional, intent(in) :: mode
    type(class_weight_map), optional, intent(in) :: class_weight(:)
    integer, optional, intent(in) :: indices(:)
    real(dp), allocatable :: weights(:),column_weights(:),values(:),column(:)
    integer, allocatable :: selected(:),ptr(:),next(:),rows(:),labels(:)
    integer :: i,j,k,pos
    character(8) :: choice
    call y%validate()
    call sample_controls(y%n_rows,y%n_cols,mode,class_weight,indices,choice,selected)
    call require(all(ieee_is_finite(y%data)),'class sample weight sparse labels must be finite')
    allocate(weights(y%n_rows));weights=1
    if(y%n_cols==0) then
      weights=[1._dp];return
    end if
    ! A column index permits one-column workspaces without densifying the target matrix.
    call require(y%n_cols<huge(1),'class sample weight sparse column count exceeds index range')
    allocate(ptr(y%n_cols+1),next(y%n_cols),rows(y%nnz()),values(y%nnz()),column(y%n_rows),labels(y%n_rows))
    ptr=0
    do k=1,y%nnz()
      ptr(y%indices(k)+1)=ptr(y%indices(k)+1)+1
    end do
    ptr(1)=1
    do j=1,y%n_cols
      ptr(j+1)=ptr(j+1)+ptr(j)
    end do
    next=ptr(:y%n_cols)
    do i=1,y%n_rows
      do k=y%indptr(i),y%indptr(i+1)-1
        j=y%indices(k);pos=next(j);next(j)=pos+1;rows(pos)=i;values(pos)=y%data(k)
      end do
    end do
    do j=1,y%n_cols
      column=0
      do k=ptr(j),ptr(j+1)-1
        i=rows(k);column(i)=ieee_difference(column(i),-values(k))
      end do
      call require(all(ieee_is_finite(column)),'class sample weight sparse label sum must be finite')
      call require(all(column>=real(-huge(1),dp)-1).and.all(column<=real(huge(1),dp)), &
        'class sample weight sparse labels exceed integer range')
      labels=nint(column)
      call require(all(real(labels,dp)==column),'class sample weight sparse labels must be integers')
      if(present(class_weight)) then
        column_weights=expanded(labels,choice,class_weight(j),selected,present(indices))
      else
        column_weights=expanded(labels,choice,selected=selected,subsample=present(indices))
      end if
      do i=1,size(weights)
        weights(i)=ieee_product(weights(i),column_weights(i))
      end do
    end do
  end function
end module
