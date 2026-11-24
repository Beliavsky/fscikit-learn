! sklearn 1.6.1 preprocessing/_label.py label_binarize (BSD-3-Clause).
! Native integer arrays and owned real64 CSR adapters are MIT integration.
module sklearn_label_binarize
 use sklearn_common,only:dp,require
 use sklearn_numpy_ordering,only:numpy_argsort
 use sklearn_sparse,only:csr_matrix
 use,intrinsic::iso_fortran_env,only:int64
 implicit none
 private
 public::label_binarize,label_binarize_csr
 interface label_binarize
  module procedure vector_dense,matrix_dense
 end interface
 interface label_binarize_csr
  module procedure vector_csr,matrix_csr
 end interface
contains
 subroutine parameters(neg_label,pos_label,negative,positive,sparse)
  integer,optional,intent(in)::neg_label,pos_label
  integer,intent(out)::negative,positive
  logical,intent(in)::sparse
  negative=0;positive=1
  if(present(neg_label)) negative=neg_label
  if(present(pos_label)) positive=pos_label
  call require(negative<positive,'label_binarize neg_label must be less than pos_label')
  if(sparse) call require(negative==0.and.positive/=0, &
   'label_binarize sparse output requires zero neg_label and nonzero pos_label')
 end subroutine
 function sorted_classes(classes) result(sorted)
  integer,intent(in)::classes(:)
  integer,allocatable::sorted(:)
  integer,allocatable::order(:)
  ! Keep repeated classes: the source function does not enforce uniqueness.
  order=numpy_argsort(real(classes,dp));sorted=classes(order)
 end function
 logical function two_values(y) result(binary)
  integer,intent(in)::y(:)
  integer::i,second
  logical::found
  binary=.true.;found=.false.;second=0
  if(size(y)==0) return
  do i=2,size(y)
   if(y(i)==y(1)) cycle
   if(.not.found) then
    second=y(i);found=.true.
   else if(y(i)/=second) then
    binary=.false.;return
   end if
  end do
 end function
 integer function search(a,value) result(index)
  integer,intent(in)::a(:),value
  integer::left,right,middle
  left=1;right=size(a)+1
  do while(left<right)
   middle=left+(right-left)/2
   if(a(middle)<value) then
    left=middle+1
   else
    right=middle
   end if
  end do
  index=left
 end function
 function columns(classes,sorted,binary) result(order)
  integer,intent(in)::classes(:),sorted(:)
  logical,intent(in)::binary
  integer,allocatable::order(:)
  integer::j
  order=[(j,j=1,size(classes))]
  if(any(classes/=sorted)) then
   do j=1,size(classes)
    order(j)=search(sorted,classes(j))
   end do
  end if
  if(binary) then
   call require(size(order)>0,'label_binarize binary output needs at least one class')
   order=order(size(order):size(order))
  end if
 end function
 function vector_map(y,classes,order) result(known)
  integer,intent(in)::y(:),classes(:)
  integer,allocatable,intent(out)::order(:)
  integer,allocatable::known(:),sorted(:)
  logical::binary
  integer::i,k
  call require(size(y)>0,'label_binarize target must be nonempty')
  binary=two_values(y).and.size(classes)<3
  sorted=sorted_classes(classes);order=columns(classes,sorted,binary)
  allocate(known(size(y)));known=0
  if(binary.and.size(classes)==1) return
  do i=1,size(y)
   k=search(sorted,y(i))
   if(k<=size(sorted)) then
    if(sorted(k)==y(i)) known(i)=k
   end if
  end do
 end function
 function vector_dense(y,classes,neg_label,pos_label) result(z)
  integer,intent(in)::y(:),classes(:)
  integer,optional,intent(in)::neg_label,pos_label
  integer,allocatable::z(:,:),known(:),order(:)
  integer::negative,positive,i,j
  call parameters(neg_label,pos_label,negative,positive,.false.)
  known=vector_map(y,classes,order)
  allocate(z(size(y),size(order)));z=negative
  do j=1,size(order)
   do i=1,size(y)
    if(known(i)==order(j)) z(i,j)=positive
   end do
  end do
 end function
 subroutine inverted(order,n,ptr,index)
  integer,intent(in)::order(:),n
  integer,allocatable,intent(out)::ptr(:),index(:)
  integer,allocatable::next(:)
  integer::j
  allocate(ptr(n+1),index(size(order)));ptr=0
  do j=1,size(order)
   ptr(order(j)+1)=ptr(order(j)+1)+1
  end do
  ptr(1)=1
  do j=2,n+1
   ptr(j)=ptr(j)+ptr(j-1)
  end do
  next=ptr
  do j=1,size(order)
   index(next(order(j)))=j;next(order(j))=next(order(j))+1
  end do
 end subroutine
 function vector_csr(y,classes,neg_label,pos_label) result(z)
  integer,intent(in)::y(:),classes(:)
  integer,optional,intent(in)::neg_label,pos_label
  type(csr_matrix)::z
  integer,allocatable::known(:),order(:),ptr(:),index(:)
  integer::negative,positive,i,j,k
  integer(int64)::stored
  call parameters(neg_label,pos_label,negative,positive,.true.)
  known=vector_map(y,classes,order);call inverted(order,size(classes),ptr,index);stored=0
  do i=1,size(y)
   if(known(i)>0) stored=stored+ptr(known(i)+1)-ptr(known(i))
  end do
  call require(stored<huge(1),'label_binarize sparse stored count overflow')
  z%n_rows=size(y);z%n_cols=size(order)
  allocate(z%indptr(size(y)+1),z%indices(int(stored)),z%data(int(stored)));k=1
  do i=1,size(y)
   z%indptr(i)=k
   if(known(i)>0) then
    do j=ptr(known(i)),ptr(known(i)+1)-1
     z%indices(k)=index(j);z%data(k)=real(positive,dp);k=k+1
    end do
   end if
  end do
  z%indptr(size(y)+1)=k;call z%validate()
 end function
 subroutine indicator_map(y,classes,order)
  integer,intent(in)::y(:,:),classes(:)
  integer,allocatable,intent(out)::order(:)
  integer,allocatable::sorted(:)
  call require(size(y,1)>0.and.size(y,2)>0,'label_binarize target must be nonempty')
  call require(two_values(reshape(y,[size(y)])),'label_binarize multioutput targets are unsupported')
  call require(size(classes)==size(y,2),'label_binarize indicator class count mismatch')
  sorted=sorted_classes(classes);order=columns(classes,sorted,.false.)
 end subroutine
 integer function mapped(value,negative,positive) result(output)
  integer,intent(in)::value,negative,positive
  output=negative
  if(value/=0) then
   output=positive
   ! Source preserves nonbinary stored indicator values when pos_label=1.
   if(positive==1) output=value
  end if
 end function
 function matrix_dense(y,classes,neg_label,pos_label) result(z)
  integer,intent(in)::y(:,:),classes(:)
  integer,optional,intent(in)::neg_label,pos_label
  integer,allocatable::z(:,:),order(:)
  integer::negative,positive,i,j
  if(size(y,2)==1) then
   z=vector_dense(y(:,1),classes,neg_label,pos_label);return
  end if
  call parameters(neg_label,pos_label,negative,positive,.false.)
  call indicator_map(y,classes,order);allocate(z(size(y,1),size(order)))
  do j=1,size(order)
   do i=1,size(y,1)
    z(i,j)=mapped(y(i,order(j)),negative,positive)
   end do
  end do
 end function
 function matrix_csr(y,classes,neg_label,pos_label) result(z)
  integer,intent(in)::y(:,:),classes(:)
  integer,optional,intent(in)::neg_label,pos_label
  type(csr_matrix)::z
  integer,allocatable::order(:),ptr(:),index(:)
  integer::negative,positive,i,j,k,column
  integer(int64)::stored
  if(size(y,2)==1) then
   z=vector_csr(y(:,1),classes,neg_label,pos_label);return
  end if
  call parameters(neg_label,pos_label,negative,positive,.true.)
  call indicator_map(y,classes,order);call inverted(order,size(classes),ptr,index);stored=0
  do j=1,size(order)
   stored=stored+count(y(:,order(j))/=0,kind=int64)
  end do
  call require(stored<huge(1),'label_binarize sparse stored count overflow')
  z%n_rows=size(y,1);z%n_cols=size(order)
  allocate(z%indptr(z%n_rows+1),z%indices(int(stored)),z%data(int(stored)));k=1
  do i=1,z%n_rows
   z%indptr(i)=k
   ! scipy CSR column slicing traverses the original stored columns first.
   do column=1,size(y,2)
    if(y(i,column)==0) cycle
    do j=ptr(column),ptr(column+1)-1
     z%indices(k)=index(j);z%data(k)=real(mapped(y(i,column),negative,positive),dp);k=k+1
    end do
   end do
  end do
  z%indptr(z%n_rows+1)=k;call z%validate()
 end function
end module
