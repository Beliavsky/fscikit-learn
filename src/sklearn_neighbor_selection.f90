! sklearn 1.6.1 utils/_heap.pyx, utils/_sorting.pyx and pairwise ArgKmin (BSD).
! One-based, serial native adapters are MIT; see THIRD_PARTY_NOTICES.md.
module sklearn_neighbor_selection
 use sklearn_common,only:dp,require
 use sklearn_numpy_ordering,only:numpy_argpartition,numpy_argsort
 implicit none
 private
 public::argkmin_order,pairwise_knn_order
 public::neighbor_heap_push,neighbor_simultaneous_sort
 interface neighbor_heap_push
  module procedure heap_push
 end interface
 interface neighbor_simultaneous_sort
  module procedure simultaneous_sort
 end interface
contains
 subroutine heap_push(values,indices,value,index)
  real(dp),intent(inout)::values(:)
  integer,intent(inout)::indices(:)
  real(dp),intent(in)::value
  integer,intent(in)::index
  integer::at,left,right,child,n
  n=size(values)
  if(value>=values(1))return
  values(1)=value;indices(1)=index;at=1
  do
   left=2*at;right=left+1
   if(left>n)exit
   if(right>n)then
    if(values(left)>value)then
     child=left
    else
     exit
    end if
   else if(values(left)>=values(right))then
    if(value<values(left))then
     child=left
    else
     exit
    end if
   else
    if(value<values(right))then
     child=right
    else
     exit
    end if
   end if
   values(at)=values(child);indices(at)=indices(child);at=child
  end do
  values(at)=value;indices(at)=index
 end subroutine
 subroutine swap(values,indices,a,b)
  real(dp),intent(inout)::values(:)
  integer,intent(inout)::indices(:)
  integer,intent(in)::a,b
  real(dp)::v
  integer::i
  v=values(a);values(a)=values(b);values(b)=v
  i=indices(a);indices(a)=indices(b);indices(b)=i
 end subroutine
 recursive subroutine simultaneous_sort(values,indices)
  real(dp),intent(inout)::values(:)
  integer,intent(inout)::indices(:)
  integer::n,pivot,store,i
  real(dp)::value
  n=size(values)
  if(n<=1)return
  if(n==2)then
   if(values(1)>values(2))call swap(values,indices,1,2)
  else if(n==3)then
   if(values(1)>values(2))call swap(values,indices,1,2)
   if(values(2)>values(3))then
    call swap(values,indices,2,3)
    if(values(1)>values(2))call swap(values,indices,1,2)
   end if
  else
   pivot=n/2+1
   if(values(1)>values(n))call swap(values,indices,1,n)
   if(values(n)>values(pivot))then
    call swap(values,indices,n,pivot)
    if(values(1)>values(n))call swap(values,indices,1,n)
   end if
   value=values(n);store=1
   do i=1,n-1
    if(values(i)<value)then
     call swap(values,indices,i,store);store=store+1
    end if
   end do
   call swap(values,indices,store,n)
   if(store>2)call simultaneous_sort(values(:store-1),indices(:store-1))
   if(store+1<n)call simultaneous_sort(values(store+1:),indices(store+1:))
  end if
 end subroutine
 function argkmin_order(surrogate,k,n_queries,selected)result(indices)
  real(dp),intent(in)::surrogate(:)
  integer,intent(in)::k,n_queries
  real(dp),optional,intent(out)::selected(:)
  integer::indices(k),j,thread_indices(k)
  real(dp)::values(k),thread_values(k)
  call require(k>=1.and.k<=size(surrogate),'internal ArgKmin k is out of range')
  values=huge(1._dp);indices=1
  do j=1,size(surrogate)
   call heap_push(values,indices,surrogate(j),j)
  end do
  ! Source auto strategy, default chunk size 256 and one effective thread.
  ! Even one parallel_on_Y heap is merged in storage order before sorting.
  if(size(surrogate)>=n_queries.and.n_queries<=1024)then
   thread_values=values;thread_indices=indices;values=huge(1._dp);indices=1
   do j=1,k
    call heap_push(values,indices,thread_values(j),thread_indices(j))
   end do
  end if
  call simultaneous_sort(values,indices)
  if(present(selected))then
   call require(size(selected)==k,'internal ArgKmin selected values shape')
   selected=values
  end if
 end function
 function pairwise_knn_order(distance,k)result(indices)
  real(dp),intent(in)::distance(:)
  integer,intent(in)::k
  integer::indices(k),partition(size(distance)),order(k)
  partition=numpy_argpartition(distance,k);indices=partition(:k)
  order=numpy_argsort(distance(indices));indices=indices(order)
 end function
end module
