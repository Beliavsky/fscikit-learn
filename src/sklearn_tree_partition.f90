! IndexComparator: sklearn 1.6.1 neighbors/_partition_nodes.pyx (BSD).
! Median/partition ordering: Microsoft STL 8eca13a3bfd44300c848bb93153cded88168aea4,
! Copyright (c) Microsoft Corporation.
! Apache-2.0 WITH LLVM-exception; see LICENSES/Microsoft-STL-* and notices.
! Native one-based coordinate adapter is MIT.
module sklearn_tree_partition
 use sklearn_common,only:dp,require
 implicit none
 private
 public::partition_node
contains
 subroutine partition_node(coordinates,indices,kth)
  real(dp),intent(in)::coordinates(:)
  integer,intent(inout)::indices(0:)
  integer,intent(in)::kth
  integer::first,last,pfirst,plast,i,j,value
  call require(kth>=1.and.kth<=size(indices),'internal tree partition index')
  first=0;last=size(indices)
  do while(last-first>32)
   call partition(first,last,pfirst,plast)
   if(plast<=kth-1)then
    first=plast
   else if(kth-1<pfirst)then
    last=pfirst
   else
    return
   end if
  end do
  do i=first+1,last-1
   value=indices(i);j=i
   do while(j>first)
    if(.not.less(value,indices(j-1)))exit
    indices(j)=indices(j-1);j=j-1
   end do
   indices(j)=value
  end do
 contains
  logical function less(a,b)
   integer,intent(in)::a,b
   less=coordinates(a)<coordinates(b)
   if(coordinates(a)==coordinates(b))less=a<b
  end function
  subroutine swap(a,b)
   integer,intent(in)::a,b
   integer::value
   value=indices(a);indices(a)=indices(b);indices(b)=value
  end subroutine
  subroutine median(first,middle,last)
   integer,intent(in)::first,middle,last
   if(less(indices(middle),indices(first)))call swap(middle,first)
   if(less(indices(last),indices(middle)))then
    call swap(last,middle)
    if(less(indices(middle),indices(first)))call swap(middle,first)
   end if
  end subroutine
  subroutine partition(first,last,pfirst,plast)
   integer,intent(in)::first,last
   integer,intent(out)::pfirst,plast
   integer::mid,step,gfirst,glast
   mid=first+(last-first)/2
   if(last-first-1>40)then
    step=(last-first)/8
    call median(first,first+step,first+2*step)
    call median(mid-step,mid,mid+step)
    call median(last-1-2*step,last-1-step,last-1)
    call median(first+step,mid,last-1-step)
   else
    call median(first,mid,last-1)
   end if
   pfirst=mid;plast=mid+1;gfirst=plast;glast=pfirst
   ! The coordinate/index comparator is a total order: distinct stored IDs
   ! cannot be equivalent, so the Source equal-pivot runs contain one item.
   do
    do while(gfirst<last)
     if(less(indices(pfirst),indices(gfirst)))then
      gfirst=gfirst+1;cycle
     else if(less(indices(gfirst),indices(pfirst)))then
      exit
     else
      if(plast/=gfirst)call swap(plast,gfirst)
      plast=plast+1
     end if
     gfirst=gfirst+1
    end do
    do while(first<glast)
     if(less(indices(glast-1),indices(pfirst)))then
      glast=glast-1;cycle
     else if(less(indices(pfirst),indices(glast-1)))then
      exit
     else
      pfirst=pfirst-1
      if(pfirst/=glast-1)call swap(pfirst,glast-1)
     end if
     glast=glast-1
    end do
    if(glast==first.and.gfirst==last)return
    if(glast==first)then
     if(plast/=gfirst)call swap(pfirst,plast)
     plast=plast+1;call swap(pfirst,gfirst);pfirst=pfirst+1;gfirst=gfirst+1
    else if(gfirst==last)then
     glast=glast-1;pfirst=pfirst-1
     if(glast/=pfirst)call swap(glast,pfirst)
     plast=plast-1;call swap(pfirst,plast)
    else
     glast=glast-1;call swap(gfirst,glast);gfirst=gfirst+1
    end if
   end do
  end subroutine
 end subroutine
end module
