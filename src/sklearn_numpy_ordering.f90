! Portable scalar adaptation of Intel x86-simd-sort index ordering, revision
! 9a1b616d5cd4eaf49f7664fb86ccc1d18bad2b8d, as bundled in NumPy 2.2.6.
! Copyright (C) 2022 Intel Corporation; BSD-3-Clause, LICENSES/x86-simd-sort-BSD.
! No SIMD instructions or C++ runtime are required. Internal real64 helpers.
module sklearn_numpy_ordering
 use sklearn_common,only:dp,require
 use,intrinsic::ieee_arithmetic
 implicit none
 private
 public::numpy_argsort,numpy_argpartition
contains
 subroutine layer(key,idx,perm,mask)
  real(dp),intent(inout)::key(0:7)
  integer,intent(inout)::idx(0:7)
  integer,intent(in)::perm(0:7),mask
  real(dp)::other(0:7),v(0:7)
  integer::ids(0:7),i
  other=key(perm);ids=idx(perm)
  do i=0,7
   if(btest(mask,i)) then
    v(i)=max(key(i),other(i))
   else
    v(i)=min(key(i),other(i))
   end if
   if(v(i)/=key(i)) idx(i)=ids(i)
  end do
  key=v
 end subroutine
 subroutine sort8(key,idx)
  real(dp),intent(inout)::key(0:7)
  integer,intent(inout)::idx(0:7)
  call layer(key,idx,[1,0,3,2,5,4,7,6],int(z'AA'))
  call layer(key,idx,[3,2,1,0,7,6,5,4],int(z'CC'))
  call layer(key,idx,[1,0,3,2,5,4,7,6],int(z'AA'))
  call layer(key,idx,[7,6,5,4,3,2,1,0],int(z'F0'))
  call layer(key,idx,[2,3,0,1,6,7,4,5],int(z'CC'))
  call layer(key,idx,[1,0,3,2,5,4,7,6],int(z'AA'))
 end subroutine
 subroutine clean8(key,idx)
  real(dp),intent(inout)::key(0:7)
  integer,intent(inout)::idx(0:7)
  call layer(key,idx,[4,5,6,7,0,1,2,3],int(z'F0'))
  call layer(key,idx,[2,3,0,1,6,7,4,5],int(z'CC'))
  call layer(key,idx,[1,0,3,2,5,4,7,6],int(z'AA'))
 end subroutine
 subroutine coex(a,b,ia,ib)
  real(dp),intent(inout)::a(0:7),b(0:7)
  integer,intent(inout)::ia(0:7),ib(0:7)
  real(dp)::v
  integer::j,k
  do j=0,7
   if(a(j)<=b(j)) cycle
   v=a(j);a(j)=b(j);b(j)=v;k=ia(j);ia(j)=ib(j);ib(j)=k
  end do
 end subroutine
 subroutine network(x,ids)
  real(dp),intent(in)::x(:)
  integer,intent(inout)::ids(0:)
  real(dp)::keys(0:7,0:31)
  integer::index(0:7,0:31),n,nv,i,j,group,start,l,r,part
  n=size(ids);nv=1
  do while(nv*8<n)
   nv=nv*2
  end do
  keys=ieee_value(0._dp,ieee_positive_inf);index=huge(1)
  do i=0,n-1
   keys(mod(i,8),i/8)=x(ids(i));index(mod(i,8),i/8)=ids(i)
  end do
  do i=0,nv-1
   call sort8(keys(:,i),index(:,i))
  end do
  group=2
  do while(group<=nv)
   do start=0,nv-1,group
    do i=0,group/2-1
     l=start+i;r=start+group-i-1
     keys(:,r)=keys(7:0:-1,r);index(:,r)=index(7:0:-1,r)
     call coex(keys(:,l),keys(:,r),index(:,l),index(:,r))
     keys(:,r)=keys(7:0:-1,r);index(:,r)=index(7:0:-1,r)
    end do
    part=group/2
    do while(part>=2)
     do j=start,start+group-1,part
      do i=0,part/2-1
       call coex(keys(:,j+i),keys(:,j+i+part/2),index(:,j+i),index(:,j+i+part/2))
      end do
     end do
     part=part/2
    end do
    do i=start,start+group-1
     call clean8(keys(:,i),index(:,i))
    end do
   end do
   group=group*2
  end do
  do i=0,n-1
   ids(i)=index(mod(i,8),i/8)
  end do
 end subroutine
 subroutine store8(x,ids,left,right,block,pivot,small,big)
  real(dp),intent(in)::x(:),pivot
  integer,intent(inout)::ids(0:),left,right
  integer,intent(in)::block(8)
  real(dp),intent(inout)::small,big
  integer::i,nlow,high(8),low(8),nhigh
  nlow=0;nhigh=0
  do i=1,8
   small=min(small,x(block(i)));big=max(big,x(block(i)))
   if(x(block(i))>=pivot) then
    nhigh=nhigh+1;high(nhigh)=block(i)
   else
    nlow=nlow+1;low(nlow)=block(i)
   end if
  end do
  ids(left:left+nlow-1)=low(:nlow)
  ids(right-nhigh:right-1)=high(:nhigh)
  left=left+nlow;right=right-nhigh
 end subroutine
 integer function partition(x,ids,first,last,pivot,small,big) result(pos)
  real(dp),intent(in)::x(:),pivot
  integer,intent(inout)::ids(0:)
  integer,intent(in)::first,last
  real(dp),intent(out)::small,big
  integer::left,right,lstore,rstore,i,j,unroll,width,t,count,bl(8,4),br(8,4),b(8,4)
  small=ieee_value(0._dp,ieee_positive_inf);big=-small
  left=first;right=last
  unroll=1;if(last-first>256) unroll=4
  width=8*unroll
  count=mod(right-left,width)
  do i=1,count
   small=min(small,x(ids(left)));big=max(big,x(ids(left)))
   if(x(ids(left))>=pivot) then
    right=right-1;t=ids(left);ids(left)=ids(right);ids(right)=t
   else
    left=left+1
   end if
  end do
  if(left==right) then
   pos=left;return
  end if
  if(right-left==8) then
   b(:,1)=ids(left:left+7);lstore=left;rstore=left+8
   call store8(x,ids,lstore,rstore,b(:,1),pivot,small,big);pos=lstore;return
  end if
  do j=1,unroll
   bl(:,j)=ids(left+(j-1)*8:left+j*8-1)
   br(:,j)=ids(right-width+(j-1)*8:right-width+j*8-1)
  end do
  lstore=left;rstore=right;left=left+width;right=right-width
  do while(right-left/=0)
   if(rstore-right<left-lstore) then
    right=right-width
    do j=1,unroll
     b(:,j)=ids(right+(j-1)*8:right+j*8-1)
    end do
   else
    do j=1,unroll
     b(:,j)=ids(left+(j-1)*8:left+j*8-1)
    end do
    left=left+width
   end if
   do j=1,unroll
    call store8(x,ids,lstore,rstore,b(:,j),pivot,small,big)
   end do
  end do
  do j=1,unroll
   call store8(x,ids,lstore,rstore,bl(:,j),pivot,small,big)
  end do
  do j=1,unroll
   call store8(x,ids,lstore,rstore,br(:,j),pivot,small,big)
  end do
  pos=lstore
 end function
 ! Fallback index sort follows Microsoft STL 8eca13a3bfd44300c848bb93153cded88168aea4.
 ! Copyright (c) Microsoft Corporation; Apache-2.0 WITH LLVM-exception.
 ! See LICENSES/Microsoft-STL-Apache-LLVM.txt and THIRD_PARTY_NOTICES.md.
 subroutine swap(ids,i,j)
  integer,intent(inout)::ids(0:)
  integer,intent(in)::i,j
  integer::t
  t=ids(i);ids(i)=ids(j);ids(j)=t
 end subroutine
 subroutine med3(x,ids,first,middle,last)
  real(dp),intent(in)::x(:)
  integer,intent(inout)::ids(0:)
  integer,intent(in)::first,middle,last
  if(x(ids(middle))<x(ids(first))) call swap(ids,middle,first)
  if(x(ids(last))<x(ids(middle))) then
   call swap(ids,last,middle)
   if(x(ids(middle))<x(ids(first))) call swap(ids,middle,first)
  end if
 end subroutine
 subroutine stl_partition(x,ids,first,last,pfirst,plast)
  real(dp),intent(in)::x(:)
  integer,intent(inout)::ids(0:)
  integer,intent(in)::first,last
  integer,intent(out)::pfirst,plast
  integer::mid,step,gfirst,glast
  mid=first+(last-first)/2
  if(last-first-1>40) then
   step=(last-first)/8
   call med3(x,ids,first,first+step,first+2*step)
   call med3(x,ids,mid-step,mid,mid+step)
   call med3(x,ids,last-1-2*step,last-1-step,last-1)
   call med3(x,ids,first+step,mid,last-1-step)
  else
   call med3(x,ids,first,mid,last-1)
  end if
  pfirst=mid;plast=mid+1
  do while(first<pfirst)
   if(x(ids(pfirst-1))/=x(ids(pfirst))) exit
   pfirst=pfirst-1
  end do
  do while(plast<last)
   if(x(ids(plast))/=x(ids(pfirst))) exit
   plast=plast+1
  end do
  gfirst=plast;glast=pfirst
  do
   do while(gfirst<last)
    if(x(ids(pfirst))<x(ids(gfirst))) then
     gfirst=gfirst+1;cycle
    else if(x(ids(gfirst))<x(ids(pfirst))) then
     exit
    else
     if(plast/=gfirst) call swap(ids,plast,gfirst)
     plast=plast+1
    end if
    gfirst=gfirst+1
   end do
   do while(first<glast)
    if(x(ids(glast-1))<x(ids(pfirst))) then
     glast=glast-1;cycle
    else if(x(ids(pfirst))<x(ids(glast-1))) then
     exit
    else
     pfirst=pfirst-1
     if(pfirst/=glast-1) call swap(ids,pfirst,glast-1)
    end if
    glast=glast-1
   end do
   if(glast==first.and.gfirst==last) return
   if(glast==first) then
    if(plast/=gfirst) call swap(ids,pfirst,plast)
    plast=plast+1;call swap(ids,pfirst,gfirst);pfirst=pfirst+1;gfirst=gfirst+1
   else if(gfirst==last) then
    glast=glast-1;pfirst=pfirst-1
    if(glast/=pfirst) call swap(ids,glast,pfirst)
    plast=plast-1;call swap(ids,pfirst,plast)
   else
    glast=glast-1;call swap(ids,gfirst,glast);gfirst=gfirst+1
   end if
  end do
 end subroutine
 subroutine heap_adjust(x,ids,first,hole,bottom,value)
  real(dp),intent(in)::x(:)
  integer,intent(inout)::ids(0:)
  integer,intent(in)::first,bottom,value
  integer,intent(inout)::hole
  integer::top,child,parent
  top=hole;child=2*hole+2
  do while(child<bottom)
   if(x(ids(first+child))<x(ids(first+child-1))) child=child-1
   ids(first+hole)=ids(first+child);hole=child;child=2*hole+2
  end do
  if(child==bottom) then
   ids(first+hole)=ids(first+child-1);hole=child-1
  end if
  do while(hole>top)
   parent=(hole-1)/2
   if(.not.(x(ids(first+parent))<x(value))) exit
   ids(first+hole)=ids(first+parent);hole=parent
  end do
  ids(first+hole)=value
 end subroutine
 subroutine stl_heap(x,ids,first,last)
  real(dp),intent(in)::x(:)
  integer,intent(inout)::ids(0:)
  integer,intent(in)::first,last
  integer::n,start,hole,value,tail
  n=last-first
  do start=n/2-1,0,-1
   hole=start;value=ids(first+hole);call heap_adjust(x,ids,first,hole,n,value)
  end do
  do tail=n-1,1,-1
   value=ids(first+tail);ids(first+tail)=ids(first);hole=0
   call heap_adjust(x,ids,first,hole,tail,value)
  end do
 end subroutine
 recursive subroutine stl_sort(x,ids,begin,end,ideal)
  real(dp),intent(in)::x(:)
  integer,intent(inout)::ids(0:)
  integer,intent(in)::begin,end,ideal
  integer::first,last,budget,pfirst,plast,i,j,value
  first=begin;last=end;budget=ideal
  do
   if(last-first<=32) then
    do i=first+1,last-1
     value=ids(i);j=i
     do while(j>first)
      if(.not.(x(value)<x(ids(j-1)))) exit
      ids(j)=ids(j-1);j=j-1
     end do
     ids(j)=value
    end do
    return
   end if
   if(budget<=0) then
    call stl_heap(x,ids,first,last);return
   end if
   call stl_partition(x,ids,first,last,pfirst,plast);budget=budget/2+budget/4
   if(pfirst-first<last-plast) then
    call stl_sort(x,ids,first,pfirst,budget);first=plast
   else
    call stl_sort(x,ids,plast,last,budget);last=pfirst
   end if
  end do
 end subroutine
 subroutine fallback(x,ids)
  real(dp),intent(in)::x(:)
  integer,intent(inout)::ids(0:)
  call stl_sort(x,ids,0,size(ids),size(ids))
 end subroutine
 recursive subroutine sort_range(x,ids,left,right,depth,kth)
  real(dp),intent(in)::x(:)
  integer,intent(inout)::ids(0:)
  integer,intent(in)::left,right,depth
  integer,optional,intent(in)::kth
  real(dp)::sample(8),pivot,small,big,tmp
  integer::i,j,stride,point,order(0:7)
  if(right<=left) return
  if(depth<=0) then
   call fallback(x,ids(left:right));return
  end if
  if(right-left+1<=256) then
   call network(x,ids(left:right));return
  end if
  stride=(right-left)/8
  do i=1,8
   sample(i)=x(ids(left+i*stride))
  end do
  order=[(i,i=1,8)];call sort8(sample,order)
  pivot=sample(5);point=partition(x,ids,left,right+1,pivot,small,big)
  if(present(kth)) then
   if(pivot/=small.and.kth<point) then
    call sort_range(x,ids,left,point-1,depth-1,kth)
   else if(pivot/=big.and.kth>=point) then
    call sort_range(x,ids,point,right,depth-1,kth)
   end if
  else
   if(pivot/=small) call sort_range(x,ids,left,point-1,depth-1)
   if(pivot/=big) call sort_range(x,ids,point,right,depth-1)
  end if
 end subroutine
 function numpy_argsort(x) result(ids)
  real(dp),intent(in)::x(:)
  integer::ids(size(x)),i,n,depth
  n=size(x);ids=[(i,i=1,n)]
  call require(.not.any(ieee_is_nan(x)),'internal NumPy ordering requires non-NaN keys')
  if(n<=1) return
  depth=2*int(log(real(n,dp))/log(2._dp));call sort_range(x,ids,0,n-1,depth)
 end function
 function numpy_argpartition(x,kth) result(ids)
  real(dp),intent(in)::x(:)
  integer,intent(in)::kth
  integer::ids(size(x)),i,n,depth
  n=size(x);ids=[(i,i=1,n)]
  call require(kth>=1.and.kth<=n,'internal NumPy partition position out of range')
  call require(.not.any(ieee_is_nan(x)),'internal NumPy ordering requires non-NaN keys')
  if(n<=1) return
  depth=2*int(log(real(n,dp))/log(2._dp));call sort_range(x,ids,0,n-1,depth,kth-1)
 end function
end module
