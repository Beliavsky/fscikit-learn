! Adapted integer-only insertion/iteration protocol from CPython 3.13.3
! Objects/setobject.c, Raymond D. Hettinger, derived from dictobject.c.
! Copyright (c) 2001-2024 Python Software Foundation; All Rights Reserved.
! PSF License Version 2: LICENSES/Python-3.13.3-LICENSE.
! Changes: scalar Fortran arrays, nonnegative default-integer identity hashes,
! no deletion/object callbacks/refcounts, owned result and checked capacities.
module sklearn_integer_sets
 use sklearn_common,only:require
 use,intrinsic::iso_fortran_env,only:int64
 implicit none
 private
 public::integer_set_order
contains
 integer function position(table,key) result(slot)
  integer,intent(in)::table(0:),key
  integer(int64)::mask,i,perturb
  integer::j,probes
  mask=int(size(table)-1,int64);perturb=int(key,int64);i=iand(perturb,mask)
  do
   probes=0
   if(i+9<=mask) probes=9
   do j=0,probes
    slot=int(i)+j
    if(table(slot)==-1.or.table(slot)==key) return
   end do
   perturb=shiftr(perturb,5);i=iand(i*5+1+perturb,mask)
  end do
 end function
 function integer_set_order(keys) result(values)
  integer,intent(in)::keys(:)
  integer,allocatable::values(:),table(:),grown(:)
  integer::i,j,slot,used
  integer(int64)::capacity,target
  call require(all(keys>=0),'integer set protocol requires nonnegative keys')
  allocate(table(0:7));table=-1;used=0
  do i=1,size(keys)
   slot=position(table,keys(i))
   if(table(slot)==keys(i)) cycle
   table(slot)=keys(i);used=used+1
   if(int(used,int64)*5<int(size(table)-1,int64)*3) cycle
   target=int(used,int64)*4
   if(used>50000) target=int(used,int64)*2
   capacity=8
   do while(capacity<=target)
    call require(capacity<=int(huge(1)/2,int64),'integer set protocol capacity overflow')
    capacity=capacity*2
   end do
   allocate(grown(0:int(capacity)-1));grown=-1
   do j=0,size(table)-1
    if(table(j)<0) cycle
    slot=position(grown,table(j));grown(slot)=table(j)
   end do
   call move_alloc(grown,table)
  end do
  values=pack(table,table>=0)
 end function
end module
