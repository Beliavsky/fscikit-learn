program reference_pairwise_reductions_driver
 use sklearn_common,only:dp
 use sklearn_metrics,only:pairwise_distances_argmin,pairwise_distances_argmin_min
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 real(dp),allocatable::x(:,:),y(:,:),saved_x(:,:),saved_y(:,:),d(:)
 integer,allocatable::idx(:),other(:)
 type(csr_matrix)::sx,sy
 integer::count,c,id,n,m,q,r,axis,storage,square,k,state,identity
 real(dp)::power
 logical::before(size(ieee_all)),after(size(ieee_all)),modes(size(ieee_all)),new_modes(size(ieee_all))
 character(32)::metric
 read(*,*)count
 do c=1,count
  read(*,*)id,metric,n,m,q,r,axis,storage,power,square
  allocate(x(n,q),y(m,r));read(*,*)x;read(*,*)y
  saved_x=x;saved_y=y;sx=csr_from_dense(x);sy=csr_from_dense(y)
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,mod(id,2)==0)
  do k=1,size(ieee_all)
   call ieee_get_flag(ieee_all(k),before(k));call ieee_get_halting_mode(ieee_all(k),modes(k))
  end do
  if(storage==0)then
   if(trim(metric)=='callback')then
    call pairwise_distances_argmin_min(x,y,idx,d,asymmetric,axis)
    other=pairwise_distances_argmin(x,y,asymmetric,axis)
   else if(power/=-1)then
    call pairwise_distances_argmin_min(x,y,idx,d,metric,axis,p=power)
    other=pairwise_distances_argmin(x,y,metric,axis,p=power)
   else if(square/=-1)then
    call pairwise_distances_argmin_min(x,y,idx,d,metric,axis,squared=square==1)
    other=pairwise_distances_argmin(x,y,metric,axis,squared=square==1)
   else
    call pairwise_distances_argmin_min(x,y,idx,d,metric,axis)
    other=pairwise_distances_argmin(x,y,metric,axis)
   end if
  end if
  if(storage==1)then
   if(trim(metric)=='callback')then
    call pairwise_distances_argmin_min(sx,sy,idx,d,asymmetric,axis)
    other=pairwise_distances_argmin(sx,sy,asymmetric,axis)
   else if(power/=-1)then
    call pairwise_distances_argmin_min(sx,sy,idx,d,metric,axis,p=power)
    other=pairwise_distances_argmin(sx,sy,metric,axis,p=power)
   else if(square/=-1)then
    call pairwise_distances_argmin_min(sx,sy,idx,d,metric,axis,squared=square==1)
    other=pairwise_distances_argmin(sx,sy,metric,axis,squared=square==1)
   else
    call pairwise_distances_argmin_min(sx,sy,idx,d,metric,axis)
    other=pairwise_distances_argmin(sx,sy,metric,axis)
   end if
  end if
  if(storage==2)then
   if(trim(metric)=='callback')then
    call pairwise_distances_argmin_min(sx,y,idx,d,asymmetric,axis)
    other=pairwise_distances_argmin(sx,y,asymmetric,axis)
   else if(power/=-1)then
    call pairwise_distances_argmin_min(sx,y,idx,d,metric,axis,p=power)
    other=pairwise_distances_argmin(sx,y,metric,axis,p=power)
   else if(square/=-1)then
    call pairwise_distances_argmin_min(sx,y,idx,d,metric,axis,squared=square==1)
    other=pairwise_distances_argmin(sx,y,metric,axis,squared=square==1)
   else
    call pairwise_distances_argmin_min(sx,y,idx,d,metric,axis)
    other=pairwise_distances_argmin(sx,y,metric,axis)
   end if
  end if
  if(storage==3)then
   if(trim(metric)=='callback')then
    call pairwise_distances_argmin_min(x,sy,idx,d,asymmetric,axis)
    other=pairwise_distances_argmin(x,sy,asymmetric,axis)
   else if(power/=-1)then
    call pairwise_distances_argmin_min(x,sy,idx,d,metric,axis,p=power)
    other=pairwise_distances_argmin(x,sy,metric,axis,p=power)
   else if(square/=-1)then
    call pairwise_distances_argmin_min(x,sy,idx,d,metric,axis,squared=square==1)
    other=pairwise_distances_argmin(x,sy,metric,axis,squared=square==1)
   else
    call pairwise_distances_argmin_min(x,sy,idx,d,metric,axis)
    other=pairwise_distances_argmin(x,sy,metric,axis)
   end if
  end if
  do k=1,size(ieee_all)
   call ieee_get_flag(ieee_all(k),after(k));call ieee_get_halting_mode(ieee_all(k),new_modes(k))
  end do
  state=0;if(all(before.eqv.after).and.all(modes.eqv.new_modes))state=1
  identity=0
  if(all(equal_values(x,saved_x)).and.all(equal_values(y,saved_y)))identity=1
  write(*,*)id,state,identity,merge(1,0,all(idx==other)),size(idx)
  write(*,*)idx
  write(*,*)d
  deallocate(x,y,idx,other,d)
 end do
contains
 function equal_values(a,b)result(equal)
  real(dp),intent(in)::a(:,:),b(:,:)
  logical::equal(size(a,1),size(a,2))
  integer::i,j
  do j=1,size(a,2)
   do i=1,size(a,1)
    equal(i,j)=.false.
    if(ieee_is_nan(a(i,j)).and.ieee_is_nan(b(i,j)))then
     equal(i,j)=.true.
    else if(.not.ieee_is_nan(a(i,j)).and..not.ieee_is_nan(b(i,j)))then
     equal(i,j)=a(i,j)==b(i,j)
    end if
   end do
  end do
 end function
 real(dp)function asymmetric(a,b)result(value)
  real(dp),intent(in)::a(:),b(:)
  value=sum(abs(a-b))+sum(a)*.125_dp
 end function
end program
