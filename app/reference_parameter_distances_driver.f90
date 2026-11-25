program reference_parameter_distances_driver
 use sklearn_common,only:dp
 use sklearn_metrics,only:pairwise_distances_argmin,pairwise_distances_argmin_min
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 real(dp),allocatable::x(:,:),y(:,:),saved_x(:,:),saved_y(:,:),d(:),w(:),variance(:),precision(:,:)
 real(dp),allocatable::saved_w(:),saved_v(:),saved_vi(:,:),power_arg
 logical,allocatable::square_arg
 integer,allocatable::idx(:),other(:)
 type(csr_matrix)::sx,sy
 integer::count,c,id,n,m,q,r,axis,storage,square,k,state,identity,self,nw,nv,ni,nj
 real(dp)::power
 logical::before(size(ieee_all)),after(size(ieee_all)),modes(size(ieee_all)),new_modes(size(ieee_all))
 character(32)::metric
 read(*,*)count
 do c=1,count
  read(*,*)id,metric,n,m,q,r,axis,storage,power,square,self,nw,nv,ni,nj
  allocate(x(n,q),y(m,r));read(*,*)x;read(*,*)y
  if(nw>=0)then
   allocate(w(nw));read(*,*)w;saved_w=w
  end if
  if(nv>=0)then
   allocate(variance(nv));read(*,*)variance;saved_v=variance
  end if
  if(ni>=0)then
   allocate(precision(ni,nj));read(*,*)precision;saved_vi=precision
  end if
  if(power/=-1)then
   allocate(power_arg);power_arg=power
  end if
  if(square/=-1)then
   allocate(square_arg);square_arg=square==1
  end if
  saved_x=x;saved_y=y;sx=csr_from_dense(x);sy=csr_from_dense(y)
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,mod(id,2)==0)
  do k=1,size(ieee_all)
   call ieee_get_flag(ieee_all(k),before(k));call ieee_get_halting_mode(ieee_all(k),modes(k))
  end do
  if(storage==0)then
   if(self==1)then
    call invoke_0(x,x)
   else
    call invoke_0(x,y)
   end if
  end if
  if(storage==1)then
   if(self==1)then
    call invoke_1(sx,sx)
   else
    call invoke_1(sx,sy)
   end if
  end if
  if(storage==2)then
   call invoke_2(sx,y)
  end if
  if(storage==3)then
   call invoke_3(x,sy)
  end if
  do k=1,size(ieee_all)
   call ieee_get_flag(ieee_all(k),after(k));call ieee_get_halting_mode(ieee_all(k),new_modes(k))
  end do
  state=0;if(all(before.eqv.after).and.all(modes.eqv.new_modes))state=1
  identity=0
  if(all(equal_values(x,saved_x)).and.all(equal_values(y,saved_y)))identity=1
  if(allocated(w))then
   if(.not.all(equal_values(reshape(w,[size(w),1]),reshape(saved_w,[size(w),1]))))identity=0
  end if
  if(allocated(variance))then
   if(.not.all(equal_values(reshape(variance,[size(variance),1]),reshape(saved_v,[size(variance),1]))))identity=0
  end if
  if(allocated(precision))then
   if(.not.all(equal_values(precision,saved_vi)))identity=0
  end if
  write(*,*)id,state,identity,merge(1,0,all(idx==other)),size(idx)
  write(*,*)idx
  write(*,*)d
  deallocate(x,y,idx,other,d)
  if(allocated(w))deallocate(w)
  if(allocated(variance))deallocate(variance)
  if(allocated(precision))deallocate(precision)
  if(allocated(power_arg))deallocate(power_arg)
  if(allocated(square_arg))deallocate(square_arg)
 end do
contains
 subroutine invoke_0(a,b)
  real(dp),intent(in)::a(:,:)
  real(dp),intent(in)::b(:,:)
  if(trim(metric)=='callback')then
   call pairwise_distances_argmin_min(a,b,idx,d,asymmetric,axis)
   other=pairwise_distances_argmin(a,b,asymmetric,axis)
  else
   call pairwise_distances_argmin_min(a,b,idx,d,metric,axis,p=power_arg,squared=square_arg,w=w,v=variance,vi=precision)
   other=pairwise_distances_argmin(a,b,metric,axis,p=power_arg,squared=square_arg,w=w,v=variance,vi=precision)
  end if
 end subroutine
 subroutine invoke_1(a,b)
  type(csr_matrix),intent(in)::a
  type(csr_matrix),intent(in)::b
  if(trim(metric)=='callback')then
   call pairwise_distances_argmin_min(a,b,idx,d,asymmetric,axis)
   other=pairwise_distances_argmin(a,b,asymmetric,axis)
  else
   call pairwise_distances_argmin_min(a,b,idx,d,metric,axis,p=power_arg,squared=square_arg,w=w,v=variance,vi=precision)
   other=pairwise_distances_argmin(a,b,metric,axis,p=power_arg,squared=square_arg,w=w,v=variance,vi=precision)
  end if
 end subroutine
 subroutine invoke_2(a,b)
  type(csr_matrix),intent(in)::a
  real(dp),intent(in)::b(:,:)
  if(trim(metric)=='callback')then
   call pairwise_distances_argmin_min(a,b,idx,d,asymmetric,axis)
   other=pairwise_distances_argmin(a,b,asymmetric,axis)
  else
   call pairwise_distances_argmin_min(a,b,idx,d,metric,axis,p=power_arg,squared=square_arg,w=w,v=variance,vi=precision)
   other=pairwise_distances_argmin(a,b,metric,axis,p=power_arg,squared=square_arg,w=w,v=variance,vi=precision)
  end if
 end subroutine
 subroutine invoke_3(a,b)
  real(dp),intent(in)::a(:,:)
  type(csr_matrix),intent(in)::b
  if(trim(metric)=='callback')then
   call pairwise_distances_argmin_min(a,b,idx,d,asymmetric,axis)
   other=pairwise_distances_argmin(a,b,asymmetric,axis)
  else
   call pairwise_distances_argmin_min(a,b,idx,d,metric,axis,p=power_arg,squared=square_arg,w=w,v=variance,vi=precision)
   other=pairwise_distances_argmin(a,b,metric,axis,p=power_arg,squared=square_arg,w=w,v=variance,vi=precision)
  end if
 end subroutine
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
