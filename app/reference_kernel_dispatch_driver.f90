program reference_kernel_dispatch_driver
 use sklearn_common,only:dp
 use sklearn_metrics,only:pairwise_kernels
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 use sklearn_gp_kernels,only:gp_kernel,rbf_covariance,white_covariance,operator(+)
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 real(dp),allocatable::x(:,:),y(:,:),saved_x(:,:),saved_y(:,:),values(:,:),gamma_arg,degree_arg,coef_arg
 type(csr_matrix)::sx,sy
 type(gp_kernel)::gp
 integer::count,c,id,n,m,q,r,storage,mode,mask,k,state,identity,filter,integer_degree
 real(dp)::gamma,degree,coef
 logical::before(size(ieee_all)),after(size(ieee_all)),modes(size(ieee_all)),new_modes(size(ieee_all))
 character(32)::metric
 read(*,*)count
 do c=1,count
  read(*,*)id,metric,n,m,q,r,storage,mode,mask,gamma,degree,coef,filter
  allocate(x(n,q),y(m,r));read(*,*)x;read(*,*)y
  if(btest(mask,0))then
   allocate(gamma_arg);gamma_arg=gamma
  end if
  if(btest(mask,1))then
   allocate(degree_arg);degree_arg=degree
  end if
  if(btest(mask,2))then
   allocate(coef_arg);coef_arg=coef
  end if
  integer_degree=3;if(btest(mask,3))integer_degree=int(degree)
  if(metric=='gp_rbf')gp=rbf_covariance(1.2_dp)
  if(metric=='gp_white')gp=white_covariance(.3_dp)
  if(metric=='gp_sum')gp=rbf_covariance(1.2_dp)+white_covariance(.3_dp)
  saved_x=x;saved_y=y;sx=csr_from_dense(x);sy=csr_from_dense(y)
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,mod(id,2)==0)
  do k=1,size(ieee_all)
   call ieee_get_flag(ieee_all(k),before(k));call ieee_get_halting_mode(ieee_all(k),modes(k))
  end do
  if(storage==0)then
   select case(mode)
   case(1)
    call invoke_0(x)
   case(2)
    call invoke_0(x,x)
   case default
    call invoke_0(x,y)
   end select
  end if
  if(storage==1)then
   select case(mode)
   case(1)
    call invoke_1(sx)
   case(2)
    call invoke_1(sx,sx)
   case default
    call invoke_1(sx,sy)
   end select
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
  write(*,*)id,state,identity,size(values,1),size(values,2)
  write(*,*)values
  deallocate(x,y,values)
  if(allocated(gamma_arg))deallocate(gamma_arg)
  if(allocated(degree_arg))deallocate(degree_arg)
  if(allocated(coef_arg))deallocate(coef_arg)
 end do
contains
 subroutine invoke_0(a,b)
  real(dp),intent(in)::a(:,:)
  real(dp),optional,intent(in)::b(:,:)
  if(trim(metric)=='callback')then
   values=pairwise_kernels(a,b,asymmetric_kernel,filter_params=filter==1)
  else if(index(metric,'gp_')==1)then
   values=pairwise_kernels(a,b,gp,filter_params=filter==1)
  else if(btest(mask,3))then
   values=pairwise_kernels(a,b,metric,gamma_arg,integer_degree,coef_arg,filter==1)
  else
   values=pairwise_kernels(a,b,metric,gamma_arg,degree_arg,coef_arg,filter==1)
  end if
 end subroutine
 subroutine invoke_1(a,b)
  type(csr_matrix),intent(in)::a
  type(csr_matrix),optional,intent(in)::b
  if(trim(metric)=='callback')then
   values=pairwise_kernels(a,b,asymmetric_kernel,filter_params=filter==1)
  else if(btest(mask,3))then
   values=pairwise_kernels(a,b,metric,gamma_arg,integer_degree,coef_arg,filter==1)
  else
   values=pairwise_kernels(a,b,metric,gamma_arg,degree_arg,coef_arg,filter==1)
  end if
 end subroutine
 subroutine invoke_2(a,b)
  type(csr_matrix),intent(in)::a
  real(dp),intent(in)::b(:,:)
  if(trim(metric)=='callback')then
   values=pairwise_kernels(a,b,asymmetric_kernel,filter_params=filter==1)
  else if(btest(mask,3))then
   values=pairwise_kernels(a,b,metric,gamma_arg,integer_degree,coef_arg,filter==1)
  else
   values=pairwise_kernels(a,b,metric,gamma_arg,degree_arg,coef_arg,filter==1)
  end if
 end subroutine
 subroutine invoke_3(a,b)
  real(dp),intent(in)::a(:,:)
  type(csr_matrix),intent(in)::b
  if(trim(metric)=='callback')then
   values=pairwise_kernels(a,b,asymmetric_kernel,filter_params=filter==1)
  else if(btest(mask,3))then
   values=pairwise_kernels(a,b,metric,gamma_arg,integer_degree,coef_arg,filter==1)
  else
   values=pairwise_kernels(a,b,metric,gamma_arg,degree_arg,coef_arg,filter==1)
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
 real(dp)function asymmetric_kernel(a,b)result(value)
  real(dp),intent(in)::a(:),b(:)
  value=sum(abs(a-b))+sum(a)*.125_dp
 end function
end program
