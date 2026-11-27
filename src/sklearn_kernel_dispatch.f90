! Kernel dispatch adapted from sklearn 1.6.1 metrics/pairwise.py (BSD-3-Clause).
! Native row views, typed adapters and quiet IEEE composition are MIT.
module sklearn_kernel_dispatch
 use sklearn_common,only:dp,require
 use sklearn_sparse,only:csr_matrix
 use sklearn_gp_kernels,only:gp_kernel
 use,intrinsic::iso_c_binding,only:c_double
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 ! Internal accumulation helpers shared with the OPTICS brute backend.
 public::kernel_blas_transpose_column
 public::kernel_blas_dot,kernel_blas_inner_product,kernel_blas_self_product,kernel_blas_self_dot
 interface kernel_blas_dot
  module procedure dense_dot
 end interface
 interface kernel_blas_inner_product
  module procedure ddot_lanes
 end interface
 public::pairwise_kernels
 type::row_view
  real(dp),pointer::dense(:,:)=>null()
  type(csr_matrix),pointer::sparse=>null()
  integer::n=0,p=0
 end type
 abstract interface
  function row_kernel(x,y)result(value)
   import dp
   real(dp),intent(in)::x(:),y(:)
   real(dp)::value
  end function
 end interface
 interface
  function c_pow(x,y)bind(C,name='pow')result(value)
   import c_double
   real(c_double),value::x,y
   real(c_double)::value
  end function
 end interface
 interface pairwise_kernels
  module procedure kernels_dd
  module procedure kernels_dd_integer
  module procedure kernels_dd_callback
  module procedure kernels_ss
  module procedure kernels_ss_integer
  module procedure kernels_ss_callback
  module procedure kernels_sd
  module procedure kernels_sd_integer
  module procedure kernels_sd_callback
  module procedure kernels_ds
  module procedure kernels_ds_integer
  module procedure kernels_ds_callback
  module procedure kernels_gp
 end interface
contains
 function kernels_dd(x,y,metric,gamma,degree,coef0,filter_params)result(values)
  real(dp),target,intent(in)::x(:,:)
  real(dp),target,optional,intent(in)::y(:,:)
  real(dp),allocatable::values(:,:)
  logical,optional,intent(in)::filter_params
  type(row_view)::a,b
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::gamma,coef0
  real(dp),optional,intent(in)::degree
  a%dense=>x
  a%n=size(x,1);a%p=size(x,2)
  b=a
  if(present(y))then
  b%dense=>y
  b%n=size(y,1);b%p=size(y,2)
  end if
  values=evaluate(a,b,metric,gamma,degree,coef0,filter_params)
 end function
 function kernels_dd_integer(x,y,metric,gamma,degree,coef0,filter_params)result(values)
  real(dp),target,intent(in)::x(:,:)
  real(dp),target,optional,intent(in)::y(:,:)
  real(dp),allocatable::values(:,:)
  logical,optional,intent(in)::filter_params
  type(row_view)::a,b
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::gamma,coef0
  integer,intent(in)::degree
  a%dense=>x
  a%n=size(x,1);a%p=size(x,2)
  b=a
  if(present(y))then
  b%dense=>y
  b%n=size(y,1);b%p=size(y,2)
  end if
  values=evaluate(a,b,metric,gamma,real(degree,dp),coef0,filter_params)
 end function
 function kernels_dd_callback(x,y,metric,filter_params)result(values)
  real(dp),target,intent(in)::x(:,:)
  real(dp),target,optional,intent(in)::y(:,:)
  real(dp),allocatable::values(:,:)
  logical,optional,intent(in)::filter_params
  type(row_view)::a,b
  procedure(row_kernel)::metric
  a%dense=>x
  a%n=size(x,1);a%p=size(x,2)
  b=a
  if(present(y))then
  b%dense=>y
  b%n=size(y,1);b%p=size(y,2)
  end if
  values=evaluate(a,b,filter_params=filter_params,callback=metric)
 end function
 function kernels_ss(x,y,metric,gamma,degree,coef0,filter_params)result(values)
  type(csr_matrix),target,intent(in)::x
  type(csr_matrix),target,optional,intent(in)::y
  real(dp),allocatable::values(:,:)
  logical,optional,intent(in)::filter_params
  type(row_view)::a,b
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::gamma,coef0
  real(dp),optional,intent(in)::degree
  a%sparse=>x
  a%n=x%n_rows;a%p=x%n_cols
  b=a
  if(present(y))then
  b%sparse=>y
  b%n=y%n_rows;b%p=y%n_cols
  end if
  values=evaluate(a,b,metric,gamma,degree,coef0,filter_params)
 end function
 function kernels_ss_integer(x,y,metric,gamma,degree,coef0,filter_params)result(values)
  type(csr_matrix),target,intent(in)::x
  type(csr_matrix),target,optional,intent(in)::y
  real(dp),allocatable::values(:,:)
  logical,optional,intent(in)::filter_params
  type(row_view)::a,b
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::gamma,coef0
  integer,intent(in)::degree
  a%sparse=>x
  a%n=x%n_rows;a%p=x%n_cols
  b=a
  if(present(y))then
  b%sparse=>y
  b%n=y%n_rows;b%p=y%n_cols
  end if
  values=evaluate(a,b,metric,gamma,real(degree,dp),coef0,filter_params)
 end function
 function kernels_ss_callback(x,y,metric,filter_params)result(values)
  type(csr_matrix),target,intent(in)::x
  type(csr_matrix),target,optional,intent(in)::y
  real(dp),allocatable::values(:,:)
  logical,optional,intent(in)::filter_params
  type(row_view)::a,b
  procedure(row_kernel)::metric
  a%sparse=>x
  a%n=x%n_rows;a%p=x%n_cols
  b=a
  if(present(y))then
  b%sparse=>y
  b%n=y%n_rows;b%p=y%n_cols
  end if
  values=evaluate(a,b,filter_params=filter_params,callback=metric)
 end function
 function kernels_sd(x,y,metric,gamma,degree,coef0,filter_params)result(values)
  type(csr_matrix),target,intent(in)::x
  real(dp),target,intent(in)::y(:,:)
  real(dp),allocatable::values(:,:)
  logical,optional,intent(in)::filter_params
  type(row_view)::a,b
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::gamma,coef0
  real(dp),optional,intent(in)::degree
  a%sparse=>x
  a%n=x%n_rows;a%p=x%n_cols
  b=a
  b%dense=>y
  nullify(b%sparse)
  b%n=size(y,1);b%p=size(y,2)
  values=evaluate(a,b,metric,gamma,degree,coef0,filter_params)
 end function
 function kernels_sd_integer(x,y,metric,gamma,degree,coef0,filter_params)result(values)
  type(csr_matrix),target,intent(in)::x
  real(dp),target,intent(in)::y(:,:)
  real(dp),allocatable::values(:,:)
  logical,optional,intent(in)::filter_params
  type(row_view)::a,b
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::gamma,coef0
  integer,intent(in)::degree
  a%sparse=>x
  a%n=x%n_rows;a%p=x%n_cols
  b=a
  b%dense=>y
  nullify(b%sparse)
  b%n=size(y,1);b%p=size(y,2)
  values=evaluate(a,b,metric,gamma,real(degree,dp),coef0,filter_params)
 end function
 function kernels_sd_callback(x,y,metric,filter_params)result(values)
  type(csr_matrix),target,intent(in)::x
  real(dp),target,intent(in)::y(:,:)
  real(dp),allocatable::values(:,:)
  logical,optional,intent(in)::filter_params
  type(row_view)::a,b
  procedure(row_kernel)::metric
  a%sparse=>x
  a%n=x%n_rows;a%p=x%n_cols
  b=a
  b%dense=>y
  nullify(b%sparse)
  b%n=size(y,1);b%p=size(y,2)
  values=evaluate(a,b,filter_params=filter_params,callback=metric)
 end function
 function kernels_ds(x,y,metric,gamma,degree,coef0,filter_params)result(values)
  real(dp),target,intent(in)::x(:,:)
  type(csr_matrix),target,intent(in)::y
  real(dp),allocatable::values(:,:)
  logical,optional,intent(in)::filter_params
  type(row_view)::a,b
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::gamma,coef0
  real(dp),optional,intent(in)::degree
  a%dense=>x
  a%n=size(x,1);a%p=size(x,2)
  b=a
  b%sparse=>y
  nullify(b%dense)
  b%n=y%n_rows;b%p=y%n_cols
  values=evaluate(a,b,metric,gamma,degree,coef0,filter_params)
 end function
 function kernels_ds_integer(x,y,metric,gamma,degree,coef0,filter_params)result(values)
  real(dp),target,intent(in)::x(:,:)
  type(csr_matrix),target,intent(in)::y
  real(dp),allocatable::values(:,:)
  logical,optional,intent(in)::filter_params
  type(row_view)::a,b
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::gamma,coef0
  integer,intent(in)::degree
  a%dense=>x
  a%n=size(x,1);a%p=size(x,2)
  b=a
  b%sparse=>y
  nullify(b%dense)
  b%n=y%n_rows;b%p=y%n_cols
  values=evaluate(a,b,metric,gamma,real(degree,dp),coef0,filter_params)
 end function
 function kernels_ds_callback(x,y,metric,filter_params)result(values)
  real(dp),target,intent(in)::x(:,:)
  type(csr_matrix),target,intent(in)::y
  real(dp),allocatable::values(:,:)
  logical,optional,intent(in)::filter_params
  type(row_view)::a,b
  procedure(row_kernel)::metric
  a%dense=>x
  a%n=size(x,1);a%p=size(x,2)
  b=a
  b%sparse=>y
  nullify(b%dense)
  b%n=y%n_rows;b%p=y%n_cols
  values=evaluate(a,b,filter_params=filter_params,callback=metric)
 end function
 function kernels_gp(x,y,metric,filter_params)result(values)
  real(dp),intent(in)::x(:,:)
  real(dp),optional,intent(in)::y(:,:)
  type(gp_kernel),intent(in)::metric
  logical,optional,intent(in)::filter_params
  real(dp),allocatable::values(:,:)
  logical::halting(size(ieee_all)),raised(size(ieee_all))
  call suspend(halting,raised)
  if(present(y))then
   values=metric%matrix(x,y)
  else
   values=metric%matrix(x,x)
  end if
  call restore(halting,raised)
 end function
 subroutine validate(a)
  type(row_view),intent(in)::a
  call require(a%n>0.and.a%p>0,'kernel inputs require nonempty samples and features')
  if(associated(a%dense))then
   call require(all(ieee_is_finite(a%dense)),'kernel inputs must be finite')
  else
   call a%sparse%validate()
   call require(all(ieee_is_finite(a%sparse%data)),'kernel CSR data must be finite')
  end if
 end subroutine
 subroutine get_row(a,i,row)
  type(row_view),intent(in)::a
  integer,intent(in)::i
  real(dp),intent(out)::row(:)
  integer::k
  if(associated(a%dense))then
   row=a%dense(i,:)
  else
   row=0
   do k=a%sparse%indptr(i),a%sparse%indptr(i+1)-1
    row(a%sparse%indices(k))=row(a%sparse%indices(k))+a%sparse%data(k)
   end do
  end if
 end subroutine
 subroutine suspend(halting,raised)
  logical,intent(out)::halting(size(ieee_all)),raised(size(ieee_all))
  integer::i
  do i=1,size(ieee_all)
   call ieee_get_flag(ieee_all(i),raised(i))
  end do
  do i=1,size(ieee_all)
   call ieee_get_halting_mode(ieee_all(i),halting(i))
  end do
  do i=1,size(ieee_all)
   call ieee_set_halting_mode(ieee_all(i),.false.)
  end do
 end subroutine
 subroutine cosine_row(a,i,norm,row)
  type(row_view),intent(in)::a
  integer,intent(in)::i
  real(dp),intent(in)::norm
  real(dp),intent(out)::row(:)
  integer::e,j
  if(associated(a%sparse))then
   row=0
   do e=a%sparse%indptr(i),a%sparse%indptr(i+1)-1
    j=a%sparse%indices(e);row(j)=row(j)+a%sparse%data(e)/norm
   end do
  else
   call get_row(a,i,row);row=row/norm
  end if
 end subroutine
 subroutine restore(halting,raised)
  logical,intent(in)::halting(size(ieee_all)),raised(size(ieee_all))
  integer::i
  do i=1,size(ieee_all)
   call ieee_set_halting_mode(ieee_all(i),halting(i))
  end do
  do i=1,size(ieee_all)
   call ieee_set_flag(ieee_all(i),raised(i))
  end do
 end subroutine
 function evaluate(a,b,metric,gamma,degree,coef0,filter_params,callback)result(values)
  type(row_view),intent(in)::a,b
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::gamma,degree,coef0
  logical,optional,intent(in)::filter_params
  procedure(row_kernel),optional::callback
  real(dp),allocatable::values(:,:),u(:),v(:),nx(:),ny(:)
  real(dp)::g,p,c,xy,value,den,difference
  integer::kind,i,j,k
  character(:),allocatable::name
  logical::filter,same,halting(size(ieee_all)),raised(size(ieee_all)),has_gamma,has_degree,has_coef
  call validate(a);call validate(b)
  name='linear';if(present(metric))name=trim(metric)
  kind=0
  if(.not.present(callback))then
   select case(name)
   case('linear');kind=1
   case('poly','polynomial');kind=2
   case('rbf');kind=3
   case('sigmoid');kind=4
   case('cosine');kind=5
   case('laplacian');kind=6
   case('additive_chi2');kind=7
   case('chi2');kind=8
   case('precomputed');kind=9
   end select
   call require(kind>0,'unknown pairwise kernel metric')
  end if
  if(kind==9)then
   call require(a%p==b%n,'precomputed kernel column count must match reference rows')
  else
   call require(a%p==b%p,'kernel feature counts differ')
  end if
  if(kind==7.or.kind==8)call require(.not.associated(a%sparse).and..not.associated(b%sparse), &
   'chi-squared kernels require dense inputs')
  filter=.false.;if(present(filter_params))filter=filter_params
  has_gamma=any(kind==[2,3,4,6,8]);has_degree=kind==2;has_coef=kind==2.or.kind==4
  if(kind/=9.and..not.filter)then
   if(present(gamma))call require(has_gamma,'gamma is not applicable to this kernel')
   if(present(degree))call require(has_degree,'degree is not applicable to this kernel')
   if(present(coef0))call require(has_coef,'coef0 is not applicable to this kernel')
  end if
  same=.false.
  if(associated(a%dense).and.associated(b%dense))same=associated(a%dense,b%dense)
  if(associated(a%sparse).and.associated(b%sparse))same=associated(a%sparse,b%sparse)
  allocate(values(a%n,b%n),u(a%p),v(b%p))
  call suspend(halting,raised)
  if(kind==9)then
   do i=1,a%n
    call get_row(a,i,u);values(i,:)=u
   end do
  else if(present(callback))then
   do i=1,a%n
    call get_row(a,i,u)
    do j=1,b%n
     if(same.and.j<=i)cycle
     call get_row(b,j,v);values(i,j)=callback(u,v)
     if(same)values(j,i)=values(i,j)+0._dp
    end do
   end do
   if(same)then
    do i=1,a%n
     call get_row(a,i,u);values(i,i)=callback(u,u)
    end do
   end if
  else
   g=1._dp/real(a%p,dp);if(kind==8)g=1
   if(present(gamma).and.has_gamma)g=gamma
   p=3;if(present(degree).and.has_degree)p=degree
   c=1;if(present(coef0).and.has_coef)c=coef0
   allocate(nx(a%n),ny(b%n));nx=0;ny=0
   if(kind==3.or.kind==5)then
    do i=1,a%n
     if(associated(a%sparse))then
      ! sklearn row_norms / CSR normalization use stored entries, including duplicates.
      do k=a%sparse%indptr(i),a%sparse%indptr(i+1)-1
       nx(i)=nx(i)+a%sparse%data(k)*a%sparse%data(k)
      end do
     else
      call get_row(a,i,u)
      do k=1,a%p
       nx(i)=nx(i)+u(k)*u(k)
      end do
     end if
     if(kind==5)then
      nx(i)=sqrt(nx(i))
      if(associated(a%sparse))then
       if(nx(i)==0)nx(i)=1
      else
       if(nx(i)<10*epsilon(1._dp))nx(i)=1
      end if
     end if
    end do
    do j=1,b%n
     if(associated(b%sparse))then
      do k=b%sparse%indptr(j),b%sparse%indptr(j+1)-1
       ny(j)=ny(j)+b%sparse%data(k)*b%sparse%data(k)
      end do
     else
      call get_row(b,j,v)
      do k=1,b%p
       ny(j)=ny(j)+v(k)*v(k)
      end do
     end if
     if(kind==5)then
      ny(j)=sqrt(ny(j))
      if(associated(b%sparse))then
       if(ny(j)==0)ny(j)=1
      else
       if(ny(j)<10*epsilon(1._dp))ny(j)=1
      end if
     end if
    end do
   end if
   do i=1,a%n
    if(kind==5)then
     call cosine_row(a,i,nx(i),u)
    else
     call get_row(a,i,u)
    end if
    if(kind==7.or.kind==8)call require(all(u>=0),'chi-squared kernel inputs must be nonnegative')
    do j=1,b%n
     if(kind==5)then
      call cosine_row(b,j,ny(j),v)
     else
      call get_row(b,j,v)
     end if
     if(kind==7.or.kind==8)call require(all(v>=0),'chi-squared kernel inputs must be nonnegative')
     xy=0
     if(kind>=1.and.kind<=4.and.associated(a%dense).and.associated(b%dense))then
      xy=dense_dot(u,v,a%n,b%n,i,j)
     else if(kind<=5)then
      do k=1,a%p
       if(kind==5)then
        xy=xy+u(k)*v(k)
       else
        xy=xy+u(k)*v(k)
       end if
      end do
     end if
     select case(kind)
     case(1,5);value=xy
     case(2);value=c_pow(g*xy+c,p)
     case(3)
      value=(-2*xy+nx(i))+ny(j)
      if(.not.ieee_is_nan(value))value=max(value,0._dp)
      if(same.and.i==j)value=0
      value=exp(-g*value)
     case(4);value=tanh(g*xy+c)
     case(6)
      value=0
      do k=1,a%p
       value=value+abs(u(k)-v(k))
      end do
      value=exp(-g*value)
     case(7,8)
      value=0
      do k=1,a%p
       den=u(k)+v(k);difference=u(k)-v(k)
       if(den/=0)value=value-(difference*difference)/den
      end do
      if(kind==8)value=exp(g*value)
     end select
     values(i,j)=value
    end do
   end do
  end if
  call restore(halting,raised)
 end function
 ! Scalar reconstruction of OpenBLAS 0.3.29 SkylakeX dot/GEMV/TN-small
 ! accumulation, with BSD notices retained in LICENSES/OpenBLAS-LICENSE and
 ! LICENSES/OpenBLAS-small-kernel-BSD. No BLAS runtime is linked.
 real(dp)function dense_dot(u,v,n,m,ii,jj)result(value)
  real(dp),intent(in)::u(:),v(:)
  integer,intent(in)::n,m,ii,jj
  real(dp)::lane(8),pair(4),part,tail
  integer::k,l,first,last,whole,count,index
  if(n==1.and.m==1)then
   value=ddot_lanes(u,v);return
  end if
  value=0
  if(n==1.or.m==1)then
   count=max(n,m);index=ii;if(n==1)index=jj
   whole=size(u)-mod(size(u),4)
   do first=1,whole,2048
    last=min(whole,first+2047);pair=0
    if(index<=count-mod(count,4))then
     do k=first,last
      l=mod(k-first,4)+1;pair(l)=ieee_fma(u(k),v(k),pair(l))
     end do
    else if(index<=count-mod(count,2))then
     do k=first,last
      l=mod(k-first,2)+1;pair(l)=pair(l)+u(k)*v(k)
     end do
    else
     do k=first,last
      l=mod(k-first,4)+1;pair(l)=pair(l)+u(k)*v(k)
     end do
    end if
    part=(pair(1)+pair(3))+(pair(2)+pair(4));value=value+part
   end do
   select case(size(u)-whole)
   case(1)
    value=ieee_fma(u(whole+1),v(whole+1),value)
   case(2)
    tail=ieee_fma(u(whole+1),v(whole+1),u(whole+2)*v(whole+2));value=value+tail
   case(3)
    tail=ieee_fma(u(whole+1),v(whole+1),u(whole+2)*v(whole+2))
    tail=ieee_fma(u(whole+3),v(whole+3),tail);value=value+tail
   end select
  else if(size(u)>=32.and.real(n,dp)*m<=1200.and.real(n,dp)*m*size(u)<=1.e6_dp)then
   lane=0
   do k=1,size(u)
    l=mod(k-1,8)+1;lane(l)=ieee_fma(u(k),v(k),lane(l))
   end do
   if(ii>n-mod(n,4).and.jj>m-mod(m,4))then
    value=((lane(1)+lane(5))+(lane(3)+lane(7)))+((lane(2)+lane(6))+(lane(4)+lane(8)))
   else
   value=((lane(1)+lane(2))+(lane(3)+lane(4)))+((lane(5)+lane(6))+(lane(7)+lane(8)))
   end if
  else
   do k=1,size(u)
    value=ieee_fma(u(k),v(k),value)
   end do
  end if
 end function
  real(dp)function kernel_blas_transpose_column(a,x,index,outputs)result(total)
    ! OpenBLAS 0.3.29 dgemv_n_4 / SkylakeX microkernel, 2014 OpenBLAS Project (BSD).
    ! X.T is Fortran-contiguous in NumPy, unlike the forward C-contiguous route.
    real(dp),intent(in)::a(:),x(:)
    integer,intent(in)::index,outputs
    real(dp)::part
    integer::i,n,whole
    n=size(a);whole=n-mod(n,4);total=0
    if(outputs==1)then
      total=kernel_blas_inner_product(a,x)
    else if(outputs<=3)then
      do i=1,whole,2
        part=ieee_fma(a(i),x(i),a(i+1)*x(i+1));total=total+part
      end do
      do i=whole+1,n
        total=ieee_fma(a(i),x(i),total)
      end do
    else if(index<=outputs-mod(outputs,4))then
      do i=1,whole,4
        part=ieee_fma(a(i),x(i),a(i+1)*x(i+1))
        part=ieee_fma(a(i+2),x(i+2),part);part=ieee_fma(a(i+3),x(i+3),part)
        total=total+part
      end do
      i=whole+1
      if(n-whole>=2)then
        total=total+ieee_fma(a(i),x(i),a(i+1)*x(i+1));i=i+2
      end if
      if(i<=n)total=total+a(i)*x(i)
    else
      do i=1,n
        total=ieee_fma(a(i),x(i),total)
      end do
    end if
  end function
 real(dp) function ddot_lanes(x,y) result(value)
  real(dp),intent(in)::x(:),y(:)
  real(dp)::wide(8,4),narrow(4,4),v(4)
  integer::n,n16,n32,i,j,k,index
  n=size(x);value=0
  if(n==1) then
   value=x(1)*y(1);return
  end if
  n16=n-mod(n,16);n32=n16-mod(n16,32);wide=0;narrow=0
  do i=1,n32,32
   do j=1,4
    do k=1,8
     index=i+(j-1)*8+k-1;wide(k,j)=ieee_fma(x(index),y(index),wide(k,j))
    end do
   end do
  end do
  do j=1,4
   do k=1,4
    narrow(k,j)=wide(k,j)+wide(k+4,j)
   end do
  end do
  if(n16>n32) then
   do j=1,4
    do k=1,4
     index=n32+(j-1)*4+k;narrow(k,j)=ieee_fma(x(index),y(index),narrow(k,j))
    end do
   end do
  end if
  if(n16>0) then
   do k=1,4
    v(k)=((narrow(k,1)+narrow(k,2))+narrow(k,3))+narrow(k,4)
   end do
   value=(v(1)+v(3))+(v(2)+v(4))
  end if
  do i=n16+1,n
   value=ieee_fma(x(i),y(i),value)
  end do
 end function
 ! NumPy 2.2.6 matmul self-transpose dispatch and OpenBLAS 0.3.29
 ! SkylakeX SYRK P/Q blocking and generic 16x2 reduction lanes.
 ! Retained BSD notices: OpenBLAS-LICENSE and OpenBLAS-level3-BSD.
 ! Scalar shape adapters are native MIT; no BLAS runtime is linked.
 integer function self_tile_width(length,index,choices)result(width)
  integer,intent(in)::length,index,choices(:)
  integer::remaining,local,whole,k
  remaining=length;local=index;width=1
  do k=1,size(choices)
   width=choices(k);whole=(remaining/width)*width
   if(local<whole)return
   local=local-whole;remaining=remaining-whole
  end do
 end function
 integer function self_row_block(remaining)result(width)
  integer,intent(in)::remaining
  if(remaining>=384)then
   width=192
  else if(remaining>192)then
   width=((remaining/2+31)/32)*32
  else
   width=remaining
  end if
 end function
 function kernel_blas_self_product(x)result(values)
  real(dp),intent(in)::x(:,:)
  real(dp)::values(size(x,1),size(x,1))
  integer::i,j
  do j=1,size(x,1)
   do i=j,size(x,1)
    values(i,j)=kernel_blas_self_dot(x(i,:),x(j,:),size(x,1),i,j)
    values(j,i)=values(i,j)
   end do
  end do
 end function
 real(dp) function kernel_blas_self_dot(u,v,n,i,j)result(value)
  real(dp),intent(in)::u(:),v(:)
  integer,intent(in)::n,i,j
  integer::row,col,start,nb,mlen,mi,nc,local,b,nn,mr,lanes,first,remaining,kcount,whole,kk,k,l
  real(dp)::sums(4),part
  if(n==1)then
   value=ddot_lanes(u,v);return
  end if
  row=max(i,j)-1;col=min(i,j)-1;start=0;nb=self_row_block(n)
  do while(row>=start+nb)
   start=start+nb;nb=self_row_block(n-start)
  end do
  if(col<start)then
   mlen=nb;mi=row-start;nc=self_tile_width(start,col,[12,10,8,6,4,2,1])
  else
   local=col-start;b=(local/32)*32;nn=min(32,nb-b)
   if(row-start<b+nn)then
    mlen=nn;mi=row-start-b
   else
    mlen=nb-b-nn;mi=row-start-b-nn
   end if
   nc=self_tile_width(nn,local-b,[12,10,8,6,4,2,1])
  end if
  mr=self_tile_width(mlen,mi,[16,8,4,2,1]);lanes=1
  if(nc==12)then
   if(mr<=2)lanes=4
   if(mr==4)lanes=2
  end if
  value=0;first=0
  do while(first<size(u))
   remaining=size(u)-first;kcount=remaining
   if(remaining>=768)then
    kcount=384
   else if(remaining>384)then
    kcount=(remaining+1)/2
   end if
   sums=0;whole=kcount-mod(kcount,lanes)
   do kk=0,whole-1
    k=first+kk+1;l=mod(kk,lanes)+1;sums(l)=ieee_fma(u(k),v(k),sums(l))
   end do
   part=sums(1)
   if(lanes==2)part=sums(1)+sums(2)
   if(lanes==4)part=(sums(1)+sums(2))+(sums(3)+sums(4))
   do kk=whole,kcount-1
    k=first+kk+1;part=ieee_fma(u(k),v(k),part)
   end do
   value=value+part;first=first+kcount
  end do
 end function
end module
