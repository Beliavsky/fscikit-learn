! Nearest-pair reductions adapted from sklearn 1.6.1 metrics/pairwise.py
! and metrics/_dist_metrics.pyx.tp. Retain sklearn BSD notices; native API is MIT.
module sklearn_pairwise_reductions
 use sklearn_common, only: dp,require
 use sklearn_sparse, only: csr_matrix
 use sklearn_pairwise_measures, only: nan_euclidean_distances
 use, intrinsic :: ieee_arithmetic
 use, intrinsic :: ieee_exceptions
 implicit none
 private
 public :: pairwise_distances_argmin,pairwise_distances_argmin_min
 type :: row_view
  real(dp),pointer :: dense(:,:)=>null()
  type(csr_matrix),pointer :: sparse=>null()
  integer :: n=0,p=0
 end type
 abstract interface
  function row_metric(x,y) result(value)
   import dp
   real(dp),intent(in)::x(:),y(:)
   real(dp)::value
  end function
 end interface
 interface pairwise_distances_argmin
  module procedure pairwise_distances_argmin_dd
  module procedure pairwise_distances_argmin_dd_callback
  module procedure pairwise_distances_argmin_ss
  module procedure pairwise_distances_argmin_ss_callback
  module procedure pairwise_distances_argmin_sd
  module procedure pairwise_distances_argmin_sd_callback
  module procedure pairwise_distances_argmin_ds
  module procedure pairwise_distances_argmin_ds_callback
 end interface
 interface pairwise_distances_argmin_min
  module procedure pairwise_distances_argmin_min_dd
  module procedure pairwise_distances_argmin_min_dd_callback
  module procedure pairwise_distances_argmin_min_ss
  module procedure pairwise_distances_argmin_min_ss_callback
  module procedure pairwise_distances_argmin_min_sd
  module procedure pairwise_distances_argmin_min_sd_callback
  module procedure pairwise_distances_argmin_min_ds
  module procedure pairwise_distances_argmin_min_ds_callback
 end interface
contains
 function pairwise_distances_argmin_dd(x,y,metric,axis,p,squared) result(indices)
  real(dp),target,intent(in)::x(:,:)
  real(dp),target,intent(in)::y(:,:)
  integer,allocatable::indices(:)
  real(dp),allocatable::distances(:)
  integer,optional,intent(in)::axis
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p
  logical,optional,intent(in)::squared
  type(row_view)::a,b
  a%dense=>x
  a%n=size(x,1);a%p=size(x,2)
  b%dense=>y
  b%n=size(y,1);b%p=size(y,2)
  call reduce_pairs(a,b,indices,distances,axis,metric,p,squared)
 end function
 function pairwise_distances_argmin_dd_callback(x,y,metric,axis) result(indices)
  real(dp),target,intent(in)::x(:,:)
  real(dp),target,intent(in)::y(:,:)
  integer,allocatable::indices(:)
  real(dp),allocatable::distances(:)
  integer,optional,intent(in)::axis
  procedure(row_metric)::metric
  type(row_view)::a,b
  a%dense=>x
  a%n=size(x,1);a%p=size(x,2)
  b%dense=>y
  b%n=size(y,1);b%p=size(y,2)
  call reduce_pairs(a,b,indices,distances,axis,callback=metric)
 end function
 function pairwise_distances_argmin_ss(x,y,metric,axis,p,squared) result(indices)
  type(csr_matrix),target,intent(in)::x
  type(csr_matrix),target,intent(in)::y
  integer,allocatable::indices(:)
  real(dp),allocatable::distances(:)
  integer,optional,intent(in)::axis
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p
  logical,optional,intent(in)::squared
  type(row_view)::a,b
  a%sparse=>x
  a%n=x%n_rows;a%p=x%n_cols
  b%sparse=>y
  b%n=y%n_rows;b%p=y%n_cols
  call reduce_pairs(a,b,indices,distances,axis,metric,p,squared)
 end function
 function pairwise_distances_argmin_ss_callback(x,y,metric,axis) result(indices)
  type(csr_matrix),target,intent(in)::x
  type(csr_matrix),target,intent(in)::y
  integer,allocatable::indices(:)
  real(dp),allocatable::distances(:)
  integer,optional,intent(in)::axis
  procedure(row_metric)::metric
  type(row_view)::a,b
  a%sparse=>x
  a%n=x%n_rows;a%p=x%n_cols
  b%sparse=>y
  b%n=y%n_rows;b%p=y%n_cols
  call reduce_pairs(a,b,indices,distances,axis,callback=metric)
 end function
 function pairwise_distances_argmin_sd(x,y,metric,axis,p,squared) result(indices)
  type(csr_matrix),target,intent(in)::x
  real(dp),target,intent(in)::y(:,:)
  integer,allocatable::indices(:)
  real(dp),allocatable::distances(:)
  integer,optional,intent(in)::axis
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p
  logical,optional,intent(in)::squared
  type(row_view)::a,b
  a%sparse=>x
  a%n=x%n_rows;a%p=x%n_cols
  b%dense=>y
  b%n=size(y,1);b%p=size(y,2)
  call reduce_pairs(a,b,indices,distances,axis,metric,p,squared)
 end function
 function pairwise_distances_argmin_sd_callback(x,y,metric,axis) result(indices)
  type(csr_matrix),target,intent(in)::x
  real(dp),target,intent(in)::y(:,:)
  integer,allocatable::indices(:)
  real(dp),allocatable::distances(:)
  integer,optional,intent(in)::axis
  procedure(row_metric)::metric
  type(row_view)::a,b
  a%sparse=>x
  a%n=x%n_rows;a%p=x%n_cols
  b%dense=>y
  b%n=size(y,1);b%p=size(y,2)
  call reduce_pairs(a,b,indices,distances,axis,callback=metric)
 end function
 function pairwise_distances_argmin_ds(x,y,metric,axis,p,squared) result(indices)
  real(dp),target,intent(in)::x(:,:)
  type(csr_matrix),target,intent(in)::y
  integer,allocatable::indices(:)
  real(dp),allocatable::distances(:)
  integer,optional,intent(in)::axis
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p
  logical,optional,intent(in)::squared
  type(row_view)::a,b
  a%dense=>x
  a%n=size(x,1);a%p=size(x,2)
  b%sparse=>y
  b%n=y%n_rows;b%p=y%n_cols
  call reduce_pairs(a,b,indices,distances,axis,metric,p,squared)
 end function
 function pairwise_distances_argmin_ds_callback(x,y,metric,axis) result(indices)
  real(dp),target,intent(in)::x(:,:)
  type(csr_matrix),target,intent(in)::y
  integer,allocatable::indices(:)
  real(dp),allocatable::distances(:)
  integer,optional,intent(in)::axis
  procedure(row_metric)::metric
  type(row_view)::a,b
  a%dense=>x
  a%n=size(x,1);a%p=size(x,2)
  b%sparse=>y
  b%n=y%n_rows;b%p=y%n_cols
  call reduce_pairs(a,b,indices,distances,axis,callback=metric)
 end function
 subroutine pairwise_distances_argmin_min_dd(x,y,indices,distances,metric,axis,p,squared)
  real(dp),target,intent(in)::x(:,:)
  real(dp),target,intent(in)::y(:,:)
  integer,allocatable,intent(out)::indices(:)
  real(dp),allocatable,intent(out)::distances(:)
  integer,optional,intent(in)::axis
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p
  logical,optional,intent(in)::squared
  type(row_view)::a,b
  a%dense=>x
  a%n=size(x,1);a%p=size(x,2)
  b%dense=>y
  b%n=size(y,1);b%p=size(y,2)
  call reduce_pairs(a,b,indices,distances,axis,metric,p,squared)
 end subroutine
 subroutine pairwise_distances_argmin_min_dd_callback(x,y,indices,distances,metric,axis)
  real(dp),target,intent(in)::x(:,:)
  real(dp),target,intent(in)::y(:,:)
  integer,allocatable,intent(out)::indices(:)
  real(dp),allocatable,intent(out)::distances(:)
  integer,optional,intent(in)::axis
  procedure(row_metric)::metric
  type(row_view)::a,b
  a%dense=>x
  a%n=size(x,1);a%p=size(x,2)
  b%dense=>y
  b%n=size(y,1);b%p=size(y,2)
  call reduce_pairs(a,b,indices,distances,axis,callback=metric)
 end subroutine
 subroutine pairwise_distances_argmin_min_ss(x,y,indices,distances,metric,axis,p,squared)
  type(csr_matrix),target,intent(in)::x
  type(csr_matrix),target,intent(in)::y
  integer,allocatable,intent(out)::indices(:)
  real(dp),allocatable,intent(out)::distances(:)
  integer,optional,intent(in)::axis
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p
  logical,optional,intent(in)::squared
  type(row_view)::a,b
  a%sparse=>x
  a%n=x%n_rows;a%p=x%n_cols
  b%sparse=>y
  b%n=y%n_rows;b%p=y%n_cols
  call reduce_pairs(a,b,indices,distances,axis,metric,p,squared)
 end subroutine
 subroutine pairwise_distances_argmin_min_ss_callback(x,y,indices,distances,metric,axis)
  type(csr_matrix),target,intent(in)::x
  type(csr_matrix),target,intent(in)::y
  integer,allocatable,intent(out)::indices(:)
  real(dp),allocatable,intent(out)::distances(:)
  integer,optional,intent(in)::axis
  procedure(row_metric)::metric
  type(row_view)::a,b
  a%sparse=>x
  a%n=x%n_rows;a%p=x%n_cols
  b%sparse=>y
  b%n=y%n_rows;b%p=y%n_cols
  call reduce_pairs(a,b,indices,distances,axis,callback=metric)
 end subroutine
 subroutine pairwise_distances_argmin_min_sd(x,y,indices,distances,metric,axis,p,squared)
  type(csr_matrix),target,intent(in)::x
  real(dp),target,intent(in)::y(:,:)
  integer,allocatable,intent(out)::indices(:)
  real(dp),allocatable,intent(out)::distances(:)
  integer,optional,intent(in)::axis
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p
  logical,optional,intent(in)::squared
  type(row_view)::a,b
  a%sparse=>x
  a%n=x%n_rows;a%p=x%n_cols
  b%dense=>y
  b%n=size(y,1);b%p=size(y,2)
  call reduce_pairs(a,b,indices,distances,axis,metric,p,squared)
 end subroutine
 subroutine pairwise_distances_argmin_min_sd_callback(x,y,indices,distances,metric,axis)
  type(csr_matrix),target,intent(in)::x
  real(dp),target,intent(in)::y(:,:)
  integer,allocatable,intent(out)::indices(:)
  real(dp),allocatable,intent(out)::distances(:)
  integer,optional,intent(in)::axis
  procedure(row_metric)::metric
  type(row_view)::a,b
  a%sparse=>x
  a%n=x%n_rows;a%p=x%n_cols
  b%dense=>y
  b%n=size(y,1);b%p=size(y,2)
  call reduce_pairs(a,b,indices,distances,axis,callback=metric)
 end subroutine
 subroutine pairwise_distances_argmin_min_ds(x,y,indices,distances,metric,axis,p,squared)
  real(dp),target,intent(in)::x(:,:)
  type(csr_matrix),target,intent(in)::y
  integer,allocatable,intent(out)::indices(:)
  real(dp),allocatable,intent(out)::distances(:)
  integer,optional,intent(in)::axis
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p
  logical,optional,intent(in)::squared
  type(row_view)::a,b
  a%dense=>x
  a%n=size(x,1);a%p=size(x,2)
  b%sparse=>y
  b%n=y%n_rows;b%p=y%n_cols
  call reduce_pairs(a,b,indices,distances,axis,metric,p,squared)
 end subroutine
 subroutine pairwise_distances_argmin_min_ds_callback(x,y,indices,distances,metric,axis)
  real(dp),target,intent(in)::x(:,:)
  type(csr_matrix),target,intent(in)::y
  integer,allocatable,intent(out)::indices(:)
  real(dp),allocatable,intent(out)::distances(:)
  integer,optional,intent(in)::axis
  procedure(row_metric)::metric
  type(row_view)::a,b
  a%dense=>x
  a%n=size(x,1);a%p=size(x,2)
  b%sparse=>y
  b%n=y%n_rows;b%p=y%n_cols
  call reduce_pairs(a,b,indices,distances,axis,callback=metric)
 end subroutine
 subroutine check_view(a,allow_nan)
  type(row_view),intent(in)::a
  logical,intent(in)::allow_nan
  call require(a%n>0.and.a%p>0,'nearest-pair inputs require nonempty samples and features')
  if(associated(a%dense)) then
   if(allow_nan) then
    call require(all(ieee_is_finite(a%dense).or.ieee_is_nan(a%dense)),'nearest-pair inputs contain infinity')
   else
    call require(all(ieee_is_finite(a%dense)),'nearest-pair inputs must be finite')
   end if
  else
   call a%sparse%validate()
   call require(all(ieee_is_finite(a%sparse%data)),'nearest-pair CSR data must be finite')
  end if
 end subroutine
 subroutine get_row(a,i,row)
  type(row_view),intent(in)::a
  integer,intent(in)::i
  real(dp),intent(out)::row(:)
  integer::k
  if(associated(a%dense)) then
   row=a%dense(i,:)
  else
   row=0
   do k=a%sparse%indptr(i),a%sparse%indptr(i+1)-1
    row(a%sparse%indices(k))=row(a%sparse%indices(k))+a%sparse%data(k)
   end do
  end if
 end subroutine
 subroutine get_haversine_row(a,i,row)
  type(row_view),intent(in)::a
  integer,intent(in)::i
  real(dp),intent(out)::row(:)
  integer::k,start,component
  if(associated(a%dense))then
   row=a%dense(i,:)
  else
   ! Preserve the pinned 1.6.1 DistanceMetric CSR coordinate convention,
   ! including its modulo by zero-based row start for later stored rows.
   row=0;start=a%sparse%indptr(i)-1
   do k=a%sparse%indptr(i),a%sparse%indptr(i+1)-1
    component=a%sparse%indices(k)-1
    if(start/=0)component=mod(component,start)
    if(component==0)then
     row(1)=a%sparse%data(k)
    else
     row(2)=a%sparse%data(k)
    end if
   end do
  end if
 end subroutine
 integer function metric_kind(metric) result(kind)
  character(*),optional,intent(in)::metric
  character(:),allocatable::name
  name='euclidean';if(present(metric))name=trim(metric)
  kind=0
  select case(name)
  case('euclidean');kind=1
  case('sqeuclidean');kind=2
  case('l2');kind=3
  case('manhattan','cityblock','l1');kind=4
  case('chebyshev');kind=5
  case('minkowski');kind=6
  case('cosine');kind=7
  case('nan_euclidean');kind=8
  case('haversine');kind=9
  case('braycurtis');kind=10
  case('canberra');kind=11
  case('correlation');kind=12
  case('hamming');kind=13
  case('dice');kind=14
  case('jaccard');kind=15
  case('rogerstanimoto');kind=16
  case('russellrao');kind=17
  case('sokalmichener');kind=18
  case('sokalsneath');kind=19
  case('yule');kind=20
  case('matching');kind=21
  end select
  call require(kind>0,'unsupported nearest-pair distance metric')
 end function
 real(dp) function row_distance(u,v,kind,p,squared,sparse_u,sparse_v) result(value)
  real(dp),intent(in)::u(:),v(:),p
  integer,intent(in)::kind
  logical,intent(in)::squared,sparse_u,sparse_v
  real(dp)::xx,yy,xy,d,nx,ny
  real(dp),allocatable::one(:,:)
  integer::k
  select case(kind)
  case(1,2)
   ! The euclidean/sqeuclidean ArgKmin path uses the ordered Gram expression.
   xx=0;yy=0;xy=0
   do k=1,size(u)
    xx=xx+u(k)*u(k);yy=yy+v(k)*v(k);xy=xy+u(k)*v(k)
   end do
   value=(-2*xy+xx)+yy
   if(.not.(sparse_u.and.sparse_v))then
    ! Cython max(0., NaN) selects its first argument in the optimized path.
    if(ieee_is_nan(value))value=0
   end if
   if(.not.ieee_is_nan(value))value=max(value,0._dp)
  case(3)
   value=0
   do k=1,size(u)
    d=u(k)-v(k);value=value+d*d
   end do
  case(4)
   value=0
   do k=1,size(u)
    value=value+abs(u(k)-v(k))
   end do
  case(5)
   value=0
   do k=1,size(u)
    value=max(value,abs(u(k)-v(k)))
   end do
  case(6)
   value=0
   if(.not.ieee_is_finite(p)) then
    do k=1,size(u)
     value=max(value,abs(u(k)-v(k)))
    end do
   else
    do k=1,size(u)
     value=value+abs(u(k)-v(k))**p
    end do
   end if
  case(7)
   nx=sqrt(sum(u*u));ny=sqrt(sum(v*v))
   if(sparse_u) then
    if(nx==0)nx=1
   else
    if(nx<10*epsilon(nx))nx=1
   end if
   if(sparse_v) then
    if(ny==0)ny=1
   else
    if(ny<10*epsilon(ny))ny=1
   end if
   xy=0
   do k=1,size(u)
    xy=xy+(u(k)/nx)*(v(k)/ny)
   end do
   value=1-xy
   if(.not.ieee_is_nan(value))value=max(0._dp,min(2._dp,value))
  case(8)
   one=nan_euclidean_distances(reshape(u,[1,size(u)]),reshape(v,[1,size(v)]),squared=squared)
   value=one(1,1)
  case(9)
   xx=sin(.5_dp*(u(1)-v(1)));yy=sin(.5_dp*(u(2)-v(2)))
   value=xx*xx+cos(u(1))*cos(v(1))*yy*yy
  end select
 end function
 subroutine reduce_pairs(x,y,indices,distances,axis,metric,p,squared,callback)
  type(row_view),intent(in)::x,y
  integer,allocatable,intent(out)::indices(:)
  real(dp),allocatable,intent(out)::distances(:)
  integer,optional,intent(in)::axis
  character(*),optional,intent(in)::metric
  real(dp),optional,intent(in)::p
  logical,optional,intent(in)::squared
  procedure(row_metric),optional::callback
  type(row_view)::a,b
  integer::direction,kind,i,j,k
  real(dp)::power,value
  real(dp),allocatable::u(:),v(:)
  logical::square,halting(size(ieee_all)),raised(size(ieee_all)),allow_nan,optimized,backend_ok,self_pair
  logical,allocatable::mask_u(:),mask_v(:)
  direction=1;if(present(axis))direction=axis
  call require(direction==0.or.direction==1,'nearest-pair axis must be 0 or 1')
  kind=0;if(.not.present(callback))kind=metric_kind(metric)
  allow_nan=kind==8
  call check_view(x,allow_nan);call check_view(y,allow_nan)
  call require(x%p==y%p,'nearest-pair feature counts differ')
  if(kind==8)call require(.not.associated(x%sparse).and..not.associated(y%sparse), &
   'nan_euclidean requires dense nearest-pair inputs')
  if(kind==2)call require(.not.(associated(x%sparse).and.associated(y%sparse)), &
   'sqeuclidean does not support two CSR inputs')
  backend_ok=.true.
  if(associated(x%sparse))backend_ok=backend_ok.and.size(x%sparse%data)>0
  if(associated(y%sparse))backend_ok=backend_ok.and.size(y%sparse%data)>0
  if(any(kind==[2,5,6,10,11,21]))call require(backend_ok, &
   'this nearest-pair metric requires nonempty CSR storage')
  if(kind==9)call require(x%p==2,'haversine requires latitude and longitude columns')
  if(kind>=12.and.kind<=20)call require(.not.associated(x%sparse).and..not.associated(y%sparse), &
   'this nearest-pair dissimilarity requires dense inputs')
  power=2
  if(present(p))then
   call require(kind==6,'p is only supported for Minkowski nearest-pair distances')
   power=p
  end if
  if(kind==6)then
   call require(.not.ieee_is_nan(power),'Minkowski p must not be NaN')
   call require(power>0,'Minkowski p must be positive')
  end if
  square=.false.
  if(present(squared))then
   call require(kind==1.or.kind==8,'squared is only supported for euclidean or nan_euclidean')
   square=squared
  end if
  self_pair=.false.
  if(associated(x%dense).and.associated(y%dense))self_pair=associated(x%dense,y%dense)
  if(associated(x%sparse).and.associated(y%sparse))self_pair=associated(x%sparse,y%sparse)
  a=x;b=y
  if(direction==0)then
   a=y;b=x
  end if
  optimized=kind/=0.and.kind/=7.and.kind/=8.and.backend_ok
  if(kind==1.and.associated(a%sparse).and.associated(b%sparse))optimized=.false.
  if(kind>=12.and.kind<=20)optimized=.false.
  allocate(indices(a%n),distances(a%n),u(a%p),v(a%p))
  if(kind>=10)then
   allocate(mask_u(a%p),mask_v(a%p))
  else
   allocate(mask_u(0),mask_v(0))
  end if
  ! Capture all flags before any halting-mode changes (Windows clears flags).
  do k=1,size(ieee_all)
   call ieee_get_flag(ieee_all(k),raised(k))
  end do
  do k=1,size(ieee_all)
   call ieee_get_halting_mode(ieee_all(k),halting(k))
  end do
  do k=1,size(ieee_all)
   call ieee_set_halting_mode(ieee_all(k),.false.)
  end do
  do i=1,a%n
   if(kind==9)then
    call get_haversine_row(a,i,u)
   else
    call get_row(a,i,u)
   end if
   if(kind>=10)call stored_row(a,i,mask_u)
   indices(i)=1;distances(i)=huge(1._dp)
   do j=1,b%n
    if(kind==9)then
     call get_haversine_row(b,j,v)
    else
     call get_row(b,j,v)
    end if
    if(kind>=10)call stored_row(b,j,mask_v)
    if(present(callback))then
     if(self_pair.and.i>j)then
      value=callback(v,u)
     else
      value=callback(u,v)
     end if
    else if(kind>=10)then
     value=extended_distance(u,v,kind,associated(a%sparse),associated(b%sparse),mask_u,mask_v)
    else
     value=row_distance(u,v,kind,power,square,associated(a%sparse),associated(b%sparse))
    end if
    if(optimized)then
     ! Fixed k=1 heap: +Inf is not pushed; NaN is pushed, as in heap_push.
     if(.not.(value>=distances(i)))then
      indices(i)=j;distances(i)=value
     end if
    else if(j==1)then
     indices(i)=j;distances(i)=value
    else if(.not.ieee_is_nan(distances(i)))then
     if(ieee_is_nan(value).or.value<distances(i))then
      indices(i)=j;distances(i)=value
     end if
    end if
   end do
   select case(kind)
   case(1)
    if(.not.square)distances(i)=sqrt(distances(i))
   case(3)
    distances(i)=sqrt(distances(i))
   case(6)
    if(ieee_is_finite(power))distances(i)=distances(i)**(1/power)
   case(9)
    distances(i)=2*asin(sqrt(distances(i)))
   end select
  end do
  do k=1,size(ieee_all)
   call ieee_set_halting_mode(ieee_all(k),halting(k))
  end do
  do k=1,size(ieee_all)
   call ieee_set_flag(ieee_all(k),raised(k))
  end do
 end subroutine
 ! Additional kernels: sklearn DistanceMetric and SciPy 1.15.3 spatial
 ! distance.py / src/distance_impl.h. Damian Eads (2007, 2008), BSD-3-Clause;
 ! see LICENSES/scipy-distance-BSD and LICENSES/scipy-COPYING.
 subroutine stored_row(a,i,mask)
  type(row_view),intent(in)::a
  integer,intent(in)::i
  logical,intent(out)::mask(:)
  integer::k
  mask=.true.
  if(associated(a%sparse))then
   mask=.false.
   do k=a%sparse%indptr(i),a%sparse%indptr(i+1)-1
    mask(a%sparse%indices(k))=.true.
   end do
  end if
 end subroutine
 recursive real(dp) function ordered_pairwise(a)result(total)
  real(dp),intent(in)::a(:)
  real(dp)::lane(8)
  integer::n,i,j,whole,split
  n=size(a);total=-0._dp
  if(n<8)then
   do i=1,n
    total=total+a(i)
   end do
  else if(n<=128)then
   lane=a(:8);whole=n-mod(n,8)
   do i=9,whole,8
    do j=1,8
     lane(j)=lane(j)+a(i+j-1)
    end do
   end do
   total=((lane(1)+lane(2))+(lane(3)+lane(4)))+((lane(5)+lane(6))+(lane(7)+lane(8)))
   do i=whole+1,n
    total=total+a(i)
   end do
  else
   split=n/2;split=split-mod(split,8)
   total=ordered_pairwise(a(:split))+ordered_pairwise(a(split+1:))
  end if
 end function
 real(dp)function ordered_sum(a)result(total)
  real(dp),intent(in)::a(:)
  integer::first,last
  total=0
  do first=1,size(a),8192
   last=min(size(a),first+8191);total=total+ordered_pairwise(a(first:last))
  end do
 end function
 real(dp)function extended_distance(u,v,kind,sparse_u,sparse_v,mask_u,mask_v)result(value)
  real(dp),intent(in)::u(:),v(:)
  integer,intent(in)::kind
  logical,intent(in)::sparse_u,sparse_v,mask_u(:),mask_v(:)
  real(dp)::num,den,mean_u,mean_v,x,y,xx,yy,xy,tt,ff,tf,ft,unequal,cosine,n
  integer::k
  logical::bu,bv
  value=0;n=real(size(u),dp)
  select case(kind)
  case(10)
   num=0;den=0
   do k=1,size(u)
    num=num+abs(u(k)-v(k));den=den+(abs(u(k))+abs(v(k)))
   end do
   if(den>0)then
    value=num/den
   end if
   ! CSR Bray-Curtis in the pinned source has exhausted-buffer tail reads.
   ! Use its dense numerical contract, including zero for a zero denominator.
  case(11)
   do k=1,size(u)
    den=abs(u(k))+abs(v(k))
    if(sparse_u.or.sparse_v)then
     if(mask_u(k).and.mask_v(k))then
      value=value+abs(u(k)-v(k))/den
     else if(mask_u(k).or.mask_v(k))then
      value=value+1
     end if
    else if(den>0)then
     value=value+abs(u(k)-v(k))/den
    end if
   end do
  case(12)
   mean_u=ordered_sum(u)/n;mean_v=ordered_sum(v)/n
   xx=0;yy=0;xy=0
   do k=1,size(u)
    x=u(k)-mean_u;y=v(k)-mean_v
    xx=xx+x*x;yy=yy+y*y;xy=xy+x*y
   end do
   cosine=xy/(sqrt(xx)*sqrt(yy))
   if(abs(cosine)>1)cosine=sign(1._dp,cosine)
   value=1-cosine
  case(13)
   do k=1,size(u)
    if(u(k)/=v(k))value=value+1
   end do
   value=value/n
  case(14:21)
   tt=0;tf=0;ft=0;ff=0
   do k=1,size(u)
    bu=u(k)/=0;bv=v(k)/=0
    if(bu.and.bv)then
     tt=tt+1
    else if(bu)then
     tf=tf+1
    else if(bv)then
     ft=ft+1
    else
     ff=ff+1
    end if
   end do
   unequal=tf+ft
   select case(kind)
   case(14)
    value=unequal/(2*tt+unequal)
   case(15)
    if(tt+unequal>0)value=unequal/(tt+unequal)
   case(16,18)
    value=(2*unequal)/(n+unequal)
   case(17)
    value=(n-tt)/n
   case(19)
    value=(2*unequal)/(2*unequal+tt)
   case(20)
    if(tf*ft/=0)value=(2*tf*ft)/(tt*ff+tf*ft)
   case(21)
    value=unequal/n
   end select
  end select
 end function
end module
