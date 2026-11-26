program reference_distance_metric_csr_driver
 use sklearn_common,only:dp
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 use sklearn_metrics,only:distance_metric,get_metric
 use,intrinsic::ieee_arithmetic
 implicit none
 type(distance_metric)::metric
 type(csr_matrix),target::xs,ys
 real(dp),allocatable,target::x(:,:),y(:,:),weights(:),variance(:),matrix(:,:)
 real(dp),allocatable::d(:,:)
 real(dp)::power,conversion(7)
 integer::n,p,m,storage,own,params,i,status,repeat
 character(24)::name
 conversion=[-ieee_value(0._dp,ieee_positive_inf),-1._dp,0._dp,.5_dp,1._dp, &
  ieee_value(0._dp,ieee_positive_inf),ieee_value(0._dp,ieee_quiet_nan)]
 do
  read(*,*,iostat=status)n,p,m,storage,own,name,power,params
  if(status/=0)exit
  allocate(x(n,p),y(m,p),weights(p),variance(p),matrix(p,p))
  do i=1,n;read(*,*)x(i,:);end do
  do i=1,m;read(*,*)y(i,:);end do
  select case(params)
  case(1);read(*,*)weights;metric=get_metric(trim(name),p=power,w=weights)
  case(2);read(*,*)variance;metric=get_metric(trim(name),v=variance)
  case(3)
   do i=1,p;read(*,*)matrix(i,:);end do
   metric=get_metric(trim(name),vi=matrix)
  case(4)
   do i=1,p;read(*,*)matrix(i,:);end do
   metric=get_metric(trim(name),v=matrix)
  case default
   if(name=='minkowski'.or.name=='p')then
    metric=get_metric(trim(name),p=power)
   else if(name=='pyfunc')then
    metric=get_metric(callback)
   else
    metric=get_metric(trim(name))
   end if
  end select
  if(btest(storage,0))then
   call read_csr(xs,n,p)
  else
   xs=csr_from_dense(x)
  end if
  if(btest(storage,1))then
   call read_csr(ys,m,p)
  else
   ys=csr_from_dense(y)
  end if
  write(*,'(a)')trim(metric%type_name())
  do repeat=1,2
   if(own==1)then
    if(btest(storage,0))then
     d=metric%pairwise(xs)
    else
     d=metric%pairwise(x)
    end if
   else
    select case(storage)
    case(0);d=metric%pairwise(x,y)
    case(1);d=metric%pairwise(xs,y)
    case(2);d=metric%pairwise(x,ys)
    case(3);d=metric%pairwise(xs,ys)
    end select
   end if
   write(*,'(*(es26.17e3,1x))')transpose(d)
   write(*,'(*(i0,1x))')transpose(merge(1,0,metric%undefined_pairs()))
  end do
  write(*,'(*(es26.17e3,1x))')metric%dist_to_rdist(conversion)
  write(*,'(*(es26.17e3,1x))')metric%rdist_to_dist(conversion)
  deallocate(x,y,weights,variance,matrix)
 end do
contains
 subroutine read_csr(a,n,p)
  type(csr_matrix),intent(out)::a
  integer,intent(in)::n,p
  integer::nnz
  read(*,*)nnz
  a%n_rows=n;a%n_cols=p
  allocate(a%indptr(n+1),a%indices(nnz),a%data(nnz))
  read(*,*)a%indptr
  if(nnz>0)then
   read(*,*)a%indices
   read(*,*)a%data
  end if
 end subroutine
 function callback(u,v)result(value)
  real(dp),intent(in)::u(:),v(:)
  real(dp)::value
  value=1000+sum(u)-2*sum(v)
 end function
end program
