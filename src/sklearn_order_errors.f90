! sklearn 1.6.1 median_absolute_error/max_error and weighted percentile (BSD).
! NumPy buffered sums and the existing Intel index-order helper retain BSD
! licenses under LICENSES/. Typed adapters/IEEE handling are MIT integration.
module sklearn_order_errors
 use sklearn_common,only:dp,require
 use sklearn_numpy_ordering,only:numpy_argsort
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::median_absolute_error,median_absolute_error_raw,max_error
 interface median_absolute_error
  module procedure median_11,median_weighted_11
  module procedure median_12,median_weighted_12
  module procedure median_21,median_weighted_21
  module procedure median_22,median_weighted_22
 end interface
 interface median_absolute_error_raw
  module procedure median_raw_11
  module procedure median_raw_12
  module procedure median_raw_21
  module procedure median_raw_22
 end interface
 interface max_error
  module procedure maximum_11
  module procedure maximum_12
  module procedure maximum_21
  module procedure maximum_22
 end interface
contains
 recursive real(dp) function pairwise(a) result(total)
  real(dp),intent(in)::a(:)
  real(dp)::r(8)
  integer::n,i,j,whole,split
  n=size(a);total=-0._dp
  if(n<8) then
   do i=1,n
    total=subtract(total,-a(i))
   end do
  else if(n<=128) then
   r=a(:8);whole=n-mod(n,8)
   do i=9,whole,8
    do j=1,8
     r(j)=subtract(r(j),-a(i+j-1))
    end do
   end do
   total=subtract(subtract(subtract(r(1),-r(2)),-subtract(r(3),-r(4))),-subtract(subtract(r(5),-r(6)),-subtract(r(7),-r(8))))
   do i=whole+1,n
    total=subtract(total,-a(i))
   end do
  else
   split=n/2;split=split-mod(split,8);total=subtract(pairwise(a(:split)),-pairwise(a(split+1:)))
  end if
 end function
 real(dp) function summed(a) result(value)
  real(dp),intent(in)::a(:)
  integer::first,last
  value=0
  do first=1,size(a),8192
   last=min(size(a),first+8191);value=subtract(value,-pairwise(a(first:last)))
  end do
 end function
 real(dp) function subtract(a,b) result(c)
  real(dp),intent(in)::a,b
  real(dp),volatile::x,y
  x=a;y=b;c=x-y
 end function
 function matrix(a) result(b)
  real(dp),intent(in)::a(..)
  real(dp),allocatable::b(:,:)
  select rank(a)
  rank(1)
   b=reshape(a,[size(a),1])
  rank(2)
   b=a
  end select
 end function
 function columns(a) result(v)
  real(dp),intent(in)::a(:,:)
  real(dp)::v(size(a,2))
  integer::i,j
  v=0
  if(size(a,2)==1) then
   v(1)=summed(a(:,1))
  else
   ! Match the slow-axis reduction of the pinned C-contiguous NumPy oracle.
   do i=1,size(a,1)
    do j=1,size(a,2)
     v(j)=subtract(v(j),-a(i,j))
    end do
   end do
  end if
 end function
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
 subroutine check_targets(y,p)
  real(dp),intent(in)::y(:,:),p(:,:)
  call require(size(y,1)>0.and.size(y,2)>0,'order errors require nonempty targets')
  call require(all(shape(y)==shape(p)),'order error target shapes differ')
  call require(all(ieee_is_finite(y)).and.all(ieee_is_finite(p)),'order error targets must be finite')
 end subroutine
 real(dp) function lower_median(errors,w) result(value)
  real(dp),intent(in)::errors(:)
  real(dp),optional,intent(in)::w(:)
  integer::order(size(errors)),n,i,left,right,mid
  real(dp)::cdf(size(errors)),target
  n=size(errors);order=numpy_argsort(errors)
  if(.not.present(w)) then
   if(mod(n,2)==1) then
    value=errors(order(n/2+1))
   else
    value=subtract(errors(order(n/2)),-errors(order(n/2+1)))/2
   end if
   return
  end if
  cdf(1)=w(order(1))
  do i=2,n
   cdf(i)=subtract(cdf(i-1),-w(order(i)))
  end do
  target=.5_dp*cdf(n)
  if(target==0) target=nearest(0._dp,1._dp)
  ! searchsorted uses a binary lower bound even for signed/nonmonotone CDFs.
  ! NaN CDF values compare after all finite/infinite values in NumPy's order.
  left=1;right=n+1
  do while(left<right)
   mid=left+(right-left)/2
   if(ieee_is_nan(target)) then
    if(ieee_is_nan(cdf(mid))) then
     right=mid
    else
     left=mid+1
    end if
   else if(ieee_is_nan(cdf(mid))) then
    right=mid
   else if(cdf(mid)<target) then
    left=mid+1
   else
    right=mid
   end if
  end do
  value=errors(order(min(left,n)))
 end function
 function evaluate(y,p,sample_weight,multioutput,output_weights,raw) result(value)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:),output_weights(:)
  character(*),optional,intent(in)::multioutput
  logical,optional,intent(in)::raw
  real(dp),allocatable::value(:),errors(:,:),medians(:)
  real(dp)::total
  logical::halting(size(ieee_all)),raised(size(ieee_all)),want_raw
  integer::q,j
  call check_targets(y,p);q=size(y,2)
  if(present(sample_weight)) then
   call require(size(sample_weight)==size(y,1),'median error sample weight length')
   call require(all(ieee_is_finite(sample_weight)),'median error sample weights must be finite')
  end if
  if(present(multioutput)) then
   call require(multioutput=='uniform_average','invalid median multioutput mode; use *_raw')
  end if
  if(present(output_weights)) then
   call require(q>1,'custom median output weights require multiple outputs')
   call require(size(output_weights)==q,'median error output weight length')
   call require(all(ieee_is_finite(output_weights)),'median error output weights must be finite')
  end if
  want_raw=.false.;if(present(raw)) want_raw=raw
  call suspend(halting,raised)
  errors=abs(p-y);allocate(medians(q))
  do j=1,q
   medians(j)=lower_median(errors(:,j),sample_weight)
  end do
  if(want_raw) then
   value=medians
  else if(present(output_weights)) then
   total=summed(output_weights)
   call require(total/=0,'median error output weights sum to zero')
   value=[summed(medians*output_weights)/total]
  else
   value=[summed(medians)/real(q,dp)]
  end if
  call restore(halting,raised)
 end function
 real(dp) function maximum(y,p) result(value)
  real(dp),intent(in)::y(:,:),p(:,:)
  logical::halting(size(ieee_all)),raised(size(ieee_all))
  call check_targets(y,p)
  call require(size(y,2)==1,'max_error rejects multioutput targets')
  call suspend(halting,raised)
  value=maxval(abs(y-p))
  call restore(halting,raised)
 end function
 function median_11(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function median_weighted_11(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function median_raw_11(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),sample_weight,raw=.true.)
 end function
 real(dp) function maximum_11(y,p) result(v)
  real(dp),intent(in)::y(:),p(:)
  v=maximum(matrix(y),matrix(p))
 end function
 function median_12(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function median_weighted_12(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function median_raw_12(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),sample_weight,raw=.true.)
 end function
 real(dp) function maximum_12(y,p) result(v)
  real(dp),intent(in)::y(:),p(:,:)
  v=maximum(matrix(y),matrix(p))
 end function
 function median_21(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function median_weighted_21(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function median_raw_21(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),sample_weight,raw=.true.)
 end function
 real(dp) function maximum_21(y,p) result(v)
  real(dp),intent(in)::y(:,:),p(:)
  v=maximum(matrix(y),matrix(p))
 end function
 function median_22(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  character(*),optional,intent(in)::multioutput
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),sample_weight,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),sample_weight,multioutput=multioutput,raw=.true.)
 end function
 function median_weighted_22(y,p,sample_weight,multioutput,raw_values) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),intent(in)::multioutput(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(p),sample_weight,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) raw_values=evaluate(matrix(y),matrix(p),sample_weight,output_weights=multioutput,raw=.true.)
 end function
 function median_raw_22(y,p,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(p),sample_weight,raw=.true.)
 end function
 real(dp) function maximum_22(y,p) result(v)
  real(dp),intent(in)::y(:,:),p(:,:)
  v=maximum(matrix(y),matrix(p))
 end function
end module
