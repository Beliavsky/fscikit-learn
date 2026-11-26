! sklearn 1.6.1 _loss/loss.py, ensemble/_gb.py, utils/stats.py (BSD-3-Clause).
! Native reductions/adapters are MIT; NumPy ordering retains existing notices.
module sklearn_boosting_loss
 use sklearn_common,only:dp,require
 use sklearn_numpy_ordering,only:numpy_argsort
 use sklearn_ieee_math,only:ieee_difference,ieee_product,ieee_quotient
 use,intrinsic::ieee_arithmetic,only:ieee_next_after
 use,intrinsic::iso_c_binding,only:c_double
 implicit none
 private
 public::boosting_sum,boosting_mean,boosting_quantile,boosting_regression_loss,boosting_logit
 interface
  function boosting_c_log1p(x)bind(C,name='log1p')result(value)
   import c_double
   real(c_double),value::x
   real(c_double)::value
  end function
 end interface
contains
 ! SciPy 1.15.3 special/xsf/log_exp.h logit, BSD-3-Clause.
 real(dp)function boosting_logit(p)result(value)
  real(dp),intent(in)::p
  real(dp)::s
  if(p<.3_dp.or.p>.65_dp)then
   value=log(p/(1-p))
  else
   s=2*(p-.5_dp);value=boosting_c_log1p(s)-boosting_c_log1p(-s)
  end if
 end function
 recursive real(dp)function boosting_sum(a)result(total)
  real(dp),intent(in)::a(:)
  real(dp)::partial(8),left,right
  integer::n,i,j,whole,split
  n=size(a);total=-0._dp
  if(n<8)then
   do i=1,n;total=ieee_difference(total,-a(i));end do
  else if(n<=128)then
   partial=a(:8);whole=n-mod(n,8)
   do i=9,whole,8
    do j=1,8;partial(j)=ieee_difference(partial(j),-a(i+j-1));end do
   end do
   left=ieee_difference(ieee_difference(partial(1),-partial(2)),-ieee_difference(partial(3),-partial(4)))
   right=ieee_difference(ieee_difference(partial(5),-partial(6)),-ieee_difference(partial(7),-partial(8)))
   total=ieee_difference(left,-right)
   do i=whole+1,n;total=ieee_difference(total,-a(i));end do
  else
   split=n/2;split=split-mod(split,8)
   total=ieee_difference(boosting_sum(a(:split)),-boosting_sum(a(split+1:)))
  end if
 end function
 real(dp)function boosting_mean(y,w)result(value)
  real(dp),intent(in)::y(:),w(:)
  real(dp)::terms(size(y)),mass
  integer::i
  mass=boosting_sum(w);call require(mass/=0,'gradient boosting weights must have nonzero total')
  do i=1,size(y);terms(i)=ieee_product(y(i),w(i));end do
  value=ieee_quotient(boosting_sum(terms),mass)
 end function
 real(dp)function boosting_quantile(y,w,q,weighted)result(value)
  real(dp),intent(in)::y(:),w(:),q
  logical,intent(in)::weighted
  integer::order(size(y)),n,i,lo,hi,mid,a,b
  real(dp)::cdf(size(y)),position,fraction,diff,target,probability
  n=size(y);call require(n>0,'gradient boosting quantile needs observations')
  order=numpy_argsort(y);probability=(100*q)/100
  if(weighted)then
   cdf(1)=w(order(1))
   do i=2,n;cdf(i)=ieee_difference(cdf(i-1),-w(order(i)));end do
   target=ieee_product(probability,cdf(n))
   if(target==0)target=ieee_next_after(target,1._dp)
   ! NumPy searchsorted(left), including the installed implementation's
   ! deterministic search when signed weights make the CDF nonmonotone.
   lo=1;hi=n+1
   do while(lo<hi)
    mid=lo+(hi-lo)/2
    if(cdf(mid)<target)then
     lo=mid+1
    else
     hi=mid
    end if
   end do
   value=y(order(min(n,lo)))
  else
   position=(n-1)*probability;a=floor(position)+1;b=min(n,a+1);fraction=position-floor(position)
   diff=ieee_difference(y(order(b)),y(order(a)))
   if(fraction>=.5_dp)then
    value=ieee_difference(y(order(b)),ieee_product(diff,1-fraction))
   else
    value=ieee_difference(y(order(a)),-ieee_product(diff,fraction))
   end if
  end if
 end function
 real(dp)function boosting_regression_loss(loss,y,prediction,w,alpha,delta)result(value)
  character(*),intent(in)::loss
  real(dp),intent(in)::y(:),prediction(:),w(:),alpha,delta
  real(dp)::terms(size(y)),r,a
  integer::i
  do i=1,size(y)
   r=ieee_difference(y(i),prediction(i));a=abs(r)
   select case(loss)
   case('squared_error');terms(i)=ieee_product(.5_dp,ieee_product(r,r))
   case('absolute_error');terms(i)=a
   case('quantile');terms(i)=max(ieee_product(alpha,r),ieee_product(alpha-1,r))
   case('huber')
    if(a<=delta)then
     terms(i)=ieee_product(.5_dp,ieee_product(r,r))
    else
     terms(i)=ieee_product(delta,ieee_difference(a,.5_dp*delta))
    end if
   end select
  end do
  value=boosting_mean(terms,w)
  if(loss=='squared_error')value=ieee_product(2._dp,value)
 end function
end module
