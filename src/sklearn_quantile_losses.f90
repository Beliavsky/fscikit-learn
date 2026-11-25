! sklearn 1.6.1 pinball loss and D2 quantile scores (BSD-3-Clause).
! NumPy buffered sums / existing Intel index ordering retain BSD notices.
! Rank, legacy-keyword, owned-result and IEEE adapters are MIT integration.
module sklearn_quantile_losses
 use sklearn_common,only:dp,require
 use sklearn_numpy_ordering,only:numpy_argsort
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::mean_pinball_loss,mean_pinball_loss_raw
 interface mean_pinball_loss
  module procedure mean_pinball_loss_legacy_11,mean_pinball_loss_char_11,mean_pinball_loss_weights_11
  module procedure mean_pinball_loss_legacy_12,mean_pinball_loss_char_12,mean_pinball_loss_weights_12
  module procedure mean_pinball_loss_legacy_21,mean_pinball_loss_char_21,mean_pinball_loss_weights_21
  module procedure mean_pinball_loss_legacy_22,mean_pinball_loss_char_22,mean_pinball_loss_weights_22
 end interface
 interface mean_pinball_loss_raw
  module procedure mean_pinball_loss_raw_11
  module procedure mean_pinball_loss_raw_12
  module procedure mean_pinball_loss_raw_21
  module procedure mean_pinball_loss_raw_22
 end interface
 public::d2_pinball_score,d2_pinball_score_raw
 interface d2_pinball_score
  module procedure d2_pinball_score_legacy_11,d2_pinball_score_char_11,d2_pinball_score_weights_11
  module procedure d2_pinball_score_legacy_12,d2_pinball_score_char_12,d2_pinball_score_weights_12
  module procedure d2_pinball_score_legacy_21,d2_pinball_score_char_21,d2_pinball_score_weights_21
  module procedure d2_pinball_score_legacy_22,d2_pinball_score_char_22,d2_pinball_score_weights_22
 end interface
 interface d2_pinball_score_raw
  module procedure d2_pinball_score_raw_11
  module procedure d2_pinball_score_raw_12
  module procedure d2_pinball_score_raw_21
  module procedure d2_pinball_score_raw_22
 end interface
 public::d2_absolute_error_score,d2_absolute_error_score_raw
 interface d2_absolute_error_score
  module procedure d2_absolute_error_score_legacy_11,d2_absolute_error_score_char_11,d2_absolute_error_score_weights_11
  module procedure d2_absolute_error_score_legacy_12,d2_absolute_error_score_char_12,d2_absolute_error_score_weights_12
  module procedure d2_absolute_error_score_legacy_21,d2_absolute_error_score_char_21,d2_absolute_error_score_weights_21
  module procedure d2_absolute_error_score_legacy_22,d2_absolute_error_score_char_22,d2_absolute_error_score_weights_22
 end interface
 interface d2_absolute_error_score_raw
  module procedure d2_absolute_error_score_raw_11
  module procedure d2_absolute_error_score_raw_12
  module procedure d2_absolute_error_score_raw_21
  module procedure d2_absolute_error_score_raw_22
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
 real(dp) function quantile(y,alpha,w) result(value)
  real(dp),intent(in)::y(:),alpha
  real(dp),optional,intent(in)::w(:)
  integer::order(size(y)),n,i,left,right,mid,index
  real(dp)::cdf(size(y)),fraction,target,position,difference,percent
  n=size(y);order=numpy_argsort(y);percent=alpha*100;fraction=percent/100
  if(.not.present(w)) then
   position=fraction*real(n-1,dp);index=floor(position)+1;fraction=position-floor(position)
   if(index>=n) then
    value=y(order(n));return
   end if
   difference=subtract(y(order(index+1)),y(order(index)))
   value=subtract(y(order(index)),-difference*fraction)
   if(fraction>=.5_dp) value=subtract(y(order(index+1)),difference*(1-fraction))
   return
  end if
  cdf(1)=w(order(1))
  do i=2,n
   cdf(i)=subtract(cdf(i-1),-w(order(i)))
  end do
  target=fraction*cdf(n)
  if(target==0) target=nearest(0._dp,1._dp)
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
  value=y(order(min(left,n)))
 end function
 function average_loss(y,p,alpha,w) result(values)
  real(dp),intent(in)::y(:,:),p(:,:),alpha
  real(dp),optional,intent(in)::w(:)
  real(dp)::values(size(y,2)),loss(size(y,1),size(y,2)),diff,sign_,first,second,mass
  integer::i,j
  do j=1,size(y,2)
   do i=1,size(y,1)
    diff=subtract(y(i,j),p(i,j));sign_=0
    if(diff>=0) sign_=1
    first=alpha*sign_;first=first*diff
    second=(1-alpha)*(1-sign_);second=second*diff
    loss(i,j)=subtract(first,second)
   end do
  end do
  mass=real(size(y,1),dp)
  if(present(w)) then
   mass=summed(w)
   do j=1,size(y,2)
    loss(:,j)=loss(:,j)*w
   end do
  end if
  call require(mass/=0,'regression metric sample weights must have nonzero total')
  values=columns(loss)/mass
 end function
 function evaluate(y,p,kind,sample_weight,alpha,multioutput,output_weights,raw) result(value)
  real(dp),intent(in)::y(:,:),p(:,:)
  integer,intent(in)::kind
  real(dp),optional,intent(in)::sample_weight(:),alpha,output_weights(:)
  character(*),optional,intent(in)::multioutput
  logical,optional,intent(in)::raw
  real(dp),allocatable::value(:),num(:),den(:),scores(:),baseline(:,:)
  real(dp)::a,total
  integer::n,q,j
  logical::halting(size(ieee_all)),raised(size(ieee_all)),want_raw
  a=.5_dp;if(present(alpha)) a=alpha
  call require(ieee_is_finite(a),'pinball alpha must be finite')
  call require(a>=0.and.a<=1,'pinball alpha must be in [0,1]')
  n=size(y,1);q=size(y,2)
  call require(n>0.and.q>0,'regression metrics require nonempty targets')
  call require(all(shape(y)==shape(p)),'regression metric target/prediction shapes differ')
  call require(all(ieee_is_finite(y)).and.all(ieee_is_finite(p)), &
   'regression metric targets and predictions must be finite')
  if(present(sample_weight)) call require(size(sample_weight)==n,'regression metric sample weight length mismatch')
  if(present(multioutput)) call require(multioutput=='uniform_average','invalid quantile multioutput mode; use *_raw')
  if(present(output_weights)) then
   call require(q>1,'custom quantile output weights require multiple outputs')
   call require(size(output_weights)==q,'regression metric output weight length mismatch')
   call require(all(ieee_is_finite(output_weights)),'regression metric output weights must be finite')
  end if
  if(kind>0.and.n<2) then
   value=[ieee_value(0._dp,ieee_quiet_nan)];return
  end if
  want_raw=.false.;if(present(raw)) want_raw=raw
  call suspend(halting,raised)
  num=average_loss(y,p,a,sample_weight)
  scores=num
  if(kind>0) then
   if(present(sample_weight)) then
    call require(all(ieee_is_finite(sample_weight)),'regression metric sample weights must be finite')
   end if
   allocate(baseline(n,q))
   do j=1,q
    baseline(:,j)=quantile(y(:,j),a,sample_weight)
   end do
   call require(all(ieee_is_finite(baseline)),'pinball quantile baseline must be finite')
   den=average_loss(y,baseline,a,sample_weight);scores=1
   do j=1,q
    if(num(j)/=0) then
     if(den(j)/=0) then
      scores(j)=1-num(j)/den(j)
     else
      scores(j)=0
     end if
    end if
   end do
  end if
  if(want_raw) then
   value=scores
  else if(present(output_weights)) then
   total=summed(output_weights)
   call require(total/=0,'regression metric output weights must have nonzero total')
   value=[summed(scores*output_weights)/total]
  else
   value=[summed(scores)/real(q,dp)]
  end if
  call restore(halting,raised)
 end function
 function mean_pinball_loss_legacy_11(y,predicted,sample_weight,alpha,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha, &
    output_weights=output_weights,raw=.true.)
  end if
 end function
 function mean_pinball_loss_char_11(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  character(*),intent(in)::multioutput
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha, &
    multioutput=multioutput,raw=.true.)
  end if
 end function
 function mean_pinball_loss_weights_11(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp),intent(in)::multioutput(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha, &
    output_weights=multioutput,raw=.true.)
  end if
 end function
 function mean_pinball_loss_raw_11(y,predicted,sample_weight,alpha) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,raw=.true.)
 end function
 function mean_pinball_loss_legacy_12(y,predicted,sample_weight,alpha,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha, &
    output_weights=output_weights,raw=.true.)
  end if
 end function
 function mean_pinball_loss_char_12(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  character(*),intent(in)::multioutput
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha, &
    multioutput=multioutput,raw=.true.)
  end if
 end function
 function mean_pinball_loss_weights_12(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp),intent(in)::multioutput(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha, &
    output_weights=multioutput,raw=.true.)
  end if
 end function
 function mean_pinball_loss_raw_12(y,predicted,sample_weight,alpha) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,raw=.true.)
 end function
 function mean_pinball_loss_legacy_21(y,predicted,sample_weight,alpha,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha, &
    output_weights=output_weights,raw=.true.)
  end if
 end function
 function mean_pinball_loss_char_21(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  character(*),intent(in)::multioutput
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha, &
    multioutput=multioutput,raw=.true.)
  end if
 end function
 function mean_pinball_loss_weights_21(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp),intent(in)::multioutput(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha, &
    output_weights=multioutput,raw=.true.)
  end if
 end function
 function mean_pinball_loss_raw_21(y,predicted,sample_weight,alpha) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,raw=.true.)
 end function
 function mean_pinball_loss_legacy_22(y,predicted,sample_weight,alpha,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha, &
    output_weights=output_weights,raw=.true.)
  end if
 end function
 function mean_pinball_loss_char_22(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  character(*),intent(in)::multioutput
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha, &
    multioutput=multioutput,raw=.true.)
  end if
 end function
 function mean_pinball_loss_weights_22(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp),intent(in)::multioutput(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha, &
    output_weights=multioutput,raw=.true.)
  end if
 end function
 function mean_pinball_loss_raw_22(y,predicted,sample_weight,alpha) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(predicted),0,sample_weight,alpha,raw=.true.)
 end function
 function d2_pinball_score_legacy_11(y,predicted,sample_weight,alpha,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha, &
    output_weights=output_weights,raw=.true.)
   if(size(y)==1.and.size(raw_values)==1) raw_values=[raw_values(1)]
  end if
 end function
 function d2_pinball_score_char_11(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  character(*),intent(in)::multioutput
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha, &
    multioutput=multioutput,raw=.true.)
   if(size(y)==1.and.size(raw_values)==1) raw_values=[raw_values(1)]
  end if
 end function
 function d2_pinball_score_weights_11(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp),intent(in)::multioutput(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha, &
    output_weights=multioutput,raw=.true.)
   if(size(y)==1.and.size(raw_values)==1) raw_values=[raw_values(1)]
  end if
 end function
 function d2_pinball_score_raw_11(y,predicted,sample_weight,alpha) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,raw=.true.)
 end function
 function d2_pinball_score_legacy_12(y,predicted,sample_weight,alpha,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha, &
    output_weights=output_weights,raw=.true.)
   if(size(y)==1.and.size(raw_values)==1) raw_values=[raw_values(1)]
  end if
 end function
 function d2_pinball_score_char_12(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  character(*),intent(in)::multioutput
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha, &
    multioutput=multioutput,raw=.true.)
   if(size(y)==1.and.size(raw_values)==1) raw_values=[raw_values(1)]
  end if
 end function
 function d2_pinball_score_weights_12(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp),intent(in)::multioutput(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha, &
    output_weights=multioutput,raw=.true.)
   if(size(y)==1.and.size(raw_values)==1) raw_values=[raw_values(1)]
  end if
 end function
 function d2_pinball_score_raw_12(y,predicted,sample_weight,alpha) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,raw=.true.)
 end function
 function d2_pinball_score_legacy_21(y,predicted,sample_weight,alpha,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha, &
    output_weights=output_weights,raw=.true.)
   if(size(y,1)==1.and.size(raw_values)==1) raw_values=spread(raw_values(1),1,size(matrix(y),2))
  end if
 end function
 function d2_pinball_score_char_21(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  character(*),intent(in)::multioutput
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha, &
    multioutput=multioutput,raw=.true.)
   if(size(y,1)==1.and.size(raw_values)==1) raw_values=spread(raw_values(1),1,size(matrix(y),2))
  end if
 end function
 function d2_pinball_score_weights_21(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp),intent(in)::multioutput(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha, &
    output_weights=multioutput,raw=.true.)
   if(size(y,1)==1.and.size(raw_values)==1) raw_values=spread(raw_values(1),1,size(matrix(y),2))
  end if
 end function
 function d2_pinball_score_raw_21(y,predicted,sample_weight,alpha) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,raw=.true.)
 end function
 function d2_pinball_score_legacy_22(y,predicted,sample_weight,alpha,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha, &
    output_weights=output_weights,raw=.true.)
   if(size(y,1)==1.and.size(raw_values)==1) raw_values=spread(raw_values(1),1,size(matrix(y),2))
  end if
 end function
 function d2_pinball_score_char_22(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  character(*),intent(in)::multioutput
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha, &
    multioutput=multioutput,raw=.true.)
   if(size(y,1)==1.and.size(raw_values)==1) raw_values=spread(raw_values(1),1,size(matrix(y),2))
  end if
 end function
 function d2_pinball_score_weights_22(y,predicted,sample_weight,alpha,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp),intent(in)::multioutput(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha, &
    output_weights=multioutput,raw=.true.)
   if(size(y,1)==1.and.size(raw_values)==1) raw_values=spread(raw_values(1),1,size(matrix(y),2))
  end if
 end function
 function d2_pinball_score_raw_22(y,predicted,sample_weight,alpha) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::alpha
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(predicted),1,sample_weight,alpha,raw=.true.)
 end function
 function d2_absolute_error_score_legacy_11(y,predicted,sample_weight,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp, &
    output_weights=output_weights,raw=.true.)
   if(size(y)==1.and.size(raw_values)==1) raw_values=[raw_values(1)]
  end if
 end function
 function d2_absolute_error_score_char_11(y,predicted,sample_weight,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  character(*),intent(in)::multioutput
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp, &
    multioutput=multioutput,raw=.true.)
   if(size(y)==1.and.size(raw_values)==1) raw_values=[raw_values(1)]
  end if
 end function
 function d2_absolute_error_score_weights_11(y,predicted,sample_weight,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp),intent(in)::multioutput(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp, &
    output_weights=multioutput,raw=.true.)
   if(size(y)==1.and.size(raw_values)==1) raw_values=[raw_values(1)]
  end if
 end function
 function d2_absolute_error_score_raw_11(y,predicted,sample_weight) result(v)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,raw=.true.)
 end function
 function d2_absolute_error_score_legacy_12(y,predicted,sample_weight,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp, &
    output_weights=output_weights,raw=.true.)
   if(size(y)==1.and.size(raw_values)==1) raw_values=[raw_values(1)]
  end if
 end function
 function d2_absolute_error_score_char_12(y,predicted,sample_weight,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  character(*),intent(in)::multioutput
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp, &
    multioutput=multioutput,raw=.true.)
   if(size(y)==1.and.size(raw_values)==1) raw_values=[raw_values(1)]
  end if
 end function
 function d2_absolute_error_score_weights_12(y,predicted,sample_weight,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp),intent(in)::multioutput(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp, &
    output_weights=multioutput,raw=.true.)
   if(size(y)==1.and.size(raw_values)==1) raw_values=[raw_values(1)]
  end if
 end function
 function d2_absolute_error_score_raw_12(y,predicted,sample_weight) result(v)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,raw=.true.)
 end function
 function d2_absolute_error_score_legacy_21(y,predicted,sample_weight,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp, &
    output_weights=output_weights,raw=.true.)
   if(size(y,1)==1.and.size(raw_values)==1) raw_values=spread(raw_values(1),1,size(matrix(y),2))
  end if
 end function
 function d2_absolute_error_score_char_21(y,predicted,sample_weight,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  character(*),intent(in)::multioutput
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp, &
    multioutput=multioutput,raw=.true.)
   if(size(y,1)==1.and.size(raw_values)==1) raw_values=spread(raw_values(1),1,size(matrix(y),2))
  end if
 end function
 function d2_absolute_error_score_weights_21(y,predicted,sample_weight,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp),intent(in)::multioutput(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp, &
    output_weights=multioutput,raw=.true.)
   if(size(y,1)==1.and.size(raw_values)==1) raw_values=spread(raw_values(1),1,size(matrix(y),2))
  end if
 end function
 function d2_absolute_error_score_raw_21(y,predicted,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,raw=.true.)
 end function
 function d2_absolute_error_score_legacy_22(y,predicted,sample_weight,output_weights,raw_values) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  tmp=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,output_weights=output_weights)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp, &
    output_weights=output_weights,raw=.true.)
   if(size(y,1)==1.and.size(raw_values)==1) raw_values=spread(raw_values(1),1,size(matrix(y),2))
  end if
 end function
 function d2_absolute_error_score_char_22(y,predicted,sample_weight,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  character(*),intent(in)::multioutput
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,multioutput=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp, &
    multioutput=multioutput,raw=.true.)
   if(size(y,1)==1.and.size(raw_values)==1) raw_values=spread(raw_values(1),1,size(matrix(y),2))
  end if
 end function
 function d2_absolute_error_score_weights_22(y,predicted,sample_weight,output_weights,raw_values,multioutput) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),optional,intent(in)::output_weights(:)
  real(dp),optional,allocatable,intent(out)::raw_values(:)
  real(dp),intent(in)::multioutput(:)
  real(dp)::v
  real(dp),allocatable::tmp(:)
  call require(.not.present(output_weights),'cannot combine multioutput and output_weights')
  tmp=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,output_weights=multioutput)
  v=tmp(1)
  if(present(raw_values)) then
   raw_values=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp, &
    output_weights=multioutput,raw=.true.)
   if(size(y,1)==1.and.size(raw_values)==1) raw_values=spread(raw_values(1),1,size(matrix(y),2))
  end if
 end function
 function d2_absolute_error_score_raw_22(y,predicted,sample_weight) result(v)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::v(:)
  v=evaluate(matrix(y),matrix(predicted),2,sample_weight,.5_dp,raw=.true.)
 end function
end module
