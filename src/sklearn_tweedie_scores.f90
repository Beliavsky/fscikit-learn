! Adapted from sklearn 1.6.1 metrics/_regression.py (BSD-3-Clause).
! NumPy 2.2.6 buffered sums retain the BSD license under LICENSES/.
! Explicit rank adapters and IEEE-state preservation are MIT integration.
module sklearn_tweedie_scores
 use sklearn_common,only:dp,require
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::mean_tweedie_deviance,d2_tweedie_score
 interface mean_tweedie_deviance
  module procedure mean_tweedie_deviance_11
  module procedure mean_tweedie_deviance_12
  module procedure mean_tweedie_deviance_21
  module procedure mean_tweedie_deviance_22
 end interface
 interface d2_tweedie_score
  module procedure d2_tweedie_score_11
  module procedure d2_tweedie_score_12
  module procedure d2_tweedie_score_21
  module procedure d2_tweedie_score_22
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
 real(dp) function kernel(y,p,power,w) result(value)
  real(dp),intent(in)::y(:),p(:),power
  real(dp),optional,intent(in)::w(:)
  real(dp)::loss(size(y)),a,b,c,ratio,p1,p2,base,mass
  integer::i
  p1=subtract(1._dp,power);p2=subtract(2._dp,power)
  do i=1,size(y)
   if(power==0) then
    a=subtract(y(i),p(i));loss(i)=a*a
   else if(power==1) then
    ratio=y(i)/p(i);a=0
    if(y(i)/=0.or.ieee_is_nan(ratio)) then
     a=y(i)*log(ratio)
    end if
    loss(i)=2*subtract(subtract(a,y(i)),-p(i))
   else if(power==2) then
    ratio=p(i)/y(i);a=log(ratio);ratio=y(i)/p(i)
    loss(i)=2*subtract(subtract(a,-ratio),1._dp)
   else
    base=y(i)
    if(power<0) base=max(base,0._dp)
    a=base**p2;a=a/(p1*p2)
    b=p(i)**p1;b=y(i)*b;b=b/p1
    c=p(i)**p2;c=c/p2
    loss(i)=2*subtract(subtract(a,b),-c)
   end if
  end do
  mass=real(size(y),dp)
  if(present(w)) then
   mass=summed(w);loss=loss*w
  end if
  call require(mass/=0,'regression metric sample weights must have nonzero total')
  value=summed(loss)/mass
 end function
 real(dp) function evaluate(y,predicted,sample_weight,power,d2) result(value)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:),power
  logical,intent(in)::d2
  real(dp)::p,num,den,average,mass,baseline(size(y,1))
  logical::halt(size(ieee_all)),raised(size(ieee_all))
  integer::n
  p=0;if(present(power)) p=power
  call require(ieee_is_finite(p),'Tweedie power must be finite')
  call require(p<=0.or.p>=1,'Tweedie power must be at most zero or at least one')
  n=size(y,1)
  call require(n>0.and.size(y,2)>0,'regression metrics require nonempty targets')
  call require(all(shape(y)==shape(predicted)),'regression metric target/prediction shapes differ')
  call require(all(ieee_is_finite(y)).and.all(ieee_is_finite(predicted)), &
   'regression metric targets and predictions must be finite')
  call require(size(y,2)==1,'Tweedie metrics reject multioutput targets')
  if(d2.and.n<2) then
   value=ieee_value(0._dp,ieee_quiet_nan);return
  end if
  if(present(sample_weight)) then
   call require(size(sample_weight)==n,'regression metric sample weight length mismatch')
  end if
  if(p/=0) call require(all(predicted>0),'Tweedie predictions must be strictly positive')
  if(p>=1.and.p<2) call require(all(y>=0),'Tweedie targets must be nonnegative')
  if(p>=2) call require(all(y>0),'Tweedie targets must be strictly positive')
  call suspend(halt,raised)
  num=kernel(y(:,1),predicted(:,1),p,sample_weight)
  value=num
  if(d2) then
   mass=real(n,dp)
   if(present(sample_weight)) then
    mass=summed(sample_weight)
    average=summed(y(:,1)*sample_weight)/mass
   else
    average=summed(y(:,1))/mass
   end if
   ! The null model is not revalidated: signed/nonfinite weights can make it
   ! nonpositive or nonfinite, which the source kernel propagates arithmetically.
   baseline=average;den=kernel(y(:,1),baseline,p,sample_weight)
   call require(den/=0,'D2 Tweedie null deviance is zero')
   value=1-num/den
  end if
  call restore(halt,raised)
 end function
 real(dp) function mean_tweedie_deviance_11(y,predicted,sample_weight,power) result(value)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:),power
  value=evaluate(matrix(y),matrix(predicted),sample_weight,power,.false.)
 end function
 real(dp) function mean_tweedie_deviance_12(y,predicted,sample_weight,power) result(value)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:),power
  value=evaluate(matrix(y),matrix(predicted),sample_weight,power,.false.)
 end function
 real(dp) function mean_tweedie_deviance_21(y,predicted,sample_weight,power) result(value)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:),power
  value=evaluate(matrix(y),matrix(predicted),sample_weight,power,.false.)
 end function
 real(dp) function mean_tweedie_deviance_22(y,predicted,sample_weight,power) result(value)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:),power
  value=evaluate(matrix(y),matrix(predicted),sample_weight,power,.false.)
 end function
 real(dp) function d2_tweedie_score_11(y,predicted,sample_weight,power) result(value)
  real(dp),intent(in)::y(:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:),power
  value=evaluate(matrix(y),matrix(predicted),sample_weight,power,.true.)
 end function
 real(dp) function d2_tweedie_score_12(y,predicted,sample_weight,power) result(value)
  real(dp),intent(in)::y(:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:),power
  value=evaluate(matrix(y),matrix(predicted),sample_weight,power,.true.)
 end function
 real(dp) function d2_tweedie_score_21(y,predicted,sample_weight,power) result(value)
  real(dp),intent(in)::y(:,:),predicted(:)
  real(dp),optional,intent(in)::sample_weight(:),power
  value=evaluate(matrix(y),matrix(predicted),sample_weight,power,.true.)
 end function
 real(dp) function d2_tweedie_score_22(y,predicted,sample_weight,power) result(value)
  real(dp),intent(in)::y(:,:),predicted(:,:)
  real(dp),optional,intent(in)::sample_weight(:),power
  value=evaluate(matrix(y),matrix(predicted),sample_weight,power,.true.)
 end function
end module
