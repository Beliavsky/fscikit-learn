! sklearn 1.6.1 metrics/_classification.py class_likelihood_ratios (BSD-3-Clause).
! Numeric adapters and quiet-IEEE integration are MIT; see THIRD_PARTY_NOTICES.md.
module sklearn_likelihood_ratios
 use sklearn_common,only:dp,require
 use sklearn_ieee_math,only:add_negative=>ieee_difference,mul=>ieee_product,div=>ieee_quotient
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic,only:ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan
 implicit none
 private
 public::class_likelihood_ratios
 interface class_likelihood_ratios
  module procedure ratios_iv_iv
  module procedure ratios_iv_im
  module procedure ratios_iv_rv
  module procedure ratios_iv_rm
  module procedure ratios_iv_lv
  module procedure ratios_iv_lm
  module procedure ratios_im_iv
  module procedure ratios_im_im
  module procedure ratios_im_rv
  module procedure ratios_im_rm
  module procedure ratios_im_lv
  module procedure ratios_im_lm
  module procedure ratios_rv_iv
  module procedure ratios_rv_im
  module procedure ratios_rv_rv
  module procedure ratios_rv_rm
  module procedure ratios_rv_lv
  module procedure ratios_rv_lm
  module procedure ratios_rm_iv
  module procedure ratios_rm_im
  module procedure ratios_rm_rv
  module procedure ratios_rm_rm
  module procedure ratios_rm_lv
  module procedure ratios_rm_lm
  module procedure ratios_lv_iv
  module procedure ratios_lv_im
  module procedure ratios_lv_rv
  module procedure ratios_lv_rm
  module procedure ratios_lv_lv
  module procedure ratios_lv_lm
  module procedure ratios_lm_iv
  module procedure ratios_lm_im
  module procedure ratios_lm_rv
  module procedure ratios_lm_rm
  module procedure ratios_lm_lv
  module procedure ratios_lm_lm
 end interface
contains
 function target_i(x) result(values)
  integer,intent(in)::x(..)
  real(dp),allocatable::values(:)
  select rank(x)
  rank(1)
   values=real(x,dp)
  rank(2)
   call require(size(x,2)==1.and.size(x,1)>0,'likelihood ratios require a nonempty single-column target')
   values=real(x(:,1),dp)
  end select
 end function
 function target_r(x) result(values)
  real(dp),intent(in)::x(..)
  real(dp),allocatable::values(:)
  select rank(x)
  rank(1)
   values=x
  rank(2)
   call require(size(x,2)==1.and.size(x,1)>0,'likelihood ratios require a nonempty single-column target')
   values=x(:,1)
  end select
 end function
 function target_l(x) result(values)
  logical,intent(in)::x(..)
  real(dp),allocatable::values(:)
  select rank(x)
  rank(1)
   values=merge(1._dp,0._dp,x)
  rank(2)
   call require(size(x,2)==1.and.size(x,1)>0,'likelihood ratios require a nonempty single-column target')
   values=merge(1._dp,0._dp,x(:,1))
  end select
 end function
 logical function equal(a,b) result(same)
  real(dp),intent(in)::a,b
  same=.false.
  if(ieee_is_nan(a).or.ieee_is_nan(b)) return
  same=a==b
 end function
 subroutine recognize(x,classes,n)
  real(dp),intent(in)::x(:)
  real(dp),intent(inout)::classes(2)
  integer,intent(inout)::n
  integer::i
  do i=1,size(x)
   call require(ieee_is_finite(x(i)),'likelihood ratio targets must be finite')
   call require(x(i)>=-2._dp**63.and.x(i)<2._dp**63,'likelihood ratio targets exceed int64 range')
   call require(real(int(x(i),int64),dp)==x(i),'likelihood ratios require integral classification targets')
   if(any(classes(:n)==x(i))) cycle
   call require(n<2,'likelihood ratios only support binary classification')
   n=n+1;classes(n)=x(i)
  end do
 end subroutine
 function calculate(y,p,labels,sample_weight) result(ratios)
  real(dp),intent(in)::y(:),p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2),classes(2),cm(2,2),tn,fp,fn,tp,support_pos,support_neg,w,tmp
  real(dp),allocatable::selected(:)
  integer::n,i,j,a,b
  logical::found
  call require(size(y)==size(p),'likelihood ratio target lengths differ')
  n=0;classes=0;call recognize(y,classes,n);call recognize(p,classes,n)
  if(n==2) then
   if(classes(1)>classes(2)) then
    tmp=classes(1);classes(1)=classes(2);classes(2)=tmp
   end if
  end if
  if(present(labels)) then
   call require(size(labels)>0,'likelihood ratio labels must not be empty')
   select type(labels)
   type is(integer)
    selected=real(labels,dp)
   type is(real(dp))
    selected=labels
   type is(logical)
    selected=merge(1._dp,0._dp,labels)
   class default
    call require(.false.,'unsupported likelihood ratio label kind')
   end select
   if(size(y)>0) then
    found=.false.
    do j=1,size(selected)
     do i=1,size(y)
      if(equal(selected(j),y(i))) found=.true.
     end do
    end do
    call require(found,'at least one likelihood ratio label must occur in y')
   end if
   n=size(selected)
  else
   call require(n>0,'empty likelihood ratio targets require explicit labels')
   selected=classes(:n)
  end if
  call require(n<=2,'likelihood ratios require one or two selected labels')
  ! sklearn's empty-target/explicit-label shortcut precedes weight validation.
  if(size(y)>0.or..not.present(labels)) then
   if(present(sample_weight)) call require(size(sample_weight)==size(y),'likelihood ratio weight length mismatch')
  end if
  ratios=ieee_value(0._dp,ieee_quiet_nan)
  if(n==1.or.size(y)==0) return
  cm=0
  do i=1,size(y)
   a=0;b=0
   do j=1,n
    ! Last duplicate wins, matching the source label-to-index dictionary.
    if(equal(y(i),selected(j))) a=j
    if(equal(p(i),selected(j))) b=j
   end do
   if(a==0.or.b==0) cycle
   w=1
   if(present(sample_weight)) w=sample_weight(i)
   cm(a,b)=add_negative(cm(a,b),-w)
  end do
  ! confusion_matrix applies nan_to_num before forming likelihood ratios.
  do j=1,2
   do i=1,2
    if(ieee_is_nan(cm(i,j))) then
     cm(i,j)=0
    else if(.not.ieee_is_finite(cm(i,j))) then
     cm(i,j)=sign(huge(1._dp),cm(i,j))
    end if
   end do
  end do
  tn=cm(1,1);fp=cm(1,2);fn=cm(2,1);tp=cm(2,2)
  support_pos=add_negative(tp,-fn);support_neg=add_negative(tn,-fp)
  if(fp/=0) ratios(1)=div(mul(tp,support_neg),mul(fp,support_pos))
  if(tn/=0) ratios(2)=div(mul(fn,support_neg),mul(tn,support_pos))
 end function
 function ratios_iv_iv(y,p,labels,sample_weight) result(ratios)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_i(y),target_i(p),labels,sample_weight)
 end function
 function ratios_iv_im(y,p,labels,sample_weight) result(ratios)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_i(y),target_i(p),labels,sample_weight)
 end function
 function ratios_iv_rv(y,p,labels,sample_weight) result(ratios)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_i(y),target_r(p),labels,sample_weight)
 end function
 function ratios_iv_rm(y,p,labels,sample_weight) result(ratios)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_i(y),target_r(p),labels,sample_weight)
 end function
 function ratios_iv_lv(y,p,labels,sample_weight) result(ratios)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_i(y),target_l(p),labels,sample_weight)
 end function
 function ratios_iv_lm(y,p,labels,sample_weight) result(ratios)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_i(y),target_l(p),labels,sample_weight)
 end function
 function ratios_im_iv(y,p,labels,sample_weight) result(ratios)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_i(y),target_i(p),labels,sample_weight)
 end function
 function ratios_im_im(y,p,labels,sample_weight) result(ratios)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_i(y),target_i(p),labels,sample_weight)
 end function
 function ratios_im_rv(y,p,labels,sample_weight) result(ratios)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_i(y),target_r(p),labels,sample_weight)
 end function
 function ratios_im_rm(y,p,labels,sample_weight) result(ratios)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_i(y),target_r(p),labels,sample_weight)
 end function
 function ratios_im_lv(y,p,labels,sample_weight) result(ratios)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_i(y),target_l(p),labels,sample_weight)
 end function
 function ratios_im_lm(y,p,labels,sample_weight) result(ratios)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_i(y),target_l(p),labels,sample_weight)
 end function
 function ratios_rv_iv(y,p,labels,sample_weight) result(ratios)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_r(y),target_i(p),labels,sample_weight)
 end function
 function ratios_rv_im(y,p,labels,sample_weight) result(ratios)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_r(y),target_i(p),labels,sample_weight)
 end function
 function ratios_rv_rv(y,p,labels,sample_weight) result(ratios)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_r(y),target_r(p),labels,sample_weight)
 end function
 function ratios_rv_rm(y,p,labels,sample_weight) result(ratios)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_r(y),target_r(p),labels,sample_weight)
 end function
 function ratios_rv_lv(y,p,labels,sample_weight) result(ratios)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_r(y),target_l(p),labels,sample_weight)
 end function
 function ratios_rv_lm(y,p,labels,sample_weight) result(ratios)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_r(y),target_l(p),labels,sample_weight)
 end function
 function ratios_rm_iv(y,p,labels,sample_weight) result(ratios)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_r(y),target_i(p),labels,sample_weight)
 end function
 function ratios_rm_im(y,p,labels,sample_weight) result(ratios)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_r(y),target_i(p),labels,sample_weight)
 end function
 function ratios_rm_rv(y,p,labels,sample_weight) result(ratios)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_r(y),target_r(p),labels,sample_weight)
 end function
 function ratios_rm_rm(y,p,labels,sample_weight) result(ratios)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_r(y),target_r(p),labels,sample_weight)
 end function
 function ratios_rm_lv(y,p,labels,sample_weight) result(ratios)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_r(y),target_l(p),labels,sample_weight)
 end function
 function ratios_rm_lm(y,p,labels,sample_weight) result(ratios)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_r(y),target_l(p),labels,sample_weight)
 end function
 function ratios_lv_iv(y,p,labels,sample_weight) result(ratios)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_l(y),target_i(p),labels,sample_weight)
 end function
 function ratios_lv_im(y,p,labels,sample_weight) result(ratios)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_l(y),target_i(p),labels,sample_weight)
 end function
 function ratios_lv_rv(y,p,labels,sample_weight) result(ratios)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_l(y),target_r(p),labels,sample_weight)
 end function
 function ratios_lv_rm(y,p,labels,sample_weight) result(ratios)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_l(y),target_r(p),labels,sample_weight)
 end function
 function ratios_lv_lv(y,p,labels,sample_weight) result(ratios)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_l(y),target_l(p),labels,sample_weight)
 end function
 function ratios_lv_lm(y,p,labels,sample_weight) result(ratios)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_l(y),target_l(p),labels,sample_weight)
 end function
 function ratios_lm_iv(y,p,labels,sample_weight) result(ratios)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_l(y),target_i(p),labels,sample_weight)
 end function
 function ratios_lm_im(y,p,labels,sample_weight) result(ratios)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_l(y),target_i(p),labels,sample_weight)
 end function
 function ratios_lm_rv(y,p,labels,sample_weight) result(ratios)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_l(y),target_r(p),labels,sample_weight)
 end function
 function ratios_lm_rm(y,p,labels,sample_weight) result(ratios)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_l(y),target_r(p),labels,sample_weight)
 end function
 function ratios_lm_lv(y,p,labels,sample_weight) result(ratios)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_l(y),target_l(p),labels,sample_weight)
 end function
 function ratios_lm_lm(y,p,labels,sample_weight) result(ratios)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp)::ratios(2)
  ratios=calculate(target_l(y),target_l(p),labels,sample_weight)
 end function
end module
