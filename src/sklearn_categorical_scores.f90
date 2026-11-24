! sklearn 1.6.1 metrics/_classification.py balanced_accuracy_score,
! matthews_corrcoef and cohen_kappa_score (BSD-3-Clause).
! NumPy 2.2.6 pairwise reductions; OpenBLAS 0.3.29 SkylakeX ddot order (BSD).
! Copyright (c) 2014, The OpenBLAS Project. All rights reserved.
! Scalar numeric adapters, owned buffers and IEEE-state preservation are MIT.
module sklearn_categorical_scores
 use sklearn_common,only:dp,require
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_arithmetic
 use,intrinsic::ieee_exceptions
 implicit none
 private
 public::balanced_accuracy_score,matthews_corrcoef,cohen_kappa_score
 interface balanced_accuracy_score
  module procedure balanced_accuracy_score_iv_iv
  module procedure balanced_accuracy_score_iv_im
  module procedure balanced_accuracy_score_iv_rv
  module procedure balanced_accuracy_score_iv_rm
  module procedure balanced_accuracy_score_iv_lv
  module procedure balanced_accuracy_score_iv_lm
  module procedure balanced_accuracy_score_im_iv
  module procedure balanced_accuracy_score_im_im
  module procedure balanced_accuracy_score_im_rv
  module procedure balanced_accuracy_score_im_rm
  module procedure balanced_accuracy_score_im_lv
  module procedure balanced_accuracy_score_im_lm
  module procedure balanced_accuracy_score_rv_iv
  module procedure balanced_accuracy_score_rv_im
  module procedure balanced_accuracy_score_rv_rv
  module procedure balanced_accuracy_score_rv_rm
  module procedure balanced_accuracy_score_rv_lv
  module procedure balanced_accuracy_score_rv_lm
  module procedure balanced_accuracy_score_rm_iv
  module procedure balanced_accuracy_score_rm_im
  module procedure balanced_accuracy_score_rm_rv
  module procedure balanced_accuracy_score_rm_rm
  module procedure balanced_accuracy_score_rm_lv
  module procedure balanced_accuracy_score_rm_lm
  module procedure balanced_accuracy_score_lv_iv
  module procedure balanced_accuracy_score_lv_im
  module procedure balanced_accuracy_score_lv_rv
  module procedure balanced_accuracy_score_lv_rm
  module procedure balanced_accuracy_score_lv_lv
  module procedure balanced_accuracy_score_lv_lm
  module procedure balanced_accuracy_score_lm_iv
  module procedure balanced_accuracy_score_lm_im
  module procedure balanced_accuracy_score_lm_rv
  module procedure balanced_accuracy_score_lm_rm
  module procedure balanced_accuracy_score_lm_lv
  module procedure balanced_accuracy_score_lm_lm
 end interface
 interface matthews_corrcoef
  module procedure matthews_corrcoef_iv_iv
  module procedure matthews_corrcoef_iv_im
  module procedure matthews_corrcoef_iv_rv
  module procedure matthews_corrcoef_iv_rm
  module procedure matthews_corrcoef_iv_lv
  module procedure matthews_corrcoef_iv_lm
  module procedure matthews_corrcoef_im_iv
  module procedure matthews_corrcoef_im_im
  module procedure matthews_corrcoef_im_rv
  module procedure matthews_corrcoef_im_rm
  module procedure matthews_corrcoef_im_lv
  module procedure matthews_corrcoef_im_lm
  module procedure matthews_corrcoef_rv_iv
  module procedure matthews_corrcoef_rv_im
  module procedure matthews_corrcoef_rv_rv
  module procedure matthews_corrcoef_rv_rm
  module procedure matthews_corrcoef_rv_lv
  module procedure matthews_corrcoef_rv_lm
  module procedure matthews_corrcoef_rm_iv
  module procedure matthews_corrcoef_rm_im
  module procedure matthews_corrcoef_rm_rv
  module procedure matthews_corrcoef_rm_rm
  module procedure matthews_corrcoef_rm_lv
  module procedure matthews_corrcoef_rm_lm
  module procedure matthews_corrcoef_lv_iv
  module procedure matthews_corrcoef_lv_im
  module procedure matthews_corrcoef_lv_rv
  module procedure matthews_corrcoef_lv_rm
  module procedure matthews_corrcoef_lv_lv
  module procedure matthews_corrcoef_lv_lm
  module procedure matthews_corrcoef_lm_iv
  module procedure matthews_corrcoef_lm_im
  module procedure matthews_corrcoef_lm_rv
  module procedure matthews_corrcoef_lm_rm
  module procedure matthews_corrcoef_lm_lv
  module procedure matthews_corrcoef_lm_lm
 end interface
 interface cohen_kappa_score
  module procedure cohen_kappa_score_iv_iv
  module procedure cohen_kappa_score_iv_im
  module procedure cohen_kappa_score_iv_rv
  module procedure cohen_kappa_score_iv_rm
  module procedure cohen_kappa_score_iv_lv
  module procedure cohen_kappa_score_iv_lm
  module procedure cohen_kappa_score_im_iv
  module procedure cohen_kappa_score_im_im
  module procedure cohen_kappa_score_im_rv
  module procedure cohen_kappa_score_im_rm
  module procedure cohen_kappa_score_im_lv
  module procedure cohen_kappa_score_im_lm
  module procedure cohen_kappa_score_rv_iv
  module procedure cohen_kappa_score_rv_im
  module procedure cohen_kappa_score_rv_rv
  module procedure cohen_kappa_score_rv_rm
  module procedure cohen_kappa_score_rv_lv
  module procedure cohen_kappa_score_rv_lm
  module procedure cohen_kappa_score_rm_iv
  module procedure cohen_kappa_score_rm_im
  module procedure cohen_kappa_score_rm_rv
  module procedure cohen_kappa_score_rm_rm
  module procedure cohen_kappa_score_rm_lv
  module procedure cohen_kappa_score_rm_lm
  module procedure cohen_kappa_score_lv_iv
  module procedure cohen_kappa_score_lv_im
  module procedure cohen_kappa_score_lv_rv
  module procedure cohen_kappa_score_lv_rm
  module procedure cohen_kappa_score_lv_lv
  module procedure cohen_kappa_score_lv_lm
  module procedure cohen_kappa_score_lm_iv
  module procedure cohen_kappa_score_lm_im
  module procedure cohen_kappa_score_lm_rv
  module procedure cohen_kappa_score_lm_rm
  module procedure cohen_kappa_score_lm_lv
  module procedure cohen_kappa_score_lm_lm
 end interface
contains
 function target_i(x) result(v)
  integer,intent(in)::x(..)
  real(dp),allocatable::v(:)
  select rank(x)
  rank(1);v=real(x,dp)
  rank(2)
   call require(size(x,1)>0.and.size(x,2)==1,'categorical scores require nonempty single-column targets')
   v=real(x(:,1),dp)
  end select
 end function
 function target_r(x) result(v)
  real(dp),intent(in)::x(..)
  real(dp),allocatable::v(:)
  select rank(x)
  rank(1);v=x
  rank(2)
   call require(size(x,1)>0.and.size(x,2)==1,'categorical scores require nonempty single-column targets')
   v=x(:,1)
  end select
 end function
 function target_l(x) result(v)
  logical,intent(in)::x(..)
  real(dp),allocatable::v(:)
  select rank(x)
  rank(1);v=merge(1._dp,0._dp,x)
  rank(2)
   call require(size(x,1)>0.and.size(x,2)==1,'categorical scores require nonempty single-column targets')
   v=merge(1._dp,0._dp,x(:,1))
  end select
 end function
 subroutine check_values(x)
  real(dp),intent(in)::x(:)
  integer::i
  do i=1,size(x)
   call require(ieee_is_finite(x(i)),'categorical score targets must be finite')
   call require(x(i)>=-2._dp**63.and.x(i)<2._dp**63,'categorical score targets exceed int64 range')
   call require(real(int(x(i),int64),dp)==x(i),'categorical scores require integral classification targets')
  end do
 end subroutine
 function sort_order(x) result(order)
  real(dp),intent(in)::x(:)
  integer::order(size(x)),work(size(x)),width,left,middle,right,i,j,k,n
  n=size(x);order=[(i,i=1,n)];width=1
  do while(width<n)
   left=1
   do while(left<=n)
    middle=left+min(width,n-left+1);right=middle+min(width,n-middle+1);i=left;j=middle
    do k=left,right-1
     if(i>=middle) then
      work(k)=order(j);j=j+1
     else if(j>=right) then
      work(k)=order(i);i=i+1
     else if(x(order(i))<=x(order(j))) then
      work(k)=order(i);i=i+1
     else
      work(k)=order(j);j=j+1
     end if
    end do
    left=right
   end do
   order=work
   if(width>=n-width) exit
   width=2*width
  end do
 end function
 subroutine label_map(x,keys,positions)
  real(dp),intent(in)::x(:)
  real(dp),allocatable,intent(out)::keys(:)
  integer,allocatable,intent(out)::positions(:)
  real(dp),allocatable::values(:),work(:)
  integer,allocatable::original(:),order(:),indices(:)
  integer::i,n
  values=pack(x,.not.ieee_is_nan(x));original=pack([(i,i=1,size(x))],.not.ieee_is_nan(x))
  order=sort_order(values);allocate(work(size(values)),indices(size(values)));n=0
  do i=1,size(order)
   if(n>0) then
    if(values(order(i))==work(n)) then
     indices(n)=original(order(i));cycle
    end if
   end if
   n=n+1;work(n)=values(order(i));indices(n)=original(order(i))
  end do
  keys=work(:n);positions=indices(:n)
 end subroutine
 integer function find_key(keys,x) result(pos)
  real(dp),intent(in)::keys(:),x
  integer::left,right,middle
  left=1;right=size(keys);pos=0
  do while(left<=right)
   middle=left+(right-left)/2
   if(keys(middle)==x) then
    pos=middle;return
   else if(keys(middle)<x) then
    left=middle+1
   else
    right=middle-1
   end if
  end do
 end function
 function build_confusion(y,p,labels,sample_weight) result(cm)
  real(dp),intent(in)::y(:),p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::sample_weight(:)
  real(dp),allocatable::cm(:,:),selected(:),keys(:)
  integer,allocatable::positions(:)
  integer::n,i,j,a,b
  logical::found
  call require(size(y)==size(p),'categorical score target lengths differ')
  call check_values(y);call check_values(p)
  if(present(labels)) then
   call require(size(labels)>0,'categorical score labels must not be empty')
   select type(labels)
   type is(integer);selected=real(labels,dp)
   type is(real(dp));selected=labels
   type is(logical);selected=merge(1._dp,0._dp,labels)
   class default;call require(.false.,'unsupported categorical score label kind')
   end select
   call label_map(selected,keys,positions);n=size(selected)
   if(size(y)>0) then
    found=.false.
    do i=1,size(y)
     if(find_key(keys,y(i))>0) then
      found=.true.;exit
     end if
    end do
    call require(found,'at least one categorical score label must occur in y')
   end if
  else
   call require(size(y)<=huge(0)/2,'categorical score class discovery exceeds native indexing capacity')
   call label_map([y,p],keys,positions);selected=keys;n=size(keys);positions=[(i,i=1,n)]
  end if
  call require(int(n,int64)*n<=huge(0),'categorical score confusion matrix exceeds native indexing capacity')
  allocate(cm(n,n));cm=0
  ! Transposed storage is contiguous in sklearn's C-order row direction.
  if(size(y)==0.and.present(labels)) return
  if(present(sample_weight)) call require(size(sample_weight)==size(y),'categorical score weight length mismatch')
  if(size(y)==0) return
  do i=1,size(y)
   a=find_key(keys,y(i));b=find_key(keys,p(i))
   if(a==0.or.b==0) cycle
   a=positions(a);b=positions(b)
   if(present(sample_weight)) then
    cm(b,a)=cm(b,a)+sample_weight(i)
   else
    cm(b,a)=cm(b,a)+1
   end if
  end do
  do j=1,n
   do i=1,n
    if(ieee_is_nan(cm(i,j))) then
     cm(i,j)=0
    else if(.not.ieee_is_finite(cm(i,j))) then
     cm(i,j)=sign(huge(1._dp),cm(i,j))
    end if
   end do
  end do
 end function
 recursive real(dp) function pairwise(x) result(value)
  real(dp),intent(in)::x(:)
  real(dp)::r(8)
  integer::n,i,j,whole,split
  n=size(x);value=-0._dp
  if(n<8) then
   do i=1,n
    value=value+x(i)
   end do
  else if(n<=128) then
   r=x(:8);whole=n-mod(n,8)
   do i=9,whole,8
    do j=1,8
     r(j)=r(j)+x(i+j-1)
    end do
   end do
   value=((r(1)+r(2))+(r(3)+r(4)))+((r(5)+r(6))+(r(7)+r(8)))
   do i=whole+1,n
    value=value+x(i)
   end do
  else
   split=n/2;split=split-mod(split,8);value=pairwise(x(:split))+pairwise(x(split+1:))
  end if
 end function
 real(dp) function summed(x) result(value)
  real(dp),intent(in)::x(:)
  integer::i,last
  value=0
  do i=1,size(x),8192
   last=min(size(x),i+8191);value=value+pairwise(x(i:last))
  end do
 end function
 real(dp) function blas_dot(x,y) result(value)
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
 ! Save all flags before changing masks: Windows mode changes clear status.
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
 ! Restore masks first, then the original flags after all mode changes.
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
 real(dp) function evaluate(y,p,mode,sample_weight,adjusted,labels,weights) result(value)
  real(dp),intent(in)::y(:),p(:)
  integer,intent(in)::mode
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  real(dp),allocatable::cm(:,:),t(:),q(:),diagonal(:),per_class(:),terms(:,:),expected(:,:)
  real(dp)::total,correct,xy,xx,yy,product,chance,coefficient,w
  logical::halting(size(ieee_all)),raised(size(ieee_all)),adjust
  integer::n,i,j
  coefficient=0
  if(present(weights)) then
   select case(weights)
   case('none');coefficient=0
   case('linear');coefficient=1
   case('quadratic');coefficient=2
   case default;call require(.false.,'categorical kappa weights must be none, linear or quadratic')
   end select
  end if
  call suspend(halting,raised)
  cm=build_confusion(y,p,labels,sample_weight);n=size(cm,1)
  allocate(t(n),q(n),diagonal(n));q=0
  do i=1,n
   t(i)=summed(cm(:,i));diagonal(i)=cm(i,i)
   do j=1,n
    q(j)=q(j)+cm(j,i)
   end do
  end do
  select case(mode)
  case(1)
   per_class=diagonal/t;per_class=pack(per_class,.not.ieee_is_nan(per_class))
   value=summed(per_class)/size(per_class)
   adjust=.false.;if(present(adjusted)) adjust=adjusted
   if(adjust) then
    call require(size(per_class)>0,'adjusted balanced accuracy has no non-NaN class recalls')
    chance=1._dp/size(per_class);value=(value-chance)/(1-chance)
   end if
  case(2)
   total=summed(q);correct=summed(diagonal)
   xy=correct*total-blas_dot(t,q);xx=total**2-blas_dot(q,q);yy=total**2-blas_dot(t,t)
   product=xx*yy;value=0
   if(product/=0) value=xy/sqrt(product)
  case(3)
   total=summed(q);allocate(terms(n,n),expected(n,n))
   do i=1,n
    do j=1,n
     w=merge(1._dp,0._dp,i/=j)
     if(coefficient==1) w=real(abs(i-j),dp)
     if(coefficient==2) w=real(i-j,dp)**2
     terms(j,i)=w*cm(j,i)
     expected(j,i)=w*((q(i)*t(j))/total)
    end do
   end do
   value=1-summed(reshape(terms,[size(terms)]))/summed(reshape(expected,[size(expected)]))
  end select
  call restore(halting,raised)
 end function
 real(dp) function balanced_accuracy_score_iv_iv(y,p,sample_weight,adjusted) result(value)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_i(y),target_i(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_iv_im(y,p,sample_weight,adjusted) result(value)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_i(y),target_i(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_iv_rv(y,p,sample_weight,adjusted) result(value)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_i(y),target_r(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_iv_rm(y,p,sample_weight,adjusted) result(value)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_i(y),target_r(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_iv_lv(y,p,sample_weight,adjusted) result(value)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_i(y),target_l(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_iv_lm(y,p,sample_weight,adjusted) result(value)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_i(y),target_l(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_im_iv(y,p,sample_weight,adjusted) result(value)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_i(y),target_i(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_im_im(y,p,sample_weight,adjusted) result(value)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_i(y),target_i(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_im_rv(y,p,sample_weight,adjusted) result(value)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_i(y),target_r(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_im_rm(y,p,sample_weight,adjusted) result(value)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_i(y),target_r(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_im_lv(y,p,sample_weight,adjusted) result(value)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_i(y),target_l(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_im_lm(y,p,sample_weight,adjusted) result(value)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_i(y),target_l(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_rv_iv(y,p,sample_weight,adjusted) result(value)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_r(y),target_i(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_rv_im(y,p,sample_weight,adjusted) result(value)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_r(y),target_i(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_rv_rv(y,p,sample_weight,adjusted) result(value)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_r(y),target_r(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_rv_rm(y,p,sample_weight,adjusted) result(value)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_r(y),target_r(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_rv_lv(y,p,sample_weight,adjusted) result(value)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_r(y),target_l(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_rv_lm(y,p,sample_weight,adjusted) result(value)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_r(y),target_l(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_rm_iv(y,p,sample_weight,adjusted) result(value)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_r(y),target_i(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_rm_im(y,p,sample_weight,adjusted) result(value)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_r(y),target_i(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_rm_rv(y,p,sample_weight,adjusted) result(value)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_r(y),target_r(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_rm_rm(y,p,sample_weight,adjusted) result(value)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_r(y),target_r(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_rm_lv(y,p,sample_weight,adjusted) result(value)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_r(y),target_l(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_rm_lm(y,p,sample_weight,adjusted) result(value)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_r(y),target_l(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_lv_iv(y,p,sample_weight,adjusted) result(value)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_l(y),target_i(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_lv_im(y,p,sample_weight,adjusted) result(value)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_l(y),target_i(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_lv_rv(y,p,sample_weight,adjusted) result(value)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_l(y),target_r(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_lv_rm(y,p,sample_weight,adjusted) result(value)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_l(y),target_r(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_lv_lv(y,p,sample_weight,adjusted) result(value)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_l(y),target_l(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_lv_lm(y,p,sample_weight,adjusted) result(value)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_l(y),target_l(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_lm_iv(y,p,sample_weight,adjusted) result(value)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_l(y),target_i(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_lm_im(y,p,sample_weight,adjusted) result(value)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_l(y),target_i(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_lm_rv(y,p,sample_weight,adjusted) result(value)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_l(y),target_r(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_lm_rm(y,p,sample_weight,adjusted) result(value)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_l(y),target_r(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_lm_lv(y,p,sample_weight,adjusted) result(value)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_l(y),target_l(p),1,sample_weight,adjusted)
 end function
 real(dp) function balanced_accuracy_score_lm_lm(y,p,sample_weight,adjusted) result(value)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  logical,optional,intent(in)::adjusted
  value=evaluate(target_l(y),target_l(p),1,sample_weight,adjusted)
 end function
 real(dp) function matthews_corrcoef_iv_iv(y,p,sample_weight) result(value)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_i(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_iv_im(y,p,sample_weight) result(value)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_i(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_iv_rv(y,p,sample_weight) result(value)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_r(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_iv_rm(y,p,sample_weight) result(value)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_r(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_iv_lv(y,p,sample_weight) result(value)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_l(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_iv_lm(y,p,sample_weight) result(value)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_l(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_im_iv(y,p,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_i(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_im_im(y,p,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_i(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_im_rv(y,p,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_r(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_im_rm(y,p,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_r(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_im_lv(y,p,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_l(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_im_lm(y,p,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_i(y),target_l(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_rv_iv(y,p,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_i(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_rv_im(y,p,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_i(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_rv_rv(y,p,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_r(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_rv_rm(y,p,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_r(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_rv_lv(y,p,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_l(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_rv_lm(y,p,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_l(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_rm_iv(y,p,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_i(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_rm_im(y,p,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_i(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_rm_rv(y,p,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_r(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_rm_rm(y,p,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_r(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_rm_lv(y,p,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_l(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_rm_lm(y,p,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_r(y),target_l(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_lv_iv(y,p,sample_weight) result(value)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_i(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_lv_im(y,p,sample_weight) result(value)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_i(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_lv_rv(y,p,sample_weight) result(value)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_r(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_lv_rm(y,p,sample_weight) result(value)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_r(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_lv_lv(y,p,sample_weight) result(value)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_l(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_lv_lm(y,p,sample_weight) result(value)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_l(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_lm_iv(y,p,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_i(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_lm_im(y,p,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_i(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_lm_rv(y,p,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_r(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_lm_rm(y,p,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_r(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_lm_lv(y,p,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_l(p),2,sample_weight)
 end function
 real(dp) function matthews_corrcoef_lm_lm(y,p,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  value=evaluate(target_l(y),target_l(p),2,sample_weight)
 end function
 real(dp) function cohen_kappa_score_iv_iv(y,p,labels,weights,sample_weight) result(value)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_i(y),target_i(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_iv_im(y,p,labels,weights,sample_weight) result(value)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_i(y),target_i(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_iv_rv(y,p,labels,weights,sample_weight) result(value)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_i(y),target_r(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_iv_rm(y,p,labels,weights,sample_weight) result(value)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_i(y),target_r(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_iv_lv(y,p,labels,weights,sample_weight) result(value)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_i(y),target_l(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_iv_lm(y,p,labels,weights,sample_weight) result(value)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_i(y),target_l(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_im_iv(y,p,labels,weights,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_i(y),target_i(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_im_im(y,p,labels,weights,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_i(y),target_i(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_im_rv(y,p,labels,weights,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_i(y),target_r(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_im_rm(y,p,labels,weights,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_i(y),target_r(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_im_lv(y,p,labels,weights,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_i(y),target_l(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_im_lm(y,p,labels,weights,sample_weight) result(value)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_i(y),target_l(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_rv_iv(y,p,labels,weights,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_r(y),target_i(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_rv_im(y,p,labels,weights,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_r(y),target_i(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_rv_rv(y,p,labels,weights,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_r(y),target_r(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_rv_rm(y,p,labels,weights,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_r(y),target_r(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_rv_lv(y,p,labels,weights,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_r(y),target_l(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_rv_lm(y,p,labels,weights,sample_weight) result(value)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_r(y),target_l(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_rm_iv(y,p,labels,weights,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_r(y),target_i(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_rm_im(y,p,labels,weights,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_r(y),target_i(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_rm_rv(y,p,labels,weights,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_r(y),target_r(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_rm_rm(y,p,labels,weights,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_r(y),target_r(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_rm_lv(y,p,labels,weights,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_r(y),target_l(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_rm_lm(y,p,labels,weights,sample_weight) result(value)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_r(y),target_l(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_lv_iv(y,p,labels,weights,sample_weight) result(value)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_l(y),target_i(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_lv_im(y,p,labels,weights,sample_weight) result(value)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_l(y),target_i(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_lv_rv(y,p,labels,weights,sample_weight) result(value)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_l(y),target_r(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_lv_rm(y,p,labels,weights,sample_weight) result(value)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_l(y),target_r(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_lv_lv(y,p,labels,weights,sample_weight) result(value)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_l(y),target_l(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_lv_lm(y,p,labels,weights,sample_weight) result(value)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_l(y),target_l(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_lm_iv(y,p,labels,weights,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_l(y),target_i(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_lm_im(y,p,labels,weights,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_l(y),target_i(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_lm_rv(y,p,labels,weights,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_l(y),target_r(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_lm_rm(y,p,labels,weights,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_l(y),target_r(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_lm_lv(y,p,labels,weights,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_l(y),target_l(p),3,sample_weight,labels=labels,weights=weights)
 end function
 real(dp) function cohen_kappa_score_lm_lm(y,p,labels,weights,sample_weight) result(value)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  real(dp),optional,intent(in)::sample_weight(:)
  class(*),optional,intent(in)::labels(:)
  character(*),optional,intent(in)::weights
  value=evaluate(target_l(y),target_l(p),3,sample_weight,labels=labels,weights=weights)
 end function
end module
