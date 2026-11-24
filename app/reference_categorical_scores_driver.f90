program reference_categorical_scores_driver
 use sklearn_categorical_scores,only:balanced_accuracy_score,matthews_corrcoef,cohen_kappa_score
 use sklearn_common,only:dp
 use,intrinsic::iso_fortran_env,only:int64
 use,intrinsic::ieee_exceptions
 implicit none
 type::holder
  integer,allocatable::i(:),ii(:,:)
  real(dp),allocatable::r(:),rr(:,:)
  logical,allocatable::l(:),ll(:,:)
 end type
 type(holder)::a,b,saved_a,saved_b
 class(*),allocatable::labels(:),saved_labels(:)
 real(dp),allocatable::weights(:),saved_weights(:)
 logical::flags_before(size(ieee_all)),modes_before(size(ieee_all)),same_state,f,m
 integer::j
 integer::ny,py,ky,np,pp,kp,kl,nl,weighted,nw,mode,adjust,kweights
 integer,allocatable::ints(:)
 interface calculate_score
  procedure calc_iv_iv
  procedure calc_iv_im
  procedure calc_iv_rv
  procedure calc_iv_rm
  procedure calc_iv_lv
  procedure calc_iv_lm
  procedure calc_im_iv
  procedure calc_im_im
  procedure calc_im_rv
  procedure calc_im_rm
  procedure calc_im_lv
  procedure calc_im_lm
  procedure calc_rv_iv
  procedure calc_rv_im
  procedure calc_rv_rv
  procedure calc_rv_rm
  procedure calc_rv_lv
  procedure calc_rv_lm
  procedure calc_rm_iv
  procedure calc_rm_im
  procedure calc_rm_rv
  procedure calc_rm_rm
  procedure calc_rm_lv
  procedure calc_rm_lm
  procedure calc_lv_iv
  procedure calc_lv_im
  procedure calc_lv_rv
  procedure calc_lv_rm
  procedure calc_lv_lv
  procedure calc_lv_lm
  procedure calc_lm_iv
  procedure calc_lm_im
  procedure calc_lm_rv
  procedure calc_lm_rm
  procedure calc_lm_lv
  procedure calc_lm_lm
 end interface
 read(*,*) ny,py,ky,np,pp,kp,kl,nl,weighted,nw,mode,adjust,kweights
 call read_data(a,ny,py,ky);call read_data(b,np,pp,kp)
 select case(kl)
 case(0)
  allocate(integer::labels(nl))
 case(1)
  allocate(real(dp)::labels(nl))
 case(2)
  allocate(logical::labels(nl))
 end select
 if(allocated(labels)) then
  select type(labels)
  type is(integer)
   read(*,*) labels
  type is(real(dp))
   read(*,*) labels
  type is(logical)
   allocate(ints(nl));read(*,*) ints;labels=ints/=0
  end select
  allocate(saved_labels,source=labels)
 end if
 if(weighted==1) then
  allocate(weights(nw));read(*,*) weights;saved_weights=weights
 end if
 saved_a=a;saved_b=b
 do j=1,size(ieee_all)
  call ieee_get_flag(ieee_all(j),flags_before(j));call ieee_get_halting_mode(ieee_all(j),modes_before(j))
 end do
 select case(ky)
 case(0);call predict_i(a%i)
 case(1);call predict_r(a%r)
 case(2);call predict_l(a%l)
 case(3);call predict_i(a%ii)
 case(4);call predict_r(a%rr)
 case(5);call predict_l(a%ll)
 end select
 print *, 'ieee_state',merge(1,0,same_state)
 print *, 'identity',merge(1,0,same(a,saved_a,ky).and.same(b,saved_b,kp).and.unchanged())
contains
 subroutine check_state()
 same_state=.true.
 do j=1,size(ieee_all)
  call ieee_get_flag(ieee_all(j),f);call ieee_get_halting_mode(ieee_all(j),m)
  same_state=same_state.and.(f.eqv.flags_before(j)).and.(m.eqv.modes_before(j))
 end do
 end subroutine
 subroutine read_data(v,n,p,k)
  type(holder),intent(out)::v
  integer,intent(in)::n,p,k
  integer,allocatable::iv(:),im(:,:)
  select case(k)
  case(0);allocate(v%i(n));read(*,*) v%i
  case(1);allocate(v%r(n));read(*,*) v%r
  case(2);allocate(iv(n));read(*,*) iv;v%l=iv/=0
  case(3);allocate(v%ii(n,p));read(*,*) v%ii
  case(4);allocate(v%rr(n,p));read(*,*) v%rr
  case(5);allocate(im(n,p));read(*,*) im;v%ll=im/=0
  end select
 end subroutine
 logical function same(v,w,k)
  type(holder),intent(in)::v,w
  integer,intent(in)::k
  same=.false.
  select case(k)
  case(0);same=all(v%i==w%i)
  case(1);same=all(transfer(v%r,[0_int64],size(v%r))==transfer(w%r,[0_int64],size(w%r)))
  case(2);same=all(v%l.eqv.w%l)
  case(3);same=all(v%ii==w%ii)
  case(4);same=all(transfer(v%rr,[0_int64],size(v%rr))==transfer(w%rr,[0_int64],size(w%rr)))
  case(5);same=all(v%ll.eqv.w%ll)
  end select
 end function
 logical function unchanged()
  unchanged=.true.
  if(allocated(weights)) unchanged=all(transfer(weights,[0_int64],size(weights))== &
   transfer(saved_weights,[0_int64],size(saved_weights)))
  if(allocated(labels)) then
   select type(labels)
   type is(integer)
    select type(saved_labels)
    type is(integer);unchanged=unchanged.and.all(labels==saved_labels)
    end select
   type is(real(dp))
    select type(saved_labels)
    type is(real(dp))
     unchanged=unchanged.and.all(transfer(labels,[0_int64],size(labels))== &
      transfer(saved_labels,[0_int64],size(saved_labels)))
    end select
   type is(logical)
    select type(saved_labels)
    type is(logical);unchanged=unchanged.and.all(labels.eqv.saved_labels)
    end select
   end select
  end if
 end function
 subroutine predict_i(y)
  integer,intent(in)::y(..)
  real(dp)::value
  select rank(y)
  rank(1)
   select case(kp)
   case(0);value=calculate_score(y,b%i,labels,weights)
   case(1);value=calculate_score(y,b%r,labels,weights)
   case(2);value=calculate_score(y,b%l,labels,weights)
   case(3);value=calculate_score(y,b%ii,labels,weights)
   case(4);value=calculate_score(y,b%rr,labels,weights)
   case(5);value=calculate_score(y,b%ll,labels,weights)
   end select
  rank(2)
   select case(kp)
   case(0);value=calculate_score(y,b%i,labels,weights)
   case(1);value=calculate_score(y,b%r,labels,weights)
   case(2);value=calculate_score(y,b%l,labels,weights)
   case(3);value=calculate_score(y,b%ii,labels,weights)
   case(4);value=calculate_score(y,b%rr,labels,weights)
   case(5);value=calculate_score(y,b%ll,labels,weights)
   end select
  end select
  print *, 'score',value
 end subroutine
 subroutine predict_r(y)
  real(dp),intent(in)::y(..)
  real(dp)::value
  select rank(y)
  rank(1)
   select case(kp)
   case(0);value=calculate_score(y,b%i,labels,weights)
   case(1);value=calculate_score(y,b%r,labels,weights)
   case(2);value=calculate_score(y,b%l,labels,weights)
   case(3);value=calculate_score(y,b%ii,labels,weights)
   case(4);value=calculate_score(y,b%rr,labels,weights)
   case(5);value=calculate_score(y,b%ll,labels,weights)
   end select
  rank(2)
   select case(kp)
   case(0);value=calculate_score(y,b%i,labels,weights)
   case(1);value=calculate_score(y,b%r,labels,weights)
   case(2);value=calculate_score(y,b%l,labels,weights)
   case(3);value=calculate_score(y,b%ii,labels,weights)
   case(4);value=calculate_score(y,b%rr,labels,weights)
   case(5);value=calculate_score(y,b%ll,labels,weights)
   end select
  end select
  print *, 'score',value
 end subroutine
 subroutine predict_l(y)
  logical,intent(in)::y(..)
  real(dp)::value
  select rank(y)
  rank(1)
   select case(kp)
   case(0);value=calculate_score(y,b%i,labels,weights)
   case(1);value=calculate_score(y,b%r,labels,weights)
   case(2);value=calculate_score(y,b%l,labels,weights)
   case(3);value=calculate_score(y,b%ii,labels,weights)
   case(4);value=calculate_score(y,b%rr,labels,weights)
   case(5);value=calculate_score(y,b%ll,labels,weights)
   end select
  rank(2)
   select case(kp)
   case(0);value=calculate_score(y,b%i,labels,weights)
   case(1);value=calculate_score(y,b%r,labels,weights)
   case(2);value=calculate_score(y,b%l,labels,weights)
   case(3);value=calculate_score(y,b%ii,labels,weights)
   case(4);value=calculate_score(y,b%rr,labels,weights)
   case(5);value=calculate_score(y,b%ll,labels,weights)
   end select
  end select
  print *, 'score',value
 end subroutine
 real(dp) function calc_iv_iv(y,p,labels,weights) result(value)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_iv_im(y,p,labels,weights) result(value)
  integer,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_iv_rv(y,p,labels,weights) result(value)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_iv_rm(y,p,labels,weights) result(value)
  integer,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_iv_lv(y,p,labels,weights) result(value)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_iv_lm(y,p,labels,weights) result(value)
  integer,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_im_iv(y,p,labels,weights) result(value)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_im_im(y,p,labels,weights) result(value)
  integer,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_im_rv(y,p,labels,weights) result(value)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_im_rm(y,p,labels,weights) result(value)
  integer,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_im_lv(y,p,labels,weights) result(value)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_im_lm(y,p,labels,weights) result(value)
  integer,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_rv_iv(y,p,labels,weights) result(value)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_rv_im(y,p,labels,weights) result(value)
  real(dp),intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_rv_rv(y,p,labels,weights) result(value)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_rv_rm(y,p,labels,weights) result(value)
  real(dp),intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_rv_lv(y,p,labels,weights) result(value)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_rv_lm(y,p,labels,weights) result(value)
  real(dp),intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_rm_iv(y,p,labels,weights) result(value)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_rm_im(y,p,labels,weights) result(value)
  real(dp),intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_rm_rv(y,p,labels,weights) result(value)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_rm_rm(y,p,labels,weights) result(value)
  real(dp),intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_rm_lv(y,p,labels,weights) result(value)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_rm_lm(y,p,labels,weights) result(value)
  real(dp),intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_lv_iv(y,p,labels,weights) result(value)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_lv_im(y,p,labels,weights) result(value)
  logical,intent(in)::y(:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_lv_rv(y,p,labels,weights) result(value)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_lv_rm(y,p,labels,weights) result(value)
  logical,intent(in)::y(:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_lv_lv(y,p,labels,weights) result(value)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_lv_lm(y,p,labels,weights) result(value)
  logical,intent(in)::y(:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_lm_iv(y,p,labels,weights) result(value)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_lm_im(y,p,labels,weights) result(value)
  logical,intent(in)::y(:,:)
  integer,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_lm_rv(y,p,labels,weights) result(value)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_lm_rm(y,p,labels,weights) result(value)
  logical,intent(in)::y(:,:)
  real(dp),intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_lm_lv(y,p,labels,weights) result(value)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
 real(dp) function calc_lm_lm(y,p,labels,weights) result(value)
  logical,intent(in)::y(:,:)
  logical,intent(in)::p(:,:)
  class(*),optional,intent(in)::labels(:)
  real(dp),optional,intent(in)::weights(:)
  select case(mode)
  case(1);value=balanced_accuracy_score(y,p,weights,adjust==1)
  case(2);value=matthews_corrcoef(y,p,weights)
  case(3)
   select case(kweights)
   case(0);value=cohen_kappa_score(y,p,labels,sample_weight=weights)
   case(1);value=cohen_kappa_score(y,p,labels,'linear',weights)
   case(2);value=cohen_kappa_score(y,p,labels,'quadratic',weights)
   end select
  end select
  call check_state()
 end function
end program
