program reference_likelihood_ratios_driver
 use sklearn_likelihood_ratios,only:class_likelihood_ratios
 use sklearn_common,only:dp
 use,intrinsic::iso_fortran_env,only:int64
 implicit none
 type::holder
  integer,allocatable::i(:),ii(:,:)
  real(dp),allocatable::r(:),rr(:,:)
  logical,allocatable::l(:),ll(:,:)
 end type
 type(holder)::a,b,saved_a,saved_b
 class(*),allocatable::labels(:),saved_labels(:)
 real(dp),allocatable::weights(:),saved_weights(:)
 integer::ny,py,ky,np,pp,kp,kl,nl,weighted,nw
 integer,allocatable::ints(:)
 read(*,*) ny,py,ky,np,pp,kp,kl,nl,weighted,nw
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
 select case(ky)
 case(0);call predict_i(a%i)
 case(1);call predict_r(a%r)
 case(2);call predict_l(a%l)
 case(3);call predict_i(a%ii)
 case(4);call predict_r(a%rr)
 case(5);call predict_l(a%ll)
 end select
 print *, 'identity',merge(1,0,same(a,saved_a,ky).and.same(b,saved_b,kp).and.unchanged())
contains
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
  real(dp)::ratios(2)
  select rank(y)
  rank(1)
   select case(kp)
   case(0);ratios=class_likelihood_ratios(y,b%i,labels,weights)
   case(1);ratios=class_likelihood_ratios(y,b%r,labels,weights)
   case(2);ratios=class_likelihood_ratios(y,b%l,labels,weights)
   case(3);ratios=class_likelihood_ratios(y,b%ii,labels,weights)
   case(4);ratios=class_likelihood_ratios(y,b%rr,labels,weights)
   case(5);ratios=class_likelihood_ratios(y,b%ll,labels,weights)
   end select
  rank(2)
   select case(kp)
   case(0);ratios=class_likelihood_ratios(y,b%i,labels,weights)
   case(1);ratios=class_likelihood_ratios(y,b%r,labels,weights)
   case(2);ratios=class_likelihood_ratios(y,b%l,labels,weights)
   case(3);ratios=class_likelihood_ratios(y,b%ii,labels,weights)
   case(4);ratios=class_likelihood_ratios(y,b%rr,labels,weights)
   case(5);ratios=class_likelihood_ratios(y,b%ll,labels,weights)
   end select
  end select
  print *, 'positive',ratios(1)
  print *, 'negative',ratios(2)
 end subroutine
 subroutine predict_r(y)
  real(dp),intent(in)::y(..)
  real(dp)::ratios(2)
  select rank(y)
  rank(1)
   select case(kp)
   case(0);ratios=class_likelihood_ratios(y,b%i,labels,weights)
   case(1);ratios=class_likelihood_ratios(y,b%r,labels,weights)
   case(2);ratios=class_likelihood_ratios(y,b%l,labels,weights)
   case(3);ratios=class_likelihood_ratios(y,b%ii,labels,weights)
   case(4);ratios=class_likelihood_ratios(y,b%rr,labels,weights)
   case(5);ratios=class_likelihood_ratios(y,b%ll,labels,weights)
   end select
  rank(2)
   select case(kp)
   case(0);ratios=class_likelihood_ratios(y,b%i,labels,weights)
   case(1);ratios=class_likelihood_ratios(y,b%r,labels,weights)
   case(2);ratios=class_likelihood_ratios(y,b%l,labels,weights)
   case(3);ratios=class_likelihood_ratios(y,b%ii,labels,weights)
   case(4);ratios=class_likelihood_ratios(y,b%rr,labels,weights)
   case(5);ratios=class_likelihood_ratios(y,b%ll,labels,weights)
   end select
  end select
  print *, 'positive',ratios(1)
  print *, 'negative',ratios(2)
 end subroutine
 subroutine predict_l(y)
  logical,intent(in)::y(..)
  real(dp)::ratios(2)
  select rank(y)
  rank(1)
   select case(kp)
   case(0);ratios=class_likelihood_ratios(y,b%i,labels,weights)
   case(1);ratios=class_likelihood_ratios(y,b%r,labels,weights)
   case(2);ratios=class_likelihood_ratios(y,b%l,labels,weights)
   case(3);ratios=class_likelihood_ratios(y,b%ii,labels,weights)
   case(4);ratios=class_likelihood_ratios(y,b%rr,labels,weights)
   case(5);ratios=class_likelihood_ratios(y,b%ll,labels,weights)
   end select
  rank(2)
   select case(kp)
   case(0);ratios=class_likelihood_ratios(y,b%i,labels,weights)
   case(1);ratios=class_likelihood_ratios(y,b%r,labels,weights)
   case(2);ratios=class_likelihood_ratios(y,b%l,labels,weights)
   case(3);ratios=class_likelihood_ratios(y,b%ii,labels,weights)
   case(4);ratios=class_likelihood_ratios(y,b%rr,labels,weights)
   case(5);ratios=class_likelihood_ratios(y,b%ll,labels,weights)
   end select
  end select
  print *, 'positive',ratios(1)
  print *, 'negative',ratios(2)
 end subroutine
end program
