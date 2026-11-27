! sklearn 1.6.1 tree/_classes.py max_features resolution (BSD-3-Clause).
! Private typed configuration and Fortran adapters are original MIT code.
module sklearn_tree_features
 use sklearn_common,only:dp,require
 use,intrinsic::ieee_arithmetic,only:ieee_is_finite
 implicit none
 private
 public::feature_count_control,sample_count_control
 type::sample_count_control
  private
  logical::fractional_=.false.
  real(dp)::value_=0
 contains
  procedure::reset=>sample_reset,set=>sample_set,resolve=>sample_resolve
  procedure::fractional=>sample_fractional,value=>sample_value
 end type
 type::feature_count_control
  private
  integer::mode=0
  real(dp)::fraction=1
  character(:),allocatable::name
 contains
  procedure::reset=>feature_reset
  procedure,private::set_fraction,set_name
  generic::set=>set_fraction,set_name
  procedure::resolve=>feature_resolve
 end type
contains
 subroutine sample_reset(self)
  class(sample_count_control),intent(inout)::self
  self%fractional_=.false.
 end subroutine
 subroutine sample_set(self,value)
  class(sample_count_control),intent(inout)::self
  real(dp),intent(in)::value
  self%fractional_=.true.;self%value_=value
 end subroutine
 logical function sample_fractional(self,count)result(active)
  class(sample_count_control),intent(in)::self
  integer,intent(in)::count
  active=self%fractional_.and.count==0
 end function
 real(dp)function sample_value(self)result(value)
  class(sample_count_control),intent(in)::self
  value=self%value_
 end function
 integer function sample_resolve(self,count,n,minimum,include_one)result(value)
  class(sample_count_control),intent(in)::self
  integer,intent(in)::count,n,minimum
  logical,intent(in)::include_one
  value=count
  if(.not.self%fractional(count))then
   call require(count>=minimum,'minimum sample count is below its integer bound')
   return
  end if
  call require(ieee_is_finite(self%value_),'minimum sample fraction must be finite')
  call require(self%value_>0,'minimum sample fraction must be positive')
  if(include_one)then
   call require(self%value_<=1,'minimum split fraction must be in (0,1]')
  else
   call require(self%value_<1,'minimum leaf fraction must be in (0,1)')
  end if
  value=max(minimum,ceiling(self%value_*real(n,dp)))
 end function
 subroutine feature_reset(self)
  class(feature_count_control),intent(inout)::self
  self%mode=0
 end subroutine
 subroutine set_fraction(self,value)
  class(feature_count_control),intent(inout)::self
  real(dp),intent(in)::value
  self%mode=1;self%fraction=value
 end subroutine
 subroutine set_name(self,value)
  class(feature_count_control),intent(inout)::self
  character(*),intent(in)::value
  self%mode=2;self%name=value
 end subroutine
 integer function feature_resolve(self,count,p,sqrt_default) result(nf)
  class(feature_count_control),intent(in)::self
  integer,intent(in)::count,p
  logical,intent(in)::sqrt_default
  call require(count>=0,'invalid tree/forest max_features')
  call require(p>0,'max_features needs at least one input feature')
  nf=count
  if(count>0)return
  select case(self%mode)
  case(0)
   nf=p
   if(sqrt_default)nf=max(1,int(sqrt(real(p,dp))))
  case(1)
   call require(ieee_is_finite(self%fraction),'max_features fraction must be finite')
   call require(self%fraction>0.and.self%fraction<=1,'max_features fraction must be in (0,1]')
   nf=max(1,int(self%fraction*real(p,dp)))
  case(2)
   select case(self%name)
   case('sqrt');nf=max(1,int(sqrt(real(p,dp))))
   case('log2');nf=max(1,bit_size(p)-leadz(p)-1)
   case('none');nf=p ! Fortran spelling for Python None, including classifier forests.
   case default;call require(.false.,'max_features name must be sqrt, log2 or none')
   end select
  end select
 end function
end module
