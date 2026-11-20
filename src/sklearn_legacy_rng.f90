! Scalar-seeded NumPy RandomState compatibility for dictionary atom resampling.
! Adapted from NumPy 2.2.6 MT19937, random_interval and legacy_gauss contracts.
! BSD/MIT upstream notices: LICENSES/numpy-* and THIRD_PARTY_NOTICES.md.
module sklearn_legacy_rng
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  private
  public :: legacy_random_state
  integer(int64), parameter :: mask32=int(z'FFFFFFFF',int64)
  type :: legacy_random_state
    private
    integer(int64) :: state(624)=0
    integer :: position=625
    logical :: initialized=.false.,has_normal=.false.
    real(dp) :: cached_normal=0
  contains
    procedure :: seed=>legacy_seed
    procedure :: raw_uint32=>legacy_uint32
    procedure :: uniform=>legacy_uniform
    procedure :: normal=>legacy_normal
    procedure :: integer_index=>legacy_index
  end type
contains
  subroutine legacy_seed(self,seed)
    class(legacy_random_state), intent(inout) :: self
    integer, intent(in) :: seed
    integer :: i
    call require(seed>=0,'legacy RNG seed must be nonnegative')
    self%state(1)=int(seed,int64)
    do i=2,624
      self%state(i)=iand(1812433253_int64*ieor(self%state(i-1),ishft(self%state(i-1),-30))+i-1,mask32)
    end do
    self%position=625; self%initialized=.true.; self%has_normal=.false.; self%cached_normal=0
  end subroutine
  integer(int64) function legacy_uint32(self) result(value)
    class(legacy_random_state), intent(inout) :: self
    integer :: i
    integer(int64) :: bits
    call require(self%initialized,'legacy RNG is not seeded')
    if(self%position==625) then
      do i=1,624
        bits=ior(iand(self%state(i),int(z'80000000',int64)), &
          iand(self%state(mod(i,624)+1),int(z'7FFFFFFF',int64)))
        self%state(i)=ieor(self%state(mod(i+396,624)+1),ishft(bits,-1))
        if(btest(bits,0)) self%state(i)=ieor(self%state(i),int(z'9908B0DF',int64))
      end do
      self%position=1
    end if
    value=self%state(self%position); self%position=self%position+1
    value=ieor(value,ishft(value,-11))
    value=ieor(value,iand(ishft(value,7),int(z'9D2C5680',int64)))
    value=ieor(value,iand(ishft(value,15),int(z'EFC60000',int64)))
    value=ieor(value,ishft(value,-18))
  end function
  real(dp) function legacy_uniform(self) result(value)
    class(legacy_random_state), intent(inout) :: self
    integer(int64) :: a,b
    a=ishft(self%raw_uint32(),-5); b=ishft(self%raw_uint32(),-6)
    value=real(a*67108864_int64+b,dp)/9007199254740992._dp
  end function
  real(dp) function legacy_normal(self) result(value)
    class(legacy_random_state), intent(inout) :: self
    real(dp) :: x1,x2,radius,factor
    call require(self%initialized,'legacy RNG is not seeded')
    if(self%has_normal) then
      value=self%cached_normal; self%has_normal=.false.; self%cached_normal=0; return
    end if
    do
      x1=2*self%uniform()-1; x2=2*self%uniform()-1; radius=x1*x1+x2*x2
      if(radius>0.and.radius<1) exit
    end do
    factor=sqrt(-2*log(radius)/radius)
    self%cached_normal=factor*x1; self%has_normal=.true.; value=factor*x2
  end function
  integer function legacy_index(self,n) result(index)
    class(legacy_random_state), intent(inout) :: self
    integer, intent(in) :: n
    integer(int64) :: mask,limit,value
    integer :: shift
    call require(self%initialized,'legacy RNG is not seeded')
    call require(n>=1,'legacy RNG index bound must be positive')
    if(n==1) then
      index=1; return
    end if
    limit=int(n-1,int64); mask=limit; shift=1
    do while(shift<=16)
      mask=ior(mask,ishft(mask,-shift)); shift=2*shift
    end do
    do
      value=iand(self%raw_uint32(),mask)
      if(value<=limit) exit
    end do
    index=int(value)+1
  end function
end module
