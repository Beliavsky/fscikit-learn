! MurmurHash3 x86_32 (Austin Appleby, public domain), with sklearn signed output.
! Native uint32 arithmetic avoids undefined signed overflow; see THIRD_PARTY_NOTICES.md.
module sklearn_murmurhash
  use sklearn_common, only: require
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  private
  integer(int64), parameter :: mask32=4294967295_int64,mask16=65535_int64
  public :: murmurhash3_32
contains
  integer(int64) function multiply32(a,b) result(c)
    integer(int64), intent(in) :: a,b
    c=iand(iand(a,mask16)*iand(b,mask16)+ishft(iand(ishft(a,-16)*iand(b,mask16)+ &
      iand(a,mask16)*ishft(b,-16),mask16),16),mask32)
  end function
  integer(int64) function rotate32(a,b) result(c)
    integer(int64), intent(in) :: a
    integer, intent(in) :: b
    c=iand(ior(ishft(a,b),ishft(a,b-32)),mask32)
  end function
  integer(int64) function murmurhash3_32(key,seed) result(hash)
    character(*), intent(in) :: key
    integer(int64), optional, intent(in) :: seed
    integer(int64) :: h,k
    integer :: i,j,n,last
    h=0
    if(present(seed)) then
      call require(seed>=0.and.seed<=mask32,'MurmurHash seed must fit uint32')
      h=seed
    end if
    n=len(key);last=n-mod(n,4)
    do i=1,last,4
      k=0
      do j=0,3
        k=ior(k,ishft(int(iachar(key(i+j:i+j)),int64),8*j))
      end do
      k=multiply32(k,3432918353_int64);k=rotate32(k,15);k=multiply32(k,461845907_int64)
      h=ieor(h,k);h=rotate32(h,13);h=iand(5*h+3864292196_int64,mask32)
    end do
    k=0
    do j=0,n-last-1
      k=ior(k,ishft(int(iachar(key(last+j+1:last+j+1)),int64),8*j))
    end do
    if(n>last) then
      k=multiply32(k,3432918353_int64);k=rotate32(k,15);k=multiply32(k,461845907_int64);h=ieor(h,k)
    end if
    h=ieor(h,int(n,int64));h=ieor(h,ishft(h,-16));h=multiply32(h,2246822507_int64)
    h=ieor(h,ishft(h,-13));h=multiply32(h,3266489909_int64);h=ieor(h,ishft(h,-16))
    hash=h
    if(h>=2147483648_int64) hash=h-4294967296_int64
  end function
end module
