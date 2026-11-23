program reference_ieee_difference_driver
  use sklearn_common, only: dp,require
  use sklearn_ieee_math, only: ieee_difference
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_negative
  implicit none
  real(dp), allocatable :: pairs(:,:),values(:,:)
  integer(int64), allocatable :: signs(:,:)
  integer :: n,i
  read(*,*) n
  call require(n>=0,'negative subtraction fixture count')
  allocate(pairs(n,2),values(n,1),signs(n,1));read(*,*) pairs
  do i=1,n
    values(i,1)=ieee_difference(pairs(i,1),pairs(i,2));signs(i,1)=0
    if(.not.ieee_is_nan(values(i,1))) signs(i,1)=merge(1_int64,0_int64,ieee_is_negative(values(i,1)))
  end do
  print *, 'F',shape(values,kind=int64)
  print '(8(es25.16e3,1x))',values
  print *, 'I',shape(signs,kind=int64)
  print '(20(i0,1x))',signs
end program
