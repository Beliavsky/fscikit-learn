program test_ieee_difference
  use sklearn_common, only: dp,require
  use sklearn_ieee_math, only: ieee_difference,ieee_total
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_positive_inf,ieee_is_finite,ieee_is_nan,ieee_is_negative
  implicit none
  real(dp) :: h,inf,value
  h=huge(1._dp);inf=ieee_value(0._dp,ieee_positive_inf)
  call require(ieee_difference(h,-1.e200_dp)==h,'rounded finite result at upper real64 boundary')
  call require(ieee_difference(1.e200_dp,-h)==h,'subtraction boundary is symmetric after sign reversal')
  call require(ieee_difference(-h,1.e200_dp)==-h.and.ieee_difference(-1.e200_dp,h)==-h,'negative boundary rounding')
  call require(ieee_difference(h,-scale(1._dp,969))==h,'quarter-ulp increment remains finite')
  value=ieee_difference(h,-scale(1._dp,970))
  call require(.not.ieee_is_finite(value).and..not.ieee_is_negative(value),'half-ulp increment overflows quietly')
  call require(ieee_difference(h,-h)==inf.and.ieee_difference(-h,h)==-inf,'true overflow retains sign')
  call require(ieee_is_nan(ieee_difference(inf,inf)),'infinity subtraction produces quiet NaN')
  call require(ieee_is_negative(ieee_difference(-0._dp,0._dp)), 'negative zero subtraction preserved')
  call require(.not.ieee_is_negative(ieee_difference(-0._dp,-0._dp)), 'equal negative zeros subtract to positive zero')
  call require(ieee_total([1.e200_dp,h])==h,'finite sequential reduction at upper boundary')
  print *, 'Quiet IEEE subtraction: PASS'
end program
