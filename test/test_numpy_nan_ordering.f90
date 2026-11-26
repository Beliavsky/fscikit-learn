program test_numpy_nan_ordering
 use sklearn_common,only:dp,require
 use sklearn_numpy_ordering,only:numpy_argsort,numpy_argpartition
 use,intrinsic::ieee_arithmetic
 implicit none
 real(dp)::x(9),nan,inf
 integer::expected(9)=[5,3,8,2,7,6,1,4,9]
 nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
 x=[nan,1._dp,0._dp,nan,-inf,inf,1._dp,0._dp,nan]
 call require(all(numpy_argsort(x)==expected),'actual NumPy NaN argsort')
 call require(all(numpy_argpartition(x,4)==expected),'actual NumPy NaN argpartition')
 call require(ieee_is_nan(x(1)).and.x(2)==1,'NaN ordering preserves input')
 print *,'NumPy NaN sort and partition passed'
end program
