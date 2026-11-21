program chained_imputation
  use sklearn, only: dp,iterative_imputer
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(iterative_imputer) :: imputer
  real(dp) :: x(8,3),missing
  real(dp), allocatable :: completed(:,:)
  integer :: j
  missing=ieee_value(0._dp,ieee_quiet_nan)
  x(:,1)=[(real(j,dp),j=1,8)]; x(:,2)=2*x(:,1); x(:,3)=3*x(:,1)
  x(2,1)=missing; x(4,2)=missing; x(6,3)=missing
  imputer%add_indicator=.true.; completed=imputer%fit_transform(x)
  print '(a,i0)', 'Chained imputation rounds: ',imputer%n_iter_
  print '(a,3f12.6)', 'Recovered values: ',completed(2,1),completed(4,2),completed(6,3)
  print '(a,i0)', 'Output columns including missing indicators: ',size(completed,2)
end program
