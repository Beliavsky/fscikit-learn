program neighbor_imputation
  use sklearn, only: dp,knn_imputer
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(knn_imputer) :: model
  real(dp) :: x(3,2),query(1,2)
  real(dp), allocatable :: result(:,:)
  x(1,:)=[0._dp,10._dp]; x(2,:)=[2._dp,20._dp]; x(3,:)=[4._dp,40._dp]
  query(1,:)=[1._dp,ieee_value(0._dp,ieee_quiet_nan)]
  model%n_neighbors=2; call model%fit(x); result=model%transform(query)
  print *, 'Completed row:',result(1,:)
end program
