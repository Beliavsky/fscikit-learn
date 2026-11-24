program sparse_tree_paths
  use sklearn, only: dp,decision_tree_classifier,random_forest_classifier,csr_matrix
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(decision_tree_classifier) :: model
  type(random_forest_classifier) :: forest
  type(csr_matrix) :: indicator
  real(dp) :: x(4,1),q(1,1)
  integer :: y(4)
  integer, allocatable :: offsets(:)
  x(:,1)=[0._dp,1._dp,2._dp,3._dp];y=[-7,-7,-7,9]
  model%min_weight_fraction_leaf=.2_dp;call model%fit(x,y,[1._dp,1._dp,1._dp,8._dp])
  indicator=model%decision_path(x)
  print *, 'CSR row pointers:',indicator%indptr
  print *, 'One-based node columns:',indicator%indices
  q=ieee_value(0._dp,ieee_quiet_nan);indicator=model%decision_path(q)
  print *, 'NaN apply leaf:',model%apply(q)
  print *, 'NaN decision path:',indicator%indices
  forest%n_estimators=3;forest%random_state=7;forest%min_weight_fraction_leaf=.2_dp
  call forest%fit(x,y);indicator=forest%decision_path(x,offsets)
  print *, 'Forest column offsets:',offsets
  print *, 'Forest path shape:',indicator%n_rows,indicator%n_cols
end program
