program resampling
  use sklearn, only: shuffle_split,repeated_stratified_kfold,leave_p_groups_out,index_split,dp
  implicit none
  type(shuffle_split) :: holdout
  type(repeated_stratified_kfold) :: repeated
  type(leave_p_groups_out) :: leave_groups
  type(index_split), allocatable :: folds(:)
  integer :: i
  holdout%n_splits=2; holdout%test_size=.25_dp; holdout%random_state=42
  folds=holdout%split(12)
  do i=1,size(folds)
    print *, 'Random holdout ',i,': train ',folds(i)%train,' test ',folds(i)%test
  end do
  repeated%n_splits=3; repeated%n_repeats=2; repeated%random_state=42
  folds=repeated%split([4,4,4,4,9,9,9,9,-2,-2,-2,-2])
  print *, 'Repeated stratified folds: ',size(folds)
  leave_groups%n_groups=2; folds=leave_groups%split([9,9,9,-3,-3,2,2,7])
  print *, 'Exhaustive two-group holdouts: ',leave_groups%get_n_splits([9,9,9,-3,-3,2,2,7])
  print *, 'First held-out sample indices: ',folds(1)%test
end program
