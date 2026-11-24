program best_first_trees
  use sklearn_common, only: dp
  use sklearn_tree, only: decision_tree_regressor
  implicit none
  type(decision_tree_regressor) :: model
  real(dp) :: x(8,1), y(8)
  integer :: i
  x(:,1)=[(real(i-1,dp),i=1,8)]
  y=[0._dp,0._dp,1._dp,1._dp,10._dp,10._dp,11._dp,11._dp]
  model%max_leaf_nodes=3
  call model%fit(x,y)
  print *, 'Leaf count:', model%get_n_leaves()
  print *, 'Predictions:', model%predict(x)
  print *, 'One-based leaves:', model%apply(x)
end program
