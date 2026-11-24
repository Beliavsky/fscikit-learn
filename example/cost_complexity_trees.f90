program cost_complexity_trees
  use sklearn, only: dp,decision_tree_classifier,tree_pruning_path
  implicit none
  type(decision_tree_classifier) :: model
  type(tree_pruning_path) :: path
  real(dp) :: x(8,1)
  integer :: y(8),i
  x(:,1)=[(real(i-1,dp),i=1,8)];y=[-7,-7,9,9,-7,9,-7,9]
  model%max_depth=5
  path=model%cost_complexity_pruning_path(x,y)
  print *, 'Pruning alphas:',path%ccp_alphas
  print *, 'Leaf risks:',path%impurities
  call model%fit(x,y)
  print *, 'Unpruned nodes:',model%node_count_
  model%ccp_alpha=.1_dp;call model%fit(x,y)
  print *, 'Pruned nodes:',model%node_count_
  print *, 'Pruned labels:',model%predict(x)
end program
