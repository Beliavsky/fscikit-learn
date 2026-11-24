program test_best_first_trees
  use sklearn_common, only: dp,require
  use sklearn_tree, only: decision_tree_regressor,decision_tree_classifier,extra_tree_regressor
  use sklearn_ensemble, only: random_forest_regressor,extra_trees_classifier
  use sklearn_sparse, only: csr_matrix
  implicit none
  type(decision_tree_regressor) :: model
  type(decision_tree_classifier) :: classifier
  type(extra_tree_regressor) :: extra
  type(random_forest_regressor) :: forest
  type(extra_trees_classifier) :: classes
  type(csr_matrix) :: indicator
  real(dp) :: x(8,1),y(8)
  integer :: i,labels(8)
  x(:,1)=[(real(i-1,dp),i=1,8)];y=[0._dp,0._dp,1._dp,1._dp,10._dp,10._dp,11._dp,11._dp]
  model%max_leaf_nodes=3;call model%fit(x,y)
  call require(model%get_n_leaves()==3.and.model%node_count_==5,'best-first leaf budget')
  call require(model%nodes_(1)%left==2.and.model%nodes_(1)%right==3,'best-first child creation indices')
  call require(model%nodes_(2)%left==4.and.model%nodes_(2)%right==5,'equal-gain frontier expands the source left record')
  call require(all(model%predict(x)==[0._dp,0._dp,1._dp,1._dp,10.5_dp,10.5_dp,10.5_dp,10.5_dp]),'best-first predictions')
  indicator=model%decision_path(x);call indicator%validate()
  model%ccp_alpha=100;call model%fit(x,y);call require(model%node_count_==1,'best-first tree post-pruning')
  labels=[-7,-7,9,9,42,42,-7,9];classifier%max_leaf_nodes=3;call classifier%fit(x,labels)
  call require(classifier%get_n_leaves()<=3,'classification best-first budget')
  extra%max_leaf_nodes=4;extra%criterion='absolute_error';call extra%fit(x,y)
  call require(extra%get_n_leaves()<=4,'ExtraTree median best-first budget')
  forest%n_estimators=3;forest%max_leaf_nodes=3;call forest%fit(x,y)
  call require(all([(forest%estimators_(i)%get_n_leaves()<=3,i=1,3)]),'forest forwards best-first budget')
  classes%n_estimators=3;classes%max_leaf_nodes=3;call classes%fit(x,labels)
  call require(all([(classes%estimators_(i)%get_n_leaves()<=3,i=1,3)]),'ExtraTrees classifier best-first budget')
  print *, 'Best-first expansion, frontier ties, leaf budgets, paths and forest forwarding: PASS'
end program
