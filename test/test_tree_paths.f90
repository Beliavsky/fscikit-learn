program test_tree_paths
  use sklearn_common, only: dp,require
  use sklearn_tree, only: decision_tree_classifier,decision_tree_regressor
  use sklearn_ensemble, only: random_forest_classifier,extra_trees_regressor
  use sklearn_sparse, only: csr_matrix
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(decision_tree_classifier) :: classifier
  type(decision_tree_regressor) :: regressor
  type(random_forest_classifier) :: forest
  type(extra_trees_regressor) :: extra
  type(csr_matrix) :: indicator
  real(dp) :: x(4,1),q(1,1)
  integer :: labels(4),leaves(1)
  integer, allocatable :: offsets(:)
  x(:,1)=[0._dp,1._dp,2._dp,3._dp];labels=[-7,-7,-7,9]
  call classifier%fit(x,labels);indicator=classifier%decision_path(x)
  call require(all(indicator%indptr==[1,3,5,7,9]),'tree path CSR row pointers')
  call require(all(indicator%indices==[1,2,1,2,1,2,1,3]),'tree path node columns')
  q=ieee_value(0._dp,ieee_quiet_nan);leaves=classifier%apply(q);indicator=classifier%decision_path(q)
  call require(leaves(1)==2.and.all(indicator%indices==[1,3]),'source NaN decision path goes right independently of apply')
  classifier%min_weight_fraction_leaf=.3_dp;call classifier%fit(x,labels,[1._dp,1._dp,1._dp,8._dp])
  call require(classifier%node_count_==1,'weighted leaf fraction blocks insufficient child mass')
  classifier%min_weight_fraction_leaf=.2_dp;call classifier%fit(x,labels,[1._dp,1._dp,1._dp,8._dp])
  call require(classifier%node_count_==3,'weighted leaf fraction accepts sufficient child mass')
  regressor%ccp_alpha=1;call regressor%fit(x,[0._dp,0._dp,2._dp,2._dp])
  indicator=regressor%decision_path(x)
  call require(indicator%n_cols==1.and.indicator%nnz()==4.and.all(indicator%indices==1),'pruned stump paths use rebuilt nodes')
  forest%n_estimators=3;forest%random_state=7;forest%min_weight_fraction_leaf=.2_dp
  call forest%fit(x,labels);indicator=forest%decision_path(x,offsets)
  call require(size(offsets)==4.and.offsets(1)==0.and.offsets(4)==indicator%n_cols,'forest zero-based column offsets')
  call require(all(indicator%data==1),'forest path data are indicators')
  extra%n_estimators=3;extra%min_weight_fraction_leaf=.5_dp;call extra%fit(x,[0._dp,0._dp,2._dp,2._dp])
  indicator=extra%decision_path(x,offsets);call indicator%validate()
  call require(indicator%n_rows==4.and.size(offsets)==4,'regression forest paths')
  print *, 'Sparse decision paths, NaN routing, weighted leaf fractions and forest offsets: PASS'
end program
