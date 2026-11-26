program test_joint_trees
 use sklearn_tree,only:decision_tree_regressor,extra_tree_regressor,tree_pruning_path
 use sklearn_common,only:dp,require
 implicit none
 type(decision_tree_regressor)::tree
 type(extra_tree_regressor)::extra
 type(tree_pruning_path)::path
 real(dp)::x(8,2),y(8,2),w(8)
 real(dp),allocatable::predicted(:,:),scalar(:)
 character(16),parameter::criteria(4)=[character(16)::'squared_error','friedman_mse','absolute_error','poisson']
 integer::i,k
 do i=1,8
  x(i,:)=[real(i,dp),real(mod(i,3),dp)];y(i,:)=[real(i,dp),real(9-i,dp)]
 end do
 w=[1._dp,2._dp,1._dp,.5_dp,3._dp,1._dp,2._dp,1._dp]
 do k=1,4
  tree%criterion=criteria(k);tree%random_state=7;tree%max_depth=10
  call tree%fit(x,y,w);predicted=tree%predict_multi(x)
  call require(tree%n_outputs_==2.and.maxval(abs(predicted-y))<1.e-12_dp,'joint CART shared tree predictions')
  call require(abs(tree%score(x,y,w)-1)<1.e-12_dp,'joint CART score')
  path=tree%cost_complexity_pruning_path(x,y,w)
  call require(size(path%ccp_alphas)>0.and.path%ccp_alphas(1)==0,'joint CART pruning path')
  extra%criterion=criteria(k);extra%random_state=42;extra%max_depth=10
  call extra%fit(x,y,w);predicted=extra%predict_multi(x)
  call require(extra%n_outputs_==2.and.maxval(abs(predicted-y))<1.e-12_dp,'joint ExtraTree predictions')
 end do
 call tree%fit(x,y(:,1),w);scalar=tree%predict(x);predicted=tree%predict_multi(x)
 call require(tree%n_outputs_==1.and.size(predicted,2)==1,'scalar refit resets output width')
 call require(maxval(abs(scalar-predicted(:,1)))==0,'single-output matrix adapter agrees with scalar prediction')
 print *,'Joint CART/ExtraTree criteria, weighted predictions, scores, pruning and scalar refit passed'
end program
