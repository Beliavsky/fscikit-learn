program test_joint_classification
 use sklearn_tree,only:decision_tree_classifier,extra_tree_classifier,tree_probability,tree_pruning_path
 use sklearn_common,only:dp,require
 implicit none
 type(decision_tree_classifier)::tree
 type(extra_tree_classifier)::extra
 type(tree_probability),allocatable::prob(:)
 type(tree_pruning_path)::path
 real(dp)::x(8,2),w(8)
 integer::y(8,3),indicator(8,2),i,k
 integer,allocatable::predicted(:,:),scalar(:)
 character(8),parameter::criteria(3)=[character(8)::'gini','entropy','log_loss']
 do i=1,8
  x(i,:)=[real(i,dp),real(mod(i,3),dp)]
  y(i,:)=[merge(-7,11,i<=4),merge(3,99,mod(i,2)==0),17]
  indicator(i,:)=[mod(i,2),merge(0,1,i<=4)]
 end do
 w=[1._dp,2._dp,1._dp,.5_dp,3._dp,1._dp,2._dp,1._dp]
 do k=1,3
  tree%criterion=criteria(k);tree%random_state=7;tree%max_depth=10
  call tree%fit(x,y,w);predicted=tree%predict_multi(x);prob=tree%predict_proba_multi(x)
  call require(tree%n_outputs_==3.and.all(tree%n_classes_==[2,2,1]),'ragged joint class metadata')
  call require(all(predicted==y),'joint CART class predictions')
  call require(all(tree%classes_multi_(1)%values==[-7,11]),'joint labels retain source sorted order')
  call require(size(prob)==3.and.size(prob(3)%values,2)==1,'ragged per-output probability matrices')
  call require(maxval(abs(prob(3)%values-1._dp))==0,'constant output probability')
  path=tree%cost_complexity_pruning_path(x,y,w)
  call require(size(path%ccp_alphas)>0.and.path%ccp_alphas(1)==0,'joint classification pruning path')
  extra%criterion=criteria(k);extra%random_state=42;extra%max_depth=10
  call extra%fit(x,y,w);predicted=extra%predict_multi(x)
  call require(all(predicted==y),'joint ExtraTree class predictions')
 end do
 call tree%fit(x,indicator,w)
 call require(abs(tree%score(x,indicator,w)-1._dp)<1.e-14_dp,'multilabel exact-match accuracy')
 call tree%fit(x,y(:,1),w);scalar=tree%predict(x);predicted=tree%predict_multi(x)
 call require(tree%n_outputs_==1.and.size(tree%classes_multi_)==1,'scalar classification refit clears joint metadata')
 call require(all(scalar==predicted(:,1)),'scalar and matrix class predictions agree')
 print *,'Joint classification criteria, ragged labels/probabilities, pruning, scores and scalar refit passed'
end program
