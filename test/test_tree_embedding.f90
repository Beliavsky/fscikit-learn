program test_tree_embedding
  use sklearn_tree_embedding
  use sklearn_common, only: dp,require
  implicit none
  type(random_trees_embedding) :: model,copy,fresh
  real(dp) :: x(60,3),w(60)
  real(dp), allocatable :: encoded(:,:),query(:,:),prior(:,:)
  integer, allocatable :: leaves(:,:),indicator(:,:),pointer(:),samples(:,:)
  character(64), allocatable :: names(:)
  integer :: i,t,old_width
  do i=1,60
    x(i,:)=[sin(real(i,dp)),cos(.37_dp*i),real(mod(i,7),dp)]
  end do
  model%n_estimators=5; model%max_depth=3; model%random_state=7
  encoded=model%fit_transform(x); leaves=model%apply(x)
  call require(all(shape(leaves)==[60,5]),'tree embedding apply dimensions')
  call require(all(encoded==0.or.encoded==1).and.all(sum(encoded,dim=2)==5),'one leaf code per tree')
  indicator=model%decision_path(x,pointer)
  do t=1,5
    do i=1,60
      call require(indicator(i,pointer(t)+leaves(i,t))==1,'embedding path contains the selected leaf')
    end do
    call require(all(indicator(:,pointer(t)+1)==1),'embedding path contains every root')
  end do
  call require(abs(sum(model%feature_importances_)-1)<1.e-13_dp,'embedding feature importance normalization')
  names=model%get_feature_names_out()
  call require(index(names(1),'randomtreesembedding_0_')==1,'embedding source feature name convention')
  samples=model%estimators_samples()
  call require(all(samples(:,1)==[(i,i=1,60)]),'embedding uses all rows without bootstrap')
  copy=model; prior=encoded; old_width=model%n_features_out_
  model%warm_start=.true.; model%n_estimators=7; call model%fit(x)
  encoded=model%transform(x)
  call require(all(encoded(:,:old_width)==prior),'warm embedding preserves original tree codes')
  fresh%n_estimators=7; fresh%max_depth=3; fresh%random_state=7; query=fresh%fit_transform(x)
  call require(all(encoded==query),'staged embedding equals one-shot forest')
  call require(all(model%estimator_seeds_==fresh%estimator_seeds_),'warm embedding seed advance')
  call model%fit(x(:6,:)); call require(model%n_added_==0,'warm no-growth fit reuses every estimator')
  call require(model%n_features_out_<=fresh%n_features_out_,'warm no-growth encoder uses current rows')
  call require(copy%n_estimators_==5.and.copy%n_features_out_==old_width,'embedding copy owns its fitted state')
  w=1; w(::4)=0; model%warm_start=.false.; call model%fit(x,w)
  encoded=model%transform(x); call require(all(sum(encoded,dim=2)==7),'zero-weight rows still receive leaf codes')
  encoded=model%fit_transform(reshape([1._dp],[1,1]))
  call require(all(shape(encoded)==[1,7]).and.all(encoded==1),'single-sample embedding has one column per tree')
  call require(all(model%feature_importances_==0),'root-only embedding has zero importances')
  print *, 'RandomTreesEmbedding: PASS'
end program
