program test_stacking
  use sklearn_stacking
  use sklearn_model_selection, only: ridge_cv,index_split
  use sklearn_linear_model, only: logistic_regression,linear_regression,ridge_classifier
  use sklearn_naive_bayes, only: gaussian_nb
  use sklearn_neighbors, only: kneighbors_regressor
  use sklearn_tree, only: decision_tree_classifier
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(stacking_classifier) :: clf,copy,prefitted
  type(stacking_regressor) :: reg
  type(stacking_estimator) :: bases(3),rbases(2),pre(2)
  type(logistic_regression) :: logistic
  type(gaussian_nb) :: nb
  type(linear_regression) :: ols
  type(kneighbors_regressor) :: knn
  type(ridge_classifier) :: rc
  type(decision_tree_classifier) :: tree
  type(index_split) :: plans(2)
  type(ridge_cv) :: loo
  real(dp) :: x(40,2),y(40),w(40)
  real(dp), allocatable :: meta(:,:),full(:,:),prediction(:),prob(:,:)
  integer :: labels(40),i
  do i=1,40
    x(i,:)=[real(i,dp)/10,sin(real(i,dp))]
  end do
  labels=merge(-7,19,x(:,1)<2); y=2*x(:,1)-x(:,2); w=1; w(1)=3; w(3)=0
  logistic%tol=1.e-10_dp; logistic%max_iter=500
  bases(1)%name='logistic'; allocate(bases(1)%estimator,source=logistic)
  bases(2)%name='bayes'; allocate(bases(2)%estimator,source=nb)
  bases(3)%name='unused'; bases(3)%drop=.true.
  call clf%fit(x,labels,bases,sample_weight=w)
  meta=clf%training_meta_features_; full=clf%transform(x)
  call require(all(shape(meta)==[40,2]),'binary stacking drops first probability column')
  call require(maxval(abs(meta-full))>1.e-6_dp,'stacking final training uses out-of-fold predictions')
  call require(all(clf%active_indices_==[1,2]),'stacking preserves configured active slots')
  call require(all(clf%stack_method_=='predict_proba'),'stacking auto prefers probabilities')
  call require(clf%score(x,labels,w)>.9_dp,'weighted stacking classification accuracy')
  prob=clf%predict_proba(x)
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-14_dp,'stacking class probabilities normalize')
  call require(all(shape(clf%decision_function(x))==[40,1]),'binary stacking decision response shape')
  copy=clf; clf%passthrough=.true.; call clf%fit(x,labels,bases,rc)
  full=clf%transform(x)
  call require(all(shape(full)==[40,4]).and.all(full(:,3:)==x),'stacking appends original features')
  call require(maxval(abs(copy%predict_proba(x)-prob))<1.e-14_dp,'copied stack retains independently fitted bases')
  call require(all(copy%training_meta_features_==meta),'stacking assignment preserves OOF training design')
  rbases(1)%name='linear'; allocate(rbases(1)%estimator,source=ols)
  rbases(2)%name='neighbors'; knn%n_neighbors=3; allocate(rbases(2)%estimator,source=knn)
  reg%passthrough=.true.; call reg%fit(x,y,rbases)
  call require(reg%score(x,y)>.99_dp,'stacking default RidgeCV regression')
  select type(model=>reg%final_estimator_)
  type is(ridge_cv)
    call require(model%cv==0.and.model%alpha_>0,'stacking default final regressor uses LOO RidgeCV')
  class default
    call require(.false.,'stacking default final regressor type')
  end select
  call require(ieee_is_nan(reg%score(x(:1,:),y(:1))),'stacking singleton R2 is NaN')
  plans(1)%test=[(i,i=40,21,-1)]; plans(1)%train=[(i,i=1,20)]
  plans(2)%test=[(i,i=20,1,-1)]; plans(2)%train=[(i,i=21,40)]
  full=reg%fit_transform(x,y,rbases,splits=plans)
  call require(all(reg%cv_splits_(1)%test==plans(1)%test),'stacking preserves explicit test index order')
  call require(maxval(abs(full-reg%transform(x)))==0,'fit_transform returns full fitted-base transform')
  call logistic%fit(x,labels); call nb%fit(x,labels)
  pre(1)%name='logistic'; allocate(pre(1)%estimator,source=logistic)
  pre(2)%name='bayes'; allocate(pre(2)%estimator,source=nb)
  prefitted%prefit=.true.; prefitted%stack_method='predict'
  call prefitted%fit(x,labels,pre)
  full=prefitted%transform(x)
  call require(all(full(:,1)==real(logistic%predict(x),dp)),'prefit preserves original classifier label predictions')
  call require(all(prefitted%training_meta_features_==full),'prefit uses actual fitted training responses')
  call require(.not.allocated(prefitted%cv_splits_),'prefit does not create CV folds')
  deallocate(pre(1)%estimator,pre(2)%estimator)
  pre(1)%name='tree'; allocate(pre(1)%estimator,source=tree)
  pre(2)%name='drop'; pre(2)%drop=.true.
  prefitted%prefit=.false.; prefitted%stack_method='auto'; prefitted%cv=2
  call prefitted%fit(x,merge(0,1,[(i==1,i=1,40)]),pre,tree)
  call require(all(shape(prefitted%training_meta_features_)==[40,1]),'stacking aligns missing fold probability classes')
  loo%cv=0; call loo%fit(x,y)
  call require(loo%score(x,y)>.99_dp,'standalone RidgeCV LOO option')
  print *, 'Stacking classifier and regressor: PASS'
end program
