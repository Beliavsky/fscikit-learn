program test_selection
  use sklearn_common
  use sklearn_model_selection
  use sklearn_feature_selection
  use sklearn_linear_model, only: linear_regression
  implicit none
  type(kfold) :: kf
  type(stratified_kfold) :: sf
  type(time_series_split) :: ts
  type(index_split), allocatable :: folds(:)
  type(variance_threshold) :: vt
  type(select_k_best) :: sk
  type(ridge_cv) :: rc
  type(linear_regression) :: ols
  integer, allocatable :: train(:),test(:)
  real(dp), allocatable :: scores(:),z(:,:)
  real(dp) :: x(20,3),y(20)
  integer :: labels(20),seen(20),i,j
  do i=1,20
    x(i,:)=[real(i,dp),cos(real(i,dp)),1._dp]; y(i)=3*i+2
    labels(i)=merge(0,1,i<=10)
  end do
  call train_test_split(20,train,test,random_state=42,stratify=labels)
  call require(size(train)==15.and.size(test)==5,'train/test split size')
  seen=0; seen(train)=seen(train)+1; seen(test)=seen(test)+1
  call require(all(seen==1),'split disjoint and exhaustive')
  folds=kf%split(20); seen=0
  do i=1,size(folds)
    seen(folds(i)%test)=seen(folds(i)%test)+1
  end do
  call require(all(seen==1),'KFold exhaustive test folds')
  folds=sf%split(labels)
  do i=1,size(folds)
    call require(count(labels(folds(i)%test)==0)==2,'stratified balance')
  end do
  ts%gap=2; folds=ts%split(20)
  do i=1,size(folds)
    call require(maxval(folds(i)%train)+2<minval(folds(i)%test),'time gap')
  end do
  scores=cross_val_score(ols,x,y,cv=5); call require(minval(scores)>0.999999_dp,'regression CV')
  rc%alphas=[0._dp,1._dp,100._dp]; call rc%fit(x,y)
  call require(rc%alpha_==0,'ridge CV selects unregularized exact model')
  call vt%fit(x); z=vt%transform(x); call require(size(z,2)==2,'variance threshold')
  sk%k=1; call sk%fit(x,y); z=sk%transform(x)
  call require(all(z(:,1)==x(:,1)),'select best regression feature')
  call sk%fit(x,labels); call require(count(sk%get_support())==1,'select classification feature')
  print *, 'model and feature selection: PASS'
end program
