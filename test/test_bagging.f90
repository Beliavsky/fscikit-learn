program test_bagging
  use sklearn_ensemble, only: bagging_classifier,bagging_regressor
  use sklearn_common, only: dp,require
  use sklearn_tree, only: decision_tree_classifier,extra_tree_classifier,extra_tree_regressor
  use sklearn_neighbors, only: kneighbors_classifier,kneighbors_regressor
  use sklearn_linear_model, only: linear_regression,ridge_classifier,ransac_regressor
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_value,ieee_quiet_nan
  implicit none
  type(bagging_classifier) :: clf,full,copy,votes
  type(bagging_regressor) :: reg
  type(extra_tree_classifier) :: extra
  type(extra_tree_regressor) :: er
  type(kneighbors_classifier) :: knn
  type(kneighbors_regressor) :: knr
  type(linear_regression) :: ols
  type(ridge_classifier) :: rc
  type(ransac_regressor) :: ransac
  type(legacy_random_state) :: rng
  real(dp) :: x(20,3),y(20),w(20),query(1,3)
  real(dp), allocatable :: p(:,:),pred(:),before(:,:)
  integer :: labels(20),i,b,seed
  integer, allocatable :: indices(:)
  do i=1,20
    x(i,:)=[real(i,dp),sin(real(i,dp)),real(mod(i,3),dp)]
  end do
  labels=merge(-3,8,x(:,1)<10); y=2*x(:,1)-x(:,2); w=1; w(1)=3; w(2)=0
  clf%n_estimators=15; clf%oob_score=.true.; call clf%fit(x,labels,sample_weight=w)
  p=clf%predict_proba(x)
  call require(all(abs(sum(p,dim=2)-1)<1.e-14_dp),'bagging probabilities normalize')
  call require(clf%score(x,labels,w)>.9_dp,'bagging weighted classifier score')
  call require(size(clf%estimators_)==15.and.size(clf%classes_)==2,'bagging fitted metadata')
  call require(allocated(clf%oob_decision_function_),'bagging classifier OOB responses')
  do b=1,size(clf%estimators_)
    indices=clf%estimators_samples_(b)%values
    select type(tree=>clf%estimators_(b)%estimator)
    class is(decision_tree_classifier)
      call require(tree%nodes_(1)%n_samples==count([(any(indices==i).and.w(i)/=0,i=1,20)]), &
        'bagging sample multiplicities exclude zero-weight rows')
    class default
      call require(.false.,'bagging default base is a decision tree')
    end select
  end do
  copy=clf; clf%oob_score=.false.; call clf%fit(x,labels,extra)
  call require(all(abs(copy%predict_proba(x)-p)<1.e-14_dp),'bagging assignment preserves fitted ensemble')
  clf%warm_start=.true.; clf%n_estimators=18; call clf%fit(x,labels,extra)
  full%n_estimators=18; call full%fit(x,labels,extra)
  call require(all(abs(full%predict_proba(x)-clf%predict_proba(x))<1.e-14_dp),'bagging warm start matches one fit')
  call require(all(full%seeds_==clf%seeds_),'bagging warm start skips prior seed draws')
  before=clf%predict_proba(x); seed=clf%seeds_(1); call clf%fit(x,labels,extra)
  call require(all(before==clf%predict_proba(x)).and.clf%seeds_(1)==seed,'bagging unchanged warm start is a no-op')
  clf%warm_start=.false.; clf%n_estimators=2; clf%max_features=2; clf%bootstrap_features=.true.
  call clf%fit(x,labels,knn)
  call require(size(clf%estimators_features_(1)%values)==2,'bagging supports duplicated feature sampling')
  votes%n_estimators=4; call votes%fit(x,labels,rc)
  p=votes%predict_proba(x)
  call require(all(abs(sum(p,dim=2)-1)<1.e-14_dp),'bagging falls back to hard votes without probabilities')
  reg%n_estimators=12; reg%oob_score=.true.; call reg%fit(x,y,er,w)
  call require(reg%score(x,y,w)>.9_dp,'bagging weighted regression score')
  call require(allocated(reg%oob_prediction_).and.reg%oob_score_>.7_dp,'bagging regression OOB score')
  reg%oob_score=.false.; reg%bootstrap=.false.; reg%max_samples_fraction=.7_dp
  call reg%fit(x,y,ols); pred=reg%predict(x)
  call require(maxval(abs(pred-y))<1.e-10_dp,'bagging weighted least-squares adapter')
  call reg%fit(x,y,ols,w); pred=reg%predict(x)
  call require(maxval(abs(pred-y))<1.e-10_dp,'bagging OLS accepts explicit nonuniform weights')
  ols%fit_intercept=.false.; call reg%fit(x,y,ols,w); pred=reg%predict(x)
  call require(maxval(abs(pred-y))<1.e-10_dp,'bagging weighted OLS without an intercept')
  call reg%fit(x,y,knr)
  call require(size(reg%estimators_samples_(1)%values)==14,'bagging resolves fractional row count')
  call reg%fit(x,y,ransac)
  call rng%seed(reg%seeds_(1)); seed=rng%integer_index(2147483647)-1
  select type(model=>reg%estimators_(1)%estimator)
  type is(ransac_regressor)
    call require(model%random_state==seed,'bagging derives random states for stochastic linear descendants')
  end select
  reg%bootstrap=.true.; reg%n_estimators=1; reg%max_samples_fraction=0; reg%max_samples=1; reg%oob_score=.true.
  call reg%fit(x,y)
  call require(count(reg%oob_prediction_==0)==1,'bagging uncovered OOB regression rows use zero')
  call require(ieee_is_nan(reg%score(x(:1,:),y(:1))),'bagging singleton regression score is NaN')
  query=x(:1,:); query(1,1)=ieee_value(0._dp,ieee_quiet_nan); pred=reg%predict(query)
  call require(size(pred)==1,'bagging defers missing queries to the tree base')
  w=0; call copy%fit(x,labels,sample_weight=w); p=copy%predict_proba(x)
  call require(all(ieee_is_nan(p)).and.all(copy%predict(x)==minval(labels)), &
    'bagging preserves undefined zero-weight class probabilities and source argmax')
  reg%oob_score=.false.; call reg%fit(x,y,sample_weight=w); pred=reg%predict(x)
  call require(all(ieee_is_nan(pred)),'bagging preserves undefined zero-weight regression responses')
  call reg%fit(x,y,ols,w); pred=reg%predict(x)
  call require(all(pred==0),'bagging zero-weight OLS without an intercept has zero coefficients')
  print *, 'Bagging classifier and regressor: PASS'
end program
