program test_hist_boosting
  use sklearn_hist_gradient_boosting
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_nan
  implicit none
  type(hist_gradient_boosting_regressor) :: reg,copy,warm,poisson
  type(hist_gradient_boosting_classifier) :: clf,multi
  real(dp) :: x(80,2),saved(80,2),y(80),q(81,2),nan,w(80)
  real(dp), allocatable :: prediction(:),staged(:,:),proba(:,:),staged_proba(:,:,:),before(:)
  integer :: j,labels(80),classes(80)
  nan=ieee_value(0._dp,ieee_quiet_nan)
  do j=1,80
    x(j,1)=real(j,dp)/80;x(j,2)=sin(real(j,dp));y(j)=2*x(j,1)+.1_dp*x(j,2)
    labels(j)=merge(-3,7,j<=40);classes(j)=mod(j,3)-1
  end do
  x(10,2)=nan;x(40,1)=nan;saved=x;w=1;w(1:4)=0
  reg%max_iter=12;reg%min_samples_leaf=3;reg%max_leaf_nodes=7;reg%max_bins=15
  call reg%fit(x,y,w);prediction=reg%predict(x);staged=reg%staged_predict(x)
  call require(maxval(abs(staged(:,reg%n_iter_)-prediction))==0,'last staged regression equals predict')
  call require(abs(reg%score(x,y)-.8937661444894699_dp)<3.e-10_dp,'histogram regression matches sklearn fixture score')
  call require(all(ieee_is_nan(x).eqv.ieee_is_nan(saved)),'fit preserves input missing mask')
  call require(all(pack(x,.not.ieee_is_nan(x))==pack(saved,.not.ieee_is_nan(saved))),'fit preserves numeric input')
  copy=reg;before=copy%predict(x)
  warm=reg;warm%max_iter=5;warm%warm_start=.false.;call warm%fit(x,y,w)
  warm%max_iter=12;warm%warm_start=.true.;call warm%fit(x,y,w)
  prediction=warm%predict(x);call require(maxval(abs(before-prediction))<1.e-12_dp,'warm continuation equals cold fit')
  reg%max_iter=1;call reg%fit(x,2*y)
  prediction=copy%predict(x);call require(all(before==prediction),'refit preserves independent fitted copy')
  reg%max_iter=30;reg%early_stopping=.true.;reg%n_iter_no_change=3;y=2
  call reg%fit(x,y);call require(reg%n_iter_==3.and.reg%early_stopped_,'training-loss early stopping')
  poisson%loss='poisson';poisson%max_iter=5;poisson%min_samples_leaf=3
  y=1+x(:,2)**2;where(ieee_is_nan(y)) y=1
  call poisson%fit(x,y);prediction=poisson%predict(x)
  call require(all(prediction>0),'Poisson predictions use exponential inverse link')
  clf%max_iter=8;clf%min_samples_leaf=3;clf%max_leaf_nodes=7;clf%class_weight='balanced'
  call clf%fit(x,labels);proba=clf%predict_proba(x);staged_proba=clf%staged_predict_proba(x)
  call require(maxval(abs(sum(proba,dim=2)-1))<1.e-15_dp,'binary probability normalization')
  call require(maxval(abs(staged_proba(:,:,clf%n_iter_)-proba))==0,'last staged probabilities equal predict_proba')
  call require(clf%score(x,labels)>.9_dp,'histogram classifier learns binary split')
  multi%max_iter=3;multi%min_samples_leaf=2;call multi%fit(x,classes)
  proba=multi%predict_proba(x);call require(size(proba,2)==3,'one Newton tree per multiclass class')
  call require(maxval(abs(sum(proba,dim=2)-1))<1.e-15_dp,'multiclass softmax normalization')
  do j=1,81
    q(j,1)=real(j-1,dp)/80;q(j,2)=0
  end do
  reg%early_stopping=.false.;reg%max_iter=10;reg%min_samples_leaf=2;reg%monotonic_cst=[1,0]
  y=[(real(j,dp)/80,j=1,80)];call reg%fit(x,y);prediction=reg%predict(q)
  call require(all(prediction(2:)>=prediction(:80)),'positive monotonic constraint')
  reg=hist_gradient_boosting_regressor();reg%max_iter=2;reg%min_samples_leaf=huge(1)
  call reg%fit(x,y)
  call require(size(reg%predictors_(1,1)%nodes_)==1,'very large minimum leaf count safely prevents splitting')
  reg%min_samples_leaf=1;reg%loss='absolute_error';y=1.e308_dp
  call reg%fit(x(:5,:),y(:5));prediction=reg%predict(q)
  call require(all(prediction==1.e308_dp),'odd median baseline avoids unnecessary mean overflow')
  call reg%fit(x,y,w);prediction=reg%predict(q)
  call require(all(prediction==1.e308_dp),'weighted median baseline avoids unnecessary mean overflow')
  reg%loss='quantile';reg%quantile=.3_dp;call reg%fit(x,y);prediction=reg%predict(q)
  call require(all(prediction==1.e308_dp),'quantile baseline avoids unnecessary mean overflow')
  clf=hist_gradient_boosting_classifier();clf%max_iter=150;clf%min_samples_leaf=1;clf%max_leaf_nodes=3
  clf%max_bins=7;clf%early_stopping=.true.;clf%n_iter_no_change=3;clf%tol=0;labels=7;w=1.e20_dp
  call clf%fit(x,labels,w)
  call require(clf%n_iter_==150,'tiny binary losses keep improving with zero stopping tolerance')
  call require(size(clf%train_score_)==151.and.all(clf%train_score_<0),'binary loss tail avoids cancellation')
  print *, 'Histogram gradient boosting: PASS'
end program
