program test_sgd_csr
  use sklearn, only: dp,sgd_regressor,sgd_classifier,csr_matrix,csr_from_arrays,csr_from_dense
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(sgd_regressor) :: r,dense,again
  type(sgd_classifier) :: c
  type(csr_matrix) :: x,stored,wide
  real(dp) :: a(2,2),y(2)
  real(dp), allocatable :: prediction(:),scores(:,:)
  integer, allocatable :: labels(:)
  a=0;a(1,1)=1;a(2,2)=2;y=[2._dp,3._dp]
  x=csr_from_dense(a);stored=csr_from_dense(a,.true.)
  r%penalty='none';r%learning_rate='constant';r%eta0=.1_dp;r%max_iter=1;r%tol=-1
  call r%fit(x,y)
  call require(abs(r%intercept_-.004998_dp)<1.e-14_dp,'sparse intercept updates use 0.01 decay')
  call require(maxval(abs(r%coef_-[.2_dp,.5996_dp]))<1.e-14_dp,'analytic sparse SGD coordinates')
  prediction=r%predict(x)
  call require(maxval(abs(prediction-r%predict(a)))<1.e-14_dp,'CSR and dense query interoperability')
  call require(abs(r%score(x,y)-r%score(a,y))<1.e-14_dp,'CSR regression scoring')
  r%fit_intercept=.false.;r%penalty='l1';r%alpha=.5_dp;dense=r;again=r
  call r%fit(x,y);call dense%fit(a,y);call again%fit(stored,y)
  call require(maxval(abs(r%coef_-[.15_dp,.5_dp]))<1.e-14_dp,'L1 truncates only stored CSR features')
  call require(maxval(abs(dense%coef_-[.1_dp,.5_dp]))<1.e-14_dp,'dense L1 also visits zero-valued features')
  call require(all(again%coef_==dense%coef_),'explicit CSR zeros receive source L1 truncation')
  r%penalty='none';r%alpha=0
  wide=csr_from_arrays(3,1000000,[1,2,3,4],[1,500000,1000000],[1._dp,2._dp,3._dp])
  call r%fit(wide,[1._dp,0._dp,-1._dp])
  prediction=r%predict(wide)
  call require(r%n_features_in_==1000000.and.size(r%coef_)==1000000,'wide CSR fit retains feature metadata')
  call require(maxval(abs(prediction-[.1_dp,0._dp,-.9_dp]))<1.e-14_dp,'wide CSR fit and query avoid dense data')
  c%loss='log_loss';c%penalty='elasticnet';c%l1_ratio=.73_dp;c%alpha=.01_dp
  c%learning_rate='constant';c%eta0=.1_dp;c%max_iter=3;c%tol=-1;c%warm_start=.true.
  call c%fit(x,[0,1],coef_init=[0._dp,0._dp],intercept_init=0._dp)
  scores=c%predict_proba(x);call require(all(abs(sum(scores,dim=2)-1)<1.e-14_dp),'CSR classifier probabilities')
  call c%partial_fit(a,[0,1]);call c%partial_fit(x,[0,1]);scores=c%decision_function(x)
  labels=c%predict(a)
  call require(all(ieee_is_finite(scores)).and.all(labels==c%predict(x)),'mixed dense/CSR updates and queries')
  call require(c%score(x,[0,1])==c%score(a,[0,1]),'CSR classification scoring')
  scores=c%predict_log_proba(x);call require(all(scores<=0),'CSR log probabilities')
  print *, 'CSR SGD and mixed-format streaming: PASS'
end program
