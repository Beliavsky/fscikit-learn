program test_linear_svr_tron
  use sklearn_common,only:dp,require
  use sklearn_linear_svm,only:linear_svr
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  implicit none
  type(linear_svr)::model
  type(csr_matrix)::sparse
  real(dp)::x(4,2),y(4),w(4)
  real(dp),allocatable::first(:),prediction(:)
  x(:,1)=[1._dp,-1._dp,0._dp,2._dp];x(:,2)=[0._dp,1._dp,2._dp,-1._dp]
  y=[1._dp,-2._dp,3._dp,.2_dp];w=[1._dp,.5_dp,0._dp,2._dp];sparse=csr_from_dense(x)
  model%loss='squared_epsilon_insensitive';model%dual='false'
  model%c=1.3_dp;model%intercept_scaling=.7_dp;model%tol=1.e-6_dp;model%max_iter=1
  call model%fit(sparse,y,w)
  call require(maxval(abs(model%coef_-[.32906608489499845_dp,-.19723006186762748_dp]))<1.e-13_dp, &
    'Source first TRON trust-region step')
  call require(abs(model%intercept_+.3173827931469114_dp)<1.e-13_dp,'Source capped synthetic intercept')
  call require(model%n_iter_==1.and..not.model%converged_,'Source capped TRON warning status')
  first=model%coef_;call model%fit(x,y,w)
  call require(all(model%coef_==first),'dense and CSR primal fits share sparse feature nodes')
  model%max_iter=1000;call model%fit(x,y,w)
  call require(maxval(abs(model%coef_-[.5335237424192859_dp,.12732622025382295_dp]))<1.e-13_dp, &
    'Source uncapped TRON optimum')
  call require(abs(model%intercept_+.4223797780517879_dp)<1.e-13_dp.and.model%n_iter_==2, &
    'Source TRON accepted-step count')
  call require(model%converged_.and..not.allocated(model%dual_variables_),'primal TRON state')
  prediction=model%predict(sparse)
  call require(maxval(abs(prediction-(matmul(x,model%coef_)+model%intercept_)))<1.e-14_dp,'TRON predictions')
  model%tol=1;call model%fit(sparse,y,w)
  call require(model%n_iter_==0.and.all(model%coef_==0).and.model%intercept_==0, &
    'Source initial relative-gradient stop for large tolerance')
  x=0;model%tol=1.e-6_dp;model%fit_intercept=.false.;call model%fit(x,y,w)
  call require(model%n_iter_==0.and.model%converged_,'stationary zero-feature TRON start')
  print *,'Source LinearSVR TRON trust-region steps, caps and stopping: PASS'
end program
