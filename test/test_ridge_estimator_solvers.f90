program test_ridge_estimator_solvers
 use sklearn_common,only:dp,require
 use sklearn_linear_model,only:ridge,linear_regression
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 implicit none
 type(ridge)::model
 type(linear_regression)::ols
 type(csr_matrix)::s
 real(dp)::x(4,1),y(4),expected,ym(4,2)
 real(dp),allocatable::p(:,:),score_weights(:)
 x(:,1)=[0._dp,1._dp,2._dp,3._dp];y=2+3*x(:,1);s=csr_from_dense(x)
 model%alpha=1;expected=15._dp/6
 call model%fit(x,y)
 call require(abs(model%coef_(1)-expected)<1.e-12_dp,'Ridge centered dense slope')
 call require(abs(model%intercept_-(6.5_dp-1.5_dp*expected))<1.e-12_dp,'Ridge dense intercept')
 model%solver='lsqr';model%tol=1.e-12_dp;call model%fit(s,y)
 call require(abs(model%coef_(1)-expected)<1.e-12_dp,'CSR implicit centering slope')
 call require(maxval(abs(model%predict(s)-model%predict(x)))<1.e-12_dp,'dense and CSR queries agree')
 call model%fit(x,y,sample_weight=2._dp)
 call require(abs(model%coef_(1)-30._dp/11)<1.e-12_dp,'scalar fit-weight broadcasting')
 ym(:,1)=y;ym(:,2)=-y;model%alpha_per_target=[1._dp,2._dp]
 call model%fit(s,ym,sample_weight=[1._dp,1._dp,1._dp,1._dp]);p=model%predict_matrix(x)
 call require(size(p,2)==2.and.model%n_targets_==2,'owned multi-target fit and prediction')
 call require(abs(model%coef_matrix_(1,1)-2.5_dp)<1.e-12_dp,'per-target first penalty')
 call require(abs(model%coef_matrix_(2,1)+15._dp/7)<1.e-12_dp,'per-target second penalty')
 deallocate(model%alpha_per_target);call model%fit(s,y)
 call require(allocated(model%coef_).and.model%n_targets_==1,'matrix-to-vector refit resets state')
 call ols%fit(x,y)
 call require(abs(ols%coef_(1)-3)<1.e-12_dp,'inherited generic binding preserves OLS fitting')
 print *,'Ridge dense/CSR centering, scalar weights, per-target penalties and generic bindings passed'
end program
