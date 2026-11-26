program test_ols_extended
 use sklearn_common,only:dp,require
 use sklearn_linear_model,only:linear_regression,ridge
 use sklearn_sparse,only:csr_matrix,csr_from_dense
 use,intrinsic::ieee_exceptions
 implicit none
 type(linear_regression)::model
 type(ridge)::penalized
 type(csr_matrix)::s
 type(csr_matrix)::large
 real(dp)::x(4,1),y(4),targets(4,2)
 real(dp),allocatable::query(:,:)
 real(dp),allocatable::large_y(:)
 integer::i
 logical::modes(size(ieee_all)),flags(size(ieee_all)),after_modes(size(ieee_all)),after_flags(size(ieee_all))
 x(:,1)=[0._dp,1._dp,2._dp,3._dp];y=2+3*x(:,1);s=csr_from_dense(x)
 call ieee_set_flag(ieee_all,.false.)
 call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
 call model%fit(x,y,sample_weight=2._dp)
 call ieee_get_halting_mode(ieee_all,after_modes);call ieee_get_flag(ieee_all,after_flags)
 call require(all(modes.eqv.after_modes).and.all(flags.eqv.after_flags),'OLS fit preserves IEEE state')
 call require(abs(model%coef_(1)-3)<1.e-12_dp.and.abs(model%intercept_-2)<1.e-12_dp,'weighted dense OLS')
 call require(model%rank_==1.and.size(model%singular_)==1,'dense rank and singular values')
 targets(:,1)=y;targets(:,2)=-y
 call model%fit(s,targets,sample_weight=[1._dp,0._dp,2._dp,1._dp])
 query=model%predict_matrix(s)
 call require(maxval(abs(query-targets))<1.e-11_dp,'CSR matrix-target fit and prediction')
 call require(.not.allocated(model%singular_).and.model%least_squares_solver_=='lsqr','sparse refit clears SVD metadata')
 call require(model%score(x,targets)>1-1.e-12_dp,'matrix-target score')
 call model%fit(s,y)
 call require(allocated(model%coef_).and..not.model%matrix_target_,'vector refit resets target rank')
 model%positive=.true.;call model%fit(x,targets)
 call require(model%coef_matrix_(1,1)>0.and.model%coef_matrix_(2,1)==0,'nonnegative multi-target fitting')
 penalized%alpha=1;penalized%solver='lsqr';penalized%tol=1.e-12_dp
 call penalized%fit(s,targets)
 call require(abs(penalized%coef_matrix_(1,1)-2.5_dp)<1.e-11_dp,'inherited Ridge overload dispatch')
 large%n_rows=50000;large%n_cols=50000
 large%indptr=[(i,i=1,50001)];large%indices=[(i,i=1,50000)];large%data=spread(1._dp,1,50000)
 allocate(large_y(50000));large_y=1
 model%positive=.false.;model%fit_intercept=.false.
 call model%fit(large,large_y,sample_weight=2._dp)
 call require(maxval(abs(model%coef_-1))<1.e-10_dp,'large CSR least squares without a dense design')
 print *,'OLS dense/CSR, weighted targets, SVD metadata, positivity and Ridge dispatch passed'
end program
