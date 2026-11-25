program ridge_estimator_solvers
 use sklearn,only:dp,ridge,csr_from_dense
 implicit none
 type(ridge)::model
 real(dp)::x(4,1),y(4,2)
 real(dp),allocatable::prediction(:,:)
 x(:,1)=[0._dp,1._dp,2._dp,3._dp]
 y(:,1)=2+3*x(:,1);y(:,2)=1-2*x(:,1)
 model%solver='lsqr';model%tol=1.e-10_dp
 model%alpha_per_target=[1._dp,2._dp]
 call model%fit(csr_from_dense(x),y,sample_weight=2._dp)
 prediction=model%predict_matrix(x)
 print *,'Selected solver: ',trim(model%solver_)
 print *,'Coefficients: ',model%coef_matrix_
 print *,'Intercepts: ',model%intercepts_
 print *,'Training R2: ',model%score(x,y)
end program
