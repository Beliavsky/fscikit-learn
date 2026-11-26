program ols_sparse_targets
 use sklearn,only:dp,linear_regression,csr_from_dense
 implicit none
 type(linear_regression)::model
 real(dp)::x(4,1),y(4,2)
 real(dp),allocatable::prediction(:,:)
 x(:,1)=[0._dp,1._dp,2._dp,3._dp];y(:,1)=2+3*x(:,1);y(:,2)=1-2*x(:,1)
 call model%fit(csr_from_dense(x),y,sample_weight=[1._dp,0._dp,2._dp,1._dp])
 prediction=model%predict_matrix(x)
 print *,'Coefficients: ',model%coef_matrix_
 print *,'Intercepts: ',model%intercepts_
 print *,'Training R2: ',model%score(x,y)
 print *,'LSQR iterations: ',model%lsqr_iterations_
end program
