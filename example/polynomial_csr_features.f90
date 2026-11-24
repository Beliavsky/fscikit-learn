program polynomial_csr_features
 use sklearn,only:dp,polynomial_count_sketch
 use sklearn_sparse,only:csr_matrix,csr_from_arrays
 implicit none
 type(polynomial_count_sketch)::model
 type(csr_matrix)::x
 real(dp)::query(1,3)
 real(dp),allocatable::z(:,:)
 x=csr_from_arrays(2,3,[1,3,4],[1,3,2],[1._dp,2._dp,-1._dp])
 model%degree=3;model%n_components=7;model%random_state=42;model%gamma=.5_dp;model%coef0=1
 z=model%fit_transform(x)
 print *, 'Feature shape:',shape(z)
 print *, 'First hash row:',model%index_hash_(1,:)
 print *, 'First CSR feature row:',z(1,:)
 query(1,:)=[.5_dp,-1._dp,2._dp];model%gamma=.25_dp;z=model%transform(query)
 print *, 'Live-gamma dense query:',z(1,:)
end program
