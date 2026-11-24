program quantile_sparse
 use sklearn,only:dp,quantile_transformer,quantile_transform
 use sklearn_sparse,only:csr_matrix,csr_from_arrays
 implicit none
 type(quantile_transformer)::model
 type(csr_matrix)::x,z,row_scaled
 x=csr_from_arrays(4,3,[1,3,4,5,5],[1,1,2,1],[1._dp,2._dp,4._dp,3._dp])
 model%n_quantiles=3;model%subsample=0;model%random_state=42
 call model%fit(x);z=model%transform(x)
 print *, 'First column quantiles:',model%quantiles_(:,1)
 print *, 'Mapped stored entries:',z%data
 row_scaled=quantile_transform(x,axis=1,n_quantiles=3,subsample=0,ignore_implicit_zeros=.true.)
 print *, 'Row-wise stored maps:',row_scaled%data
end program
