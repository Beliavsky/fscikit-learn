program additive_csr_features
 use sklearn,only:dp,additive_chi2_sampler
 use sklearn_sparse,only:csr_matrix,csr_from_arrays
 implicit none
 type(additive_chi2_sampler)::model
 type(csr_matrix)::x,z
 x=csr_from_arrays(3,3,[1,3,3,4],[1,3,2],[1._dp,2._dp,3._dp])
 z=model%fit_transform(x)
 print *, 'Feature shape:',z%n_rows,z%n_cols
 print *, 'One-based row pointers:',z%indptr
 print *, 'One-based feature columns:',z%indices
 print *, 'Feature values:',z%data
end program
