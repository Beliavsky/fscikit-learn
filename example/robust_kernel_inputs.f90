program robust_kernel_inputs
 use sklearn,only:dp,robust_scaler,robust_scale
 use sklearn_sparse,only:csr_matrix,csr_from_arrays
 implicit none
 type(robust_scaler)::model
 type(csr_matrix)::x,z
 x=csr_from_arrays(5,3,[1,3,4,5,5,5],[1,3,2,1],[1._dp,10._dp,2._dp,3._dp])
 model%with_centering=.false.;model%unit_variance=.true.;call model%fit(x)
 z=model%transform(x)
 print *, 'Fitted width:',model%n_features_in_
 print *, 'Robust scales:',model%scale_
 print *, 'Stored mapped values:',z%data
 z=robust_scale(x,with_centering=.false.,axis=1)
 print *, 'Row-scaled stored values:',z%data
end program
