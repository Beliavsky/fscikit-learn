program maxabs_streaming
 use sklearn,only:dp,maxabs_scaler,maxabs_scale
 use sklearn_sparse,only:csr_matrix,csr_from_arrays
 implicit none
 type(maxabs_scaler)::model
 type(csr_matrix)::x,z
 real(dp)::batch(1,3)
 x=csr_from_arrays(2,3,[1,3,4],[1,3,2],[2._dp,-4._dp,1._dp])
 call model%partial_fit(x);batch(1,:)=[-3._dp,2._dp,0._dp];call model%partial_fit(batch)
 z=model%transform(x)
 print *, 'Samples seen:',model%n_samples_seen_
 print *, 'Feature maxima:',model%max_abs_
 print *, 'Stored scaled values:',z%data
 z=maxabs_scale(x,axis=1)
 print *, 'Row-scaled values:',z%data
end program
