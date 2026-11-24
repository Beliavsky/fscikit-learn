program minmax_streaming
 use sklearn,only:dp,minmax_scaler,minmax_scale
 implicit none
 type(minmax_scaler)::model
 real(dp)::x(2,2),batch(1,2),query(1,2)
 real(dp),allocatable::z(:,:)
 x(1,:)=[1._dp,-2._dp];x(2,:)=[3._dp,2._dp]
 call model%partial_fit(x);batch(1,:)=[5._dp,-4._dp];call model%partial_fit(batch)
 query(1,:)=[7._dp,-8._dp];model%clip=.true.;z=model%transform(query)
 print *, 'Samples seen:',model%n_samples_seen_
 print *, 'Data range:',model%data_range_
 print *, 'Clipped query:',z(1,:)
 z=minmax_scale(x,feature_min=-1._dp,feature_max=1._dp,axis=1)
 print *, 'First row free map:',z(1,:)
end program
