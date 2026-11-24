program standard_streaming
 use sklearn,only:dp,standard_scaler
 implicit none
 type(standard_scaler)::scaler
 real(dp)::x(4,2),w(4)
 real(dp),allocatable::z(:,:)
 x(:,1)=[1._dp,2._dp,3._dp,4._dp];x(:,2)=2*x(:,1);w=[1._dp,2._dp,3._dp,4._dp]
 call scaler%partial_fit(x(:2,:),sample_weight=w(:2))
 call scaler%partial_fit(x(3:,:),sample_weight=w(3:))
 z=scaler%transform(x)
 print *, 'Weighted means:',scaler%mean_
 print *, 'Population variances:',scaler%var_
 print *, 'Per-feature weight sums:',scaler%n_samples_seen_
 print *, 'First standardized sample:',z(1,:)
end program
