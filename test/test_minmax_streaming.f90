program test_minmax_streaming
 use sklearn_common,only:dp,require
 use sklearn_minmax_scaling,only:minmax_scaler,minmax_scale
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_is_nan,ieee_is_finite
 implicit none
 type(minmax_scaler)::model
 real(dp)::x(2,2),b(1,2),nan
 real(dp),allocatable::z(:,:),back(:,:)
 nan=ieee_value(0._dp,ieee_quiet_nan)
 x(1,:)=[1._dp,nan];x(2,:)=[3._dp,nan];call model%partial_fit(x)
 call require(model%n_samples_seen_==2.and.model%n_features_in_==2,'MinMax initial streaming metadata')
 call require(model%data_min_(1)==1.and.model%data_max_(1)==3.and.ieee_is_nan(model%scale_(2)), &
  'MinMax missing extrema')
 b(1,:)=[5._dp,2._dp];call model%partial_fit(b)
 call require(model%n_samples_seen_==3.and.model%data_range_(1)==4.and.ieee_is_nan(model%scale_(2)), &
  'MinMax streaming extrema and source NaN propagation')
 z=model%fit_transform(b);call require(all(z==0).and.model%n_samples_seen_==1,'MinMax refit clears missing state')
 model=minmax_scaler();call model%fit(reshape([0._dp,2._dp],[2,1]))
 model%feature_min=2;model%feature_max=1;model%clip=.true.
 z=model%transform(reshape([-1._dp,1._dp,3._dp],[3,1]))
 call require(all(z==1).and.model%scale_(1)==.5_dp,'live reversed clip bounds retain cached scale')
 model=minmax_scaler();call model%fit(reshape([2._dp,4._dp],[2,1]))
 back=model%inverse_transform(reshape([0._dp,.5_dp,1._dp],[1,3]))
 call require(all(back(1,:)==[2._dp,3._dp,4._dp]),'MinMax inverse broadcasts learned scalar scale')
 call model%fit(reshape([-1.e308_dp,1.e308_dp],[2,1]));z=model%transform(reshape([0._dp],[1,1]))
 call require(model%data_range_(1)>huge(1._dp).and.model%scale_(1)==0.and.z(1,1)==0,'source overflowing range')
 back=model%inverse_transform(reshape([1._dp],[1,1]))
 call require(.not.ieee_is_finite(back(1,1)).and..not.ieee_is_nan(back(1,1)),'quiet inverse infinite result')
 call require(maxval(abs(minmax_scale([-2._dp,0._dp,1._dp])-[0._dp,2._dp/3,1._dp]))<1.e-14_dp,'vector scale')
 call require(all(minmax_scale([-2._dp,0._dp,1._dp],feature_min=-2._dp,feature_max=3._dp,axis=1)==-2), &
  'vector row constants map to lower endpoint')
 print *, 'MinMax streaming, NaNs, live clipping, inverse broadcasting and free maps: PASS'
end program
