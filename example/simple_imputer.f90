program simple_imputer_example
 use sklearn,only:dp,simple_imputer
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan
 implicit none
 type(simple_imputer)::model
 real(dp)::x(3,3),nan
 real(dp),allocatable::z(:,:),back(:,:)
 nan=ieee_value(0._dp,ieee_quiet_nan)
 x(:,1)=[1._dp,nan,3._dp];x(:,2)=nan;x(:,3)=[4._dp,5._dp,6._dp]
 model%add_indicator=.true.;z=model%fit_transform(x);back=model%inverse_transform(z)
 print *, 'Input/output features:',model%n_features_in_,size(z,2)
 print *, 'Learned statistics:',model%statistics_
 print *, 'Second transformed sample:',z(2,:)
 print *, 'Restored second sample:',back(2,:)
end program
