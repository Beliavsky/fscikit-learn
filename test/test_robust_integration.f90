program test_robust_integration
 use sklearn,only:dp,robust_scaler,pipeline,frozen_estimator
 use sklearn_common,only:require
 implicit none
 type(robust_scaler),target::scaler
 type(frozen_estimator)::view
 type(pipeline)::graph
 real(dp)::x(3,2),expected(3,2)
 real(dp),allocatable::z(:,:),back(:,:)
 x(1,:)=[2._dp,-4._dp];x(2,:)=[-3._dp,1._dp];x(3,:)=0
 expected(:,1)=x(:,1)/2.5_dp;expected(:,2)=x(:,2)/2.5_dp
 allocate(graph%steps(1));graph%steps(1)%name='maximum';allocate(graph%steps(1)%model,source=scaler)
 z=graph%fit_transform(x);call require(maxval(abs(z-expected))<1.e-14_dp,'Robust pipeline type dispatch')
 back=graph%inverse_transform(z);call require(maxval(abs(back-x))<1.e-14_dp,'Robust pipeline inverse')
 call require(.not.allocated(scaler%scale_),'pipeline preserves Robust prototype')
 call scaler%fit(x);view%estimator=>scaler
 call require(view%is_fitted().and.view%can_transform(),'frozen Robust capabilities')
 z=view%transform(x);call require(maxval(abs(z-expected))<1.e-14_dp,'frozen Robust transform delegation')
 back=view%inverse_transform(z);call require(maxval(abs(back-x))<1.e-14_dp,'frozen Robust inverse delegation')
 call require(size(view%get_feature_names_out())==2,'frozen Robust feature width')
 scaler%with_centering=.false.;scaler%with_scaling=.false.;call scaler%fit(x)
 call require(view%is_fitted().and.size(view%get_feature_names_out())==2,'frozen robust no-op fitted state')
 z=view%transform(x);call require(all(z==x),'frozen robust no-op delegation')
 print *, 'Robust pipeline and frozen type dispatch, inverse maps and prototypes: PASS'
end program
