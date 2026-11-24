program test_standard_integration
 use sklearn,only:dp,standard_scaler,pipeline,frozen_estimator
 use sklearn_common,only:require
 implicit none
 type(standard_scaler),target::scaler
 type(frozen_estimator)::view
 type(pipeline)::graph
 real(dp)::x(3,2),expected(3,2)
 real(dp),allocatable::z(:,:),back(:,:)
 x(1,:)=[2._dp,-4._dp];x(2,:)=[-3._dp,1._dp];x(3,:)=0
 expected(:,1)=(x(:,1)+1._dp/3)/sqrt(38._dp/9)
 expected(:,2)=(x(:,2)+1._dp)/sqrt(14._dp/3)
 allocate(graph%steps(1));graph%steps(1)%name='maximum';allocate(graph%steps(1)%model,source=scaler)
 z=graph%fit_transform(x);call require(maxval(abs(z-expected))<1.e-14_dp,'Standard pipeline type dispatch')
 back=graph%inverse_transform(z);call require(maxval(abs(back-x))<1.e-14_dp,'Standard pipeline inverse')
 call require(.not.allocated(scaler%scale_),'pipeline preserves Standard prototype')
 call scaler%fit(x);view%estimator=>scaler
 call require(view%is_fitted().and.view%can_transform(),'frozen Standard capabilities')
 z=view%transform(x);call require(maxval(abs(z-expected))<1.e-14_dp,'frozen Standard transform delegation')
 back=view%inverse_transform(z);call require(maxval(abs(back-x))<1.e-14_dp,'frozen Standard inverse delegation')
 call require(size(view%get_feature_names_out())==2,'frozen Standard feature width')
 scaler%with_mean=.false.;scaler%with_std=.false.;call scaler%fit(x)
 call require(view%is_fitted().and.size(view%get_feature_names_out())==2,'frozen standard no-op fitted state')
 z=view%transform(x);call require(all(z==x),'frozen standard no-op delegation')
 print *, 'Standard pipeline and frozen type dispatch, inverse maps and prototypes: PASS'
end program
