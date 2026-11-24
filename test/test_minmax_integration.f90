program test_minmax_integration
 use sklearn,only:dp,minmax_scaler,pipeline,frozen_estimator
 use sklearn_common,only:require
 implicit none
 type(minmax_scaler),target::scaler
 type(frozen_estimator)::view
 type(pipeline)::graph
 real(dp)::x(3,2),expected(3,2)
 real(dp),allocatable::z(:,:),back(:,:)
 x(1,:)=[2._dp,-4._dp];x(2,:)=[-3._dp,1._dp];x(3,:)=0
 expected(:,1)=(x(:,1)+3)/5;expected(:,2)=(x(:,2)+4)/5
 allocate(graph%steps(1));graph%steps(1)%name='maximum';allocate(graph%steps(1)%model,source=scaler)
 z=graph%fit_transform(x);call require(maxval(abs(z-expected))<1.e-14_dp,'MinMax pipeline type dispatch')
 back=graph%inverse_transform(z);call require(maxval(abs(back-x))<1.e-14_dp,'MinMax pipeline inverse')
 call require(.not.allocated(scaler%scale_),'pipeline preserves MinMax prototype')
 call scaler%fit(x);view%estimator=>scaler
 call require(view%is_fitted().and.view%can_transform(),'frozen MinMax capabilities')
 z=view%transform(x);call require(maxval(abs(z-expected))<1.e-14_dp,'frozen MinMax transform delegation')
 back=view%inverse_transform(z);call require(maxval(abs(back-x))<1.e-14_dp,'frozen MinMax inverse delegation')
 call require(size(view%get_feature_names_out())==2,'frozen MinMax feature width')
 print *, 'MinMax pipeline and frozen type dispatch, inverse maps and prototypes: PASS'
end program
