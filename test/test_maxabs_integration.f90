program test_maxabs_integration
 use sklearn,only:dp,maxabs_scaler,pipeline,frozen_estimator
 use sklearn_common,only:require
 implicit none
 type(maxabs_scaler),target::scaler
 type(frozen_estimator)::view
 type(pipeline)::graph
 real(dp)::x(3,2),expected(3,2)
 real(dp),allocatable::z(:,:),back(:,:)
 x(1,:)=[2._dp,-4._dp];x(2,:)=[-3._dp,1._dp];x(3,:)=0
 expected(:,1)=x(:,1)/3;expected(:,2)=x(:,2)/4
 allocate(graph%steps(1));graph%steps(1)%name='maximum';allocate(graph%steps(1)%model,source=scaler)
 z=graph%fit_transform(x);call require(maxval(abs(z-expected))<1.e-14_dp,'MaxAbs pipeline type dispatch')
 back=graph%inverse_transform(z);call require(maxval(abs(back-x))<1.e-14_dp,'MaxAbs pipeline inverse')
 call require(.not.allocated(scaler%scale_),'pipeline preserves MaxAbs prototype')
 call scaler%fit(x);view%estimator=>scaler
 call require(view%is_fitted().and.view%can_transform(),'frozen MaxAbs capabilities')
 z=view%transform(x);call require(maxval(abs(z-expected))<1.e-14_dp,'frozen MaxAbs transform delegation')
 back=view%inverse_transform(z);call require(maxval(abs(back-x))<1.e-14_dp,'frozen MaxAbs inverse delegation')
 call require(size(view%get_feature_names_out())==2,'frozen MaxAbs feature width')
 print *, 'MaxAbs pipeline and frozen type dispatch, inverse maps and prototypes: PASS'
end program
