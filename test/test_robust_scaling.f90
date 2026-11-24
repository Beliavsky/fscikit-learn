program test_robust_scaling
 use sklearn_common,only:dp,require
 use sklearn_robust_scaling,only:robust_scaler,robust_scale
 use sklearn_sparse,only:csr_matrix,csr_from_arrays
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_is_nan,ieee_is_finite
 implicit none
 type(robust_scaler)::model
 type(csr_matrix)::x,z,big,q
 real(dp)::a(4,2),nan
 real(dp),allocatable::v(:,:),values(:)
 nan=ieee_value(0._dp,ieee_quiet_nan)
 a(:,1)=[1._dp,2._dp,3._dp,4._dp];a(:,2)=0
 v=model%fit_transform(a)
 call require(all(model%center_==[2.5_dp,0._dp]).and.all(model%scale_==[1.5_dp,1._dp]),'Robust median/IQR')
 model%unit_variance=.true.;call model%fit(a)
 call require(abs(model%scale_(1)-1.5_dp/1.3489795003921634_dp)<1.e-13_dp,'normal reference unit variance')
 a(:,1)=nan;call model%fit(a)
 call require(ieee_is_nan(model%center_(1)).and.ieee_is_nan(model%scale_(1)),'Robust all-missing statistics')
 model%with_centering=.false.;model%with_scaling=.false.;call model%fit(a)
 call require(.not.allocated(model%center_).and..not.allocated(model%scale_).and.model%n_features_in_==2, &
  'disabled robust statistics are absent')
 model=robust_scaler();model%with_centering=.false.
 x=csr_from_arrays(5,2,[1,3,4,4,4,4],[1,1,2],[1._dp,3._dp,2._dp]);z=model%fit_transform(x)
 call require(all(model%scale_==1),'sparse raw-entry quantiles and implicit zeros')
 call require(all(z%indices==x%indices).and.all(z%data==x%data),'robust map preserves duplicate storage')
 model%with_centering=.true.;z=model%transform(x)
 call require(all(z%data==x%data),'sparse queries ignore live centering as source')
 model=robust_scaler();a=1.e308_dp;call model%fit(a)
 call require(all(.not.ieee_is_finite(model%center_)),'masked median source overflow')
 model=robust_scaler();model%quantile_low=50;model%quantile_high=50;model%unit_variance=.true.
 a(:,1)=[1._dp,2._dp,3._dp,4._dp];a(:,2)=0;call model%fit(a)
 call require(all(.not.ieee_is_finite(model%scale_)),'equal quantiles retain source zero-adjust behavior')
 values=robust_scale([1._dp,2._dp,3._dp],axis=1)
 call require(all(values==0),'robust vector row centering')
 ! Empty columns are handled by implicit-zero multiplicities, not dense buffers.
 big%n_rows=100000;big%n_cols=100000
 allocate(big%indptr(100001),big%indices(1),big%data(1))
 big%indptr=2;big%indptr(1)=1;big%indices=100000;big%data=2
 model=robust_scaler();model%with_centering=.false.;call model%fit(big)
 q=csr_from_arrays(2,100000,[1,2,2],[100000],[2._dp]);z=model%transform(q)
 call require(z%nnz()==1.and.z%data(1)==2.and.size(model%scale_)==100000,'large sparse robust fit/map')
 print *, 'Robust dense/CSR quantiles, nullable statistics, unit variance and sparse shapes: PASS'
end program
