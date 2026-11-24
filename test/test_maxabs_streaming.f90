program test_maxabs_streaming
 use sklearn_common,only:dp,require
 use sklearn_maxabs_scaling,only:maxabs_scaler,maxabs_scale
 use sklearn_sparse,only:csr_matrix,csr_from_arrays
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_is_nan
 implicit none
 type(maxabs_scaler)::model
 type(csr_matrix)::x,z,back,big,q,original
 real(dp)::a(1,3),nan
 real(dp),allocatable::dense(:,:)
 nan=ieee_value(0._dp,ieee_quiet_nan)
 x=csr_from_arrays(2,3,[1,3,4],[1,3,1],[nan,nan,nan]);original=x
 z=model%fit_transform(x)
 call require(ieee_is_nan(model%max_abs_(1)).and.all(model%max_abs_(2:)==0),'CSR missing versus implicit zero maxima')
 call require(model%n_features_in_==3.and.model%n_samples_seen_==2,'MaxAbs fitted metadata')
 a(1,:)=[3._dp,nan,4._dp];call model%partial_fit(a)
 call require(all(ieee_is_nan(model%max_abs_(:2))).and.model%max_abs_(3)==4,'streaming source NaN propagation')
 call require(model%n_samples_seen_==3,'mixed dense/CSR sample count')
 call require(all(x%indptr==original%indptr).and.all(ieee_is_nan(x%data)),'NaN sparse input preservation')
 a(1,:)=[2._dp,-4._dp,0._dp];dense=model%fit_transform(a)
 call require(model%n_samples_seen_==1.and.all(dense(1,:)==[1._dp,-1._dp,0._dp]),'fit resets streaming NaNs')
 x=csr_from_arrays(1,2,[1,5],[2,1,1,2],[2._dp,3._dp,1._dp,0._dp]);original=x
 z=maxabs_scale(x,axis=1)
 call require(all(z%indices==[1,2]).and.all(z%data==[1._dp,.5_dp]),'row scale canonicalizes duplicate coordinates')
 call require(x%nnz()==4.and.all(x%data==original%data),'free row scale preserves raw caller storage')
 call require(all(maxabs_scale([-2._dp,0._dp,1._dp])==[-1._dp,0._dp,.5_dp]),'one-dimensional default scale')
 call require(all(maxabs_scale([-2._dp,0._dp,1._dp],axis=1)==[-1._dp,0._dp,1._dp]),'one-dimensional row scale')
 call model%fit(reshape([2._dp],[1,1]));dense=model%inverse_transform(reshape([1._dp,2._dp,3._dp],[1,3]))
 call require(all(dense(1,:)==[2._dp,4._dp,6._dp]),'inverse uses source length-one scale broadcasting')
 x=csr_from_arrays(1,1,[1,3],[1,1],[1.e308_dp,1.e308_dp]);z=model%fit_transform(x)
 call require(model%max_abs_(1)>huge(1._dp).and.all(z%data==0),'finite duplicates yield source infinite scale')
 back=model%inverse_transform(z);call require(all(ieee_is_nan(back%data)),'sparse inverse zero times infinite scale')
 big%n_rows=100000;big%n_cols=100000
 allocate(big%indptr(100001),big%indices(1),big%data(1))
 big%indptr=2;big%indptr(1)=1;big%indices=100000;big%data=2
 call model%fit(big);q=csr_from_arrays(2,100000,[1,2,2],[100000],[2._dp]);z=model%transform(q)
 call require(model%n_samples_seen_==100000.and.z%nnz()==1.and.z%data(1)==1,'large sparse fit and map')
 print *, 'MaxAbs dense/CSR streaming, NaNs, free axes, refits and sparse shapes: PASS'
end program
