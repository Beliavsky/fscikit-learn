program test_free_scale
 use sklearn,only:dp,scale,standard_scaler
 use sklearn_common,only:require
 use sklearn_sparse,only:csr_matrix,csr_from_arrays
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_is_nan
 implicit none
 type(standard_scaler)::model
 type(csr_matrix)::x,z,big
 real(dp)::a(3,2),tiny(3),nan
 real(dp),allocatable::v(:),mapped(:,:),expected(:,:)
 logical::first,second
 a(:,1)=[1._dp,2._dp,3._dp];a(:,2)=[4._dp,4._dp,4._dp]
 mapped=scale(a,centering_corrected=first,scaling_corrected=second)
 call require(maxval(abs(mapped(:,1)-[-sqrt(1.5_dp),0._dp,sqrt(1.5_dp)]))<1.e-14_dp, &
  'free scale population standard deviation')
 call require(all(mapped(:,2)==0).and..not.first.and..not.second,'constant feature and correction flags')
 expected=transpose(scale(transpose(a)));mapped=scale(a,axis=1)
 call require(maxval(abs(mapped-expected))<1.e-14_dp,'free scale row axis')
 tiny=[1.e-17_dp,2.e-17_dp,3.e-17_dp];v=scale(tiny)
 call require(maxval(abs(v-[-sqrt(1.5_dp),0._dp,sqrt(1.5_dp)]))<1.e-14_dp,'vector scalar zero handling')
 a(:,1)=tiny;a(:,2)=0;mapped=scale(a)
 call require(maxval(abs(mapped(:,1)))<2.e-17_dp,'matrix standard deviation threshold differs from vector')
 x=csr_from_arrays(3,1,[1,2,3,4],[1,1,1],tiny);z=scale(x,with_mean=.false.)
 call require(all(z%data==tiny),'sparse thresholds variance before sqrt')
 model%with_mean=.false.;call model%fit(x);z=model%transform(x)
 call require(maxval(abs(z%data))>1,'estimator threshold differs from sparse free scale')
 nan=ieee_value(0._dp,ieee_quiet_nan);a(:,1)=nan;a(:,2)=[1._dp,2._dp,3._dp]
 mapped=scale(a,centering_corrected=first,scaling_corrected=second)
 call require(all(ieee_is_nan(mapped(:,1))).and.first.and.second,'all-missing triggers global source corrections')
 call require(maxval(abs(mapped(:,2)-[-sqrt(1.5_dp),0._dp,sqrt(1.5_dp)]))<1.e-14_dp, &
  'global corrections preserve finite feature')
 x=csr_from_arrays(4,2,[1,3,4,5,5],[1,1,2,1],[1._dp,2._dp,4._dp,3._dp])
 z=scale(x,with_mean=.false.)
 call require(all(z%indices==x%indices).and.all(z%indptr==x%indptr),'sparse raw storage retained')
 call require(abs(z%data(1)-1/sqrt(1.25_dp))<1.e-14_dp,'raw duplicate statistics')
 big%n_rows=100000;big%n_cols=100000
 allocate(big%indptr(100001),big%indices(1),big%data(1))
 big%indptr=2;big%indptr(1)=1;big%indices=100000;big%data=2
 z=scale(big,with_mean=.false.)
 call require(z%nnz()==1.and.z%data(1)>300.and.z%data(1)<320,'large free sparse scale without densification')
 print *, 'Free scale dense/vector/CSR, correction flags, thresholds and sparse shapes: PASS'
end program
