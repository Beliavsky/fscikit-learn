program test_standard_streaming
 use sklearn_common,only:dp,require
 use sklearn_preprocessing,only:standard_scaler
 use sklearn_sparse,only:csr_matrix,csr_from_arrays
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_is_nan
 implicit none
 type(standard_scaler)::model,whole
 type(csr_matrix)::x,z,big
 real(dp)::a(4,2),w(4),nan
 real(dp),allocatable::mapped(:,:)
 nan=ieee_value(0._dp,ieee_quiet_nan)
 a(:,1)=[1._dp,2._dp,3._dp,4._dp];a(:,2)=2*a(:,1);w=[1._dp,2._dp,3._dp,4._dp]
 call model%fit(a,w)
 call require(maxval(abs(model%mean_-[3._dp,6._dp]))<1.e-14_dp,'weighted mean')
 call require(maxval(abs(model%var_-[1._dp,4._dp]))<1.e-14_dp,'weighted population variance')
 call require(all(model%n_samples_seen_==10).and.model%sample_count_scalar_.and.model%sample_count_real_, &
  'weighted count metadata')
 call whole%fit(a,w);call model%fit(a(:2,:),w(:2));call model%partial_fit(a(3:,:),w(3:))
 call require(maxval(abs(model%mean_-whole%mean_))<1.e-14_dp.and. &
  maxval(abs(model%var_-whole%var_))<1.e-14_dp,'incremental weighted statistics')
 call model%fit(a,.15_dp);call model%partial_fit(a)
 call require(all(model%n_samples_seen_==4).and..not.model%sample_count_real_,'source scalar count cast')
 a(:,2)=nan;call model%fit(a)
 call require(all(model%n_samples_seen_==[4._dp,0._dp]).and..not.model%sample_count_scalar_, &
  'per-feature missing counts')
 call require(ieee_is_nan(model%mean_(2)).and.ieee_is_nan(model%var_(2)),'all-missing statistics')
 model=standard_scaler();model%with_mean=.false.;model%with_std=.false.;call model%fit(a,w)
 call require(.not.allocated(model%mean_).and..not.allocated(model%var_).and..not.allocated(model%scale_), &
  'disabled statistics are absent')
 call require(all(model%n_samples_seen_==[4._dp,0._dp]),'disabled dense counts ignore weights as source')
 mapped=model%transform(a);call require(all(mapped(:,1)==a(:,1)),'disabled transform')
 x=csr_from_arrays(4,2,[1,3,4,5,5],[1,1,2,1],[1._dp,2._dp,4._dp,3._dp])
 model=standard_scaler();model%with_mean=.false.;call model%fit(x)
 call require(all(abs(model%mean_-[1.5_dp,1._dp])<1.e-14_dp),'raw duplicate sparse mean')
 call require(all(abs(model%var_-[1.25_dp,3._dp])<1.e-14_dp),'raw duplicate sparse variance')
 z=model%transform(x);call require(all(z%indices==x%indices).and.all(z%indptr==x%indptr),'CSR storage retained')
 model%with_std=.false.;z=model%transform(x)
 call require(abs(z%data(1)-1/sqrt(1.25_dp))<1.e-14_dp,'CSR queries use retained scale')
 big%n_rows=100000;big%n_cols=100000
 allocate(big%indptr(100001),big%indices(1),big%data(1))
 big%indptr=2;big%indptr(1)=1;big%indices=100000;big%data=2
 model=standard_scaler();model%with_mean=.false.;call model%fit(big)
 call require(size(model%scale_)==100000.and.model%mean_(100000)==2.e-5_dp,'large sparse fit')
 print *, 'StandardScaler weighted streaming, missing values, counts and CSR: PASS'
end program
