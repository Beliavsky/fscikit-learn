program test_quantile_sparse_seeded
 use sklearn,only:dp,quantile_transformer,quantile_transform
 use sklearn_common,only:require
 use sklearn_sparse,only:csr_matrix,csr_from_arrays
 use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_is_nan
 implicit none
 type(quantile_transformer)::model
 type(csr_matrix)::x,z,big
 real(dp)::dense(10,1),nan
 real(dp),allocatable::a(:,:)
 integer,allocatable::before(:),after(:)
 integer::i,n
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 dense(:,1)=[(real(i,dp),i=1,10)]
 model%n_quantiles=5;model%subsample=6;model%random_state=42;call model%fit(dense)
 call require(all(model%quantiles_(:,1)==[1._dp,2.25_dp,4.5_dp,7.5_dp,9._dp]),'NumPy seeded row sampling')
 call random_seed(get=after);call require(all(before==after),'quantile fit preserves global RNG')
 model%n_quantiles=0;model%subsample=-1;model%random_state=-1;a=model%transform(dense)
 call require(all(a>=0).and.all(a<=1),'query ignores fit-only controls')
 x=csr_from_arrays(4,3,[1,3,4,5,5],[1,1,2,1],[1._dp,2._dp,4._dp,3._dp])
 model=quantile_transformer();model%n_quantiles=3;model%subsample=0;call model%fit(x)
 call require(all(model%quantiles_(:,1)==[0._dp,1.5_dp,3._dp]),'raw duplicate and implicit-zero quantiles')
 call require(all(model%quantiles_(:,3)==0),'empty column quantiles')
 z=model%transform(x);call require(all(z%indices==x%indices).and.all(z%indptr==x%indptr),'raw CSR map layout')
 model%ignore_implicit_zeros=.true.;call model%fit(x)
 call require(all(model%quantiles_(:,1)==[1._dp,2._dp,3._dp]),'ignore implicit zeros')
 x=csr_from_arrays(3,1,[1,2,2,3],[1,1],[0._dp,2._dp]);model%ignore_implicit_zeros=.false.
 model%output_distribution='normal';call model%fit(x);z=model%transform(x)
 call require(z%data(1)<-5.and.z%indptr(3)==z%indptr(2),'explicit versus implicit zero normal maps')
 x=csr_from_arrays(1,5,[1,5],[5,1,3,2],[9._dp,1._dp,5._dp,3._dp])
 z=quantile_transform(x,axis=1,n_quantiles=2,subsample=2,random_state=42,ignore_implicit_zeros=.true.)
 call require(all(z%data==[1._dp,0._dp,1._dp,1._dp]),'row sampling preserves unsorted source storage')
 nan=ieee_value(0._dp,ieee_quiet_nan);dense=nan
 model=quantile_transformer();model%n_quantiles=1;model%subsample=0;call model%fit(dense)
 dense=1;a=model%transform(dense)
 call require(all(a==0).and.all(ieee_is_nan(model%quantiles_)),'singleton missing-table source interpolation')
 model=quantile_transformer();model%n_quantiles=3;model%subsample=0
 dense(:,1)=[(real(i,dp),i=1,10)];call model%fit(dense)
 x=csr_from_arrays(1,1,[1,2],[1],[-1._dp]);model%ignore_implicit_zeros=.true.;z=model%transform(x)
 call require(z%data(1)==0,'live ignore flag permits signed sparse query after dense fit')
 model%ignore_implicit_zeros=.false.;z=model%inverse_transform(x)
 call require(z%data(1)==1,'sparse inverse permits negatives with implicit zeros included')
 big%n_rows=100000;big%n_cols=100000
 allocate(big%indptr(100001),big%indices(1),big%data(1));big%indptr=2;big%indptr(1)=1
 big%indices=100000;big%data=2;model%n_quantiles=3;call model%fit(big);z=model%transform(big)
 call require(z%nnz()==1.and.z%data(1)==1.and.all(model%quantiles_(:,1)==0),'compact sparse zero multiplicities')
 print *, 'Quantile seeded dense/CSR, free axes, missing tables and sparse shapes: PASS'
end program
