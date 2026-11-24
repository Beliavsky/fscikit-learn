program test_additive_csr
 use sklearn_common,only:dp,require
 use sklearn_kernel_approximation,only:additive_chi2_sampler
 use sklearn_sparse,only:csr_matrix,csr_from_arrays
 use,intrinsic::ieee_arithmetic,only:ieee_is_nan,ieee_is_finite
 implicit none
 type(additive_chi2_sampler)::model
 type(csr_matrix)::x,z,large
 real(dp),allocatable::dense(:,:)
 integer,allocatable::before(:),after(:)
 integer::n
 x=csr_from_arrays(2,3,[1,4,5],[3,1,3,2],[0._dp,2._dp,1._dp,3._dp])
 call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
 z=model%transform(x);call z%validate()
 call require(all(z%indptr==[1,10,13]),'AdditiveChi2 CSR horizontal-stack row pointers')
 call require(all(z%indices==[3,1,3,6,4,6,9,7,9,2,5,8]),'AdditiveChi2 CSR duplicate/unsorted columns')
 call require(ieee_is_nan(z%data(4)).and.ieee_is_nan(z%data(7)),'explicit CSR zeros retain source NaN harmonics')
 call require(z%data(1)==0.and.z%data(2)==1,'AdditiveChi2 source zeroth CSR component')
 call require(model%n_features_in_==0,'stateless CSR transform does not fit feature width')
 call model%fit(x);model%sample_steps=3;z=model%transform(x)
 call require(z%n_cols==15.and.z%nnz()==20,'AdditiveChi2 live sample steps')
 model%sample_interval=0;z=model%transform(x)
 call require(any(ieee_is_nan(z%data)),'CSR zero interval keeps explicit-zero undefined harmonics')
 model=additive_chi2_sampler();model%sample_interval=1.e308_dp
 dense=model%fit_transform(reshape([1.e308_dp,0._dp],[1,2]))
 call require(.not.ieee_is_finite(dense(1,1)).and.ieee_is_nan(dense(1,3)), &
   'dense source intermediate overflow retained')
 large%n_rows=100000;large%n_cols=10000
 allocate(large%indptr(large%n_rows+1),large%indices(1),large%data(1))
 large%indptr=2;large%indptr(1)=1;large%indices=10000;large%data=1
 model=additive_chi2_sampler();z=model%fit_transform(large)
 call require(z%n_rows==100000.and.z%n_cols==30000.and.z%nnz()==3,'sparse feature map avoids 24 GB dense output')
 call random_seed(get=after);call require(all(before==after),'AdditiveChi2 preserves intrinsic RNG')
 print *, 'AdditiveChi2 CSR stacking, stateless maps, duplicates, IEEE values and sparse shapes: PASS'
end program
