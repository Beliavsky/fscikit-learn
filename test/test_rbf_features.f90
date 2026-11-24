program test_rbf_features
  use sklearn_common,only: dp,require
  use sklearn_kernel_approximation,only: rbf_sampler
  use sklearn_sparse,only: csr_matrix,csr_from_arrays
  use,intrinsic::ieee_arithmetic,only: ieee_is_nan
  implicit none
  type(rbf_sampler)::dense,sparse
  type(csr_matrix)::x,large,q
  real(dp),allocatable::z(:,:),expected(:,:),base(:,:)
  integer,allocatable::before(:),after(:)
  integer::n
  x=csr_from_arrays(3,3,[1,4,4,6],[3,1,3,2,1],[2._dp,1._dp,-1._dp,3._dp,0._dp])
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  dense%n_components=2;dense%gamma=.5_dp;sparse=dense
  z=dense%fit_transform(x%to_dense());expected=sparse%fit_transform(x)
  call require(maxval(abs(dense%random_weights_(1,:)-[1.764052345967664_dp,.4001572083672233_dp])) &
    <1.e-15_dp,'RBF NumPy row-major normal stream')
  call require(all(dense%random_weights_==sparse%random_weights_),'dense/CSR RBF seeded weights')
  call require(maxval(abs(z-expected))<1.e-14_dp,'CSR RBF empty rows and duplicates')
  sparse%n_components=8;expected=sparse%transform(x)
  call require(maxval(abs(expected-z/2))<1.e-14_dp,'RBF source live component normalization')
  sparse%n_components=2;sparse%gamma_scale=.true.;call sparse%fit(x)
  dense%gamma_scale=.true.;call dense%fit(x%to_dense())
  call require(abs(sparse%gamma_-dense%gamma_)<1.e-14_dp,'dense and CSR scaled RBF variance')
  large%n_rows=100000;large%n_cols=10000
  allocate(large%indptr(large%n_rows+1),large%indices(0),large%data(0));large%indptr=1
  call sparse%fit(large);call require(sparse%gamma_==1,'large CSR zero variance fallback')
  q=csr_from_arrays(2,10000,[1,2,2],[10000],[1._dp]);expected=sparse%transform(q)
  call require(all(shape(expected)==[2,2]),'large CSR query shape without dense input copy')
  dense=rbf_sampler();dense%n_components=5;dense%gamma=.7_dp
  z=dense%fit_transform(spread(spread(1.e154_dp,1,7),2,3))
  call require(maxval(abs(z(:,2)-0.2848542851068746_dp))<1.e-14_dp,'RBF source fused large finite phases')
  dense=rbf_sampler();dense%n_components=2;dense%gamma=1.e308_dp
  z=dense%fit_transform(reshape([1._dp,2._dp,3._dp,4._dp],[2,2]))
  call require(all(ieee_is_nan(z)),'RBF quiet nonfinite source phases')
  call random_seed(get=after);call require(all(before==after),'RBF sampler preserves intrinsic RNG')
  print *, 'RBF source seeds, dense/CSR scale, mutation, IEEE phases and sparse shapes: PASS'
end program
