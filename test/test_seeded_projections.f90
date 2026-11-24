program test_seeded_projections
  use sklearn_common, only: dp,require
  use sklearn_random_projection
  implicit none
  type(gaussian_random_projection) :: gaussian
  type(sparse_random_projection) :: sparse
  real(dp) :: x(2,3)
  real(dp),allocatable :: z(:,:),back(:,:)
  integer,allocatable :: before(:),after(:)
  integer :: n
  x=reshape([1._dp,2._dp,3._dp,4._dp,5._dp,6._dp],[2,3])
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  gaussian%n_components=2;gaussian%random_state=0;z=gaussian%fit_transform(x)
  call require(maxval(abs(gaussian%components_(1,:)-[1.2473733762017727_dp,.28295387557714186_dp, &
    .6920722655660196_dp]))<1.e-15_dp,'NumPy row-major Gaussian component stream')
  call require(gaussian%n_features_in_==3,'projection fitted feature width')
  back=gaussian%inverse_transform(z)
  call require(maxval(abs(gaussian%transform(back)-z))<1.e-12_dp,'seeded Gaussian inverse coordinates')
  sparse%n_components=2;sparse%random_state=0;sparse%density=1;call sparse%fit(x)
  call require(all(sparse%components_==reshape([1._dp,1._dp,1._dp,-1._dp,1._dp,1._dp],[2,3])/sqrt(2._dp)), &
    'NumPy dense binomial signs')
  sparse%density=.2_dp;sparse%compute_inverse_components=.true.;call sparse%fit(x)
  sparse%compute_inverse_components=.false.;call sparse%fit(x)
  call require(.not.allocated(sparse%inverse_components_),'refit clears inverse cache')
  call random_seed(get=after);call require(all(before==after),'projection fits preserve intrinsic RNG')
  print *, 'Seeded Gaussian and sparse projections, inverse cache and RNG isolation: PASS'
end program
