program test_seeded_skewed_features
  use sklearn_common,only: dp,require
  use sklearn_kernel_approximation,only: skewed_chi2_sampler
  use,intrinsic::ieee_arithmetic,only: ieee_is_nan
  implicit none
  type(skewed_chi2_sampler)::model
  real(dp)::x(2,2)
  real(dp),allocatable::z(:,:),again(:,:)
  integer,allocatable::before(:),after(:)
  integer::n
  x=reshape([0._dp,1._dp,2._dp,3._dp],[2,2])
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  model%n_components=2;z=model%fit_transform(x)
  call require(maxval(abs(model%random_weights_(1,:)-[0.04900596005816545_dp,0.23374427833691244_dp])) &
    <1.e-14_dp,'SkewedChi2 source uniform stream and inverse CDF')
  model%n_components=8;again=model%transform(x)
  call require(maxval(abs(again-z/2))<1.e-14_dp,'SkewedChi2 source live normalization')
  model%n_components=2;call model%fit(x-100);again=model%transform(x)
  call require(maxval(abs(again-z))<1.e-14_dp,'SkewedChi2 fit validates shape without shifted-domain restriction')
  model%skewedness=1.e308_dp;again=model%transform(spread(spread(1.e308_dp,1,2),2,2))
  call require(all(ieee_is_nan(again)),'SkewedChi2 source overflowing shifts return quiet NaN')
  call random_seed(get=after);call require(all(before==after),'SkewedChi2 preserves intrinsic RNG')
  print *, 'SkewedChi2 source seeds, domain, mutation, IEEE shifts and RNG isolation: PASS'
end program
