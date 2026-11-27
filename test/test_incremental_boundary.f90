program test_incremental_boundary
  use sklearn_common,only:dp,require
  use sklearn_incremental_pca,only:incremental_pca
  use sklearn_sparse,only:csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_is_nan
  implicit none
  type(incremental_pca)::m
  real(dp),allocatable::z(:,:)
  real(dp)::single(1,3),constant(9,3)
  single(1,:)=[2._dp,-3._dp,7._dp];constant=spread(single(1,:),1,9)
  m%n_components=1;m%whiten=.true.;call m%fit(single)
  call require(m%n_samples_seen_==1,'IncrementalPCA accepts a singleton fit')
  call require(all(ieee_is_nan(m%explained_variance_)),'singleton explained variance is NaN')
  call require(all(ieee_is_nan(m%explained_variance_ratio_)),'singleton explained variance ratio is NaN')
  call require(all(m%var_==0),'singleton population variance is zero')
  z=m%transform(single)
  call require(all(ieee_is_nan(z)),'singleton whitened transform preserves NaN')
  call m%fit(csr_from_dense(single))
  call require(all(ieee_is_nan(m%explained_variance_)),'CSR singleton variance is NaN')
  call m%fit(constant)
  call require(all(m%explained_variance_==0),'constant explained variance is zero')
  call require(all(ieee_is_nan(m%explained_variance_ratio_)),'constant ratio is NaN')
  z=m%transform(single)
  call require(all(z==0),'constant whitened projection uses epsilon clipping')
  m=incremental_pca();m%n_components=1;m%whiten=.true.
  call m%partial_fit(reshape([2._dp],[1,1]));call m%partial_fit(reshape([4._dp],[1,1]))
  call require(m%n_samples_seen_==2.and.abs(m%mean_(1)-3)<1.e-14_dp,'singleton streaming mean')
  call require(abs(m%var_(1)-1)<1.e-14_dp,'singleton streaming population variance')
  call require(abs(m%explained_variance_(1)-2)<1.e-14_dp,'singleton streaming explained variance')
  z=m%transform(reshape([2._dp,4._dp],[2,1]))
  call require(maxval(abs(z(:,1)-[-1._dp,1._dp]/sqrt(2._dp)))<1.e-14_dp,'singleton streaming whitening')
  print *,'IncrementalPCA singleton/constant fits, IEEE moments and streaming: PASS'
end program
