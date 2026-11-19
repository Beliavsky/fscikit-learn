program test_incremental_pca
  use sklearn, only: dp,incremental_pca,pca
  use sklearn_common, only: require
  implicit none
  type(incremental_pca) :: model,stream
  type(pca) :: full
  real(dp) :: x(8,3),y(8,3),expected(3)
  real(dp), allocatable :: z(:,:),back(:,:)
  integer :: i
  do i=1,8
    x(i,:)=[real(i,dp),real(mod(i,3),dp),real(i*i,dp)]
  end do
  model%n_components=3; model%batch_size=3
  call model%fit(x); call full%fit(x)
  call require(model%n_samples_seen_==8,'batched sample count')
  call require(maxval(abs(model%mean_-full%mean_))<1.e-12_dp,'streaming mean')
  call require(maxval(abs(model%explained_variance_-full%explained_variance_))<1.e-10_dp,'full rank equals PCA')
  expected=sum((x-spread(model%mean_,1,8))**2,dim=1)/8
  call require(maxval(abs(model%var_-expected))<1.e-11_dp,'population variance')
  z=model%transform(x); back=model%inverse_transform(z)
  call require(maxval(abs(back-x))<1.e-11_dp,'full rank reconstruction')
  model%whiten=.true.; z=model%transform(x)
  call require(maxval(abs(sum(z**2,dim=1)/7-1))<1.e-10_dp,'unit sample variance after whitening')
  stream%n_components=3; call stream%partial_fit(x(:4,:)); call stream%partial_fit(x(5:7,:))
  call stream%partial_fit(x(8:,:))
  call require(maxval(abs(stream%explained_variance_-full%explained_variance_))<1.e-10_dp,'one row update')
  y=x+10; call model%fit(y)
  call require(model%n_samples_seen_==8.and.maxval(abs(model%mean_-sum(y,dim=1)/8))<1.e-12_dp,'fit resets state')
  y=5; model%whiten=.false.; call model%fit(y)
  call require(all(model%var_==0).and.all(model%explained_variance_ratio_==0),'constant data finite ratios')
  print *, 'IncrementalPCA tests passed'
end program
