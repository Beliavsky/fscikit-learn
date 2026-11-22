program kernel_preimage
  use sklearn_common, only: dp
  use sklearn_kernel_pca, only: kernel_pca
  implicit none
  type(kernel_pca) :: model
  real(dp) :: x(20,2)
  real(dp), allocatable :: latent(:,:),reconstructed(:,:)
  integer :: i
  do i=1,20
    x(i,1)=cos(.15_dp*i);x(i,2)=sin(.15_dp*i)
  end do
  model%kernel='rbf';model%gamma=2;model%n_components=3
  model%fit_inverse_transform=.true.;model%alpha=.01_dp
  latent=model%fit_transform(x);reconstructed=model%inverse_transform(latent)
  print '(a,2i5)', 'Latent shape: ',shape(latent)
  print '(a,f14.8)', 'Reconstruction RMSE: ',sqrt(sum((x-reconstructed)**2)/size(x))
end program
