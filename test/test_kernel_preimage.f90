program test_kernel_preimage
  use sklearn_common, only: dp,require
  use sklearn_kernel_pca, only: kernel_pca
  implicit none
  type(kernel_pca) :: model
  real(dp) :: x(8,3),a(8,8)
  real(dp), allocatable :: z(:,:),back(:,:)
  integer :: i,j
  do j=1,3
    do i=1,8
      x(i,j)=sin(.31_dp*i*j)+2*j
    end do
  end do
  model%n_components=3;model%alpha=.2_dp;model%fit_inverse_transform=.true.
  z=model%fit_transform(x);back=model%inverse_transform(z)
  a=matmul(z,transpose(z))
  do i=1,8
    a(i,i)=a(i,i)+model%alpha
  end do
  call require(maxval(abs(matmul(a,model%dual_coef_)-x))<1.e-11_dp,'kernel preimage ridge equations')
  call require(maxval(abs(back-matmul(matmul(z,transpose(z)),model%dual_coef_)))<1.e-12_dp, &
    'linear preimage uses uncentered latent kernel without restoring mean')
  call require(maxval(abs(sum(back,dim=1)))<1.e-10_dp,'linear kernel preimage does not restore feature mean')
  model%kernel='rbf';model%gamma=.2_dp;model%alpha=.01_dp
  call model%fit(x);back=model%inverse_transform(model%transform(x))
  call require(all(shape(back)==shape(x)),'rbf preimage reconstruction shape')
  call require(maxval(abs(model%x_transformed_fit_-model%transform(x)))<1.e-10_dp,'stored latent training coordinates')
  model%fit_inverse_transform=.false.;call model%fit(x)
  call require(.not.allocated(model%dual_coef_).and..not.allocated(model%x_transformed_fit_),'refit clears inverse state')
  print *, 'KernelPCA learned inverse: PASS'
end program
