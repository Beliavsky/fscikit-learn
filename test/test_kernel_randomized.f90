program test_kernel_randomized
  use sklearn_kernel_pca,only:kernel_pca
  use sklearn_common,only:dp,require
  implicit none
  type(kernel_pca)::model,copy,dense
  real(dp)::x(64,4)
  real(dp),allocatable::z(:,:),again(:,:),changed(:,:),exact(:,:)
  integer::i
  do i=1,64
    x(i,:)=[sin(real(i,dp)),cos(.7_dp*i),sin(.3_dp*i),real(mod(i,7),dp)/7]
  end do
  model%kernel='rbf';model%gamma=.4_dp;model%n_components=3
  model%eigen_solver='randomized';model%iterated_power=0;model%random_state=42
  z=model%fit_transform(x);copy=model;again=model%fit_transform(x)
  call require(maxval(abs(z-again))<1.e-12_dp,'randomized KernelPCA seeded repeat')
  call require(maxval(abs(z-model%transform(x)))>1.e-6_dp,'approximate fit coordinates differ from projection')
  model%random_state=7;changed=model%fit_transform(x)
  call require(maxval(abs(z-changed))>1.e-5_dp,'randomized KernelPCA uses seed')
  dense%kernel='rbf';dense%gamma=.4_dp;dense%n_components=3;exact=dense%fit_transform(x)
  model%iterated_power=7;again=model%fit_transform(x)
  call require(maxval(abs(model%eigenvalues_-dense%eigenvalues_))<1.e-7_dp,'randomized KernelPCA converges to dense')
  call require(maxval(abs(again-exact))<1.e-6_dp,'randomized KernelPCA converged coordinates')
  call require(all(model%eigenvalues_>0),'randomized KernelPCA PSD eigenvalues')
  model%fit_inverse_transform=.true.;again=model%fit_transform(x)
  changed=model%inverse_transform(again)
  call require(all(shape(changed)==shape(x)),'randomized KernelPCA inverse path')
  print *,'Randomized KernelPCA seeds, transforms, spectrum and inverse: PASS'
end program
