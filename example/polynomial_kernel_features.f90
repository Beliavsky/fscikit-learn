program polynomial_kernel_features
  use sklearn, only: dp,polynomial_count_sketch
  implicit none
  type(polynomial_count_sketch) :: features
  real(dp) :: x(3,2),exact,approximate
  real(dp), allocatable :: z(:,:)
  x(1,:)=[1._dp,2._dp]; x(2,:)=[.5_dp,1._dp]; x(3,:)=[-1._dp,.5_dp]
  features%degree=3; features%gamma=.5_dp; features%coef0=1
  features%n_components=1009; features%random_state=42
  z=features%fit_transform(x)
  exact=(features%gamma*dot_product(x(1,:),x(2,:))+features%coef0)**features%degree
  approximate=dot_product(z(1,:),z(2,:))
  print *, 'Mapped samples, components:',shape(z)
  print *, 'Exact polynomial kernel for samples 1 and 2:',exact
  print *, 'TensorSketch approximation:',approximate
  print *, 'Hash table shape (degree, augmented features):',shape(features%index_hash_)
end program
