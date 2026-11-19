program chi2_kernel_features
  use sklearn
  implicit none
  type(additive_chi2_sampler) :: additive
  type(skewed_chi2_sampler) :: skewed
  real(dp) :: x(3,2)
  real(dp), allocatable :: z(:,:)
  x(1,:)=[0._dp,1._dp]; x(2,:)=[1._dp,2._dp]; x(3,:)=[2._dp,0._dp]
  z=additive%fit_transform(x)
  print *, 'Additive feature shape:',shape(z)
  skewed%n_components=100; skewed%skewedness=.5_dp; skewed%random_state=42
  z=skewed%fit_transform(x)
  print *, 'Skewed kernel approximation:'
  print *, matmul(z,transpose(z))
end program
