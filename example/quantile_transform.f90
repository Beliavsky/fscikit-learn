program quantile_transform_demo
  use sklearn
  implicit none
  type(quantile_transformer) :: model
  real(dp) :: x(5,1)
  real(dp), allocatable :: z(:,:)
  x(:,1)=[0._dp,1._dp,2._dp,4._dp,8._dp]
  model%n_quantiles=5; model%subsample=0
  z=model%fit_transform(x)
  print *, 'Uniform quantiles:',z(:,1)
  model%output_distribution='normal'; z=model%transform(x)
  print *, 'Normal scores:',z(:,1)
  z=model%inverse_transform(z)
  print *, 'Recovered data:',z(:,1)
end program
