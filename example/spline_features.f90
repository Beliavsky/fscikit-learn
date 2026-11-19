program spline_features
  use sklearn, only: dp,spline_transformer
  implicit none
  type(spline_transformer) :: model
  real(dp) :: x(5,1)
  real(dp), allocatable :: basis(:,:)
  x(:,1)=[0._dp,.5_dp,1._dp,1.5_dp,2._dp]
  model%n_knots=3; basis=model%fit_transform(x)
  print *, 'Number of spline features:',model%n_features_out_
  print *, 'Basis sums:',sum(basis,dim=2)
end program
