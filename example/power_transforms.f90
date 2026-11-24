program power_transforms
  use sklearn
  implicit none
  type(power_transformer) :: model
  real(dp) :: x(6,1)
  real(dp), allocatable :: z(:,:),restored(:,:)
  x(:,1)=[-3._dp,-1._dp,0._dp,1._dp,4._dp,12._dp]
  z=model%fit_transform(x); restored=model%inverse_transform(z)
  print *, 'Free map agreement:',maxval(abs(power_transform(x)-z))
  print *, 'Yeo-Johnson power:',model%lambdas_
  print *, 'Standardized values:',z(:,1)
  print *, 'Maximum reconstruction error:',maxval(abs(restored-x))
end program
