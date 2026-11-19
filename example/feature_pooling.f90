program feature_pooling
  use sklearn, only: dp,feature_agglomeration
  implicit none
  type(feature_agglomeration) :: model
  real(dp) :: x(3,4)
  real(dp), allocatable :: z(:,:)
  x(:,1)=[1._dp,2._dp,3._dp]; x(:,2)=x(:,1)+.1_dp
  x(:,3)=[20._dp,30._dp,40._dp]; x(:,4)=x(:,3)+.2_dp
  z=model%fit_transform(x)
  print *, 'Feature cluster labels:',model%labels_
  print *, 'Pooled first row:',z(1,:)
end program
