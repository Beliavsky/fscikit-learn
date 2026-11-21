program target_encoding_example
  use sklearn, only: dp,target_encoder
  implicit none
  type(target_encoder) :: encoder
  real(dp) :: x(8,1),y(8)
  real(dp), allocatable :: training(:,:),query(:,:)
  x(:,1)=[1._dp,1._dp,2._dp,2._dp,3._dp,3._dp,4._dp,4._dp]
  y=[1._dp,3._dp,4._dp,6._dp,9._dp,7._dp,2._dp,4._dp]
  encoder%target_type='continuous'; encoder%cv=4; encoder%random_state=42
  training=encoder%fit_transform(x,y)
  query=encoder%transform(reshape([1._dp,2._dp,99._dp],[3,1]))
  print *, 'Cross-fitted training features:',training(:,1)
  print *, 'Query category encodings:',query(:,1)
  print *, 'Unseen category uses target mean:',encoder%target_mean_(1)
end program
