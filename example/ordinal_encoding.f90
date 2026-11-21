program ordinal_encoding_example
  use sklearn, only: dp,ordinal_encoder
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(ordinal_encoder) :: encoder
  real(dp) :: x(7,1)
  real(dp), allocatable :: encoded(:,:),query(:,:),restored(:,:)
  logical, allocatable :: unknown(:,:),infrequent(:,:)
  x(:,1)=[10._dp,10._dp,10._dp,20._dp,20._dp,30._dp,40._dp]
  encoder%min_frequency_count=2
  encoder%handle_unknown='use_encoded_value'; encoder%unknown_value=-1
  encoder%encoded_missing_value=-2
  encoded=encoder%fit_transform(x)
  query=encoder%transform(reshape([10._dp,30._dp,99._dp,ieee_value(0._dp,ieee_quiet_nan)],[4,1]))
  restored=encoder%inverse_transform(query,unknown,infrequent)
  print *, 'Training codes:',encoded(:,1)
  print *, 'Query codes:',query(:,1)
  print *, 'Unknown query rows:',unknown(:,1)
  print *, 'Grouped query rows:',infrequent(:,1)
end program
