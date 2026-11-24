program balanced_class_weights
  use sklearn, only: dp,compute_class_weight,compute_sample_weight,logistic_regression
  implicit none
  integer :: y(6)=[1,1,1,1,0,0]
  real(dp) :: x(6,1)
  real(dp), allocatable :: weights(:),class_weights(:)
  type(logistic_regression) :: model
  x(:,1)=[1._dp,2._dp,3._dp,4._dp,-1._dp,-2._dp]
  class_weights=compute_class_weight([0,1],y,'balanced')
  weights=compute_sample_weight(y,'balanced')
  call model%fit(x,y,sample_weight=weights)
  print *, 'Class weights:',class_weights
  print *, 'Total class masses:',sum(pack(weights,y==0)),sum(pack(weights,y==1))
  print *, 'Correct training predictions:',count(model%predict(x)==y)
end program
