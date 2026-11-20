program online_outliers
  use sklearn, only: dp,sgd_one_class_svm
  implicit none
  type(sgd_one_class_svm) :: model
  real(dp) :: x(60,2),query(3,2)
  integer :: i
  do i=1,60
    x(i,:)=[2+.2_dp*sin(real(i,dp)),2+.2_dp*cos(real(i,dp))]
  end do
  model%learning_rate='constant'; model%eta0=.01_dp; model%nu=.1_dp; model%average=1
  do i=1,30
    call model%partial_fit(x)
  end do
  query(:,1)=[2._dp,2.1_dp,0._dp]; query(:,2)=[2._dp,1.9_dp,0._dp]
  print *, 'Processed samples:',model%t_-1
  print *, 'Decision scores:',model%decision_function(query)
  print *, 'Inlier (+1) / outlier (-1) labels:',model%predict(query)
end program
