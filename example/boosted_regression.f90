program boosted_regression
  use sklearn, only: dp,adaboost_regressor
  implicit none
  type(adaboost_regressor) :: model
  real(dp) :: x(40,1),y(40)
  integer :: i
  do i=1,40
    x(i,1)=real(i-1,dp)/10; y(i)=sin(x(i,1))+x(i,1)/4
  end do
  model%n_estimators=30; model%random_state=42
  call model%fit(x,y)
  print *, 'Fitted trees:',model%n_estimators_
  print *, 'Training R2:',model%score(x,y)
  print *, 'Predictions:',model%predict(reshape([.5_dp,1.5_dp,2.5_dp],[3,1]))
end program
