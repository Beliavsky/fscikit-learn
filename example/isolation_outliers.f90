program isolation_outliers
  use sklearn, only: dp,isolation_forest
  implicit none
  type(isolation_forest) :: model
  real(dp) :: x(100,2),query(3,2)
  integer :: i
  do i=1,100
    x(i,:)=[real(mod(i-1,10),dp)/5-.9_dp,real((i-1)/10,dp)/5-.9_dp]
  end do
  model%random_state=42; model%contamination=.1_dp
  call model%fit(x)
  query(1,:)=[0._dp,0._dp]; query(2,:)=[.5_dp,.5_dp]; query(3,:)=[8._dp,8._dp]
  print *, 'Tree count:',model%n_estimators_
  print *, 'Anomaly scores:',model%score_samples(query)
  print *, 'Inlier (+1) / outlier (-1) labels:',model%predict(query)
end program
