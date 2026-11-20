program recursive_selection
  use sklearn, only: dp,rfecv,ridge
  implicit none
  type(rfecv) :: selector
  type(ridge) :: estimator
  real(dp) :: x(48,5),y(48)
  integer :: i
  do i=1,48
    x(i,:)=[sin(real(i,dp)),cos(real(2*i,dp)),sin(real(3*i,dp)),cos(real(4*i,dp)),sin(real(5*i,dp))]
  end do
  y=3*x(:,2)-2*x(:,4); estimator%alpha=.1_dp; selector%cv=4
  call selector%fit(x,y,estimator)
  print *, 'Selected mask: ',selector%get_support()
  print *, 'Rankings: ',selector%ranking_
  print *, 'Feature counts: ',selector%cv_results_%n_features
  print *, 'Mean validation R2: ',selector%cv_results_%mean_test_score
  print *, 'Full-data R2: ',selector%score(x,y)
end program
