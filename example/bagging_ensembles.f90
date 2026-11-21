program bagging_ensembles
  use sklearn, only: dp,bagging_classifier,bagging_regressor,extra_tree_regressor
  implicit none
  type(bagging_classifier) :: classifier
  type(bagging_regressor) :: regressor
  type(extra_tree_regressor) :: extra
  real(dp) :: x(40,2),target(40)
  integer :: y(40),i
  do i=1,40
    x(i,:)=[real(i,dp)/10,sin(real(i,dp))]
  end do
  y=merge(0,1,x(:,1)<2); target=x(:,1)**2+x(:,2)
  classifier%n_estimators=30; classifier%oob_score=.true.; classifier%random_state=42
  call classifier%fit(x,y)
  regressor%n_estimators=30; regressor%oob_score=.true.; regressor%random_state=42
  call regressor%fit(x,target,extra)
  print *, 'Bagging classification accuracy:',classifier%score(x,y)
  print *, 'Bagging classifier OOB accuracy:',classifier%oob_score_
  print *, 'Bagging regression R2:',regressor%score(x,target)
  print *, 'Bagging regressor OOB R2:',regressor%oob_score_
end program
