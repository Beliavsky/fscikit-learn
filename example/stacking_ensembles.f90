program stacking_ensembles
  use sklearn, only: dp,stacking_classifier,stacking_regressor,stacking_estimator, &
    logistic_regression,gaussian_nb,linear_regression,kneighbors_regressor
  implicit none
  type(stacking_classifier) :: classifier
  type(stacking_regressor) :: regressor
  type(stacking_estimator) :: classifiers(2),regressors(2)
  type(logistic_regression) :: logistic
  type(gaussian_nb) :: nb
  type(linear_regression) :: linear
  type(kneighbors_regressor) :: neighbors
  real(dp) :: x(60,2),target(60)
  integer :: y(60),i
  do i=1,60
    x(i,:)=[real(i,dp)/10,sin(real(i,dp))]
  end do
  y=merge(0,1,x(:,1)<3); target=2*x(:,1)-x(:,2)
  classifiers(1)%name='logistic'; allocate(classifiers(1)%estimator,source=logistic)
  classifiers(2)%name='bayes'; allocate(classifiers(2)%estimator,source=nb)
  regressors(1)%name='linear'; allocate(regressors(1)%estimator,source=linear)
  regressors(2)%name='neighbors'; allocate(regressors(2)%estimator,source=neighbors)
  call classifier%fit(x,y,classifiers)
  regressor%passthrough=.true.; call regressor%fit(x,target,regressors)
  print *, 'Stacking classification accuracy:',classifier%score(x,y)
  print *, 'Stacking regression R2:',regressor%score(x,target)
  print *, 'Classifier meta feature count:',classifier%n_features_out_
  print *, 'Regressor meta feature count:',regressor%n_features_out_
end program
