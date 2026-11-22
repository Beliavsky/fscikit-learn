program model_learning_curves
  use sklearn, only: dp, ridge, learning_curve_options, learning_curve_result, learning_curve, &
    cross_validation_options, validation_curve_result, validation_curve, permutation_score_result, permutation_test_score
  use sklearn_parameter_space, only: param
  implicit none
  type(ridge) :: model
  type(learning_curve_options) :: options
  type(cross_validation_options) :: cv
  type(learning_curve_result) :: learning
  type(validation_curve_result) :: validation
  type(permutation_score_result) :: significance
  real(dp) :: x(40,3),y(40)
  integer :: i
  do i=1,40
    x(i,:)=[sin(real(i,dp)),cos(real(2*i,dp)),real(i,dp)/40]
  end do
  y=2*x(:,1)-x(:,2)+.2_dp*x(:,3)
  model%alpha=.4_dp
  options%cv=4;options%train_sizes=[.25_dp,.5_dp,1._dp]
  learning=learning_curve(model,x,y,options)
  print '(a,3i5)', 'Training sizes:',learning%train_sizes
  print '(a,3f12.7)', 'Mean test R2:  ',sum(learning%test_score,dim=2)/4
  cv%cv=4
  validation=validation_curve(model,x,y,'alpha',[param(.01_dp),param(.4_dp),param(10._dp)],cv)
  print '(a,3f12.7)', 'Alpha test R2: ',sum(validation%test_score,dim=2)/4
  significance=permutation_test_score(model,x,y,cv,n_permutations=19,random_state=42)
  print '(a,f12.7,a,f7.4)', 'Observed R2:   ',significance%score,'  permutation p: ',significance%pvalue
end program
