program regression_example
  use sklearn
  implicit none
  type(standard_scaler) :: scaler
  type(ridge_cv) :: model
  real(dp), allocatable :: x(:,:),y(:),train_x(:,:),test_x(:,:),prediction(:)
  integer, allocatable :: train(:),test(:)
  call make_regression(200,6,x,y,noise=2._dp,random_state=42)
  call train_test_split(size(y),train,test,test_size=0.2_dp,random_state=7)
  call scaler%fit(x(train,:)); train_x=scaler%transform(x(train,:)); test_x=scaler%transform(x(test,:))
  model%alphas=[0.01_dp,0.1_dp,1._dp,10._dp]; call model%fit(train_x,y(train))
  prediction=model%predict(test_x)
  print *, 'Selected alpha: ',model%alpha_
  print *, 'Test R2: ',r2_score(y(test),prediction)
  print *, 'Test RMSE: ',root_mean_squared_error(y(test),prediction)
end program
