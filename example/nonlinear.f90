program nonlinear_example
  use sklearn
  implicit none
  type(standard_scaler) :: scaler
  type(svc) :: model
  real(dp), allocatable :: x(:,:),train_x(:,:),test_x(:,:)
  integer, allocatable :: y(:),train(:),test(:)
  call make_moons(120,x,y,noise=0.12_dp,random_state=42)
  call train_test_split(size(y),train,test,test_size=0.25_dp,random_state=7,stratify=y)
  call scaler%fit(x(train,:)); train_x=scaler%transform(x(train,:)); test_x=scaler%transform(x(test,:))
  model%c=10; model%gamma=1; call model%fit(train_x,y(train))
  print *, 'Test accuracy: ',model%score(test_x,y(test))
  print *, 'Converged: ',model%converged_,' iterations: ',model%n_iter_
end program
