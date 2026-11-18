program bayesian_regression
  use sklearn
  implicit none
  type(bayesian_ridge) :: model
  type(ard_regression) :: sparse_model
  real(dp), allocatable :: x(:,:),y(:),prediction(:),std(:)
  call make_regression(n_samples=120,n_features=8,x=x,y=y,noise=0.2_dp,random_state=21)
  call model%fit(x(:90,:),y(:90))
  prediction=model%predict(x(91:,:)); std=model%predict_std(x(91:,:))
  print *, 'Bayesian held-out R2:',model%score(x(91:,:),y(91:))
  print *, 'First prediction and predictive standard deviation:',prediction(1),std(1)
  call sparse_model%fit(x(:90,:),y(:90))
  print *, 'ARD held-out R2:',sparse_model%score(x(91:,:),y(91:))
  print *, 'ARD retained coefficients:',count(sparse_model%coef_/=0._dp)
end program
