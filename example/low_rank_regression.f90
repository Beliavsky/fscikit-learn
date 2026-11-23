program low_rank_regression
  use sklearn, only: dp,make_low_rank_matrix,make_regression,linear_regression
  implicit none
  type(linear_regression) :: model
  real(dp), allocatable :: matrix(:,:),x(:,:),y(:,:),coef(:,:)
  matrix=make_low_rank_matrix(20,5,effective_rank=2,tail_strength=.1_dp,random_state=42)
  call make_regression(60,8,x,y,n_targets=2,n_informative=4,effective_rank=3, &
    tail_strength=.1_dp,bias=2._dp,random_state=42,coef=coef)
  call model%fit(x,y(:,1))
  print *, 'Low-rank matrix shape:',shape(matrix)
  print *, 'Regression samples, features and targets:',size(x,1),size(x,2),size(y,2)
  print *, 'Noiseless target reconstruction error:',maxval(abs(y-matmul(x,coef)-2))
  print *, 'First-target training R2:',model%score(x,y(:,1))
end program
