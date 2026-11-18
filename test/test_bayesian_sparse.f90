program test_bayesian_sparse
  use sklearn
  use sklearn_common, only: require,normal_random,seed_rng,cholesky_factor,solve_lower,solve_upper
  implicit none
  type(bayesian_ridge) :: br
  type(ard_regression) :: ar
  type(orthogonal_matching_pursuit) :: omp
  real(dp) :: x(100,8),y(100),zero(100),query(2,8),std(2),wide(4,8),xc(4,8),a(8,8),l(8,8),expected(4)
  real(dp), allocatable :: covariance(:,:),scores(:)
  integer :: i
  call seed_rng(91); x=normal_random(100,8)
  y=3*x(:,1)-2*x(:,4)+1+0.01_dp*x(:,7); zero=0
  call br%fit(x,y)
  call require(br%converged_.and.br%score(x,y)>0.999_dp,'Bayesian sparse signal fit')
  scores=cross_val_score(br,x,y,cv=4)
  call require(minval(scores)>0.999_dp,'Bayesian cross-validation refits inherited regressor')
  query=0; query(2,1)=100; std=br%predict_std(query)
  call require(std(2)>std(1).and.all(std>0),'Bayesian uncertainty away from data')
  call ar%fit(x,y)
  call require(ar%converged_.and.ar%score(x,y)>0.999_dp,'ARD sparse signal fit')
  call require(count(ar%coef_/=0._dp)<8.and.abs(ar%coef_(1)-3)<0.02_dp,'ARD prunes irrelevant features')
  call ar%fit(x,zero)
  call require(all(ar%coef_==0).and.size(ar%sigma_,1)==0,'ARD all features pruned')
  std=ar%predict_std(query); call require(all(std>0),'ARD empty posterior uncertainty')
  ar%threshold_lambda=1.e-10_dp; ar%max_iter=1; call ar%fit(x,y)
  call require(size(ar%sigma_,1)==0,'ARD all-pruned refit')
  omp%n_nonzero_coefs=2; call omp%fit(x,y)
  call require(omp%n_iter_==2.and.count(omp%coef_/=0._dp)==2,'OMP exact support size')
  call require(abs(omp%coef_(1)-3)<0.02_dp.and.abs(omp%coef_(4)+2)<0.02_dp,'OMP support recovery')
  omp%tol=0.1_dp; call omp%fit(x,y)
  call require(sum((y-omp%predict(x))**2)<=omp%tol,'OMP residual tolerance')
  call omp%fit(x,zero); call require(omp%n_iter_==0.and.all(omp%coef_==0),'OMP zero target')
  x(:,2)=2*x(:,1); y=x(:,1)+5
  omp%tol=1.e-15_dp; call omp%fit(x,y)
  call require(maxval(abs(omp%predict(x)-y))<1.e-10_dp,'OMP dependent dictionary')
  wide=x(:4,:); br%fit_intercept=.false.; call br%fit(wide,y(:4))
  covariance=br%sigma_
  call require(all(br%predict_std(wide)>0).and.all(shape(covariance)==[8,8]),'wide Bayesian covariance')
  br%fit_intercept=.true.; call br%fit(wide,y(:4))
  xc=wide-spread(sum(wide,dim=1)/4,1,4)
  a=br%alpha_*matmul(transpose(xc),xc)
  do i=1,8
    a(i,i)=a(i,i)+br%lambda_
  end do
  l=cholesky_factor(a)
  do i=1,4
    expected(i)=sqrt(dot_product(xc(i,:),solve_upper(transpose(l),solve_lower(l,xc(i,:))))+1/br%alpha_)
  end do
  call require(maxval(abs(br%predict_std(xc)-expected))<1.e-8_dp,'centered wide posterior on identifiable row space')
  call br%fit(x,zero); call require(all(br%coef_==0),'Bayesian constant target refit')
  print *, 'Bayesian and sparse regression: PASS'
end program
