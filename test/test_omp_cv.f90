program test_omp_cv
  use sklearn_omp_cv, only: orthogonal_matching_pursuit_cv
  use sklearn_common, only: dp,require
  implicit none
  type(orthogonal_matching_pursuit_cv) :: model
  real(dp) :: x(18,3),y(18),before(18,3),yb(18)
  real(dp), allocatable :: prediction(:)
  integer :: i,j
  x=0
  do i=1,18
    j=mod(i-1,6)/2+1; x(i,j)=merge(1._dp,-1._dp,mod(i,2)==1)
  end do
  y=5+2*x(:,1)+x(:,2); before=x; yb=y
  model%cv=3; call model%fit(x,y)
  call require(model%n_nonzero_coefs_==2.and.model%n_iter_==2,'OMP CV analytic selected sparsity')
  call require(maxval(abs(model%coef_-[2._dp,1._dp,0._dp]))<1.e-13_dp,'OMP CV analytic coefficients')
  call require(abs(model%intercept_-5)<1.e-13_dp,'OMP CV analytic intercept')
  call require(all(shape(model%mse_path_)==[2,3]),'OMP CV common early-stop path')
  call require(maxval(abs(model%mse_path_(1,:)-1._dp/3))<1.e-13_dp,'OMP CV analytic first-step MSE')
  call require(maxval(abs(model%mse_path_(2,:)))<1.e-25_dp,'OMP CV analytic selected MSE')
  call require(all(model%fold_n_iter_==2),'OMP CV early-stop metadata')
  call require(all(x==before).and.all(y==yb),'OMP CV preserves inputs')
  call require(abs(model%score(x,y)-1)<1.e-13_dp,'OMP CV analytic score')
  model%max_iter=1; call model%fit(x,y)
  call require(model%n_nonzero_coefs_==1.and.size(model%mse_path_,1)==1,'OMP CV repeated shorter fit')
  model%test_fold=[(merge(9,-1,i<=6),i=1,18)]; model%max_iter=0
  call model%fit(x,y)
  call require(model%n_splits_==1.and.model%fold_test_sizes_(1)==6,'OMP CV predefined single holdout')
  deallocate(model%test_fold); allocate(model%train_mask(18,2),model%test_mask(18,2))
  model%train_mask=.false.; model%test_mask=.false.
  model%train_mask(1:6,1)=.true.; model%test_mask(7:12,1)=.true.
  model%train_mask(1:12,2)=.true.; model%test_mask(13:18,2)=.true.
  call model%fit(x,y)
  call require(all(model%fold_train_sizes_==[6,12]),'OMP CV arbitrary training masks')
  prediction=model%predict(x)
  call require(maxval(abs(prediction-y))<1.e-13_dp,'OMP CV arbitrary fold refit')
  print *, 'OrthogonalMatchingPursuitCV tests passed'
end program
