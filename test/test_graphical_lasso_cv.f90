program test_graphical_lasso_cv
  use sklearn_covariance, only: graphical_lasso_cv,graphical_lasso
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_exceptions, only: ieee_invalid,ieee_overflow,ieee_divide_by_zero, &
    ieee_get_halting_mode,ieee_get_flag
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(graphical_lasso_cv) :: model,copy,ties
  type(graphical_lasso) :: plain
  real(dp) :: x(60,3),t,identity(3,3)
  real(dp), allocatable :: residual(:,:)
  logical :: before(3),after(3),before_flags(3),after_flags(3)
  integer :: i,j
  do i=1,60
    t=real(i,dp)/60
    x(i,:)=[sin(17*t)+1,cos(23*t)+.7_dp*sin(17*t)-2,sin(37*t)+.3_dp*cos(23*t)]
  end do
  call ieee_get_halting_mode(ieee_invalid,before(1)); call ieee_get_halting_mode(ieee_overflow,before(2))
  call ieee_get_halting_mode(ieee_divide_by_zero,before(3))
  call ieee_get_flag(ieee_invalid,before_flags(1)); call ieee_get_flag(ieee_overflow,before_flags(2))
  call ieee_get_flag(ieee_divide_by_zero,before_flags(3))
  model%cv=3; model%n_refinements=2; model%max_iter=500; model%tol=1.e-9_dp; model%enet_tol=1.e-11_dp
  call model%fit(x)
  call ieee_get_halting_mode(ieee_invalid,after(1)); call ieee_get_halting_mode(ieee_overflow,after(2))
  call ieee_get_halting_mode(ieee_divide_by_zero,after(3))
  call require(all(before.eqv.after),'GraphicalLassoCV restores FPE modes')
  call require(size(model%cv_results_%alphas)==9,'GraphicalLassoCV refined grid plus empirical endpoint')
  call require(model%cv_results_%alphas(9)==0.and.model%alpha_>0,'GraphicalLassoCV zero endpoint is report-only')
  call require(all(shape(model%cv_results_%split_test_score)==[9,3]),'GraphicalLassoCV result orientation')
  call require(all(model%cv_splits_(1)%test==[(i,i=1,20)]),'GraphicalLassoCV nonshuffled folds')
  identity=0
  do j=1,3
    identity(j,j)=1
  end do
  residual=matmul(model%covariance_,model%precision_)-identity
  call require(maxval(abs(residual))<1.e-7_dp,'GraphicalLassoCV fitted covariance inverse')
  call require(model%converged_.and.abs(model%dual_gap_)<model%tol,'GraphicalLassoCV final convergence')
  plain%alpha=model%alpha_; plain%max_iter=model%max_iter; plain%tol=model%tol; plain%enet_tol=model%enet_tol
  call plain%fit(x)
  call require(maxval(abs(plain%precision_-model%precision_))<1.e-8_dp,'GraphicalLassoCV final standalone equivalence')
  call require(abs(model%score(x)-plain%score(x))<1.e-8_dp,'GraphicalLassoCV inherited score convention')
  copy=model
  ties%alphas=[10._dp,5._dp,2._dp]; ties%cv=3; call ties%fit(x)
  call require(ties%alpha_==2,'GraphicalLassoCV smaller alpha wins exact score ties')
  call require(size(ties%cv_results_%alphas)==4,'GraphicalLassoCV explicit grid ignores refinements')
  model%alphas=[2._dp,1._dp]; model%max_iter=0; call model%fit(x)
  call require(size(model%costs_,1)==0.and.model%n_iter_==1,'GraphicalLassoCV zero-iteration source convention')
  call require(copy%n_iter_>0.and.size(copy%cv_results_%alphas)==9,'GraphicalLassoCV independent model copy')
  model=copy; allocate(model%splits(1))
  model%splits(1)%train=[(i,i=1,40)]; model%splits(1)%test=[(i,i=41,60)]
  call model%fit(x)
  call require(size(model%cv_results_%split_test_score,2)==1,'GraphicalLassoCV explicit single split')
  call require(all(model%cv_results_%std_test_score==0),'GraphicalLassoCV single-fold standard deviation')
  model=graphical_lasso_cv(); model%n_refinements=2
  allocate(model%splits(1)); model%splits(1)%train=[1,2,3,4]; model%splits(1)%test=[1,2,3,4]
  call model%fit(reshape([1._dp,-1._dp,0._dp,0._dp,0._dp,0._dp,1._dp,-1._dp],[4,2]))
  call require(model%alpha_==0.and.model%n_iter_==0,'GraphicalLassoCV zero off-diagonal source grid')
  call require(size(model%costs_,1)==1,'GraphicalLassoCV direct inverse costs')
  call ieee_get_halting_mode(ieee_invalid,after(1)); call ieee_get_halting_mode(ieee_overflow,after(2))
  call ieee_get_halting_mode(ieee_divide_by_zero,after(3))
  call ieee_get_flag(ieee_invalid,after_flags(1)); call ieee_get_flag(ieee_overflow,after_flags(2))
  call ieee_get_flag(ieee_divide_by_zero,after_flags(3))
  call require(all(before.eqv.after).and.all(before_flags.eqv.after_flags),'GraphicalLassoCV restores failed-path FPE state')
  model=graphical_lasso_cv(); model%alphas=[1._dp,.1_dp]
  allocate(model%splits(1)); model%splits(1)%train=[1]; model%splits(1)%test=[2,3,4]
  call model%fit(x)
  call require(model%alpha_==.1_dp.and.all(ieee_is_nan(model%cv_results_%std_test_score)), &
    'GraphicalLassoCV failed-fold scores and negative-infinity tie selection')
  print *, 'GraphicalLassoCV: PASS'
end program
