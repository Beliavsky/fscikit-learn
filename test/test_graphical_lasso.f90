program test_graphical_lasso
  use sklearn
  use sklearn_common, only: require
  implicit none
  type(graphical_lasso) :: model,precomputed,plain,capped
  type(empirical_covariance) :: empirical
  real(dp) :: covariance(2,2),expected(2,2),x(80,4),identity(4,4),residual(4,4),t
  real(dp), allocatable :: distances(:)
  integer :: i,j,k
  covariance=reshape([2._dp,0.8_dp,0.8_dp,1._dp],[2,2])
  model%covariance='precomputed'; model%alpha=0.3_dp; model%tol=1.e-12_dp; model%enet_tol=1.e-12_dp
  call model%fit(covariance)
  expected=reshape([1._dp,-0.5_dp,-0.5_dp,2._dp],[2,2])/1.75_dp
  call require(model%converged_,'analytic graphical lasso convergence')
  call require(maxval(abs(model%precision_-expected))<1.e-11_dp,'analytic sparse precision')
  call require(abs(model%covariance_(1,2)-0.5_dp)<1.e-12_dp,'analytic covariance soft threshold')
  model%alpha=2; call model%fit(covariance)
  call require(model%precision_(1,2)==0.and.model%precision_(2,1)==0,'large alpha diagonal precision')
  call require(abs(model%precision_(1,1)-0.5_dp)<1.e-12_dp,'unpenalized precision diagonal')
  do i=1,80
    t=real(i,dp)/80
    x(i,:)=[sin(17*t)+2,sin(17*t)*0.7_dp+cos(23*t)-1,sin(31*t)+0.2_dp*cos(23*t),cos(41*t)+3]
  end do
  call empirical%fit(x)
  model%covariance='empirical'; model%alpha=0.06_dp; model%tol=1.e-10_dp; model%enet_tol=1.e-12_dp
  model%max_iter=500; call model%fit(x)
  identity=0
  do j=1,4
    identity(j,j)=1
  end do
  call require(model%converged_,'dense graphical lasso convergence')
  call require(maxval(abs(matmul(model%precision_,model%covariance_)-identity))<1.e-8_dp,'covariance inverse')
  call require(maxval(abs(model%location_-empirical%location_))==0,'graphical training location')
  call require(abs(model%dual_gap_)<model%tol,'graphical dual gap')
  call require(all(shape(model%costs_)==[model%n_iter_,2]),'graphical history shape')
  call require(model%costs_(model%n_iter_,1)==model%objective_,'final graphical cost recorded')
  residual=empirical%covariance_-model%covariance_
  do j=1,4
    call require(abs(residual(j,j))<1.e-12_dp,'covariance diagonal is unpenalized')
    do k=1,4
      if(j==k) cycle
      if(abs(model%precision_(j,k))>1.e-8_dp) then
        call require(abs(residual(j,k)+model%alpha*sign(1._dp,model%precision_(j,k)))<1.e-7_dp, &
          'graphical active KKT condition')
      else
        call require(abs(residual(j,k))<=model%alpha+1.e-7_dp,'graphical inactive KKT condition')
      end if
    end do
  end do
  precomputed=model; precomputed%covariance='precomputed'; call precomputed%fit(empirical%covariance_)
  call require(maxval(abs(precomputed%precision_-model%precision_))<1.e-10_dp,'precomputed covariance equivalence')
  distances=model%mahalanobis(x)
  call require(all(distances>=0).and.model%score(x)<0,'inherited distances and likelihood')
  plain%alpha=0; call plain%fit(x)
  call require(plain%n_iter_==0.and.plain%converged_,'unregularized direct inverse')
  call require(maxval(abs(plain%precision_-empirical%precision_))<1.e-10_dp,'unregularized empirical equivalence')
  capped=model; capped%max_iter=1; capped%tol=1.e-14_dp; call capped%fit(x)
  call require(capped%n_iter_==1.and..not.capped%converged_,'iteration cap status')
  ! Rank-deficient sample covariance remains estimable with positive alpha.
  x(:,4)=x(:,1)+x(:,2); model%alpha=0.2_dp; call model%fit(x)
  call require(model%converged_,'regularized singular sample covariance')
  print *, 'GraphicalLasso tests passed'
end program
