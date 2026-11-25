program test_coordinate_paths
  use sklearn,only:dp,enet_path,lasso_path,coordinate_path_options,coordinate_path_result
  use sklearn_common,only:require
  implicit none
  type(coordinate_path_options)::options
  type(coordinate_path_result)::fit,direct
  real(dp)::x(3,3),y(3),ym(3,2),expected(3),grid(3),w(3),norm
  integer::i,j
  x=0
  do i=1,3
    x(i,i)=1
  end do
  y=[3._dp,-2._dp,1._dp];grid=[.01_dp,.5_dp,.01_dp]
  options%precompute='false';options%tol=1.e-12_dp
  direct=enet_path(x,y,options,alphas=grid)
  call require(all(direct%alphas==[.5_dp,.01_dp,.01_dp]),'coordinate path descending duplicate alphas')
  do i=1,3
    expected=sign(max(abs(y)-3*direct%alphas(i)*.5_dp,0._dp),y)/(1+3*direct%alphas(i)*.5_dp)
    call require(maxval(abs(direct%coefficients(1,:,i)-expected))<1.e-12_dp,'elastic path orthogonal closed form')
  end do
  call require(all(grid==[.01_dp,.5_dp,.01_dp]),'coordinate path retains alpha input')
  options%precompute='true';fit=enet_path(x,y,options,alphas=grid)
  call require(maxval(abs(fit%coefficients-direct%coefficients))<1.e-12_dp,'Gram/direct elastic paths')
  call require(fit%backend=='gram','coordinate Gram backend')
  options%precompute='false';fit=enet_path(x,y,options,alphas=grid,xy=[99._dp])
  call require(maxval(abs(fit%coefficients-direct%coefficients))<1.e-12_dp,'unused correlation shape is ignored')
  options%l1_ratio=.25_dp;fit=lasso_path(x,y,options,alphas=[.25_dp])
  expected=sign(max(abs(y)-.75_dp,0._dp),y)
  call require(maxval(abs(fit%coefficients(1,:,1)-expected))<1.e-12_dp,'Lasso overrides elastic mixing')
  options%l1_ratio=.5_dp;options%eps=.01_dp;options%n_alphas=3
  fit=enet_path(x,y,options)
  call require(maxval(abs(fit%alphas-[2._dp,.2_dp,.02_dp]))<1.e-12_dp,'automatic alpha grid')
  ym(:,1)=y;ym(:,2)=[1._dp,2._dp,0._dp]
  fit=enet_path(x,ym,options,alphas=[.25_dp])
  call require(fit%backend=='multitask','matrix targets select grouped penalty')
  do j=1,3
    norm=sqrt(sum(ym(j,:)**2))
    call require(maxval(abs(fit%coefficients(:,j,1)-ym(j,:)*max(1-.375_dp/norm,0._dp)/1.375_dp)) &
      <1.e-12_dp,'multi-task path orthogonal block solution')
  end do
  options%max_iter=7
  fit=enet_path(x,0*y,options,alphas=[.5_dp])
  call require(fit%n_iter(1)==7.and..not.fit%converged(1),'scalar strict zero dual tolerance')
  fit=enet_path(x,reshape(0*y,[3,1]),options,alphas=[.5_dp])
  call require(fit%n_iter(1)==1.and.fit%converged(1),'multi-task non-strict zero dual tolerance')
  options%max_iter=0;w=[1._dp,2._dp,3._dp]
  fit=enet_path(x,y,options,alphas=[.1_dp,.01_dp],coef_init=w)
  call require(all(fit%n_iter==1).and..not.any(fit%converged),'source zero-sweep convention')
  call require(all(fit%coefficients(1,:,1)==w).and.all(fit%coefficients(1,:,2)==w),'zero sweeps retain warm state')
  call require(all(abs(fit%dual_gaps-(1+options%tol)/3)<1.e-14_dp),'zero-sweep gap initialization')
  fit=enet_path(x,y,options,alphas=[real(dp)::])
  call require(size(fit%alphas)==0.and.size(fit%coefficients,3)==0,'empty explicit coordinate path')
  print *,'Elastic-net and Lasso coordinate paths: PASS'
end program
