program omp_paths
  use sklearn,only:dp,orthogonal_mp,orthogonal_mp_gram,omp_result
  implicit none
  real(dp)::x(3,3),y(3,2)
  type(omp_result)::fit
  x=0;x(1,1)=1;x(2,2)=1;x(3,3)=1
  y(:,1)=[3._dp,0._dp,-2._dp];y(:,2)=[0._dp,4._dp,0._dp]
  fit=orthogonal_mp(x,y,tol=.01_dp,return_path=.true.)
  print '(a,*(1x,i0))','Target steps:',fit%n_iter
  print '(a,*(1x,f6.2))','First target coefficients:',fit%coefficients(:,1)
  fit=orthogonal_mp_gram(matmul(transpose(x),x),matmul(transpose(x),y), &
    tol=.01_dp,norms_squared=sum(y**2,dim=1))
  print '(a,*(1x,f6.2))','Gram second target:',fit%coefficients(:,2)
end program
