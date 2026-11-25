program coordinate_paths
  use sklearn,only:dp,enet_path,lasso_path,coordinate_path_options,coordinate_path_result
  implicit none
  type(coordinate_path_options)::options
  type(coordinate_path_result)::fit
  real(dp)::x(3,3),y(3)
  x=reshape([1._dp,0._dp,0._dp,0._dp,1._dp,0._dp,0._dp,0._dp,1._dp],[3,3]);y=[3._dp,-2._dp,1._dp]
  options%l1_ratio=.5_dp
  fit=enet_path(x,y,options,alphas=[.5_dp,.1_dp])
  print '(a,*(1x,f8.4))','Elastic alpha=0.1:',fit%coefficients(1,:,2)
  options%positive=.true.
  fit=lasso_path(x,y,options,alphas=[.1_dp])
  print '(a,*(1x,f8.4))','Positive Lasso:',fit%coefficients(1,:,1)
end program
