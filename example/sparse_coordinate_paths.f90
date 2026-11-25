program sparse_coordinate_paths
  use sklearn,only:dp,csr_matrix,csr_from_arrays,enet_path,lasso_path,coordinate_path_options,coordinate_path_result
  implicit none
  type(csr_matrix)::x
  type(coordinate_path_options)::options
  type(coordinate_path_result)::fit
  real(dp)::y(3)
  x=csr_from_arrays(3,3,[1,2,3,4],[1,2,3],[1._dp,1._dp,1._dp]);y=[3._dp,-2._dp,1._dp]
  fit=enet_path(x,y,options,alphas=[.5_dp],sample_weight=[1._dp,2._dp,.5_dp])
  print '(a,*(1x,f8.4))','Weighted sparse elastic:',fit%coefficients(1,:,1)
  options%positive=.true.;fit=lasso_path(x,y,options,alphas=[.1_dp])
  print '(a,*(1x,f8.4))','Sparse positive Lasso:',fit%coefficients(1,:,1)
end program
