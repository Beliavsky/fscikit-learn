program test_sparse_coordinate_paths
  use sklearn,only:dp,csr_matrix,csr_from_arrays,enet_path,lasso_path,coordinate_path_options,coordinate_path_result
  use sklearn_common,only:require
  implicit none
  type(csr_matrix)::x,z
  type(coordinate_path_options)::options
  type(coordinate_path_result)::fit,other
  real(dp)::y(3),weights(3),expected(3)
  x=csr_from_arrays(3,3,[1,2,3,4],[1,2,3],[1._dp,1._dp,1._dp])
  y=[3._dp,-2._dp,1._dp];weights=[1._dp,2._dp,.5_dp]
  options%tol=1.e-12_dp
  fit=enet_path(x,y,options,alphas=[.5_dp],sample_weight=weights)
  expected=sign(max(abs(weights*y)-.75_dp,0._dp),weights*y)/(weights+.75_dp)
  call require(maxval(abs(fit%coefficients(1,:,1)-expected))<1.e-12_dp,'weighted sparse orthogonal closed form')
  call require(fit%backend=='sparse','sparse backend overrides auto Gram')
  other=enet_path(x,y,options,alphas=[.5_dp],sample_weight=weights,gram=reshape([99._dp],[1,1]),xy=[99._dp])
  call require(all(other%coefficients==fit%coefficients),'sparse Gram and Xy are ignored')
  options%n_alphas=3;fit=enet_path(x,y,options,sample_weight=weights)
  call require(abs(fit%alphas(1)-2)<1.e-12_dp,'sparse automatic grid ignores sample weights')
  call require(abs(fit%coefficients(1,2,1))>.1_dp,'unweighted sparse grid need not start at weighted zero model')
  z=csr_from_arrays(3,1,[1,1,1,1],[integer::],[real(dp)::])
  fit=enet_path(z,[2._dp,2._dp,2._dp],options,alphas=[.1_dp],x_offset=[2._dp],x_scale=[2._dp])
  call require(abs(fit%coefficients(1,1,1)+(6-.15_dp)/(3+.15_dp))<1.e-12_dp,'implicit sparse centering')
  options%positive=.true.;fit=lasso_path(x,y,options,alphas=[.1_dp])
  call require(maxval(abs(fit%coefficients(1,:,1)-[2.7_dp,0._dp,.7_dp]))<1.e-12_dp,'sparse positive Lasso')
  options%positive=.false.;options%max_iter=0
  fit=enet_path(x,y,options,alphas=[.1_dp],coef_init=[1._dp,2._dp,3._dp])
  call require(fit%n_iter(1)==1.and..not.fit%converged(1),'sparse zero-sweep convention')
  call require(all(fit%coefficients(1,:,1)==[1._dp,2._dp,3._dp]),'sparse zero-sweep initial state')
  options%max_iter=1
  z=csr_from_arrays(2,1,[1,3,5],[1,1,1,1],[1._dp,1._dp,1._dp,1._dp])
  fit=enet_path(z,[2._dp,2._dp],options,alphas=[.1_dp],x_offset=[1._dp],x_scale=[1._dp])
  call require(abs(fit%coefficients(1,1,1)-(4-.1_dp)/(4294967294._dp+.1_dp))<1.e-20_dp, &
    'source unsigned implicit-zero count for duplicate storage')
  other=enet_path(z,[2._dp,2._dp],options,alphas=[.1_dp],sample_weight=[1._dp,1._dp], &
    x_offset=[1._dp],x_scale=[1._dp])
  call require(abs(other%coefficients(1,1,1)-(4-.1_dp)/(-2+.1_dp))<1.e-12_dp, &
    'source weighted duplicate norm uses separate accumulation')
  print *,'Sparse elastic-net and Lasso paths: PASS'
end program
