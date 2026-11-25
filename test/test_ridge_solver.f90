program test_ridge_solver
  use sklearn,only:dp,ridge_regression,ridge_solver_options,ridge_solution,csr_from_dense,csr_from_arrays,csr_matrix
  use sklearn_common,only:require
  implicit none
  real(dp)::x(3,3),y(3),w(3),expected(3)
  type(ridge_solver_options)::options
  type(ridge_solution)::fit,other
  type(csr_matrix)::large
  integer::i,j
  character(16),parameter::solvers(*)=[character(16)::'svd','cholesky','sparse_cg','lsqr','lbfgs']
  x=0
  do i=1,3
    x(i,i)=1
  end do
  y=[3._dp,-2._dp,1._dp];w=[1._dp,2._dp,.5_dp]
  options%tol=1.e-12_dp
  do j=1,size(solvers)
    options%solver=solvers(j);options%positive=options%solver=='lbfgs'
    expected=w*y/(w+2)
    if(options%positive)expected=max(expected,0._dp)
    fit=ridge_regression(x,y,2._dp,options,w)
    call require(maxval(abs(fit%coefficients(1,:)-expected))<1.e-8_dp,'weighted ridge closed form')
    other=ridge_regression(x,y,[2._dp],options,w)
    call require(all(fit%coefficients==other%coefficients),'dense vector penalty overload')
    other=ridge_regression(x,reshape(y,[3,1]),2._dp,options,w)
    call require(all(fit%coefficients==other%coefficients),'dense matrix scalar overload')
    other=ridge_regression(x,reshape(y,[3,1]),[2._dp],options,w)
    call require(all(fit%coefficients==other%coefficients),'dense matrix vector overload')
    if(options%solver=='svd')cycle
    other=ridge_regression(csr_from_dense(x),y,2._dp,options,w)
    call require(maxval(abs(other%coefficients-fit%coefficients))<1.e-10_dp,'CSR scalar overload')
    other=ridge_regression(csr_from_dense(x),y,[2._dp],options,w)
    call require(maxval(abs(other%coefficients-fit%coefficients))<1.e-10_dp,'CSR vector overload')
    other=ridge_regression(csr_from_dense(x),reshape(y,[3,1]),2._dp,options,w)
    call require(maxval(abs(other%coefficients-fit%coefficients))<1.e-10_dp,'CSR matrix scalar overload')
    other=ridge_regression(csr_from_dense(x),reshape(y,[3,1]),[2._dp],options,w)
    call require(maxval(abs(other%coefficients-fit%coefficients))<1.e-10_dp,'CSR matrix vector overload')
  end do
  options%positive=.false.;options%solver='lsqr';options%max_iter=0
  fit=ridge_regression(x,y,1._dp,options)
  call require(all(fit%n_iter==0).and.all(fit%coefficients==0),'zero LSQR iteration limit')
  call require(.not.any(fit%converged),'zero LSQR cap is unconverged')
  options%solver='auto';options%max_iter=-1
  large=csr_from_arrays(1,100000,[1,2],[100000],[2._dp])
  fit=ridge_regression(large,[3._dp],1._dp,options)
  call require(abs(fit%coefficients(1,100000)-1.2_dp)<1.e-12_dp,'wide matrix-free CSR ridge')
  call require(all(y==[3._dp,-2._dp,1._dp]).and.all(w==[1._dp,2._dp,.5_dp]),'input preservation')
  print *,'ridge solver tests passed'
end program
