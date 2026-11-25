program test_omp_paths
  use sklearn,only:dp,orthogonal_mp,orthogonal_mp_gram,omp_result
  use sklearn_common,only:require
  implicit none
  type(omp_result)::fit,direct,gram_fit
  real(dp)::x(4,4),y(4,2),g(4,4),xy(4,2)
  integer::i
  x=0
  do i=1,4
    x(i,i)=1
  end do
  y(:,1)=[3._dp,0._dp,-2._dp,0._dp];y(:,2)=[0._dp,4._dp,0._dp,0._dp]
  fit=orthogonal_mp(x,y,n_nonzero_coefs=3,return_path=.true.)
  call require(all(fit%n_iter==[2,1]),'OMP ragged target iteration counts')
  call require(all(fit%premature),'OMP premature dependent/zero residual status')
  call require(all(fit%coefficients==y),'OMP orthogonal coefficients')
  call require(all(shape(fit%path)==[4,2,2]),'OMP full ragged path extent')
  call require(all(fit%path(:,2,2)==0),'OMP ragged path zero padding')
  call require(all(fit%path(:,1,2)==y(:,1)),'OMP last nonempty path column')
  fit=orthogonal_mp(x,y,tol=1._dp,return_path=.true.)
  call require(all(fit%n_iter==[2,1]),'OMP squared residual tolerance')
  call require(.not.any(fit%premature),'OMP tolerance convergence status')
  g=matmul(transpose(x),x);xy=matmul(transpose(x),y)
  gram_fit=orthogonal_mp_gram(g,xy,tol=1._dp,norms_squared=sum(y**2,dim=1),return_path=.true.)
  call require(all(gram_fit%coefficients==fit%coefficients),'Gram/direct OMP coefficients')
  call require(all(gram_fit%path==fit%path),'Gram/direct OMP paths')
  direct=orthogonal_mp(x,y(:,1),n_nonzero_coefs=1)
  call require(.not.allocated(direct%path),'OMP no-path allocation')
  call require(all(direct%coefficients(:,1)==[3._dp,0._dp,0._dp,0._dp]),'OMP first selected atom')
  fit=orthogonal_mp(x,y(:,1),precompute='auto',n_nonzero_coefs=1)
  call require(.not.fit%gram_used,'OMP equal-size auto selects direct')
  fit=orthogonal_mp(x,y(:,1),precompute=.true.,n_nonzero_coefs=1)
  call require(fit%gram_used.and.all(fit%n_iter==1),'OMP precomputed iteration metadata')
  fit=orthogonal_mp(x,0*y,tol=0._dp,return_path=.true.)
  call require(all(fit%n_iter==0).and.size(fit%path,3)==0,'OMP zero-target empty paths')
  print *,'Direct and Gram OMP paths: PASS'
end program
