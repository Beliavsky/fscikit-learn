program test_lars_gram
  use sklearn,only:dp,lars_path,lars_path_gram,lars_path_result,graphical_lasso,graphical_lasso_cv
  use sklearn_common,only:require
  implicit none
  type(lars_path_result)::direct,gram_path
  type(graphical_lasso)::cd,lar
  type(graphical_lasso_cv)::cv
  real(dp)::x(40,4),y(40),gram(4,4),xy(4)
  integer::i,j
  do i=1,40
    x(i,:)=[sin(real(i,dp)),cos(real(i,dp)),sin(real(2*i,dp)),cos(real(3*i,dp))]
  end do
  x(:,2)=x(:,2)+.7_dp*x(:,1)
  y=2+3*x(:,2)-2*x(:,4);gram=matmul(transpose(x),x);xy=matmul(transpose(x),y)
  do j=1,2
    if(j==1)then
      direct=lars_path(x,y,method='lar',alpha_min=.03_dp)
      gram_path=lars_path_gram(xy,gram,40,method='lar',alpha_min=.03_dp)
    else
      direct=lars_path(x,y,method='lasso',positive=.true.)
      gram_path=lars_path_gram(xy,gram,40,method='lasso',positive=.true.)
    end if
    call require(all(direct%alphas==gram_path%alphas),'Gram and XY alpha paths')
    call require(all(direct%coefficients==gram_path%coefficients),'Gram and XY coefficient paths')
    call require(all(direct%active==gram_path%active),'Gram and XY active paths')
  end do
  cd%alpha=.1_dp;cd%tol=1.e-9_dp;cd%enet_tol=1.e-11_dp
  lar=cd;lar%mode='lars'
  call cd%fit(x);call lar%fit(x)
  call require(maxval(abs(cd%precision_-lar%precision_))<1.e-7_dp,'GraphicalLasso solver agreement')
  cv%mode='lars';cv%alphas=[.3_dp,.1_dp];cv%cv=3
  call cv%fit(x)
  call require(any(cv%alpha_==cv%alphas),'LARS CV selects supplied alpha')
  print *,'Gram LARS and graphical LARS tests passed'
end program
