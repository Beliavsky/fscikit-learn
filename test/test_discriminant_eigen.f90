program test_discriminant_eigen
  use sklearn_common, only: dp,require
  use sklearn_discriminant_analysis, only: linear_discriminant_analysis
  implicit none
  type(linear_discriminant_analysis) :: model,least,small
  real(dp) :: x(18,5),total(5,5),between(5,5),trace
  integer :: y(18),i,j
  real(dp), allocatable :: z(:,:),white(:,:),residual(:,:)
  do i=1,18
    y(i)=mod(i,3)*4-2
    do j=1,3
      x(i,j)=sin(.71_dp*i*j)+y(i)*.3_dp*j
    end do
  end do
  x(:,4)=x(:,1);x(:,5)=7
  model%solver='eigen';model%shrinkage=.2_dp;z=model%fit_transform(x,y)
  call require(all(shape(model%scalings_)==[5,5]).and.size(z,2)==2,'eigen LDA stores full covariance-normalized basis')
  white=matmul(transpose(model%scalings_),matmul(model%covariance_(:,:,1),model%scalings_))
  do j=1,5
    white(j,j)=white(j,j)-1
  end do
  call require(maxval(abs(white))<1.e-10_dp,'eigen LDA axes normalize within covariance')
  white=x-spread(sum(x,dim=1)/18,1,18);total=matmul(transpose(white),white)/18;trace=0
  do j=1,5
    trace=trace+total(j,j)/5
  end do
  total=.8_dp*total
  do j=1,5
    total(j,j)=total(j,j)+.2_dp*trace
  end do
  between=total-model%covariance_(:,:,1)
  residual=matmul(between,model%scalings_)- &
    matmul(model%covariance_(:,:,1),model%scalings_)*spread(model%eigenvalues_,1,5)
  call require(maxval(abs(residual))<1.e-10_dp,'eigen LDA generalized eigenproblem')
  call require(maxval(abs(z-matmul(x,model%scalings_(:,:2))))<1.e-12_dp,'eigen LDA projection is uncentered')
  least%shrinkage=.2_dp;call least%fit(x,y)
  call require(maxval(abs(least%coef_-model%coef_))<1.e-10_dp,'eigen and LSQR LDA coefficients agree')
  small=model;small%n_components=1;call small%fit(x,y)
  call require(maxval(abs(small%predict_proba(x)-model%predict_proba(x)))<1.e-12_dp, &
    'eigen LDA transform cap does not reduce classification basis')
  model%solver='svd';model%shrinkage=0;call model%fit(x,y)
  call require(.not.allocated(model%eigenvalues_),'SVD refit clears eigen diagnostics')
  print *, 'Eigen linear discriminant analysis: PASS'
end program
