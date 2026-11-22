program test_discriminant_svd
  use sklearn_common, only: dp,require
  use sklearn_discriminant_analysis, only: linear_discriminant_analysis
  implicit none
  type(linear_discriminant_analysis) :: model,small
  real(dp) :: x(18,5),cov(5,5),delta(5)
  integer :: y(18),i,j,c
  real(dp), allocatable :: z(:,:),prob(:,:),white(:,:)
  do i=1,18
    y(i)=mod(i,3)*4-2
    do j=1,3
      x(i,j)=sin(.71_dp*i*j)+y(i)*.3_dp*j
    end do
  end do
  x(:,4)=x(:,1);x(:,5)=7
  model%solver='svd';z=model%fit_transform(x,y)
  call require(model%within_rank_==3.and.model%n_components_==2,'SVD discriminant rank-deficient fit')
  call require(maxval(abs(z-model%transform(x)))<1.e-12_dp,'SVD discriminant fit and query coordinates')
  cov=0
  do i=1,18
    c=findloc(model%classes_,y(i),dim=1);delta=x(i,:)-model%means_(c,:)
    do j=1,5
      cov(:,j)=cov(:,j)+delta*delta(j)/15
    end do
  end do
  white=matmul(transpose(model%scalings_),matmul(cov,model%scalings_))
  call require(maxval(abs(white-reshape([1._dp,0._dp,0._dp,1._dp],[2,2])))<1.e-10_dp, &
    'SVD discriminant axes whiten unbiased within-class covariance')
  small=model;small%n_components=1;call small%fit(x,y)
  prob=model%predict_proba(x)
  call require(maxval(abs(prob-small%predict_proba(x)))<1.e-12_dp,'component cap affects transform, not classification')
  model%solver='lsqr';model%shrinkage=.2_dp;call model%fit(x,y)
  call require(.not.allocated(model%scalings_).and..not.allocated(model%xbar_), 'LSQR refit clears SVD diagnostics')
  print *, 'SVD linear discriminant analysis: PASS'
end program
