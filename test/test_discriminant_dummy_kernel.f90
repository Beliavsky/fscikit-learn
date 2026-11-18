program test_discriminant_dummy_kernel
  use sklearn_common
  use sklearn_discriminant_analysis
  use sklearn_dummy
  use sklearn_kernel_ridge
  implicit none
  type(linear_discriminant_analysis) :: lda
  type(quadratic_discriminant_analysis) :: qda
  type(dummy_classifier) :: dc
  type(dummy_regressor) :: dr
  type(kernel_ridge) :: kr
  real(dp) :: x(20,2),y(20)
  integer :: labels(20),i
  real(dp), allocatable :: prob(:,:)
  do i=1,20
    x(i,:)=[real(i,dp)/5,cos(real(i,dp))]; y(i)=sin(x(i,1))
    labels(i)=merge(1,-1,i>10)
  end do
  call lda%fit(x,labels); call require(lda%score(x,labels)==1,'LDA predictions')
  prob=lda%predict_proba(x)
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-12_dp,'LDA probabilities')
  call qda%fit(x,labels); call require(qda%score(x,labels)==1,'QDA predictions')
  call dc%fit(x,labels); prob=dc%predict_proba(x)
  call require(all(prob==0.5_dp),'dummy prior probabilities')
  call dr%fit(x,y); call require(abs(dr%score(x,y))<1.e-12_dp,'mean dummy R2')
  kr%kernel='rbf'; kr%alpha=0.001_dp; call kr%fit(x,y)
  call require(kr%score(x,y)>0.999_dp,'kernel ridge nonlinear fit')
  print *, 'discriminant, dummy and kernel ridge: PASS'
end program
