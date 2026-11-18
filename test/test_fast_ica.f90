program test_fast_ica
  use sklearn
  use sklearn_common, only: require
  implicit none
  type(fast_ica) :: model,repeat
  real(dp) :: sources(1000,2),x(1000,2),t,corr(2,2)
  real(dp), allocatable :: z(:,:),restored(:,:),again(:,:)
  integer :: i,j
  do i=1,1000
    t=real(i,dp)/1000
    sources(i,1)=sin(31*t)
    sources(i,2)=merge(1._dp,-1._dp,sin(73*t)>0)
  end do
  do j=1,2
    sources(:,j)=sources(:,j)-sum(sources(:,j))/1000
    sources(:,j)=sources(:,j)/sqrt(sum(sources(:,j)**2)/1000)
  end do
  x(:,1)=sources(:,1)+0.5_dp*sources(:,2)+3
  x(:,2)=0.2_dp*sources(:,1)+sources(:,2)-1
  model%tol=1.e-8_dp; model%max_iter=1000; model%random_state=42
  z=model%fit_transform(x); restored=model%inverse_transform(z)
  call require(model%converged_,'ICA source recovery converges')
  call require(maxval(abs(restored-x))<1.e-10_dp,'ICA full reconstruction')
  call require(maxval(abs(sum(z,dim=1)))<1.e-10_dp,'ICA centered sources')
  call require(maxval(abs(sum(z*z,dim=1)/1000-1))<1.e-10_dp,'ICA unit variance')
  corr=abs(matmul(transpose(z),sources)/1000)
  call require(all(maxval(corr,dim=1)>0.99_dp),'ICA recovers independent sources')
  repeat=model; again=repeat%fit_transform(x)
  call require(maxval(abs(z-again))<1.e-12_dp,'ICA random seed reproducibility')
  model%n_components=1; z=model%fit_transform(x); restored=model%inverse_transform(z)
  again=model%transform(restored)
  call require(maxval(abs(again-z))<1.e-10_dp,'ICA reduced inverse is a projection')
  model%n_components=0; model%algorithm='deflation'; z=model%fit_transform(x)
  call require(model%converged_,'ICA deflation converges')
  model%whiten='arbitrary-variance'; z=model%fit_transform(x)
  call require(maxval(abs(sum(z*z,dim=1)-1))<1.e-10_dp,'ICA arbitrary variance scaling')
  print *, 'FastICA tests passed'
end program
