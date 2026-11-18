program test_datasets_gp
  use sklearn_common
  use sklearn_datasets
  use sklearn_gaussian_process
  implicit none
  type(gaussian_process_regressor) :: gp
  real(dp), allocatable :: x(:,:),y(:),coef(:),cov(:,:),std(:),blobs(:,:)
  integer, allocatable :: labels(:)
  integer :: i
  call make_regression(40,3,x,y,noise=0._dp,random_state=4,coef=coef)
  call require(maxval(abs(y-matmul(x,coef)))<1.e-12_dp,'synthetic noiseless regression')
  call make_blobs(30,2,3,blobs,labels,random_state=7)
  call require(count(labels==0)==10.and.count(labels==2)==10,'blob balance')
  call make_moons(20,x,labels,shuffle=.false.)
  call require(maxval(abs(x(1,:)-[1._dp,0._dp]))<1.e-12_dp,'moon endpoint')
  call make_circles(20,x,labels,factor=0.5_dp,shuffle=.false.)
  call require(abs(sum(x(11,:)**2)-0.25_dp)<1.e-12_dp,'circle radius')
  deallocate(x); allocate(x(15,1))
  do i=1,15
    x(i,1)=real(i,dp)/5
  end do
  y=sin(x(:,1)); gp%normalize_y=.true.; call gp%fit(x,y)
  call require(gp%score(x,y)>0.999999_dp,'GP interpolation')
  std=gp%predict_std(x); cov=gp%predict_cov(x)
  call require(maxval(std)<1.e-4_dp,'GP training uncertainty')
  do i=1,15
    call require(abs(std(i)**2-cov(i,i))<1.e-12_dp,'GP variance consistency')
  end do
  print *, 'datasets and Gaussian process: PASS'
end program
