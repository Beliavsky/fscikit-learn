program test_gp_kernels
  use sklearn_gp_kernels
  use sklearn_common, only: dp,require
  implicit none
  type(gp_kernel) :: kernel,copy,up,down
  real(dp) :: x(6,2)
  real(dp), allocatable :: value(:,:),cross(:,:),gradient(:,:,:),theta(:),plus(:),minus(:),finite(:,:),diagonal(:)
  integer :: i,j
  do i=1,6
    x(i,:)=[real(i,dp)/5,sin(real(i,dp))]
  end do
  kernel=2._dp*rbf_covariance([.7_dp,1.2_dp])+white_covariance(.1_dp)
  value=kernel%matrix(x);cross=kernel%matrix(x,x);diagonal=kernel%diag(x)
  call require(kernel%n_dims()==4,'kernel log-parameter dimension includes scalar constants')
  call require(maxval(abs(kernel%theta()-log([2._dp,.7_dp,1.2_dp,.1_dp])))<1.e-14_dp,'kernel theta order')
  do i=1,6
    call require(abs(value(i,i)-2.1_dp)<1.e-14_dp.and.abs(cross(i,i)-2._dp)<1.e-14_dp, &
      'white noise contributes to auto covariance, never cross covariance')
    call require(abs(diagonal(i)-value(i,i))<1.e-14_dp,'efficient kernel diagonal')
  end do
  gradient=kernel%gradient(x);theta=kernel%theta()
  do j=1,size(theta)
    plus=theta;minus=theta;plus(j)=plus(j)+1.e-5_dp;minus(j)=minus(j)-1.e-5_dp
    up=kernel%with_theta(plus);down=kernel%with_theta(minus)
    finite=(up%matrix(x)-down%matrix(x))/2.e-5_dp
    call require(maxval(abs(finite-gradient(:,:,j)))<1.e-8_dp,'composed kernel log gradients')
  end do
  copy=kernel%with_theta(theta+.1_dp)
  call require(maxval(abs(kernel%matrix(x)-value))==0,'kernel copies have independent parameters')
  kernel=(rbf_covariance(.8_dp,.true.)+white_covariance(.03_dp,.true.))**2
  gradient=kernel%gradient(x);call require(size(gradient,3)==0,'fixed kernel has zero gradient width')
  kernel=matern_covariance([.7_dp,1.2_dp],nu=.7_dp)
  value=kernel%matrix(x)
  call require(maxval(abs(value-transpose(value)))<1.e-14_dp,'general Matern symmetry')
  call require(all(value>0).and.all(value<=1+1.e-14_dp),'general Matern correlation range')
  print *, 'Gaussian process covariance kernels: PASS'
end program
