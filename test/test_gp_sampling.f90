program test_gp_sampling
  use sklearn_common, only: dp,require
  use sklearn_gaussian_process, only: gaussian_process_regressor
  use sklearn_gp_kernels
  implicit none
  type(gaussian_process_regressor) :: model
  real(dp) :: x(5,1),query(3,1),y(5,2)
  real(dp), allocatable :: a(:,:),b(:,:),multi(:,:,:),again(:,:,:)
  integer, allocatable :: before(:),after(:)
  integer :: i,n
  x(:,1)=[-1._dp,-.5_dp,0._dp,.5_dp,1._dp];query(:,1)=[-.8_dp,.1_dp,.7_dp]
  model%kernel=rbf_covariance(.7_dp)+white_covariance(.03_dp);model%random_state=99
  call random_seed(size=n);allocate(before(n),after(n));call random_seed(get=before)
  a=model%sample_y(query,7);b=model%sample_y(query,7,0)
  call random_seed(get=after)
  call require(all(before==after),'GP sampling leaves intrinsic RNG untouched')
  call require(all(shape(a)==[3,7]).and.all(a==b),'GP sampling default seed and shape')
  b=model%sample_y(query,7,42);call require(any(a/=b),'GP sampling seed changes draws')
  b=model%sample_y(query,0);call require(all(shape(b)==[3,0]),'GP empty sample axis')
  model%n_targets=2;multi=model%sample_y_multi(query,7,42);again=model%sample_y_multi(query,7,42)
  call require(all(shape(multi)==[3,2,7]).and.all(multi==again),'GP multi-target prior reproducibility')
  call require(any(multi(:,1,:)/=multi(:,2,:)),'GP prior targets have independent draws')
  do i=1,5
    y(i,:)=[sin(x(i,1)),10+2*x(i,1)]
  end do
  model%normalize_y=.true.;call model%fit(x,y)
  multi=model%sample_y_multi(query,7,42);again=model%sample_y_multi(query,7,42)
  call require(all(multi==again),'GP multi-target posterior reproducibility')
  model=gaussian_process_regressor();model%kernel=constant_covariance(0._dp,.true.)
  a=model%sample_y(query,7,42);call require(all(a==0),'GP zero-covariance prior samples')
  model%kernel=constant_covariance(1._dp,.true.);a=model%sample_y(query,7,42)
  call require(maxval(abs(a(1,:)-a(2,:)))<1.e-7_dp,'GP singular prior repeats equal query values')
  print *, 'Gaussian process prior and posterior sampling: PASS'
end program
