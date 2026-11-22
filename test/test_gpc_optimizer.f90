program test_gpc_optimizer
  use sklearn_gp_classification, only: gaussian_process_classifier
  use sklearn_gp_kernels
  use sklearn_common, only: dp,require
  implicit none
  type(gaussian_process_classifier) :: model
  type(gp_kernel) :: kernel
  real(dp) :: x(18,1),bounds(3,2)
  real(dp), allocatable :: theta(:),probability(:,:),refit(:,:)
  integer :: i,k,labels(18)
  do i=1,18
    x(i,1)=-1.5_dp+3*real(i-1,dp)/17
    labels(i)=merge(-4,merge(9,17,i<=12),i<=6)
  end do
  kernel=1.2_dp*rbf_covariance(.8_dp)+white_covariance(.05_dp)
  bounds(:,1)=log([.1_dp,.1_dp,.0001_dp]);bounds(:,2)=log([4._dp,4._dp,.4_dp])
  model%kernel=kernel%with_bounds(bounds);model%optimizer='fmin_l_bfgs_b'
  model%warm_start=.true.;model%n_restarts_optimizer=1;model%random_state=42
  call model%fit(x,labels)
  call require(size(model%kernels_)==3.and.model%n_optimization_runs_==2,'GPC independent OVR kernel optimization')
  do k=1,3
    theta=model%kernels_(k)%theta()
    call require(all(theta>=bounds(:,1)).and.all(theta<=bounds(:,2)),'GPC optimized theta satisfies bounds')
    call require(model%n_function_evaluations_(k)>0,'GPC optimizer evaluation metadata')
    call require(maxval(abs(model%optimization_initial_points_(2,:,k)- &
      [0.39813993347866738_dp,3.3350423523880255_dp,0.043319764968450393_dp]))<1.e-13_dp, &
      'GPC sklearn 1.6.1 exponential restart-theta convention')
  end do
  call require(maxval(abs(model%kernels_(1)%theta()-model%kernels_(2)%theta()))>1.e-4_dp, &
    'GPC heads retain distinct learned kernels')
  probability=model%predict_proba(x)
  call require(maxval(abs(sum(probability,dim=2)-1))<1.e-14_dp,'GPC multiclass probabilities normalize')
  call require(model%score(x,labels)>.9_dp,'GPC learned training classification')
  call model%fit(x,labels);refit=model%predict_proba(x)
  call require(maxval(abs(refit-probability))<1.e-12_dp,'GPC repeated fits reset optimizer and warm caches')
  print *, 'GPC kernel learning and warm starts: PASS'
end program
