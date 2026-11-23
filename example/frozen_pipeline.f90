program frozen_pipeline
  use sklearn, only: dp,standard_scaler,frozen_estimator,pipeline,linear_regression
  implicit none
  type(standard_scaler), target :: pretrained_scale
  type(frozen_estimator) :: fixed
  type(pipeline) :: model
  real(dp) :: original(6,2),training(6,2),y(6),mean_before(2)
  integer :: i
  do i=1,6
    original(i,:)=[real(i,dp),real(mod(i,2),dp)]
    y(i)=1+2*original(i,1)-.5_dp*original(i,2)
  end do
  call pretrained_scale%fit(original);mean_before=pretrained_scale%mean_
  fixed%estimator=>pretrained_scale
  allocate(model%steps(2));model%steps(1)%name='fixed_scale';model%steps(2)%name='regression'
  allocate(model%steps(1)%model,source=fixed)
  allocate(model%steps(2)%model,source=linear_regression())
  training=original+10;call model%fit(training,y)
  print *, 'Scaler mean before pipeline fit:',mean_before
  print *, 'Scaler mean after pipeline fit:',pretrained_scale%mean_
  print *, 'Pipeline training R2:',model%score(training,y)
end program
