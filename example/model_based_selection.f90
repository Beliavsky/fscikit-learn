program model_based_selection
  use sklearn, only: dp,select_from_model,ridge,pipeline,standard_scaler
  implicit none
  type(select_from_model) :: selection
  type(ridge) :: base
  type(standard_scaler) :: scale
  type(pipeline) :: model
  real(dp) :: x(40,5),y(40)
  integer, allocatable :: retained(:)
  integer :: i
  do i=1,40
    x(i,:)=[sin(real(i,dp)),cos(real(2*i,dp)),sin(real(3*i,dp)),cos(real(4*i,dp)),sin(real(5*i,dp))]
  end do
  y=3*x(:,2)-2*x(:,4);base%alpha=.1_dp
  allocate(selection%estimator,source=base)
  allocate(model%steps(3))
  model%steps(1)%name='scale';allocate(model%steps(1)%model,source=scale)
  model%steps(2)%name='select';allocate(model%steps(2)%model,source=selection)
  model%steps(3)%name='regress';allocate(model%steps(3)%model,source=base)
  call model%fit(x,y)
  select type(fitted=>model%steps(2)%model)
  type is(select_from_model)
    retained=fitted%get_support_indices()
  end select
  write(*,'(a,*(i0,1x))') 'Retained columns: ',retained
  write(*,'(a,f10.7)') 'Training R2: ',model%score(x,y)
end program
