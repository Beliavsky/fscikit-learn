program parameter_search
  use sklearn, only: dp,grid_search_cv,ridge,param
  implicit none
  type(grid_search_cv) :: search
  type(ridge) :: base
  real(dp) :: x(40,3),y(40)
  integer :: i
  do i=1,40
    x(i,:)=[sin(real(i,dp)),cos(real(2*i,dp)),sin(real(3*i,dp))]
  end do
  y=2*x(:,1)-x(:,2)+.2_dp*x(:,3)
  allocate(search%estimator,source=base)
  allocate(search%param_grid(1));allocate(search%param_grid(1)%axes(1))
  search%param_grid(1)%axes(1)%name='alpha'
  search%param_grid(1)%axes(1)%values=[param(.01_dp),param(.3_dp),param(3._dp)]
  search%cv=4;search%scoring='neg_mean_squared_error'
  call search%fit(x,y)
  select type(fitted=>search%best_estimator_)
  type is(ridge)
    write(*,'(a,f7.3)') 'Selected alpha: ',fitted%alpha
  end select
  write(*,'(a,i0)') 'Candidates: ',size(search%cv_results_%params)
  write(*,'(a,es14.6)') 'Best mean CV score: ',search%best_score_
end program
