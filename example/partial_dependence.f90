program inspect_dependence
  use sklearn, only: dp,ridge,partial_dependence,partial_dependence_result
  implicit none
  type(ridge) :: model
  type(partial_dependence_result) :: dependence
  real(dp) :: x(12,2),y(12)
  integer :: i
  x(:,1)=[(real(i,dp),i=1,12)]; x(:,2)=[(real(mod(i,3),dp),i=1,12)]
  y=2*x(:,1)+sin(x(:,2)); model%alpha=.7_dp
  call model%fit(x,y)
  dependence=partial_dependence(x,[1],predict,grid_resolution=5,percentiles=[0._dp,1._dp])
  print *, 'Dependence grid:',dependence%grid_values(1)%values
  print *, 'Mean response:',dependence%average(1,:)
contains
  function predict(features) result(values)
    real(dp), intent(in) :: features(:,:)
    real(dp), allocatable :: values(:,:)
    allocate(values(size(features,1),1)); values(:,1)=model%predict(features)
  end function
end program
