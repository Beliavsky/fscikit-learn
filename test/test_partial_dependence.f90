program test_partial_dependence
  use sklearn, only: dp,partial_dependence,partial_dependence_result
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_nan
  implicit none
  type(partial_dependence_result) :: result,unweighted,saved
  real(dp) :: x(8,3),original(8,3),weights(8)=1,expected
  integer :: i,r,calls=0
  x(:,1)=[(real(i,dp),i=1,8)]; x(:,2)=[(real(mod(i,2),dp),i=1,8)]; x(:,3)=x(:,1)**2
  original=x
  result=partial_dependence(x,[1],predict,grid_resolution=4,percentiles=[0._dp,1._dp],kind='both')
  call require(calls==4,'partial dependence evaluates one response matrix per grid point')
  call require(all(shape(result%average)==[2,4]),'partial dependence average layout')
  call require(all(shape(result%individual)==[2,8,4]),'partial dependence individual layout')
  do r=1,4
    expected=result%grid_values(1)%values(r)+sum(2*x(:,2)*x(:,3))/8
    call require(abs(result%average(1,r)-expected)<1.e-12_dp,'partial dependence averages substituted features')
    call require(maxval(abs(result%individual(2,:,r)-x(:,3)*result%grid_values(1)%values(r)))<1.e-12_dp, &
      'partial dependence individual curves preserve remaining features')
  end do
  saved=result; unweighted=result; weights(1)=-.1_dp
  result=partial_dependence(x,[1],predict,grid_resolution=4,percentiles=[0._dp,1._dp],kind='both',sample_weight=weights)
  call require(all(result%individual==unweighted%individual),'partial dependence weights leave individual curves unchanged')
  call require(all(saved%average==unweighted%average),'partial dependence result assignment deep copies')
  result=partial_dependence(x,[1,1],predict,grid_resolution=3,percentiles=[0._dp,1._dp])
  call require(size(result%average,2)==9,'duplicate target features retain Cartesian axes')
  call require(all(result%average(:,1)==result%average(:,4)).and.all(result%average(:,4)==result%average(:,7)), &
    'duplicate target features follow the last substitution')
  call require(.not.allocated(result%individual),'average-only results omit individual data')
  result=partial_dependence(x,[1],predict,grid_resolution=3,categorical_features=[1],kind='individual')
  call require(size(result%grid_values(1)%values)==8,'categorical grids use all unique values')
  call require(.not.allocated(result%average),'individual-only results omit the averaged field')
  result=partial_dependence(x,[2],predict,grid_resolution=2,categorical_features=[.false.,.true.,.false.])
  call require(all(result%grid_values(1)%values==[0._dp,1._dp]),'categorical boolean masks follow original features')
  call require(all(x==original),'partial dependence preserves original inputs')
  x(1,1)=ieee_value(0._dp,ieee_quiet_nan)
  result=partial_dependence(x,[1],predict,grid_resolution=3,kind='both')
  call require(all(ieee_is_nan(result%grid_values(1)%values)),'NaN quantile endpoints propagate into source grids')
  print *, 'partial dependence: PASS'
contains
  function predict(features) result(values)
    real(dp), intent(in) :: features(:,:)
    real(dp), allocatable :: values(:,:),clean(:,:)
    calls=calls+1; clean=features
    where(ieee_is_nan(clean)) clean=0
    allocate(values(size(features,1),2))
    values(:,1)=clean(:,1)+2*clean(:,2)*clean(:,3)
    values(:,2)=clean(:,1)*clean(:,3)
  end function
end program
