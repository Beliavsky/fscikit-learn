program test_parameter_space
  use sklearn_common, only: dp,require
  use sklearn_parameter_space
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  type(parameter_grid) :: grid,empty,large,grids(2)
  type(parameter_set), allocatable :: candidates(:),sampled(:)
  type(parameter_value) :: value
  integer :: i
  allocate(grid%axes(2));grid%axes(1)%name='z';grid%axes(2)%name='a'
  grid%axes(1)%values=[param(1),param(2),param(3)]
  grid%axes(2)%values=[param(.false.),param(.true.)]
  candidates=grid%enumerate()
  call require(size(candidates)==6,'Cartesian grid size')
  call require(candidates(1)%items(1)%name=='a','sorted parameter names')
  value=candidates(4)%get('a');call require(value%as_logical(),'rightmost parameter varies fastest')
  value=candidates(4)%get('z');call require(value%as_integer()==1,'Cartesian mixed-radix order')
  candidates=empty%enumerate()
  call require(size(candidates)==1.and.size(candidates(1)%items)==0,'empty dictionary has one candidate')
  grids(1)=grid;grids(2)=empty;candidates=enumerate_grids(grids)
  call require(size(candidates)==7.and.size(candidates(7)%items)==0,'union of grids preserves order')
  sampled=sample_parameters([grid],6,42)
  do i=1,6
    value=sampled(i)%get('z')
    call require(value%as_integer()==1+modulo(i-1,3),'complete finite sampling preserves enumeration')
  end do
  allocate(large%axes(2));large%axes(1)%name='a';large%axes(2)%name='b'
  large%axes(1)%values=[(param(i),i=1,1000)];large%axes(2)%values=large%axes(1)%values
  sampled=sample_parameters([large],3,0)
  call require(size(sampled)==3.and.large%length()==1000000,'large finite space is sampled lazily')
  grids(1)=empty;allocate(grids(1)%axes(1));grids(1)%axes(1)%name='alpha';grids(1)%axes(1)%rvs=>uniform_parameter
  sampled=sample_parameters(grids,10,0)
  call require(size(sampled)==10,'mixed distribution/list sampling with replacement')
  value=param('linear');call require(value%as_text()=='linear','text parameter')
  value=param(3);call require(value%as_real()==3._dp,'integer numeric conversion')
  print *, 'Parameter grids and samplers: PASS'
contains
  function uniform_parameter(rng) result(value)
    type(legacy_random_state), intent(inout) :: rng
    type(parameter_value) :: value
    value=param(rng%uniform())
  end function
end program
