program reference_parameter_space_driver
  use sklearn_common, only: dp
  use sklearn_parameter_space
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  type(parameter_grid), allocatable :: grids(:)
  type(parameter_set), allocatable :: candidates(:)
  type(parameter_value) :: value
  integer :: sampled,n_iter,seed,n_grids,n_axes,i,j,k,distribution,width,kind,integer_value
  real(dp) :: number
  character(256) :: text
  read(*,*) sampled,n_iter,seed,n_grids;allocate(grids(n_grids))
  do i=1,n_grids
    read(*,*) n_axes;allocate(grids(i)%axes(n_axes))
    do j=1,n_axes
      read(*,'(a)') text;grids(i)%axes(j)%name=trim(text)
      read(*,*) distribution,width
      if(distribution==1) then
        grids(i)%axes(j)%rvs=>uniform_parameter
      else
        allocate(grids(i)%axes(j)%values(width))
        do k=1,width
          read(*,*) kind
          select case(kind)
          case(1);read(*,*) number;value=param(number)
          case(2);read(*,*) integer_value;value=param(integer_value)
          case(3);read(*,*) integer_value;value=param(integer_value/=0)
          case(4);read(*,'(a)') text;value=param(trim(text))
          end select
          grids(i)%axes(j)%values(k)=value
        end do
      end if
    end do
  end do
  if(sampled==1) then
    candidates=sample_parameters(grids,n_iter,seed)
  else
    candidates=enumerate_grids(grids)
  end if
  write(*,*) size(candidates)
  do i=1,size(candidates)
    write(*,*) size(candidates(i)%items)
    do j=1,size(candidates(i)%items)
      write(*,'(a)') candidates(i)%items(j)%name
      value=candidates(i)%items(j)%value;write(*,*) value%kind
      select case(value%kind)
      case(1);write(*,'(es26.17e3)') value%real
      case(2);write(*,*) value%integer
      case(3);write(*,*) merge(1,0,value%logical)
      case(4);write(*,'(a)') value%text
      end select
    end do
  end do
contains
  function uniform_parameter(rng) result(value)
    type(legacy_random_state), intent(inout) :: rng
    type(parameter_value) :: value
    value=param(rng%uniform())
  end function
end program
