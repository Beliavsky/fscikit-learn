! sklearn 1.6.1 model_selection/_search.py and utils/_random.pyx (BSD-3-Clause).
! Typed parameter storage/adapters are original MIT contributions.
module sklearn_parameter_space
  use sklearn_common, only: dp,require
  use sklearn_legacy_rng, only: legacy_random_state
  implicit none
  private
  public :: parameter_value,parameter_item,parameter_set,parameter_axis,parameter_grid
  public :: param,enumerate_grids,sample_parameters
  type :: parameter_value
    integer :: kind=0,integer=0
    real(dp) :: real=0
    logical :: logical=.false.
    character(:), allocatable :: text
  contains
    procedure :: as_real,as_integer,as_logical,as_text
  end type
  abstract interface
    function parameter_distribution(rng) result(value)
      import legacy_random_state,parameter_value
      type(legacy_random_state), intent(inout) :: rng
      type(parameter_value) :: value
    end function
  end interface
  type :: parameter_item
    character(:), allocatable :: name
    type(parameter_value) :: value
  end type
  type :: parameter_set
    type(parameter_item), allocatable :: items(:)
  contains
    procedure :: find=>find_parameter
    procedure :: get=>get_parameter
  end type
  type :: parameter_axis
    character(:), allocatable :: name
    type(parameter_value), allocatable :: values(:)
    procedure(parameter_distribution), pointer, nopass :: rvs=>null()
  end type
  type :: parameter_grid
    type(parameter_axis), allocatable :: axes(:)
  contains
    procedure :: enumerate=>enumerate_grid
    procedure :: length=>grid_length
    procedure :: at=>grid_item
  end type
  interface param
    module procedure real_param,integer_param,logical_param,text_param
  end interface
contains
  function real_param(number) result(value)
    real(dp), intent(in) :: number
    type(parameter_value) :: value
    value%kind=1;value%real=number
  end function
  function integer_param(number) result(value)
    integer, intent(in) :: number
    type(parameter_value) :: value
    value%kind=2;value%integer=number
  end function
  function logical_param(flag) result(value)
    logical, intent(in) :: flag
    type(parameter_value) :: value
    value%kind=3;value%logical=flag
  end function
  function text_param(text) result(value)
    character(*), intent(in) :: text
    type(parameter_value) :: value
    value%kind=4;value%text=text
  end function
  real(dp) function as_real(self) result(value)
    class(parameter_value), intent(in) :: self
    call require(self%kind==1.or.self%kind==2,'parameter value must be numeric')
    value=self%real;if(self%kind==2) value=real(self%integer,dp)
  end function
  integer function as_integer(self) result(value)
    class(parameter_value), intent(in) :: self
    call require(self%kind==2,'parameter value must be integer')
    value=self%integer
  end function
  logical function as_logical(self) result(value)
    class(parameter_value), intent(in) :: self
    call require(self%kind==3,'parameter value must be logical')
    value=self%logical
  end function
  function as_text(self) result(value)
    class(parameter_value), intent(in) :: self
    character(:), allocatable :: value
    call require(self%kind==4,'parameter value must be text')
    call require(allocated(self%text),'parameter text is unallocated');value=self%text
  end function
  integer function find_parameter(self,name) result(index)
    class(parameter_set), intent(in) :: self
    character(*), intent(in) :: name
    integer :: i
    index=0;if(.not.allocated(self%items)) return
    do i=1,size(self%items)
      if(.not.allocated(self%items(i)%name)) cycle
      if(len(name)/=len(self%items(i)%name)) cycle
      if(name==self%items(i)%name) then
        index=i;return
      end if
    end do
  end function
  function get_parameter(self,name) result(value)
    class(parameter_set), intent(in) :: self
    character(*), intent(in) :: name
    type(parameter_value) :: value
    integer :: i
    i=self%find(name);call require(i>0,'parameter name was not found');value=self%items(i)%value
  end function
  function sorted_axes(grid,allow_distributions) result(order)
    class(parameter_grid), intent(in) :: grid
    logical, intent(in) :: allow_distributions
    integer, allocatable :: order(:)
    integer :: n,i,j,key
    n=0;if(allocated(grid%axes)) n=size(grid%axes)
    order=[(i,i=1,n)]
    do i=1,n
      call require(allocated(grid%axes(i)%name),'parameter axis name is unallocated')
      call require(.not.(associated(grid%axes(i)%rvs).and.allocated(grid%axes(i)%values)), &
        'parameter axis must choose values or a distribution')
      if(associated(grid%axes(i)%rvs)) then
        call require(allow_distributions,'ParameterGrid does not accept distributions')
      else
        call require(allocated(grid%axes(i)%values),'parameter axis values are unallocated')
        call require(size(grid%axes(i)%values)>0,'parameter axis values must be nonempty')
        do j=1,size(grid%axes(i)%values)
          call validate_value(grid%axes(i)%values(j))
        end do
      end if
    end do
    do i=2,n
      key=order(i);j=i-1
      do while(j>=1)
        if(.not.name_less(grid%axes(key)%name,grid%axes(order(j))%name)) exit
        order(j+1)=order(j);j=j-1
      end do
      order(j+1)=key
    end do
    do i=2,n
      call require(grid%axes(order(i))%name/=grid%axes(order(i-1))%name.or. &
        len(grid%axes(order(i))%name)/=len(grid%axes(order(i-1))%name),'parameter axis names must be unique')
    end do
  end function
  logical function name_less(a,b) result(less)
    character(*), intent(in) :: a,b
    integer :: i
    do i=1,min(len(a),len(b))
      if(iachar(a(i:i))==iachar(b(i:i))) cycle
      less=iachar(a(i:i))<iachar(b(i:i));return
    end do
    less=len(a)<len(b)
  end function
  subroutine validate_value(value)
    type(parameter_value), intent(in) :: value
    call require(value%kind>=1.and.value%kind<=4,'parameter value has invalid kind')
    if(value%kind==4) call require(allocated(value%text),'parameter text is unallocated')
  end subroutine
  integer function grid_length(self) result(total)
    class(parameter_grid), intent(in) :: self
    integer, allocatable :: order(:)
    integer :: i,width
    order=sorted_axes(self,.false.);total=1
    do i=1,size(order)
      width=size(self%axes(order(i))%values)
      call require(total<=huge(1)/width,'parameter grid size exceeds native integer range')
      total=total*width
    end do
  end function
  function grid_item(self,index) result(candidate)
    class(parameter_grid), intent(in) :: self
    integer, intent(in) :: index
    type(parameter_set) :: candidate
    integer, allocatable :: order(:)
    integer :: j,width,pick,offset
    call require(index>=1.and.index<=self%length(),'parameter grid index out of bounds')
    order=sorted_axes(self,.false.);allocate(candidate%items(size(order)));offset=index-1
    do j=size(order),1,-1
      associate(axis=>self%axes(order(j)))
        width=size(axis%values);pick=modulo(offset,width)+1;offset=offset/width
        candidate%items(j)%name=axis%name;candidate%items(j)%value=axis%values(pick)
      end associate
    end do
  end function
  function enumerate_grid(self) result(candidates)
    class(parameter_grid), intent(in) :: self
    type(parameter_set), allocatable :: candidates(:)
    integer :: i
    allocate(candidates(self%length()))
    do i=1,size(candidates)
      candidates(i)=self%at(i)
    end do
  end function
  function enumerate_grids(grids) result(candidates)
    type(parameter_grid), intent(in) :: grids(:)
    type(parameter_set), allocatable :: candidates(:),part(:)
    integer :: i
    allocate(candidates(0))
    do i=1,size(grids)
      part=grids(i)%enumerate()
      call require(size(part)<=huge(1)-size(candidates),'parameter grid size exceeds native integer range')
      candidates=[candidates,part]
    end do
  end function
  function sample_parameters(grids,n_iter,random_state) result(candidates)
    type(parameter_grid), intent(in) :: grids(:)
    integer, intent(in) :: n_iter,random_state
    type(parameter_set), allocatable :: candidates(:)
    type(legacy_random_state) :: rng
    integer, allocatable :: order(:),indices(:),pool(:),counts(:)
    integer :: i,j,k,n,total,pick,temporary
    real(dp) :: ratio
    logical :: all_lists
    call require(n_iter>0,'ParameterSampler n_iter must be positive')
    call require(size(grids)>0,'ParameterSampler requires at least one grid')
    call rng%seed(random_state);all_lists=.true.
    do i=1,size(grids)
      order=sorted_axes(grids(i),.true.)
      do j=1,size(order)
        if(associated(grids(i)%axes(order(j))%rvs)) all_lists=.false.
      end do
    end do
    if(all_lists) then
      allocate(counts(size(grids)));total=0
      do i=1,size(grids)
        counts(i)=grids(i)%length()
        call require(counts(i)<=huge(1)-total,'parameter grid size exceeds native integer range')
        total=total+counts(i)
      end do
      n=min(n_iter,total);allocate(indices(n))
      ratio=real(n,dp)/total
      if(ratio>.01_dp.and.ratio<.99_dp) then
        pool=[(i,i=1,total)]
        do i=total,2,-1
          pick=rng%integer_index(i);temporary=pool(i);pool(i)=pool(pick);pool(pick)=temporary
        end do
        indices=pool(:n)
      else if(ratio<.2_dp) then
        do i=1,n
          do
            pick=rng%integer_index(total)
            if(.not.any(indices(:i-1)==pick)) exit
          end do
          indices(i)=pick
        end do
      else
        indices=[(i,i=1,n)]
        do i=n+1,total
          pick=rng%integer_index(i)
          if(pick<=n) indices(pick)=i
        end do
      end if
      allocate(candidates(n))
      do i=1,n
        pick=indices(i);k=1
        do while(pick>counts(k))
          pick=pick-counts(k);k=k+1
        end do
        candidates(i)=grids(k)%at(pick)
      end do
    else
      allocate(candidates(n_iter))
      do i=1,n_iter
        k=rng%integer_index(size(grids));order=sorted_axes(grids(k),.true.)
        allocate(candidates(i)%items(size(order)))
        do j=1,size(order)
          associate(axis=>grids(k)%axes(order(j)))
            candidates(i)%items(j)%name=axis%name
            if(associated(axis%rvs)) then
              candidates(i)%items(j)%value=axis%rvs(rng)
              call validate_value(candidates(i)%items(j)%value)
            else
              pick=rng%integer_index(size(axis%values));candidates(i)%items(j)%value=axis%values(pick)
            end if
          end associate
        end do
      end do
    end if
  end function
end module
