! Brute partial dependence adapted from sklearn 1.6.1 inspection/_partial_dependence.py.
! Quantile grid: SciPy 1.15.3 stats/_mstats_basic.py mquantiles (alphap=betap=.4).
! Retained notices: THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_partial_dependence
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value, &
    ieee_quiet_nan,ieee_positive_inf,ieee_negative_inf
  implicit none
  private
  public :: partial_dependence,partial_dependence_result,dependence_axis,dependence_predictor
  type :: dependence_axis
    real(dp), allocatable :: values(:)
  end type
  type :: partial_dependence_result
    type(dependence_axis), allocatable :: grid_values(:)
    real(dp), allocatable :: average(:,:),individual(:,:,:)
  end type
  abstract interface
    function dependence_predictor(features) result(values)
      import dp
      real(dp), intent(in) :: features(:,:)
      real(dp), allocatable :: values(:,:)
    end function
  end interface
contains
  function partial_dependence(x,features,predictor,percentiles,grid_resolution,categorical_features, &
      kind,sample_weight) result(results)
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: features(:)
    procedure(dependence_predictor) :: predictor
    real(dp), intent(in), optional :: percentiles(:),sample_weight(:)
    integer, intent(in), optional :: grid_resolution
    class(*), intent(in), optional :: categorical_features(:)
    character(*), intent(in), optional :: kind
    type(partial_dependence_result) :: results
    real(dp), allocatable :: ordered(:),unique(:),work(:,:),response(:,:),weights(:)
    logical, allocatable :: categorical(:)
    integer, allocatable :: indices(:)
    real(dp) :: limits(2),lo,hi,step,value,mass
    integer :: resolution,points,j,r,k,id,outputs,rows,columns
    character(:), allocatable :: which
    call require(size(x,1)>0.and.size(x,2)>0,'partial dependence features must be nonempty')
    call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'partial dependence features contain infinity')
    rows=size(x,1); columns=size(x,2)
    call require(size(features)>0,'partial dependence needs at least one target feature')
    call require(all(features>=1).and.all(features<=columns),'partial dependence feature indices must be in 1..n_features')
    limits=[.05_dp,.95_dp]
    if(present(percentiles)) then
      call require(size(percentiles)==2,'partial dependence percentiles need two values')
      limits=percentiles
    end if
    call require(all(ieee_is_finite(limits)),'partial dependence percentiles must be finite')
    call require(all(limits>=0).and.all(limits<=1),'partial dependence percentiles must be in [0,1]')
    call require(limits(1)<limits(2),'partial dependence percentiles must increase')
    resolution=100; if(present(grid_resolution)) resolution=grid_resolution
    call require(resolution>1,'partial dependence grid resolution must exceed one')
    which='average'; if(present(kind)) which=trim(kind)
    call require(any(which==[character(10) :: 'average','individual','both']),'unknown partial dependence kind')
    allocate(weights(rows)); weights=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==rows,'partial dependence sample weight length mismatch')
      call require(all(ieee_is_finite(sample_weight)),'partial dependence sample weights must be finite')
      weights=sample_weight
    end if
    mass=total(weights)
    if(.not.ieee_is_nan(mass)) &
      call require(mass/=0,'partial dependence sample weights must have a nonzero sum')
    allocate(categorical(size(features))); categorical=.false.
    if(present(categorical_features)) then
      select type(categorical_features)
      type is(logical)
        call require(size(categorical_features)==columns,'partial dependence categorical mask length mismatch')
        categorical=categorical_features(features)
      type is(integer)
        do j=1,size(features)
          categorical(j)=any(categorical_features==features(j))
        end do
      type is(integer(int64))
        do j=1,size(features)
          categorical(j)=any(categorical_features==features(j))
        end do
      class default
        call require(.false.,'partial dependence categorical features must be logical flags or integer indices')
      end select
    end if
    allocate(results%grid_values(size(features)),indices(size(features))); points=1
    do j=1,size(features)
      ordered=sorted(x(:,features(j))); unique=distinct(ordered)
      if(categorical(j).or.size(unique)<resolution) then
        results%grid_values(j)%values=unique
      else
        lo=quantile(ordered,limits(1)); hi=quantile(ordered,limits(2))
        if(.not.ieee_is_nan(lo).and..not.ieee_is_nan(hi)) then
          if(ieee_is_finite(lo).and.ieee_is_finite(hi)) then
            call require(abs(difference(lo,hi))>1.e-8_dp+1.e-5_dp*abs(hi), &
              'partial dependence percentiles are too close to build a grid')
          else
            call require(lo/=hi,'partial dependence percentiles are too close to build a grid')
          end if
        end if
        allocate(results%grid_values(j)%values(resolution))
        step=quotient(difference(hi,lo),real(resolution-1,dp))
        do r=1,resolution
          results%grid_values(j)%values(r)=difference(product(real(r-1,dp),step),-lo)
        end do
        results%grid_values(j)%values(resolution)=hi
      end if
      call require(points<=huge(1)/size(results%grid_values(j)%values),'partial dependence grid exceeds integer range')
      points=points*size(results%grid_values(j)%values)
    end do
    work=x; outputs=0
    do r=1,points
      id=r-1
      do j=size(features),1,-1
        indices(j)=mod(id,size(results%grid_values(j)%values))+1
        id=id/size(results%grid_values(j)%values)
      end do
      do j=1,size(features)
        work(:,features(j))=results%grid_values(j)%values(indices(j))
      end do
      response=predictor(work)
      call require(allocated(response),'partial dependence predictor returned an unallocated response')
      call require(size(response,1)==rows,'partial dependence predictor row count mismatch')
      call require(size(response,2)>0,'partial dependence predictor returned no outputs')
      if(r==1) then
        outputs=size(response,2)
        if(which=='average'.or.which=='both') allocate(results%average(outputs,points))
        if(which=='individual'.or.which=='both') allocate(results%individual(outputs,rows,points))
      end if
      call require(size(response,2)==outputs,'partial dependence predictor changed output count')
      do k=1,outputs
        value=0
        do j=1,rows
          value=difference(value,-product(response(j,k),weights(j)))
        end do
        value=quotient(value,mass)
        if(allocated(results%average)) results%average(k,r)=value
        if(allocated(results%individual)) results%individual(k,:,r)=response(:,k)
      end do
    end do
  end function

  function sorted(values) result(ordered)
    real(dp), intent(in) :: values(:)
    real(dp) :: ordered(size(values)),key
    integer :: i,j
    ordered=values
    do i=2,size(values)
      key=ordered(i)
      if(ieee_is_nan(key)) cycle
      j=i-1
      do while(j>=1)
        if(.not.ieee_is_nan(ordered(j))) then
          if(ordered(j)<=key) exit
        end if
        ordered(j+1)=ordered(j); j=j-1
      end do
      ordered(j+1)=key
    end do
  end function

  function distinct(ordered) result(values)
    real(dp), intent(in) :: ordered(:)
    real(dp), allocatable :: values(:)
    real(dp) :: work(size(ordered))
    integer :: i,n
    n=1; work(1)=ordered(1)
    do i=2,size(ordered)
      if(ieee_is_nan(ordered(i))) then
        if(ieee_is_nan(work(n))) cycle
      else
        if(ordered(i)==work(n)) cycle
      end if
      n=n+1; work(n)=ordered(i)
    end do
    values=work(:n)
  end function

  real(dp) function quantile(ordered,p) result(value)
    real(dp), intent(in) :: ordered(:),p
    real(dp) :: position,gamma
    integer :: k,n
    n=size(ordered)
    if(n==1) then
      value=ordered(1); return
    end if
    position=n*p+(.4_dp+p*(1._dp-.4_dp-.4_dp))
    k=floor(max(1._dp,min(position,real(n-1,dp))))
    gamma=max(0._dp,min(1._dp,position-k))
    value=difference(product(1-gamma,ordered(k)),-product(gamma,ordered(k+1)))
  end function

  real(dp) function total(values) result(value)
    real(dp), intent(in) :: values(:)
    integer :: j
    value=0
    do j=1,size(values)
      value=difference(value,-values(j))
    end do
  end function

  real(dp) function product(a,b) result(value)
    real(dp), intent(in) :: a,b
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(a).or.ieee_is_nan(b)) return
    if(.not.ieee_is_finite(a).or..not.ieee_is_finite(b)) then
      if(a==0.or.b==0) return
      value=ieee_value(0._dp,ieee_positive_inf)
      if((a<0).neqv.(b<0)) value=-value
    else
      if(abs(a)>1) then
        if(abs(b)>huge(a)/abs(a)) then
          value=ieee_value(0._dp,ieee_positive_inf)
          if((a<0).neqv.(b<0)) value=-value
          return
        end if
      end if
      value=a*b
    end if
  end function

  real(dp) function quotient(a,b) result(value)
    real(dp), intent(in) :: a,b
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(a).or.ieee_is_nan(b)) return
    if(.not.ieee_is_finite(a).and..not.ieee_is_finite(b)) return
    if(abs(b)<1) then
      if(abs(a)>huge(a)*abs(b)) then
        value=ieee_value(0._dp,ieee_positive_inf)
        if((a<0).neqv.(b<0)) value=-value
        return
      end if
    end if
    value=a/b
  end function
  real(dp) function difference(a,b) result(value)
    real(dp), intent(in) :: a,b
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(a).or.ieee_is_nan(b)) return
    if(.not.ieee_is_finite(a).or..not.ieee_is_finite(b)) then
      if(a==b) return
      if(a>0.and..not.ieee_is_finite(a).or.b<0.and..not.ieee_is_finite(b)) then
        value=ieee_value(0._dp,ieee_positive_inf)
      else
        value=ieee_value(0._dp,ieee_negative_inf)
      end if
    else
      if(b<0) then
        if(a>huge(a)+b) then
          value=ieee_value(0._dp,ieee_positive_inf); return
        end if
      else if(b>0) then
        if(a< -huge(a)+b) then
          value=ieee_value(0._dp,ieee_negative_inf); return
        end if
      end if
      value=a-b
    end if
  end function

end module
