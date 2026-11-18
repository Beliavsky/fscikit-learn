! sklearn/isotonic.py contracts, with native pool-adjacent-violators fitting.
! See THIRD_PARTY_NOTICES.md for upstream BSD-3-Clause attribution.
module sklearn_isotonic
  use sklearn_common
  use sklearn_metrics, only: r2_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: isotonic_regression,isotonic_fit,check_increasing
  type :: isotonic_regression
    logical :: increasing=.true.,auto_increasing=.false.,increasing_=.true.
    real(dp) :: y_min=-huge(1._dp),y_max=huge(1._dp),x_min_=0,x_max_=0
    character(8) :: out_of_bounds='nan'
    real(dp), allocatable :: x_thresholds_(:),y_thresholds_(:)
  contains
    procedure :: fit=>iso_fit
    procedure :: predict=>iso_predict
    procedure :: transform=>iso_predict
    procedure :: fit_transform=>iso_fit_transform
    procedure :: score=>iso_score
  end type
contains
  subroutine check_vector(x)
    real(dp), intent(in) :: x(:)
    call require(size(x)>0,'isotonic requires nonempty vectors')
    call require(all(ieee_is_finite(x)),'isotonic input contains NaN or infinity')
  end subroutine
  function ranks(x) result(r)
    real(dp), intent(in) :: x(:)
    real(dp) :: r(size(x))
    integer :: order(size(x)),i,j
    order=argsort(x); i=1
    do while(i<=size(x))
      j=i
      do while(j<size(x))
        if(x(order(j+1))/=x(order(i))) exit
        j=j+1
      end do
      r(order(i:j))=(real(i,dp)+j)/2; i=j+1
    end do
  end function
  logical function check_increasing(x,y) result(increasing)
    real(dp), intent(in) :: x(:),y(:)
    real(dp), allocatable :: rx(:),ry(:)
    call check_vector(x); call check_vector(y)
    call require(size(x)==size(y),'isotonic sample counts differ')
    rx=ranks(x); ry=ranks(y); rx=rx-sum(rx)/size(rx); ry=ry-sum(ry)/size(ry)
    ! sklearn's undefined Spearman coefficient compares false against zero.
    increasing=.false.
    if(sum(rx**2)>0.and.sum(ry**2)>0) increasing=sum(rx*ry)>=0
  end function
  function isotonic_fit(y,sample_weight,increasing,y_min,y_max) result(fitted)
    real(dp), intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:),y_min,y_max
    logical, intent(in), optional :: increasing
    real(dp), allocatable :: fitted(:),weights(:),means(:),mass(:)
    integer, allocatable :: first(:),last(:)
    real(dp) :: direction,total,lo,hi
    integer :: n,i,j,blocks
    call check_vector(y); n=size(y); lo=-huge(1._dp); hi=huge(1._dp)
    if(present(y_min)) lo=y_min
    if(present(y_max)) hi=y_max
    call require(ieee_is_finite(lo).and.ieee_is_finite(hi).and.lo<=hi,'invalid isotonic bounds')
    allocate(weights(n),means(n),mass(n),first(n),last(n),fitted(n)); weights=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'isotonic weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'invalid isotonic weights')
      call require(all(sample_weight>0),'isotonic_fit weights must be positive')
      weights=sample_weight
    end if
    weights=weights/maxval(weights)
    call require(all(weights>0),'isotonic weight dynamic range is too large')
    direction=1
    if(present(increasing)) then
      if(.not.increasing) direction=-1
    end if
    blocks=0
    do i=1,n
      blocks=blocks+1; means(blocks)=direction*y(i); mass(blocks)=weights(i)
      first(blocks)=i; last(blocks)=i
      do while(blocks>1)
        if(means(blocks-1)<=means(blocks)) exit
        total=mass(blocks-1)+mass(blocks)
        means(blocks-1)=means(blocks-1)*(mass(blocks-1)/total)+means(blocks)*(mass(blocks)/total)
        mass(blocks-1)=total; last(blocks-1)=last(blocks); blocks=blocks-1
      end do
    end do
    do j=1,blocks
      fitted(first(j):last(j))=max(lo,min(hi,direction*means(j)))
    end do
  end function
  subroutine iso_fit(self,x,y,sample_weight)
    class(isotonic_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: weights(:),ux(:),uy(:),uw(:),fitted(:)
    integer, allocatable :: order(:)
    logical, allocatable :: keep(:)
    integer :: n,i,j,k,m
    real(dp) :: total
    call check_vector(x); call check_vector(y); n=size(x)
    call require(size(y)==n,'isotonic sample counts differ')
    call require(self%out_of_bounds=='nan'.or.self%out_of_bounds=='clip'.or. &
      self%out_of_bounds=='raise','invalid isotonic out_of_bounds')
    allocate(weights(n),ux(n),uy(n),uw(n)); weights=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'isotonic weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'invalid isotonic weights')
      call require(all(sample_weight>=0).and.any(sample_weight>0),'invalid isotonic weights')
      weights=sample_weight
    end if
    weights=weights/maxval(weights); order=argsort(x); m=0
    self%increasing_=self%increasing
    if(self%auto_increasing) self%increasing_=check_increasing(x,y)
    do j=1,n
      i=order(j)
      if(weights(i)==0) cycle
      if(m>0) then
        if(x(i)-ux(m)<1.e-15_dp) then
          total=uw(m)+weights(i)
          uy(m)=uy(m)*(uw(m)/total)+y(i)*(weights(i)/total); uw(m)=total
          cycle
        end if
      end if
      m=m+1; ux(m)=x(i); uy(m)=y(i); uw(m)=weights(i)
    end do
    fitted=isotonic_fit(uy(:m),uw(:m),self%increasing_,self%y_min,self%y_max)
    allocate(keep(m)); keep=.true.
    do k=2,m-1
      keep(k)=fitted(k)/=fitted(k-1).or.fitted(k)/=fitted(k+1)
    end do
    self%x_min_=ux(1); self%x_max_=ux(m)
    self%x_thresholds_=pack(ux(:m),keep); self%y_thresholds_=pack(fitted,keep)
  end subroutine
  function iso_predict(self,x) result(y)
    class(isotonic_regression), intent(in) :: self
    real(dp), intent(in) :: x(:)
    real(dp), allocatable :: y(:)
    real(dp) :: t,ratio
    integer :: i,lo,hi,mid,n
    call require(allocated(self%x_thresholds_),'isotonic regression is not fitted')
    call check_vector(x); allocate(y(size(x))); n=size(self%x_thresholds_)
    do i=1,size(x)
      t=x(i)
      ! A one-threshold sklearn fit predicts a constant even outside its domain.
      if(n==1) then
        y(i)=self%y_thresholds_(1); cycle
      end if
      if(t<self%x_min_.or.t>self%x_max_) then
        select case(self%out_of_bounds)
        case('raise'); call require(.false.,'isotonic query outside fitted bounds')
        case('nan'); y(i)=ieee_value(0._dp,ieee_quiet_nan); cycle
        case('clip'); t=max(self%x_min_,min(self%x_max_,t))
        end select
      end if
      lo=1; hi=n
      do while(hi-lo>1)
        mid=(lo+hi)/2
        if(self%x_thresholds_(mid)<=t) then
          lo=mid
        else
          hi=mid
        end if
      end do
      ratio=(t-self%x_thresholds_(lo))/(self%x_thresholds_(hi)-self%x_thresholds_(lo))
      y(i)=(1-ratio)*self%y_thresholds_(lo)+ratio*self%y_thresholds_(hi)
    end do
  end function
  function iso_fit_transform(self,x,y,sample_weight) result(z)
    class(isotonic_regression), intent(inout) :: self
    real(dp), intent(in) :: x(:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: z(:)
    call self%fit(x,y,sample_weight); z=self%predict(x)
  end function
  real(dp) function iso_score(self,x,y) result(value)
    class(isotonic_regression), intent(in) :: self
    real(dp), intent(in) :: x(:),y(:)
    value=r2_score(y,self%predict(x))
  end function
end module
