! SplineTransformer contracts: sklearn 1.6.1 preprocessing/_polynomial.py.
! BSD-3-Clause attribution retained in LICENSES/scikit-learn-COPYING.
! Native Cox-de Boor evaluation and derivative recurrence replace scipy BSpline.
module sklearn_spline
  use sklearn_common, only: dp,require,check_x,argsort
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  private
  public :: spline_transformer
  type :: spline_transformer
    integer :: n_knots=5,degree=3,n_features_in_=0,n_features_out_=0
    character(16) :: knots='uniform',extrapolation='constant'
    logical :: include_bias=.true.
    real(dp), allocatable :: base_knots_(:,:),knots_(:,:)
    integer, private :: fitted_degree=3,n_splines=0
    character(16), private :: fitted_extrapolation='constant'
    logical, private :: fitted_bias=.true.
  contains
    procedure :: fit=>spline_fit
    procedure :: transform=>spline_transform
    procedure :: fit_transform=>spline_fit_transform
  end type
contains
  subroutine spline_fit(self,x,knots)
    class(spline_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: knots(:,:)
    real(dp), allocatable :: base(:,:),sorted(:)
    integer, allocatable :: order(:)
    real(dp) :: q,fraction,period,first_step,last_step
    integer(int64) :: raw,width,total
    integer :: p,m,j,k,lo,hi
    call check_x(x)
    call require(size(x,1)>=2,'SplineTransformer needs at least two samples')
    call require(self%n_knots>=2,'SplineTransformer n_knots must be at least two')
    call require(self%degree>=1,'SplineTransformer degree must be positive')
    call require(self%extrapolation=='constant'.or.self%extrapolation=='linear'.or. &
      self%extrapolation=='continue'.or.self%extrapolation=='periodic'.or.self%extrapolation=='error', &
      'unknown SplineTransformer extrapolation')
    p=size(x,2); m=self%n_knots
    if(present(knots)) m=size(knots,1)
    total=int(m,int64)+2_int64*self%degree
    raw=int(m,int64)+self%degree-1
    if(self%extrapolation=='periodic') raw=int(m,int64)-1
    width=int(p,int64)*(raw-merge(0,1,self%include_bias))
    call require(total<=huge(0).and.width<=huge(0),'SplineTransformer output dimension overflow')
    if(present(knots)) then
      call check_x(knots)
      call require(size(knots,1)>=2,'SplineTransformer explicit knots need at least two rows')
      call require(size(knots,2)==p,'SplineTransformer knot feature count mismatch')
      base=knots
    else
      call require(self%knots=='uniform'.or.self%knots=='quantile','unknown SplineTransformer knots mode')
      allocate(base(self%n_knots,p))
      do j=1,p
        order=argsort(x(:,j)); sorted=x(order,j)
        do k=1,self%n_knots
          fraction=real(k-1,dp)/real(self%n_knots-1,dp)
          if(self%knots=='uniform') then
            base(k,j)=sorted(1)+(sorted(size(x,1))-sorted(1))*fraction
          else
            q=fraction*(size(x,1)-1); lo=floor(q)+1; hi=min(lo+1,size(x,1))
            base(k,j)=sorted(lo)+(q-floor(q))*(sorted(hi)-sorted(lo))
          end if
        end do
        ! np.linspace/quantile preserve the exact extrema, including cancellation-sensitive endpoints.
        base(1,j)=sorted(1); base(self%n_knots,j)=sorted(size(x,1))
      end do
    end if
    m=size(base,1)
    call require(all(base(2:,:)>base(:m-1,:)),'SplineTransformer knots must be strictly increasing')
    if(self%extrapolation=='periodic') &
      call require(m>self%degree,'periodic SplineTransformer requires degree < knot count')
    total=int(m,int64)+2_int64*self%degree
    raw=int(m,int64)+self%degree-1
    if(self%extrapolation=='periodic') raw=m-1
    width=int(p,int64)*(raw-merge(0,1,self%include_bias))
    call require(total<=huge(0).and.width<=huge(0),'SplineTransformer output dimension overflow')
    self%base_knots_=base
    if(allocated(self%knots_)) deallocate(self%knots_)
    allocate(self%knots_(int(total),p)); self%knots_(self%degree+1:self%degree+m,:)=base
    do j=1,p
      period=base(m,j)-base(1,j); first_step=base(2,j)-base(1,j); last_step=base(m,j)-base(m-1,j)
      do k=1,self%degree
        if(self%extrapolation=='periodic') then
          self%knots_(k,j)=base(m-self%degree+k-1,j)-period
          self%knots_(self%degree+m+k,j)=base(k+1,j)+period
        else
          self%knots_(k,j)=base(1,j)-real(self%degree-k+1,dp)*first_step
          self%knots_(self%degree+m+k,j)=base(m,j)+real(k,dp)*last_step
        end if
      end do
    end do
    self%n_features_in_=p; self%n_features_out_=int(width); self%n_splines=int(raw)
    self%fitted_degree=self%degree; self%fitted_extrapolation=self%extrapolation; self%fitted_bias=self%include_bias
  end subroutine

  subroutine spline_basis(t,degree,x,basis,derivative)
    real(dp), intent(in) :: t(:),x
    integer, intent(in) :: degree
    real(dp), allocatable, intent(out) :: basis(:),derivative(:)
    real(dp), allocatable :: old(:),next(:)
    integer :: m,n,span,d,i
    m=size(t); n=m-degree-1; span=degree+1
    if(x>=t(n+1)) then
      span=n
    else if(x>t(degree+1)) then
      do while(span<n)
        if(x<t(span+1)) exit
        span=span+1
      end do
    end if
    allocate(old(m-1),derivative(n)); old=0; old(span)=1
    do d=1,degree
      if(d==degree) then
        do i=1,n
          derivative(i)=degree*(old(i)/(t(i+degree)-t(i))-old(i+1)/(t(i+degree+1)-t(i+1)))
        end do
      end if
      allocate(next(m-d-1))
      do i=1,size(next)
        next(i)=(x-t(i))*old(i)/(t(i+d)-t(i))+(t(i+d+1)-x)*old(i+1)/(t(i+d+1)-t(i+1))
      end do
      call move_alloc(next,old)
    end do
    call move_alloc(old,basis)
  end subroutine

  function spline_transform(self,x) result(z)
    class(spline_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),basis(:),derivative(:),folded(:)
    real(dp) :: value,point,left,right
    integer :: i,j,k,width,column
    call require(allocated(self%knots_),'SplineTransformer is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'SplineTransformer feature count mismatch')
    width=self%n_splines-merge(0,1,self%fitted_bias)
    allocate(z(size(x,1),self%n_features_out_),folded(self%n_splines))
    do j=1,size(x,2)
      left=self%base_knots_(1,j); right=self%base_knots_(size(self%base_knots_,1),j)
      do i=1,size(x,1)
        value=x(i,j); point=value
        select case(self%fitted_extrapolation)
        case('error')
          call require(value>=left.and.value<=right,'SplineTransformer input outside knot range')
        case('constant','linear')
          point=max(left,min(right,value))
        case('periodic')
          point=left+modulo(value-left,right-left)
        end select
        call spline_basis(self%knots_(:,j),self%fitted_degree,point,basis,derivative)
        if(self%fitted_extrapolation=='linear') basis=basis+(value-point)*derivative
        folded=0
        do k=1,size(basis)
          column=k
          if(column>self%n_splines) column=column-self%n_splines
          folded(column)=folded(column)+basis(k)
        end do
        z(i,(j-1)*width+1:j*width)=folded(:width)
      end do
    end do
  end function

  function spline_fit_transform(self,x,knots) result(z)
    class(spline_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), intent(in), optional :: knots(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x,knots); z=self%transform(x)
  end function
end module
