! QuantileTransformer contracts: sklearn 1.6.1 preprocessing/_data.py.
! Retained BSD-3-Clause notice: LICENSES/scikit-learn-COPYING.
module sklearn_quantile_transform
  use sklearn_common, only: dp,require,argsort,seed_rng
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: quantile_transformer
  real(dp), parameter :: bounds_threshold=1.e-7_dp
  type :: quantile_transformer
    integer :: n_quantiles=1000,subsample=10000,random_state=0,n_quantiles_=0,n_features_in_=0
    character(16) :: output_distribution='uniform'
    real(dp), allocatable :: quantiles_(:,:),references_(:)
  contains
    procedure :: fit=>quantile_fit
    procedure :: transform=>quantile_transform
    procedure :: inverse_transform=>quantile_inverse
    procedure :: fit_transform=>quantile_fit_transform
  end type
contains
  subroutine quantile_check(x)
    real(dp), intent(in) :: x(:,:)
    call require(size(x,1)>0.and.size(x,2)>0,'QuantileTransformer X must be nonempty')
    call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'QuantileTransformer input contains infinity')
  end subroutine

  subroutine quantile_controls(self)
    class(quantile_transformer), intent(in) :: self
    call require(self%n_quantiles>=1,'QuantileTransformer n_quantiles must be positive')
    call require(self%subsample>=0,'QuantileTransformer subsample must be nonnegative')
    if(self%subsample>0) call require(self%n_quantiles<=self%subsample, &
      'QuantileTransformer n_quantiles exceeds subsample')
    call require(self%output_distribution=='uniform'.or.self%output_distribution=='normal', &
      'invalid QuantileTransformer output_distribution')
  end subroutine

  subroutine quantile_fit(self,x)
    class(quantile_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: rows(:),order(:)
    real(dp), allocatable :: column(:),values(:)
    integer :: n,m,p,i,j,k,left,pick,tmp
    real(dp) :: u,position,fraction
    call quantile_controls(self); call quantile_check(x)
    n=size(x,1); p=size(x,2); m=n
    if(self%subsample>0) m=min(n,self%subsample)
    rows=[(i,i=1,n)]
    if(m<n) then
      call seed_rng(self%random_state)
      do i=1,m
        call random_number(u); pick=i+int(u*(n-i+1))
        tmp=rows(i); rows(i)=rows(pick); rows(pick)=tmp
      end do
    end if
    self%n_quantiles_=min(self%n_quantiles,n); self%n_features_in_=p
    if(allocated(self%quantiles_)) deallocate(self%quantiles_,self%references_)
    allocate(self%quantiles_(self%n_quantiles_,p),self%references_(self%n_quantiles_))
    self%references_=0
    if(self%n_quantiles_>1) self%references_=[(real(i-1,dp)/real(self%n_quantiles_-1,dp), &
      i=1,self%n_quantiles_)]
    do j=1,p
      column=x(rows(:m),j); values=pack(column,.not.ieee_is_nan(column))
      k=size(values)
      if(k==0) then
        self%quantiles_(:,j)=ieee_value(0._dp,ieee_quiet_nan); cycle
      end if
      order=argsort(values); values=values(order)
      do i=1,self%n_quantiles_
        position=self%references_(i)*real(k-1,dp)
        left=min(int(position)+1,k); fraction=position-real(left-1,dp)
        self%quantiles_(i,j)=values(left)
        if(left<k) self%quantiles_(i,j)=(1-fraction)*values(left)+fraction*values(left+1)
        if(i>1) self%quantiles_(i,j)=max(self%quantiles_(i,j),self%quantiles_(i-1,j))
      end do
    end do
  end subroutine

  ! numpy.interp's rightmost duplicate convention, including singleton tables.
  real(dp) function interpolate(x,xp,yp) result(y)
    real(dp), intent(in) :: x,xp(:),yp(:)
    integer :: n,lo,hi,mid
    n=size(xp)
    if(x<xp(1)) then
      y=yp(1); return
    end if
    if(x>=xp(n)) then
      y=yp(n); return
    end if
    lo=1; hi=n
    do while(hi-lo>1)
      mid=(lo+hi)/2
      if(x>=xp(mid)) then
        lo=mid
      else
        hi=mid
      end if
    end do
    y=yp(lo)+(x-xp(lo))/(xp(hi)-xp(lo))*(yp(hi)-yp(lo))
  end function

  real(dp) function normal_cdf(x) result(p)
    real(dp), intent(in) :: x
    p=.5_dp*erfc(-x/sqrt(2._dp))
  end function

  ! Invert erfc monotonically in the lower tail, reflecting the upper tail.
  ! No third-party approximation coefficients or SciPy runtime are required.
  real(dp) function normal_quantile(p) result(x)
    real(dp), intent(in) :: p
    real(dp) :: tail,lo,hi,mid
    integer :: i
    tail=min(p,1-p); lo=-40; hi=0
    do i=1,80
      mid=.5_dp*(lo+hi)
      if(normal_cdf(mid)<tail) then
        lo=mid
      else
        hi=mid
      end if
    end do
    x=.5_dp*(lo+hi)
    if(p>.5_dp) x=-x
    if(p==.5_dp) x=0
  end function

  function quantile_apply(self,x,inverse) result(z)
    class(quantile_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    logical, intent(in) :: inverse
    real(dp), allocatable :: z(:,:),q(:),rev_q(:),rev_ref(:)
    real(dp) :: v,y,lower,upper,lower_value,upper_value,clip_p,low_p,high_p
    logical :: low,high,normal
    integer :: i,j,nq
    call require(allocated(self%quantiles_),'QuantileTransformer is not fitted')
    call quantile_controls(self); call quantile_check(x)
    call require(size(x,2)==self%n_features_in_,'QuantileTransformer feature count mismatch')
    nq=self%n_quantiles_; normal=self%output_distribution=='normal'; z=x
    clip_p=bounds_threshold-epsilon(1._dp); low_p=clip_p; high_p=1-clip_p
    do j=1,size(x,2)
      q=self%quantiles_(:,j)
      if(any(ieee_is_nan(q))) then
        z(:,j)=ieee_value(0._dp,ieee_quiet_nan); cycle
      end if
      rev_q=-q(nq:1:-1); rev_ref=-self%references_(nq:1:-1)
      lower=q(1); upper=q(nq); lower_value=0; upper_value=1
      if(inverse) then
        lower=0; upper=1; lower_value=q(1); upper_value=q(nq)
      end if
      do i=1,size(x,1)
        if(ieee_is_nan(x(i,j))) cycle
        v=x(i,j)
        if(inverse.and.normal) v=normal_cdf(v)
        if(normal) then
          low=v-bounds_threshold<lower; high=v+bounds_threshold>upper
        else
          low=v==lower; high=v==upper
        end if
        if(inverse) then
          y=interpolate(v,self%references_,q)
        else
          y=.5_dp*(interpolate(v,q,self%references_)-interpolate(-v,rev_q,rev_ref))
        end if
        if(high) y=upper_value
        if(low) y=lower_value
        if(.not.inverse.and.normal) y=normal_quantile(max(low_p,min(high_p,y)))
        z(i,j)=y
      end do
    end do
  end function

  function quantile_transform(self,x) result(z)
    class(quantile_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    z=quantile_apply(self,x,.false.)
  end function

  function quantile_inverse(self,x) result(z)
    class(quantile_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    z=quantile_apply(self,x,.true.)
  end function

  function quantile_fit_transform(self,x) result(z)
    class(quantile_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
end module
