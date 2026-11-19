! PowerTransformer contracts adapted from sklearn 1.6.1 preprocessing/_data.py.
! Retained BSD-3-Clause notice: LICENSES/scikit-learn-COPYING.
! Native log-scaled likelihood and golden-section optimizer; no SciPy dependency.
module sklearn_power_transform
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  private
  public :: power_transformer
  type :: power_transformer
    character(16) :: method='yeo-johnson'
    logical :: standardize=.true.
    real(dp), allocatable :: lambdas_(:),mean_(:),scale_(:)
    character(16), private :: fitted_method=''
    logical, private :: fitted_standardize=.true.
  contains
    procedure :: fit=>power_fit
    procedure :: transform=>power_transform
    procedure :: fit_transform=>power_fit_transform
    procedure :: inverse_transform=>power_inverse
  end type
contains
  real(dp) function log_one_plus(x) result(y)
    real(dp), intent(in) :: x
    real(dp) :: u
    u=1+x
    if(u==1) then
      y=x
    else
      y=log(u)*x/(u-1)
    end if
  end function

  real(dp) function exp_minus_one(x) result(y)
    real(dp), intent(in) :: x
    real(dp) :: term
    integer :: k
    if(abs(x)<0.01_dp) then
      y=x; term=x
      do k=2,12
        term=term*x/k; y=y+term
      end do
    else
      y=exp(x)-1
    end if
  end function

  ! Represent each transformed value as sign * exp(log_magnitude). This
  ! permits likelihood evaluation without overflowing trial powers.
  subroutine power_log(x,lambda,method,log_magnitude,sgn)
    real(dp), intent(in) :: x,lambda
    character(*), intent(in) :: method
    real(dp), intent(out) :: log_magnitude,sgn
    real(dp) :: base,exponent,a
    if(method=='box-cox') then
      base=log(x); exponent=lambda
    else
      base=log_one_plus(abs(x)); exponent=lambda
      if(x<0) exponent=2-lambda
    end if
    if(base==0) then
      log_magnitude=-huge(1._dp); sgn=0; return
    end if
    sgn=sign(1._dp,base)
    if(method/='box-cox'.and.x<0) sgn=-1
    if(abs(exponent)<epsilon(1._dp)) then
      log_magnitude=log(abs(base)); return
    end if
    a=exponent*base
    if(a>50) then
      log_magnitude=a+log(1-exp(-a))-log(abs(exponent))
    else if(a< -50) then
      log_magnitude=log(1-exp(a))-log(abs(exponent))
    else
      log_magnitude=log(abs(exp_minus_one(a)))-log(abs(exponent))
    end if
  end subroutine

  real(dp) function likelihood(x,lambda,method) result(value)
    real(dp), intent(in) :: x(:),lambda
    character(*), intent(in) :: method
    real(dp) :: logs(size(x)),signs(size(x)),z(size(x)),largest,variance,jacobian,average
    integer :: i,n
    n=size(x)
    do i=1,n
      call power_log(x(i),lambda,method,logs(i),signs(i))
    end do
    largest=maxval(logs)
    if(largest== -huge(1._dp)) then
      value=huge(1._dp)/100; return
    end if
    z=0
    do i=1,n
      if(signs(i)/=0) z(i)=signs(i)*exp(logs(i)-largest)
    end do
    average=sum(z)/n; variance=sum((z-average)**2)/n
    if(variance<=tiny(1._dp)) then
      value=huge(1._dp)/100; return
    end if
    jacobian=0
    do i=1,n
      if(method=='box-cox') then
        jacobian=jacobian+log(x(i))
      else
        jacobian=jacobian+sign(1._dp,x(i))*log_one_plus(abs(x(i)))
      end if
    end do
    value=real(n,dp)/2*(2*largest+log(variance))-(lambda-1)*jacobian
  end function

  real(dp) function optimal_power(x,method) result(lambda)
    real(dp), intent(in) :: x(:)
    character(*), intent(in) :: method
    real(dp), parameter :: gold=1.6180339887498948482_dp,ratio=0.6180339887498948482_dp
    real(dp) :: a,b,c,fa,fb,fc,left,right,u,v,fu,fv,tmp
    integer :: k
    a=-2; b=2; fa=likelihood(x,a,method); fb=likelihood(x,b,method)
    if(fb>fa) then
      tmp=a; a=b; b=tmp; tmp=fa; fa=fb; fb=tmp
    end if
    c=b+gold*(b-a); fc=likelihood(x,c,method)
    do k=1,60
      if(fc>=fb) exit
      a=b; b=c; fb=fc; c=b+gold*(b-a); fc=likelihood(x,c,method)
    end do
    call require(k<=60,'PowerTransformer could not bracket likelihood optimum')
    left=min(a,c); right=max(a,c)
    u=right-ratio*(right-left); v=left+ratio*(right-left)
    fu=likelihood(x,u,method); fv=likelihood(x,v,method)
    do k=1,200
      if(right-left<=1.e-9_dp*(1+abs(u)+abs(v))) exit
      if(fu<fv) then
        right=v; v=u; fv=fu; u=right-ratio*(right-left); fu=likelihood(x,u,method)
      else
        left=u; u=v; fu=fv; v=left+ratio*(right-left); fv=likelihood(x,v,method)
      end if
    end do
    lambda=(left+right)/2
  end function

  subroutine validate(x)
    real(dp), intent(in) :: x(:,:)
    call require(size(x,1)>0.and.size(x,2)>0,'PowerTransformer needs a nonempty matrix')
    call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'PowerTransformer rejects infinity')
  end subroutine

  subroutine power_fit(self,x)
    class(power_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: observed(:),z(:,:)
    real(dp) :: average,variance,upper
    integer :: j,n
    call validate(x)
    call require(self%method=='yeo-johnson'.or.self%method=='box-cox','invalid PowerTransformer method')
    self%fitted_method=self%method; self%fitted_standardize=self%standardize
    self%lambdas_=spread(1._dp,1,size(x,2)); self%mean_=self%lambdas_*0; self%scale_=self%lambdas_
    do j=1,size(x,2)
      observed=pack(x(:,j),.not.ieee_is_nan(x(:,j))); n=size(observed)
      call require(n>0,'PowerTransformer all-missing feature')
      if(self%method=='box-cox') then
        call require(all(observed>0),'Box-Cox requires strictly positive data')
        call require(maxval(observed)>minval(observed),'Box-Cox requires nonconstant data')
      end if
      average=sum(observed)/n; variance=sum((observed-average)**2)/n
      upper=n*epsilon(1._dp)*variance+(n*average*epsilon(1._dp))**2
      if(self%method=='yeo-johnson'.and.variance<=upper) cycle
      self%lambdas_(j)=optimal_power(observed,self%method)
    end do
    ! Transform first with neutral statistics, then fit population moments.
    z=self%transform(x)
    if(self%standardize) then
      do j=1,size(x,2)
        observed=pack(z(:,j),.not.ieee_is_nan(z(:,j))); n=size(observed)
        self%mean_(j)=sum(observed)/n
        variance=sum((observed-self%mean_(j))**2)/n
        upper=n*epsilon(1._dp)*variance+(n*self%mean_(j)*epsilon(1._dp))**2
        if(variance>upper) self%scale_(j)=sqrt(variance)
      end do
    end if
  end subroutine

  function power_transform(self,x) result(z)
    class(power_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: z(size(x,1),size(x,2)),magnitude,sgn
    integer :: i,j
    call require(allocated(self%lambdas_),'PowerTransformer is not fitted')
    call validate(x)
    call require(size(x,2)==size(self%lambdas_),'PowerTransformer feature count mismatch')
    z=x
    do j=1,size(x,2)
      do i=1,size(x,1)
        if(ieee_is_nan(x(i,j))) cycle
        if(self%fitted_method=='box-cox') &
          call require(x(i,j)>0,'Box-Cox requires strictly positive data')
        call power_log(x(i,j),self%lambdas_(j),self%fitted_method,magnitude,sgn)
        call require(magnitude<log(huge(1._dp)),'PowerTransformer power overflow')
        z(i,j)=0
        if(sgn/=0) z(i,j)=sgn*exp(magnitude)
        if(self%fitted_standardize) z(i,j)=(z(i,j)-self%mean_(j))/self%scale_(j)
      end do
    end do
  end function

  function power_fit_transform(self,x) result(z)
    class(power_transformer), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function

  function power_inverse(self,x) result(z)
    class(power_transformer), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: z(size(x,1),size(x,2)),v,exponent,t,argument
    integer :: i,j
    call require(allocated(self%lambdas_),'PowerTransformer is not fitted')
    call validate(x)
    call require(size(x,2)==size(self%lambdas_),'PowerTransformer feature count mismatch')
    z=x
    do j=1,size(x,2)
      do i=1,size(x,1)
        if(ieee_is_nan(x(i,j))) cycle
        v=x(i,j)
        if(self%fitted_standardize) v=v*self%scale_(j)+self%mean_(j)
        exponent=self%lambdas_(j)
        if(self%fitted_method/='box-cox'.and.v<0) exponent=2-exponent
        if(self%fitted_method/='box-cox') v=abs(v)
        if(abs(exponent)<epsilon(1._dp)) then
          t=v
        else
          argument=exponent*v
          call require(argument> -1,'PowerTransformer inverse outside power domain')
          t=log_one_plus(argument)/exponent
        end if
        call require(t<log(huge(1._dp)),'PowerTransformer inverse overflow')
        if(self%fitted_method=='box-cox') then
          z(i,j)=exp(t)
        else
          z(i,j)=sign(exp_minus_one(t),v)
          ! v was made positive above; recover the unstandardized sign.
          if(self%fitted_standardize) then
            if(x(i,j)*self%scale_(j)+self%mean_(j)<0) z(i,j)=-z(i,j)
          else
            if(x(i,j)<0) z(i,j)=-z(i,j)
          end if
        end if
      end do
    end do
  end function
end module
