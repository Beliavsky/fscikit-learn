! Dense covariance estimators. Shrinkage formulas follow sklearn/covariance/
! _shrunk_covariance.py (scikit-learn 1.6.1, BSD-3-Clause; see THIRD_PARTY_NOTICES).
module sklearn_covariance
  use sklearn_common
  implicit none
  private
  public :: empirical_covariance,shrunk_covariance,ledoit_wolf,oas
  type :: empirical_covariance
    logical :: assume_centered=.false.
    real(dp), allocatable :: location_(:),covariance_(:,:),precision_(:,:)
  contains
    procedure :: fit=>empirical_fit
    procedure :: mahalanobis=>cov_mahalanobis
    procedure :: score=>cov_score
    procedure :: error_norm=>cov_error
  end type
  type, extends(empirical_covariance) :: shrunk_covariance
    real(dp) :: shrinkage=0.1_dp
  contains
    procedure :: fit=>shrunk_fit
  end type
  type, extends(empirical_covariance) :: ledoit_wolf
    real(dp) :: shrinkage_=0
  contains
    procedure :: fit=>ledoit_fit
  end type
  type, extends(empirical_covariance) :: oas
    real(dp) :: shrinkage_=0
  contains
    procedure :: fit=>oas_fit
  end type
contains
  subroutine empirical_fit(self,x)
    class(empirical_covariance), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: centered(size(x,1),size(x,2))
    call check_x(x); self%location_=sum(x,dim=1)/size(x,1)
    if(self%assume_centered) self%location_=0
    centered=x-spread(self%location_,1,size(x,1))
    self%covariance_=matmul(transpose(centered),centered)/size(x,1)
    call set_precision(self)
  end subroutine
  subroutine set_precision(self)
    class(empirical_covariance), intent(inout) :: self
    real(dp) :: values(size(self%covariance_,1)),vectors(size(self%covariance_,1),size(self%covariance_,1))
    real(dp) :: inverse(size(self%covariance_,1)),cutoff
    call symmetric_eigen(self%covariance_,values,vectors)
    cutoff=epsilon(1._dp)*size(values)*maxval(abs(values)); inverse=0
    where(values>cutoff) inverse=1/max(values,tiny(1._dp))
    self%precision_=matmul(vectors*spread(inverse,1,size(values)),transpose(vectors))
  end subroutine
  subroutine shrink(self,amount)
    class(empirical_covariance), intent(inout) :: self
    real(dp), intent(in) :: amount
    real(dp) :: mu
    integer :: j,p
    call require(amount>=0.and.amount<=1,'invalid covariance shrinkage')
    p=size(self%covariance_,1); mu=0
    do j=1,p
      mu=mu+self%covariance_(j,j)/p
    end do
    self%covariance_=(1-amount)*self%covariance_
    do j=1,p
      self%covariance_(j,j)=self%covariance_(j,j)+amount*mu
    end do
    call set_precision(self)
  end subroutine
  subroutine shrunk_fit(self,x)
    class(shrunk_covariance), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    call empirical_fit(self,x); call shrink(self,self%shrinkage)
  end subroutine
  subroutine ledoit_fit(self,x)
    class(ledoit_wolf), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: centered(size(x,1),size(x,2)),mu,beta,delta,squared_cov
    integer :: n,p,j
    call empirical_fit(self,x); n=size(x,1); p=size(x,2); self%shrinkage_=0
    if(p==1) return
    centered=x-spread(self%location_,1,n); mu=0
    do j=1,p
      mu=mu+self%covariance_(j,j)/p
    end do
    squared_cov=sum(self%covariance_**2)
    beta=(sum(sum(centered**2,dim=2)**2)/n-squared_cov)/(p*n)
    delta=(squared_cov-p*mu**2)/p
    if(delta>0) self%shrinkage_=max(0._dp,min(1._dp,beta/delta))
    call shrink(self,self%shrinkage_)
  end subroutine
  subroutine oas_fit(self,x)
    class(oas), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: mu,alpha,den
    integer :: p,j
    call empirical_fit(self,x); p=size(x,2); self%shrinkage_=0
    if(p==1) return
    mu=0
    do j=1,p
      mu=mu+self%covariance_(j,j)/p
    end do
    alpha=sum(self%covariance_**2)/(p*p); den=(size(x,1)+1)*(alpha-mu**2/p)
    self%shrinkage_=1
    if(den>0) self%shrinkage_=min(1._dp,(alpha+mu**2)/den)
    call shrink(self,self%shrinkage_)
  end subroutine
  function cov_mahalanobis(self,x) result(distances)
    class(empirical_covariance), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: distances(size(x,1)),centered(size(x,1),size(x,2))
    call require(allocated(self%precision_),'covariance estimator is not fitted')
    call check_xy(transpose(x),size(self%location_)); centered=x-spread(self%location_,1,size(x,1))
    distances=max(0._dp,sum(matmul(centered,self%precision_)*centered,dim=2))
  end function
  real(dp) function cov_score(self,x) result(v)
    class(empirical_covariance), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: values(size(x,2)),vectors(size(x,2),size(x,2)),distances(size(x,1))
    distances=self%mahalanobis(x); call symmetric_eigen(self%precision_,values,vectors)
    call require(all(values>0),'log likelihood undefined for singular precision')
    v=0.5_dp*(sum(log(values))-size(x,2)*log(2*acos(-1._dp))-sum(distances)/size(x,1))
  end function
  real(dp) function cov_error(self,comp_cov,norm,scaling,squared) result(v)
    class(empirical_covariance), intent(in) :: self
    real(dp), intent(in) :: comp_cov(:,:)
    character(*), intent(in), optional :: norm
    logical, intent(in), optional :: scaling,squared
    real(dp), allocatable :: error(:,:),values(:),vectors(:,:)
    integer :: p
    logical :: scale,square
    character(16) :: kind
    call require(allocated(self%covariance_),'covariance estimator is not fitted')
    call require(all(shape(comp_cov)==shape(self%covariance_)),'covariance error shape mismatch')
    kind='frobenius'; if(present(norm)) kind=norm
    scale=.true.; square=.true.
    if(present(scaling)) scale=scaling
    if(present(squared)) square=squared
    error=comp_cov-self%covariance_; p=size(error,1)
    select case(trim(kind))
    case('frobenius'); v=sum(error**2)
    case('spectral')
      allocate(values(p),vectors(p,p)); call symmetric_eigen(matmul(transpose(error),error),values,vectors)
      v=maxval(values)
    case default; call require(.false.,'unknown covariance error norm')
    end select
    if(scale) v=v/p
    if(.not.square) v=sqrt(v)
  end function
end module
