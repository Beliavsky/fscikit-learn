! Dense full-SVD implementation of sklearn.decomposition.FactorAnalysis.
! See THIRD_PARTY_NOTICES.md for upstream attribution.
module sklearn_factor_analysis
  use sklearn_common
  implicit none
  private
  public :: factor_analysis
  type :: factor_analysis
    integer :: n_components=-1,max_iter=1000,n_iter_=0
    real(dp) :: tol=1.e-2_dp
    logical :: converged_=.false.
    real(dp), allocatable :: components_(:,:),mean_(:),noise_variance_(:),loglike_(:)
    real(dp), allocatable :: noise_variance_init(:)
  contains
    procedure :: fit=>fa_fit
    procedure :: transform=>fa_transform
    procedure :: fit_transform=>fa_fit_transform
    procedure :: get_covariance=>fa_covariance
    procedure :: get_precision=>fa_precision
    procedure :: score_samples=>fa_score_samples
    procedure :: score=>fa_score
  end type
contains
  subroutine fa_fit(self,x)
    class(factor_analysis), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: xc(size(x,1),size(x,2)),a(size(x,1),size(x,2))
    real(dp) :: s(size(x,2)),u(size(x,1),size(x,2)),v(size(x,2),size(x,2))
    real(dp) :: variance(size(x,2)),root(size(x,2)),history(self%max_iter),ll,old,constant
    integer :: n,p,k,it,j
    call check_x(x); n=size(x,1); p=size(x,2); k=self%n_components
    if(k==-1) k=p
    call require(k>=0.and.k<=min(n,p),'invalid FactorAnalysis n_components')
    call require(self%max_iter>0.and.self%tol>=0,'invalid FactorAnalysis iteration parameters')
    self%mean_=sum(x,dim=1)/n; xc=x-spread(self%mean_,1,n)
    variance=sum(xc**2,dim=1)/n
    self%noise_variance_=spread(1._dp,1,p)
    if(allocated(self%noise_variance_init)) then
      call require(size(self%noise_variance_init)==p,'FactorAnalysis noise variance shape mismatch')
      call require(all(self%noise_variance_init>0),'FactorAnalysis noise variance must be positive')
      self%noise_variance_=self%noise_variance_init
    end if
    constant=p*log(2*acos(-1._dp))+k; old=-huge(1._dp); self%converged_=.false.
    if(allocated(self%components_)) deallocate(self%components_)
    allocate(self%components_(k,p))
    do it=1,self%max_iter
      root=sqrt(self%noise_variance_)+1.e-12_dp
      a=xc/spread(root*sqrt(real(n,dp)),1,n)
      call dense_svd(a,s,u,v)
      do j=1,k
        self%components_(j,:)=sqrt(max(s(j)**2-1,0._dp))*v(:,j)*root
      end do
      ! Floor only zero singular values so singular fixtures remain finite.
      ll=-n*0.5_dp*(constant+sum(log(max(s(:k)**2,tiny(1._dp)))) &
        +sum(s(k+1:min(n,p))**2)+sum(log(self%noise_variance_)))
      history(it)=ll; self%n_iter_=it
      if(ll-old<self%tol) then
        self%converged_=.true.; exit
      end if
      old=ll
      self%noise_variance_=max(variance-sum(self%components_**2,dim=1),1.e-12_dp)
    end do
    self%loglike_=history(:self%n_iter_)
  end subroutine
  function fa_covariance(self) result(cov)
    class(factor_analysis), intent(in) :: self
    real(dp), allocatable :: cov(:,:)
    integer :: j
    call require(allocated(self%components_),'FactorAnalysis is not fitted')
    cov=matmul(transpose(self%components_),self%components_)
    do j=1,size(cov,1)
      cov(j,j)=cov(j,j)+self%noise_variance_(j)
    end do
  end function
  function inverse(a) result(inv)
    real(dp), intent(in) :: a(:,:)
    real(dp) :: inv(size(a,1),size(a,2)),l(size(a,1),size(a,2)),e(size(a,1))
    integer :: j
    l=cholesky_factor(a)
    do j=1,size(a,1)
      e=0; e(j)=1; inv(:,j)=solve_upper(transpose(l),solve_lower(l,e))
    end do
  end function
  function fa_precision(self) result(precision)
    class(factor_analysis), intent(in) :: self
    real(dp), allocatable :: precision(:,:)
    precision=inverse(self%get_covariance())
  end function
  function fa_transform(self,x) result(z)
    class(factor_analysis), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),w(:,:),c(:,:)
    integer :: k,j
    call require(allocated(self%components_),'FactorAnalysis is not fitted')
    call check_xy(transpose(x),size(self%mean_)); k=size(self%components_,1)
    w=self%components_/spread(self%noise_variance_,1,k)
    c=matmul(w,transpose(self%components_))
    do j=1,k
      c(j,j)=c(j,j)+1
    end do
    z=matmul(matmul(x-spread(self%mean_,1,size(x,1)),transpose(w)),inverse(c))
  end function
  function fa_fit_transform(self,x) result(z)
    class(factor_analysis), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
  function fa_score_samples(self,x) result(scores)
    class(factor_analysis), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: scores(size(x,1))
    real(dp), allocatable :: xc(:,:),l(:,:)
    integer :: i
    call require(allocated(self%components_),'FactorAnalysis is not fitted')
    call check_xy(transpose(x),size(self%mean_))
    l=cholesky_factor(self%get_covariance()); xc=x-spread(self%mean_,1,size(x,1))
    do i=1,size(x,1)
      scores(i)=-0.5_dp*(sum(solve_lower(l,xc(i,:))**2)+size(x,2)*log(2*acos(-1._dp)))
    end do
    do i=1,size(l,1)
      scores=scores-log(l(i,i))
    end do
  end function
  real(dp) function fa_score(self,x) result(score)
    class(factor_analysis), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    score=sum(self%score_samples(x))/size(x,1)
  end function
end module
