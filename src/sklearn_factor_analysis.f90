! Dense full/randomized-SVD sklearn.decomposition.FactorAnalysis and orthogonal rotations.
! See THIRD_PARTY_NOTICES.md for upstream attribution.
module sklearn_factor_analysis
  use sklearn_common
  use sklearn_randomized_svd_core, only: randomized_svd
  use sklearn_legacy_rng, only: legacy_random_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: factor_analysis
  type :: factor_analysis
    integer :: n_components=-1,max_iter=1000,n_iter_=0
    real(dp) :: tol=1.e-2_dp
    character(16) :: rotation='none',svd_method='lapack'
    integer :: iterated_power=3,random_state=0,n_rotation_iter_=0
    logical :: rotation_converged_=.true.
    real(dp), allocatable :: rotation_matrix_(:,:)
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
    real(dp) :: variance(size(x,2)),root(size(x,2)),ll,old,constant,unexplained
    real(dp), allocatable :: history(:),rs(:),ru(:,:),rvt(:,:)
    type(legacy_random_state) :: rng
    integer :: n,p,k,it,j
    call check_x(x); n=size(x,1); p=size(x,2); k=self%n_components
    if(k==-1) k=p
    call require(k>=0.and.k<=min(n,p),'invalid FactorAnalysis n_components')
    call require(ieee_is_finite(self%tol),'FactorAnalysis tol must be finite')
    call require(self%max_iter>0.and.self%tol>=0,'invalid FactorAnalysis iteration parameters')
    call require(any(self%rotation==[character(16)::'none','varimax','quartimax']),'unsupported FactorAnalysis rotation')
    call require(any(self%svd_method==[character(16)::'lapack','randomized']),'unsupported FactorAnalysis svd_method')
    call require(self%iterated_power>=0,'FactorAnalysis iterated_power must be nonnegative')
    call require(self%random_state>=0,'FactorAnalysis random_state must be nonnegative')
    allocate(history(self%max_iter));call rng%seed(self%random_state)
    self%mean_=sum(x,dim=1)/n; xc=x-spread(self%mean_,1,n)
    variance=sum(xc**2,dim=1)/n
    self%noise_variance_=spread(1._dp,1,p)
    if(allocated(self%noise_variance_init)) then
      call require(size(self%noise_variance_init)==p,'FactorAnalysis noise variance shape mismatch')
      call require(all(ieee_is_finite(self%noise_variance_init)),'FactorAnalysis noise variance must be finite')
      call require(all(self%noise_variance_init>0),'FactorAnalysis noise variance must be positive')
      self%noise_variance_=self%noise_variance_init
    end if
    constant=p*log(2*acos(-1._dp))+k; old=-huge(1._dp); self%converged_=.false.
    if(allocated(self%components_)) deallocate(self%components_)
    allocate(self%components_(k,p))
    do it=1,self%max_iter
      root=sqrt(self%noise_variance_)+1.e-12_dp
      a=xc/spread(root*sqrt(real(n,dp)),1,n)
      if(self%svd_method=='randomized'.and.k>0) then
        call randomized_svd(a,k,rng,rs,ru,rvt,n_iter=self%iterated_power)
        s=0;s(:k)=rs;v(:,:k)=transpose(rvt)
        unexplained=sum(a*a)-sum(rs*rs)
      else
        call dense_svd(a,s,u,v);unexplained=sum(s(k+1:min(n,p))**2)
      end if
      do j=1,k
        self%components_(j,:)=sqrt(max(s(j)**2-1,0._dp))*v(:,j)*root
      end do
      ! Floor only zero singular values so singular fixtures remain finite.
      ll=-n*0.5_dp*(constant+sum(log(max(s(:k)**2,tiny(1._dp)))) &
        +unexplained+sum(log(self%noise_variance_)))
      history(it)=ll; self%n_iter_=it
      if(ll-old<self%tol) then
        self%converged_=.true.; exit
      end if
      old=ll
      self%noise_variance_=max(variance-sum(self%components_**2,dim=1),1.e-12_dp)
    end do
    self%loglike_=history(:self%n_iter_)
    self%n_rotation_iter_=0;self%rotation_converged_=.true.
    if(allocated(self%rotation_matrix_)) deallocate(self%rotation_matrix_)
    allocate(self%rotation_matrix_(k,k));self%rotation_matrix_=0
    do j=1,k
      self%rotation_matrix_(j,j)=1
    end do
    if(self%rotation/='none'.and.k>0) call rotate_components(self)
  end subroutine
  subroutine rotate_components(self)
    ! sklearn 1.6.1 _factor_analysis.py _ortho_rotation, fixed 100 / 1e-6 controls.
    class(factor_analysis), intent(inout) :: self
    real(dp), allocatable :: loadings(:,:),rotated(:,:),tmp(:,:),matrix(:,:),u(:,:),v(:,:),s(:),candidate(:),best(:)
    real(dp) :: previous,current,norm,best_norm
    integer :: p,k,it,j,i,rank,pass
    loadings=transpose(self%components_);p=size(loadings,1);k=size(loadings,2)
    allocate(u(k,k),v(k,k),s(k),tmp(p,k));previous=0;self%rotation_converged_=.false.
    do it=1,100
      rotated=matmul(loadings,self%rotation_matrix_);tmp=0
      if(self%rotation=='varimax') tmp=rotated*spread(sum(rotated**2,dim=1)/p,1,p)
      matrix=matmul(transpose(loadings),rotated**3-tmp);call dense_svd(matrix,s,u,v)
      ! Native SVD leaves zero singular columns of U empty. Complete those
      ! columns orthogonally so the polar factor remains orthogonal at rank loss.
      rank=count(s>epsilon(1._dp)*k*maxval(s))
      do j=rank+1,k
        best_norm=-1
        do i=1,k
          allocate(candidate(k));candidate=0;candidate(i)=1
          do pass=1,2
            candidate=candidate-matmul(u(:,:j-1),matmul(transpose(u(:,:j-1)),candidate))
          end do
          norm=sum(candidate*candidate)
          if(norm>best_norm) then
            best=candidate;best_norm=norm
          end if
          deallocate(candidate)
        end do
        call require(best_norm>epsilon(1._dp),'FactorAnalysis rotation orthogonal completion failed')
        u(:,j)=best/sqrt(best_norm)
      end do
      self%rotation_matrix_=matmul(u,transpose(v));current=sum(s);self%n_rotation_iter_=it
      if(previous/=0) then
        if(current<previous*(1+1.e-6_dp)) then
          self%rotation_converged_=.true.;exit
        end if
      end if
      previous=current
    end do
    self%components_=transpose(matmul(loadings,self%rotation_matrix_))
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
