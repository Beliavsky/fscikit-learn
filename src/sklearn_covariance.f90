! Dense covariance estimators. Shrinkage formulas follow sklearn/covariance/
! _shrunk_covariance.py (scikit-learn 1.6.1, BSD-3-Clause; see THIRD_PARTY_NOTICES).
module sklearn_covariance
  use sklearn_robust_covariance, only: min_cov_det,elliptic_envelope,fast_mcd
  use sklearn_common
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: min_cov_det,elliptic_envelope,fast_mcd
  public :: empirical_covariance,shrunk_covariance,ledoit_wolf,oas,graphical_lasso
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
  type, extends(empirical_covariance) :: graphical_lasso
    real(dp) :: alpha=0.01_dp,tol=1.e-4_dp,enet_tol=1.e-4_dp,eps=epsilon(1._dp)
    real(dp) :: objective_=0,dual_gap_=0
    integer :: max_iter=100,n_iter_=0
    character(16) :: covariance='empirical',mode='cd'
    logical :: converged_=.false.
    real(dp), allocatable :: costs_(:,:)
  contains
    procedure :: fit=>graphical_fit
  end type
contains
  ! sklearn 1.6.1 covariance/_graph_lasso.py and linear_model/_cd_fast.pyx.
  ! Cyclic Gram coordinate descent, including the upstream inner dual-gap test.
  subroutine graphical_cd(gram,row,alpha,coefs,max_iter,tol)
    real(dp), intent(in) :: gram(:,:),row(:),alpha,tol
    real(dp), intent(inout) :: coefs(:)
    integer, intent(in) :: max_iter
    real(dp) :: h(size(row)),old,value,change,largest,dual_norm,constant,gap,residual2,row_norm2,product
    integer :: it,j
    logical :: check_gap
    h=matmul(gram,coefs); row_norm2=sum(row**2)
    do it=1,max_iter
      change=0; largest=0
      do j=1,size(row)
        old=coefs(j); h=h-old*gram(:,j); value=row(j)-h(j)
        coefs(j)=sign(max(abs(value)-alpha,0._dp),value)/gram(j,j)
        h=h+coefs(j)*gram(:,j)
        change=max(change,abs(coefs(j)-old)); largest=max(largest,abs(coefs(j)))
      end do
      check_gap=largest==0.or.it==max_iter
      if(largest>0) check_gap=check_gap.or.change/largest<tol
      if(.not.check_gap) cycle
      product=dot_product(row,coefs); dual_norm=maxval(abs(row-h))
      residual2=row_norm2+dot_product(coefs,h)-2*product
      constant=1
      if(dual_norm>alpha) constant=alpha/dual_norm
      gap=0.5_dp*residual2*(1+constant**2)+alpha*sum(abs(coefs))-constant*row_norm2+constant*product
      if(gap<tol*row_norm2) exit
      ! A zero right-hand side has an exact zero solution and zero gap.
      if(row_norm2==0.and.all(coefs==0)) exit
    end do
  end subroutine

  function graphical_inverse(a) result(inverse)
    real(dp), intent(in) :: a(:,:)
    real(dp) :: inverse(size(a,1),size(a,2)),l(size(a,1),size(a,2)),unit(size(a,1))
    integer :: j
    l=cholesky_factor(a)
    do j=1,size(a,1)
      unit=0; unit(j)=1; inverse(:,j)=solve_upper(transpose(l),solve_lower(l,unit))
    end do
    inverse=0.5_dp*(inverse+transpose(inverse))
  end function

  subroutine graphical_diagnostics(emp_cov,precision,alpha,cost,gap)
    real(dp), intent(in) :: emp_cov(:,:),precision(:,:),alpha
    real(dp), intent(out) :: cost,gap
    real(dp) :: l(size(precision,1),size(precision,2)),penalty,trace_product,logdet
    integer :: j,p
    p=size(precision,1); l=cholesky_factor(precision); penalty=0; logdet=0
    do j=1,p
      penalty=penalty+sum(abs(precision(:j-1,j)))+sum(abs(precision(j+1:,j)))
      logdet=logdet+2*log(l(j,j))
    end do
    trace_product=sum(emp_cov*precision)
    gap=trace_product-p+alpha*penalty
    ! Match sklearn's shifted objective, including its Gaussian constant.
    cost=-logdet+trace_product+2*p*log(2*acos(-1._dp))+alpha*penalty
  end subroutine

  subroutine graphical_fit(self,x)
    class(graphical_lasso), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: empirical(:,:),centered(:,:),gram(:,:),coefs(:),row(:),history(:,:)
    real(dp) :: denominator
    real(dp) :: values(size(x,2)),vectors(size(x,2),size(x,2))
    integer :: p,n,i,j,it,indices(size(x,2)-1)
    call check_x(x); n=size(x,1); p=size(x,2)
    call require(n>=2.and.p>=2,'GraphicalLasso requires at least two samples and features')
    call require(all(ieee_is_finite([self%alpha,self%tol,self%enet_tol,self%eps])), &
      'nonfinite GraphicalLasso controls')
    call require(self%alpha>=0,'GraphicalLasso alpha must be nonnegative')
    call require(self%tol>0.and.self%enet_tol>0.and.self%eps>0,'invalid GraphicalLasso tolerances')
    call require(self%max_iter>0,'GraphicalLasso max_iter must be positive')
    call require(self%mode=='cd','GraphicalLasso supports mode cd only')
    call require(self%covariance=='empirical'.or.self%covariance=='precomputed', &
      'invalid GraphicalLasso covariance mode')
    if(self%covariance=='precomputed') then
      call require(n==p,'GraphicalLasso precomputed covariance must be square')
      call require(all(abs(x-transpose(x))<=1.e-10_dp*max(1._dp,abs(x),abs(transpose(x)))), &
        'GraphicalLasso precomputed covariance must be symmetric')
      empirical=0.5_dp*(x+transpose(x)); self%location_=spread(0._dp,1,p)
      call symmetric_eigen(empirical,values,vectors)
      call require(minval(values)>= -1.e-10_dp*maxval(abs(values)), &
        'GraphicalLasso covariance must be positive semidefinite')
    else
      self%location_=sum(x,dim=1)/n
      if(self%assume_centered) self%location_=0
      centered=x-spread(self%location_,1,n); empirical=matmul(transpose(centered),centered)/n
    end if
    do j=1,p
      call require(empirical(j,j)>0,'GraphicalLasso requires positive feature variances')
    end do
    self%n_iter_=0; self%converged_=.false.
    if(self%alpha==0) then
      self%covariance_=empirical; self%precision_=graphical_inverse(empirical)
      call graphical_diagnostics(empirical,self%precision_,self%alpha,self%objective_,self%dual_gap_)
      if(allocated(self%costs_)) deallocate(self%costs_)
      allocate(self%costs_(1,2)); self%costs_(1,:)=[self%objective_,self%dual_gap_]
      self%converged_=.true.; return
    end if
    self%covariance_=0.95_dp*empirical
    do j=1,p
      self%covariance_(j,j)=empirical(j,j)
    end do
    self%precision_=graphical_inverse(self%covariance_); allocate(history(self%max_iter,2))
    do it=1,self%max_iter
      do j=1,p
        indices=pack([(i,i=1,p)],[(i/=j,i=1,p)])
        gram=self%covariance_(indices,indices); row=empirical(indices,j)
        coefs=-self%precision_(indices,j)/(self%precision_(j,j)+1000*self%eps)
        call graphical_cd(gram,row,self%alpha,coefs,self%max_iter,self%enet_tol)
        denominator=self%covariance_(j,j)-dot_product(self%covariance_(indices,j),coefs)
        call require(denominator>0,'GraphicalLasso nonpositive Schur complement; ill-conditioned fit')
        self%precision_(j,j)=1/denominator
        self%precision_(indices,j)=-self%precision_(j,j)*coefs
        self%precision_(j,indices)=self%precision_(indices,j)
        self%covariance_(indices,j)=matmul(gram,coefs)
        self%covariance_(j,indices)=self%covariance_(indices,j)
      end do
      call require(all(ieee_is_finite(self%precision_)),'GraphicalLasso nonfinite precision')
      call graphical_diagnostics(empirical,self%precision_,self%alpha,self%objective_,self%dual_gap_)
      history(it,:)=[self%objective_,self%dual_gap_]
      if(abs(self%dual_gap_)<self%tol) then
        self%converged_=.true.; exit
      end if
    end do
    self%n_iter_=min(it,self%max_iter); self%costs_=history(:self%n_iter_,:)
  end subroutine

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
