! Multi-task coordinate descent contracts: sklearn 1.6.1 _coordinate_descent.py/_cd_fast.pyx.
! BSD-3-Clause attribution retained in LICENSES/scikit-learn-COPYING.
module sklearn_multitask
  use sklearn_common, only: dp,require,check_x,check_xy
  use sklearn_metrics, only: r2_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: multi_task_elastic_net,multi_task_lasso
  type :: multi_task_elastic_net
    real(dp) :: alpha=1,l1_ratio=.5_dp,tol=1.e-4_dp
    integer :: max_iter=1000,n_iter_=0,n_features_in_=0
    logical :: fit_intercept=.true.,warm_start=.false.,converged_=.false.
    real(dp) :: dual_gap_=0,eps_=0
    real(dp), allocatable :: coef_(:,:),intercept_(:)
  contains
    procedure :: fit=>multitask_fit
    procedure :: predict=>multitask_predict
    procedure :: score=>multitask_score
  end type
  type, extends(multi_task_elastic_net) :: multi_task_lasso
  contains
    procedure :: fit=>multitask_lasso_fit
  end type
contains
  subroutine multitask_fit(self,x,y)
    class(multi_task_elastic_net), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    call require(ieee_is_finite(self%l1_ratio),'nonfinite MultiTask l1_ratio')
    call require(self%l1_ratio>0.and.self%l1_ratio<=1,'MultiTask l1_ratio must be in (0,1]')
    call multitask_solve(self,x,y,self%l1_ratio)
  end subroutine

  subroutine multitask_lasso_fit(self,x,y)
    class(multi_task_lasso), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    call multitask_solve(self,x,y,1._dp)
  end subroutine

  subroutine multitask_solve(self,x,y,mix)
    class(multi_task_elastic_net), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:,:),mix
    real(dp), allocatable :: a(:,:),b(:,:),xm(:),ym(:),residual(:,:),norms(:),old(:),z(:),gradient(:,:)
    real(dp) :: l1,l2,znorm,change,largest,dualnorm,scale,gap,rnorm2,l21
    integer :: n,p,t,j,iteration
    call check_xy(x,size(y,1)); call check_x(y)
    call require(ieee_is_finite(self%alpha),'nonfinite MultiTask alpha')
    call require(self%alpha>0,'MultiTask alpha must be positive')
    call require(ieee_is_finite(self%tol),'nonfinite MultiTask tolerance')
    call require(self%tol>=0,'MultiTask tolerance must be nonnegative')
    call require(self%max_iter>0,'MultiTask max_iter must be positive')
    n=size(x,1); p=size(x,2); t=size(y,2)
    xm=sum(x,dim=1)/real(n,dp); ym=sum(y,dim=1)/real(n,dp)
    if(.not.self%fit_intercept) then
      xm=0; ym=0
    end if
    a=x-spread(xm,1,n); b=y-spread(ym,1,n)
    if(self%warm_start.and.allocated(self%coef_)) then
      call require(size(self%coef_,1)==t.and.size(self%coef_,2)==p,'MultiTask warm-start dimensions changed')
    else
      self%coef_=spread(spread(0._dp,1,t),2,p)
    end if
    residual=b-matmul(a,transpose(self%coef_)); norms=sum(a*a,dim=1)
    allocate(old(t),z(t)); l1=self%alpha*mix*n; l2=self%alpha*(1-mix)*n
    self%eps_=self%tol*sum(b*b); self%converged_=.false.
    do iteration=1,self%max_iter
      change=0; largest=0
      do j=1,p
        if(norms(j)==0) cycle
        old=self%coef_(:,j)
        residual=residual+spread(a(:,j),2,t)*spread(old,1,n)
        z=matmul(a(:,j),residual); znorm=sqrt(sum(z*z))
        self%coef_(:,j)=0
        if(znorm>l1) self%coef_(:,j)=z*((1-l1/znorm)/(norms(j)+l2))
        residual=residual-spread(a(:,j),2,t)*spread(self%coef_(:,j),1,n)
        change=max(change,maxval(abs(self%coef_(:,j)-old)))
        largest=max(largest,maxval(abs(self%coef_(:,j))))
      end do
      if(largest==0.or.change<self%tol*largest.or.iteration==self%max_iter) then
        gradient=matmul(transpose(a),residual)-l2*transpose(self%coef_)
        dualnorm=maxval(sqrt(sum(gradient*gradient,dim=2)))
        scale=1
        if(dualnorm>l1) scale=l1/dualnorm
        rnorm2=sum(residual*residual); l21=sum(sqrt(sum(self%coef_**2,dim=1)))
        gap=.5_dp*rnorm2*(1+scale**2)+l1*l21-scale*sum(residual*b) &
          +.5_dp*l2*(1+scale**2)*sum(self%coef_**2)
        self%dual_gap_=gap/real(n,dp)
        if(gap<=self%eps_) then
          self%converged_=.true.; exit
        end if
      end if
    end do
    self%n_iter_=min(iteration,self%max_iter); self%n_features_in_=p
    self%intercept_=ym-matmul(self%coef_,xm)
  end subroutine

  function multitask_predict(self,x) result(y)
    class(multi_task_elastic_net), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:,:)
    call require(allocated(self%coef_),'MultiTask model is not fitted')
    call check_x(x)
    call require(size(x,2)==self%n_features_in_,'MultiTask feature count mismatch')
    y=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function

  real(dp) function multitask_score(self,x,y) result(score)
    class(multi_task_elastic_net), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:,:)
    real(dp), allocatable :: prediction(:,:)
    integer :: j
    prediction=self%predict(x); call check_xy(y,size(x,1))
    call require(size(y,2)==size(prediction,2),'MultiTask target count mismatch')
    score=0
    do j=1,size(y,2)
      score=score+r2_score(y(:,j),prediction(:,j))
    end do
    score=score/real(size(y,2),dp)
  end function
end module
