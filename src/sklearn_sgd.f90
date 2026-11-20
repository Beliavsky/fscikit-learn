! SGDRegressor: sklearn 1.6.1 _stochastic_gradient.py and _sgd_fast.pyx.tp.
! BSD attribution: LICENSES/scikit-learn-COPYING; native dense updates.
module sklearn_sgd
  use sklearn_common, only: dp,require,check_x,check_xy,seed_rng
  use sklearn_metrics, only: r2_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: sgd_regressor
  type :: sgd_regressor
    character(32) :: loss='squared_error',penalty='l2',learning_rate='invscaling'
    real(dp) :: alpha=1.e-4_dp,eta0=.01_dp,power_t=.25_dp,epsilon=.1_dp,tol=1.e-3_dp
    integer :: max_iter=1000,n_iter_no_change=5,average=0,random_state=0,n_iter_=0,n_features_in_=0
    logical :: fit_intercept=.true.,shuffle=.false.,converged_=.false.
    real(dp) :: intercept_=0,t_=1,loss_=0
    real(dp), allocatable :: coef_(:)
    real(dp), allocatable, private :: standard_coef(:),average_coef(:)
    real(dp), private :: standard_intercept=0,average_intercept=0
    integer, private :: fitted_average=0
  contains
    procedure :: fit=>sgd_fit
    procedure :: partial_fit=>sgd_partial
    procedure :: predict=>sgd_predict
    procedure :: score=>sgd_score
  end type
contains
  subroutine sgd_controls(self)
    class(sgd_regressor), intent(in) :: self
    call require(any(self%loss==[character(32)::'squared_error','huber', &
      'epsilon_insensitive','squared_epsilon_insensitive']),'unknown SGD regression loss')
    call require(self%penalty=='l2'.or.self%penalty=='none','unsupported SGD penalty')
    call require(self%learning_rate=='constant'.or.self%learning_rate=='invscaling','unsupported SGD learning rate')
    call require(all(ieee_is_finite([self%alpha,self%eta0,self%power_t,self%epsilon,self%tol])),'nonfinite SGD controls')
    call require(self%alpha>=0,'SGD alpha must be nonnegative')
    call require(self%eta0>0,'SGD eta0 must be positive')
    call require(self%power_t>=0,'SGD power_t must be nonnegative')
    call require(self%epsilon>=0,'SGD epsilon must be nonnegative')
    call require(self%tol>=0.or.self%tol== -1,'SGD tol must be nonnegative or -1')
    call require(self%max_iter>0.and.self%n_iter_no_change>0,'invalid SGD iteration controls')
    call require(self%average>=0,'SGD average must be nonnegative')
  end subroutine
  subroutine sgd_initialize(self,p)
    class(sgd_regressor), intent(inout) :: self
    integer, intent(in) :: p
    self%standard_coef=spread(0._dp,1,p); self%average_coef=self%standard_coef; self%coef_=self%standard_coef
    self%standard_intercept=0; self%average_intercept=0; self%intercept_=0; self%t_=1
    self%n_features_in_=p; self%fitted_average=self%average
  end subroutine
  subroutine sgd_fit(self,x,y,sample_weight)
    class(sgd_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    call sgd_controls(self); call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call sgd_initialize(self,size(x,2)); call sgd_run(self,x,y,self%max_iter,.true.,sample_weight)
  end subroutine
  subroutine sgd_partial(self,x,y,sample_weight)
    class(sgd_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    call sgd_controls(self); call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    if(.not.allocated(self%standard_coef)) call sgd_initialize(self,size(x,2))
    call require(size(x,2)==self%n_features_in_,'SGD feature count mismatch')
    call require(self%average==self%fitted_average,'SGD averaging cannot change during partial_fit')
    call sgd_run(self,x,y,1,.false.,sample_weight)
  end subroutine
  subroutine sgd_loss(self,residual,value,gradient)
    class(sgd_regressor), intent(in) :: self
    real(dp), intent(in) :: residual
    real(dp), intent(out) :: value,gradient
    real(dp) :: excess
    select case(self%loss)
    case('squared_error'); value=.5_dp*residual**2; gradient=residual
    case('huber')
      if(abs(residual)<=self%epsilon) then
        value=.5_dp*residual**2; gradient=residual
      else
        value=self%epsilon*(abs(residual)-.5_dp*self%epsilon); gradient=sign(self%epsilon,residual)
      end if
    case('epsilon_insensitive')
      excess=max(0._dp,abs(residual)-self%epsilon); value=excess; gradient=0
      if(excess>0) gradient=sign(1._dp,residual)
    case('squared_epsilon_insensitive')
      excess=max(0._dp,abs(residual)-self%epsilon); value=excess**2; gradient=2*sign(excess,residual)
    end select
  end subroutine
  subroutine sgd_run(self,x,y,epochs,stopping,sample_weight)
    class(sgd_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    integer, intent(in) :: epochs
    logical, intent(in) :: stopping
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: weights(size(y)),sumloss,best_loss,value,gradient,eta,update,count,coin
    integer :: order(size(y)),n,i,j,it,pick,tmp,stale
    n=size(y); weights=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'SGD sample weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'nonfinite SGD sample weights')
      call require(all(sample_weight>=0),'negative SGD sample weights'); weights=sample_weight
    end if
    order=[(i,i=1,n)]; best_loss=huge(1._dp); stale=0; self%converged_=.false.
    if(self%shuffle) call seed_rng(self%random_state)
    do it=1,epochs
      if(self%shuffle) then
        do j=n,2,-1
          call random_number(coin); pick=min(j,1+int(coin*j)); tmp=order(j); order(j)=order(pick); order(pick)=tmp
        end do
      end if
      sumloss=0
      do j=1,n
        i=order(j)
        call sgd_loss(self,dot_product(self%standard_coef,x(i,:))+self%standard_intercept-y(i),value,gradient)
        sumloss=sumloss+value; eta=self%eta0
        if(self%learning_rate=='invscaling') eta=eta/self%t_**self%power_t
        gradient=max(-1.e12_dp,min(1.e12_dp,gradient)); update=-eta*gradient*weights(i)
        if(self%penalty=='l2') self%standard_coef=self%standard_coef*max(0._dp,1-eta*self%alpha)
        self%standard_coef=self%standard_coef+update*x(i,:)
        if(self%fit_intercept) self%standard_intercept=self%standard_intercept+update
        if(self%average>0.and.self%t_>=self%average) then
          count=self%t_-self%average+1
          self%average_coef=self%average_coef+(self%standard_coef-self%average_coef)/count
          self%average_intercept=self%average_intercept+(self%standard_intercept-self%average_intercept)/count
        end if
        self%t_=self%t_+1
      end do
      call require(all(ieee_is_finite(self%standard_coef)).and.ieee_is_finite(self%standard_intercept), &
        'SGD fit overflow; scale the inputs')
      if(stopping.and.self%tol>=0) then
        if(sumloss>best_loss-self%tol*n) then
          stale=stale+1
        else
          stale=0
        end if
        best_loss=min(best_loss,sumloss)
        if(stale>=self%n_iter_no_change) then
          self%converged_=.true.; exit
        end if
      end if
    end do
    self%n_iter_=min(it,epochs); self%loss_=sumloss/n
    self%coef_=self%standard_coef; self%intercept_=self%standard_intercept
    if(self%average>0.and.self%t_-1>=self%average) then
      self%coef_=self%average_coef; self%intercept_=self%average_intercept
    end if
  end subroutine
  function sgd_predict(self,x) result(y)
    class(sgd_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: y(:)
    call require(allocated(self%coef_),'SGDRegressor is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'SGD feature count mismatch')
    y=matmul(x,self%coef_)+self%intercept_
  end function
  real(dp) function sgd_score(self,x,y) result(value)
    class(sgd_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    value=r2_score(y,self%predict(x))
  end function
end module
