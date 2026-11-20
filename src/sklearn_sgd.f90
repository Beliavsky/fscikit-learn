! SGDRegressor: sklearn 1.6.1 _stochastic_gradient.py and _sgd_fast.pyx.tp.
! BSD attribution: LICENSES/scikit-learn-COPYING; native dense updates.
module sklearn_sgd
  use sklearn_common, only: dp,require,check_x,check_xy,seed_rng,unique_labels,label_index,sigmoid
  use sklearn_metrics, only: r2_score,accuracy_score
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: sgd_regressor,sgd_classifier
  type :: sgd_regressor
    character(32) :: loss='squared_error',penalty='l2',learning_rate='invscaling'
    real(dp) :: alpha=1.e-4_dp,eta0=.01_dp,power_t=.25_dp,epsilon=.1_dp,tol=1.e-3_dp
    integer :: max_iter=1000,n_iter_no_change=5,average=0,random_state=0,n_iter_=0,n_features_in_=0
    logical :: fit_intercept=.true.,shuffle=.false.,converged_=.false.
    real(dp) :: intercept_=0,t_=1,loss_=0
    real(dp), allocatable :: coef_(:)
    real(dp), allocatable, private :: standard_coef(:),average_coef(:)
    real(dp), private :: standard_intercept=0,average_intercept=0
    logical, private :: classification=.false.
    integer, private :: fitted_average=0
  contains
    procedure :: fit=>sgd_fit
    procedure :: partial_fit=>sgd_partial
    procedure :: predict=>sgd_predict
    procedure :: score=>sgd_score
  end type
  ! Classification wrappers use one signed binary SGD head per OVR channel.
  type :: sgd_classifier
    character(32) :: loss='hinge',penalty='l2',learning_rate='invscaling'
    real(dp) :: alpha=1.e-4_dp,eta0=.01_dp,power_t=.25_dp,tol=1.e-3_dp,t_=1
    integer :: max_iter=1000,n_iter_no_change=5,average=0,random_state=0,n_iter_=0,n_features_in_=0
    logical :: fit_intercept=.true.,shuffle=.false.,converged_=.false.,class_weight_balanced=.false.
    integer, allocatable :: classes_(:),n_iter_per_class_(:)
    logical, allocatable :: converged_per_class_(:)
    real(dp), allocatable :: coef_(:,:),intercept_(:),loss_(:)
    type(sgd_regressor), allocatable, private :: heads(:)
  contains
    procedure :: fit=>sgdc_fit
    procedure :: partial_fit=>sgdc_partial
    procedure :: decision_function=>sgdc_decision
    procedure :: predict=>sgdc_predict
    procedure :: predict_proba=>sgdc_proba
    procedure :: score=>sgdc_score
  end type
contains
  subroutine sgd_controls(self)
    class(sgd_regressor), intent(in) :: self
    if(self%classification) then
      call require(any(self%loss==[character(32)::'hinge','squared_hinge','perceptron', &
        'log_loss','modified_huber']),'unknown SGD classification loss')
    else
      call require(any(self%loss==[character(32)::'squared_error','huber', &
        'epsilon_insensitive','squared_epsilon_insensitive']),'unknown SGD regression loss')
    end if
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
  subroutine sgd_loss(self,prediction,target,value,gradient)
    class(sgd_regressor), intent(in) :: self
    real(dp), intent(in) :: prediction,target
    real(dp), intent(out) :: value,gradient
    real(dp) :: excess,residual,margin,e
    residual=prediction-target; margin=prediction*target
    select case(self%loss)
    case('hinge','perceptron')
      excess=1
      if(self%loss=='perceptron') excess=0
      value=max(0._dp,excess-margin); gradient=0
      if(margin<=excess) gradient=-target
    case('squared_hinge')
      excess=max(0._dp,1-margin); value=excess**2; gradient=-2*target*excess
    case('modified_huber')
      if(margin>=1) then
        value=0; gradient=0
      else if(margin>= -1) then
        value=(1-margin)**2; gradient=-2*target*(1-margin)
      else
        value=-4*margin; gradient=-4*target
      end if
    case('log_loss')
      if(margin>=0) then
        e=exp(-margin); value=log(1+e); gradient=-target*e/(1+e)
      else
        e=exp(margin); value=-margin+log(1+e); gradient=-target/(1+e)
      end if
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
        call sgd_loss(self,dot_product(self%standard_coef,x(i,:))+self%standard_intercept,y(i),value,gradient)
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
  subroutine sgdc_configure(self,head)
    class(sgd_classifier), intent(in) :: self
    type(sgd_regressor), intent(inout) :: head
    head%classification=.true.; head%loss=self%loss; head%penalty=self%penalty; head%learning_rate=self%learning_rate
    head%alpha=self%alpha; head%eta0=self%eta0; head%power_t=self%power_t; head%tol=self%tol
    head%max_iter=self%max_iter; head%n_iter_no_change=self%n_iter_no_change; head%average=self%average
    head%random_state=self%random_state; head%fit_intercept=self%fit_intercept; head%shuffle=self%shuffle
    call sgd_controls(head)
  end subroutine
  subroutine sgdc_fit(self,x,y,sample_weight,class_weight)
    class(sgd_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:),class_weight(:)
    call sgdc_train(self,x,y,.false.,sample_weight,class_weight)
  end subroutine
  subroutine sgdc_partial(self,x,y,classes,sample_weight,class_weight)
    class(sgd_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    integer, intent(in), optional :: classes(:)
    real(dp), intent(in), optional :: sample_weight(:),class_weight(:)
    call sgdc_train(self,x,y,.true.,sample_weight,class_weight,classes)
  end subroutine
  subroutine sgdc_train(self,x,y,incremental,sample_weight,class_weight,classes)
    class(sgd_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    logical, intent(in) :: incremental
    real(dp), intent(in), optional :: sample_weight(:),class_weight(:)
    integer, intent(in), optional :: classes(:)
    type(sgd_regressor) :: settings
    integer, allocatable :: labels(:),encoded(:)
    real(dp), allocatable :: weights(:),cw(:),target(:),cost(:)
    integer :: n,p,k,channels,c,positive,i
    real(dp) :: start_t
    logical :: fresh
    call sgdc_configure(self,settings); call check_xy(x,size(y)); n=size(y); p=size(x,2)
    fresh=.not.incremental
    if(incremental) fresh=.not.allocated(self%heads)
    if(fresh) then
      if(incremental) then
        call require(present(classes),'SGDClassifier first partial_fit requires classes')
        labels=unique_labels(classes)
        call require(size(labels)==size(classes),'SGDClassifier classes must be distinct')
      else
        labels=unique_labels(y)
      end if
      call require(size(labels)>=2,'SGDClassifier requires at least two classes')
    else
      labels=self%classes_
      call require(p==self%n_features_in_,'SGDClassifier feature count mismatch')
      call require(self%average==self%heads(1)%fitted_average,'SGD averaging cannot change during partial_fit')
      if(present(classes)) then
        encoded=unique_labels(classes)
        call require(size(encoded)==size(labels),'SGDClassifier classes changed')
        call require(all(encoded==labels),'SGDClassifier classes changed')
      end if
    end if
    k=size(labels)
    call require(.not.(incremental.and.self%average>0.and.self%fit_intercept), &
      'averaged SGDClassifier partial_fit requires fit_intercept=false')
    encoded=[(label_index(labels,y(i)),i=1,n)]
    call require(all(encoded>0),'SGDClassifier target is outside known classes')
    weights=spread(1._dp,1,n)
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'SGD sample weight count mismatch')
      call require(all(ieee_is_finite(sample_weight)),'nonfinite SGD sample weights')
      call require(all(sample_weight>=0),'negative SGD sample weights'); weights=sample_weight
    end if
    cw=spread(1._dp,1,k)
    call require(.not.(incremental.and.self%class_weight_balanced),'balanced SGD class weights require fit')
    if(present(class_weight)) then
      call require(.not.self%class_weight_balanced,'choose balanced or explicit SGD class weights')
      call require(size(class_weight)==k,'SGD class weight count mismatch')
      call require(all(ieee_is_finite(class_weight)),'nonfinite SGD class weights')
      call require(all(class_weight>0),'SGD class weights must be positive'); cw=class_weight
    else if(self%class_weight_balanced) then
      do c=1,k
        cw(c)=real(n,dp)/(k*real(count(encoded==c),dp))
      end do
    end if
    channels=k
    if(k==2) channels=1
    if(fresh) then
      if(allocated(self%heads)) deallocate(self%heads)
      allocate(self%heads(channels)); self%classes_=labels; self%n_features_in_=p; self%t_=1
      do c=1,channels
        call sgdc_configure(self,self%heads(c)); call sgd_initialize(self%heads(c),p)
      end do
    end if
    self%coef_=spread(spread(0._dp,1,channels),2,p)
    self%intercept_=spread(0._dp,1,channels); self%loss_=self%intercept_
    self%n_iter_per_class_=spread(0,1,channels); self%converged_per_class_=spread(.false.,1,channels)
    start_t=self%t_
    do c=1,channels
      positive=c
      if(k==2) positive=2
      target=merge(1._dp,-1._dp,encoded==positive)
      if(k==2) then
        cost=weights*cw(encoded)
      else
        cost=weights*merge(cw(c),1._dp,encoded==c)
      end if
      call sgdc_configure(self,self%heads(c)); self%heads(c)%t_=start_t
      if(incremental) then
        call sgd_run(self%heads(c),x,target,1,.false.,cost)
      else
        call sgd_run(self%heads(c),x,target,self%max_iter,.true.,cost)
      end if
      self%n_iter_per_class_(c)=self%heads(c)%n_iter_; self%converged_per_class_(c)=self%heads(c)%converged_
      self%loss_(c)=self%heads(c)%loss_
    end do
    self%n_iter_=maxval(self%n_iter_per_class_); self%t_=start_t+real(self%n_iter_,dp)*n
    self%converged_=all(self%converged_per_class_)
    do c=1,channels
      if(self%average>0.and.self%t_-1>=self%average) then
        self%coef_(c,:)=self%heads(c)%average_coef; self%intercept_(c)=self%heads(c)%average_intercept
      else
        self%coef_(c,:)=self%heads(c)%standard_coef; self%intercept_(c)=self%heads(c)%standard_intercept
      end if
    end do
  end subroutine
  function sgdc_decision(self,x) result(scores)
    class(sgd_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: scores(:,:)
    call require(allocated(self%heads),'SGDClassifier is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'SGDClassifier feature count mismatch')
    scores=matmul(x,transpose(self%coef_))+spread(self%intercept_,1,size(x,1))
  end function
  function sgdc_predict(self,x) result(y)
    class(sgd_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, allocatable :: y(:)
    real(dp), allocatable :: scores(:,:)
    scores=self%decision_function(x)
    if(size(self%classes_)==2) then
      y=self%classes_(merge(2,1,scores(:,1)>0))
    else
      y=self%classes_(maxloc(scores,dim=2))
    end if
  end function
  function sgdc_proba(self,x) result(prob)
    class(sgd_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:),scores(:,:)
    real(dp) :: total
    integer :: i,c,k
    scores=self%decision_function(x); k=size(self%classes_)
    call require(self%loss=='log_loss'.or.self%loss=='modified_huber','SGD loss does not provide probabilities')
    allocate(prob(size(x,1),k))
    do i=1,size(x,1)
      do c=1,size(scores,2)
        if(self%loss=='log_loss') then
          prob(i,c)=sigmoid(scores(i,c))
        else
          prob(i,c)=(min(1._dp,max(-1._dp,scores(i,c)))+1)/2
        end if
      end do
      if(k==2) then
        prob(i,2)=prob(i,1); prob(i,1)=1-prob(i,2)
      else
        total=sum(prob(i,:))
        if(total>0) then
          prob(i,:)=prob(i,:)/total
        else
          prob(i,:)=1._dp/k
        end if
      end if
    end do
  end function
  real(dp) function sgdc_score(self,x,y) result(value)
    class(sgd_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    value=accuracy_score(y,self%predict(x))
  end function
end module
